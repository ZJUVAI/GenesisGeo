"""Prepare exact models and certify guarded polynomial implications."""

import ast
from collections import Counter, OrderedDict
from functools import lru_cache
import math
from time import perf_counter

import sympy as sp
from flint import fmpq, fmpq_mpoly_ctx

from .exact_geometry import parse_statement, substitute_coordinates
from .native_basis import NativeBasis, NativeBasisLimit, integer_terms, rational_terms
from .orientation_conditions import (
    nonzero_consequences,
    side_nonzero_consequences,
    side_strict_guards,
)
from .predicate_encodings import EXPLICIT_NONZERO, encode
from .singular import search
from .strict_certificate import build_model
from ..rules import split_rule_text


def parse_polynomial(text, variables):
    """Parse native CAS output without eval/sympify of executable strings."""
    tree = ast.parse(text.replace("^", "**"), mode="eval")
    allowed = {str(v): v for v in variables}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.Name) and node.id in allowed:
            return allowed[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div) and b.is_Rational and b != 0:
                return a / b
            if isinstance(node.op, ast.Pow) and b.is_Integer and 0 <= b <= 512:
                return a ** int(b)
        raise ValueError("invalid polynomial certificate syntax")

    stack = [(tree.body, 1)]
    terms = []
    while stack:
        node, sign = stack.pop()
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            stack.append((node.right, -sign if isinstance(node.op, ast.Sub) else sign))
            stack.append((node.left, sign))
        else:
            terms.append(sign * visit(node))
    return sp.expand(sp.Add(*terms))


def _check_reduction(native, goal, factors, used, powers, exponent):
    if exponent not in (1, 2) or len(used) != len(powers):
        return False
    if any(not 0 <= i < len(factors) or factors[i] == 0 for i in used) or any(
        k < 0 for k in powers
    ):
        return False
    target = native.convert(goal) ** exponent
    for i, power in zip(used, powers):
        target *= native.convert(factors[i]) ** power
    return not native.remainder(target)


def check_membership(inputs, basis, goal, factors, used, powers, exponent, variables):
    """Verify a new CAS basis against the original equations."""
    import flint

    native = _reuse_verified_basis(basis, variables)
    original = flint.fmpz_mpoly_vec(
        [native.convert(p) for p in inputs if p != 0],
        native.context,
    )
    if not native.basis.is_groebner(original):
        return False
    if any(native.remainder(native.convert(p)) for p in inputs if p != 0):
        return False
    return _check_reduction(native, goal, factors, used, powers, exponent)


def _small_basis(inputs, variables):
    """Compute a strictly bounded native basis for small polynomial systems."""
    if len(inputs) > 4 or len(variables) > 16:
        return None
    if sum(len(sp.Add.make_args(p)) for p in inputs) > 96:
        return None
    try:
        return NativeBasis(inputs, variables, limits=(32, 512, 128))
    except NativeBasisLimit:
        return None


def _goal_remainders(native, goal):
    if len(sp.Add.make_args(goal)) > 128:
        return None
    first = native.remainder(native.convert(goal))
    if len(first) > 16 or first.total_degree() > 12:
        return None
    return first, native.remainder(first**2) if first else first


def _native_domains_hold(native, factors):
    return bool(native.remainder(native.convert(1))) and all(
        native.remainder(native.convert(factor)) for factor in factors
    )


def _reuse_verified_basis(initial_basis, variables):
    import flint

    native = NativeBasis([], variables)
    native.basis = flint.fmpz_mpoly_vec(
        [native.convert(p) for p in initial_basis if p != 0],
        native.context,
    )
    native.empty = not tuple(native.basis)
    return native


_prepared_contexts = OrderedDict()
_verified_bases = OrderedDict()
_AFFINE_PREDICATES = frozenset({"coll", "ncoll", "para", "midp", "diff"})


def _compile_model(axiom, g):
    """Compile only symbolic equations; each goal still needs its own certificate."""
    key = None
    _, premises, goals, *guards = axiom.split("|")
    facts = tuple(
        p.strip()
        for text in (premises, *guards[:1])
        for p in text.split(",")
        if p.strip()
    )
    parsed = [parse_statement(p) for p in facts]
    # Without explicit nonzero assumptions, the collapsed-model check
    # depends on the goal and must run independently.
    if any(kind in EXPLICIT_NONZERO for kind, _, _ in parsed):
        parsed.extend(parse_statement(p.strip()) for p in goals.split(",") if p.strip())
        names = frozenset(p for _, points, _ in parsed for p in points)
        affine = all(kind in _AFFINE_PREDICATES for kind, _, _ in parsed)
        key = (facts, names, affine)
    if key is not None and key in _prepared_contexts:
        _prepared_contexts.move_to_end(key)
        model, context = _prepared_contexts[key]
    else:
        model = build_model(axiom)
        if "coordinates" not in model:
            return model, None
        context = _encode_context(model)
        if key is not None:
            _prepared_contexts[key] = (model, context)
            if len(_prepared_contexts) > 8:
                _prepared_contexts.popitem(last=False)
    substituted_coordinates, cas_variables, integral, inputs, required_factors = context
    goal_equations, _ = encode(g, substituted_coordinates, premise=False)
    if not goal_equations:
        return model, None
    # Over real coordinates, zero sum of squares requires every component zero.
    goal = integral(
        goal_equations[0]
        if len(goal_equations) == 1
        else sum(e * e for e in goal_equations)
    )
    return model, (
        substituted_coordinates,
        cas_variables,
        integral,
        inputs,
        goal,
        required_factors,
    )


def _encode_context(model):
    coordinates = model["coordinates"]
    variables = model["variables"]
    substitutions = model["substitutions"]
    substituted_coordinates = {
        name: tuple(substitute_coordinates(c, substitutions) for c in pair)
        for name, pair in coordinates.items()
    }
    cas_variables = list(
        sp.symbols(" ".join("z" + str(i + 1) for i in range(len(variables))))
        if len(variables) > 1
        else ([sp.Symbol("z1")] if variables else [sp.Symbol("z0")])
    )
    renaming = dict(zip(variables, cas_variables))

    def integral(expression):
        # Equations/factors are already substituted by the model compiler;
        # candidate predicates use substituted coordinates directly. Avoid a
        # full symbolic substitution walk for every candidate polynomial.
        e = sp.expand(sp.sympify(expression).xreplace(renaming))
        return _primitive_polynomial(e, cas_variables)

    inputs = [integral(e) for e in model["equations"] if e != 0] or [sp.Integer(0)]
    required_factors = list(
        dict.fromkeys(integral(f) for f in model["explicit_nonzero_factors"])
    )
    return substituted_coordinates, cas_variables, integral, inputs, required_factors


def _primitive_polynomial(expression, variables):
    terms = integer_terms(expression, variables)
    if not terms:
        return sp.S.Zero
    content = math.gcd(*terms.values())
    if terms[max(terms)] < 0:
        content = -content
    return sp.Add(
        *(
            sp.Integer(coefficient // content)
            * sp.Mul(
                *(
                    variable**exponent
                    for variable, exponent in zip(variables, monomial)
                    if exponent
                )
            )
            for monomial, coefficient in terms.items()
        )
    )


def _guard_coordinates(coordinates, variables, renamed):
    """Move substituted coordinates to FLINT once before testing guard polynomials."""
    context = fmpq_mpoly_ctx.get(tuple(map(str, variables)), "degrevlex")

    @lru_cache(maxsize=None)
    def convert(expression):
        expression = expression.xreplace(renamed)
        return context.from_dict(
            {
                monomial: fmpq(int(value.p), int(value.q))
                for monomial, value in rational_terms(expression, variables).items()
                if value
            }
        )

    return {
        name: tuple(map(convert, pair)) for name, pair in coordinates.items()
    }, convert


def _factor_key(polynomial):
    return str(polynomial / polynomial.leading_coefficient())


@lru_cache(maxsize=1)
def _model(rule_text, required):
    premises, goal = split_rule_text(rule_text)
    axiom = "candidate|" + ", ".join([*premises, *required]) + "|" + goal
    return _compile_model(axiom, goal)


def _optional_factors(
    candidates, coordinates, variables, renaming, required, affine_frame
):
    priority = {
        "npara": 0,
        "ncoll": 1,
        "diff": 2,
        "nperp": 3,
        "sameclock": 4,
        "sameside": 4,
    }
    candidates = [
        guard
        for guard in candidates
        if not (affine_frame and guard.startswith("nperp "))
    ]
    if not candidates:
        return [], []
    native, convert = _guard_coordinates(coordinates, variables, renaming)
    seen = {_factor_key(poly) for factor in required if (poly := convert(factor))}
    factors, guards = [], []
    for guard in sorted(
        set(candidates), key=lambda text: (priority[text.split()[0]], text)
    ):
        kind = guard.split()[0]
        if kind == "sameclock":
            expressions = nonzero_consequences(guard, native)
        elif kind == "sameside":
            expressions = side_nonzero_consequences(guard, native)
        else:
            _, obligations = encode(guard, native, premise=True)
            last = obligations[-1]
            expressions = [
                sum(x * x for x in last) if isinstance(last, tuple) else last
            ]
        for polynomial in expressions:
            if polynomial and polynomial.total_degree() > 0:
                key = _factor_key(polynomial)
                if key not in seen:
                    seen.add(key)
                    factors.append(polynomial)
                    guards.append(guard)
    return factors, guards


def _strict_domains(selected):
    for guard in selected:
        kind, *points = guard.split()
        if kind == "sameclock":
            yield "ncoll " + " ".join(sorted(points[:3]))
            yield "ncoll " + " ".join(sorted(points[3:]))
        elif kind == "sameside":
            yield from side_strict_guards(guard)


def prove(rule_text, required, propose, *, seconds, timings):
    started = perf_counter()
    model, encoded = _model(rule_text, tuple(required))
    timings["model"] = perf_counter() - started
    if "coordinates" not in model:
        return "rejected_model", (), None
    if encoded is None:
        return "unsupported_goal", (), None

    coordinates, variables, _, inputs, goal, required_factors = encoded
    renaming = dict(zip(model["variables"], variables))
    occurrences = Counter(
        symbol for equation in inputs for symbol in equation.free_symbols
    )
    variables = sorted(variables, key=occurrences.__getitem__)
    key = (tuple(variables), tuple(inputs))
    cached = _verified_bases.get(key)
    if cached is not None:
        _verified_bases.move_to_end(key)
    basis, native = (
        cached if cached is not None else ((), _small_basis(inputs, variables))
    )
    if native is not None:
        remainders = _goal_remainders(native, goal)
        if (
            remainders is not None
            and not remainders[1]
            and _native_domains_hold(native, required_factors)
        ):
            if cached is None:
                _remember(
                    key, tuple(native.expression(p) for p in native.basis), native
                )
            timings["proof"] = perf_counter() - started - timings["model"]
            return "certified", (), "flint"

    premises, conclusion = split_rule_text(rule_text)
    facts = [*premises, *required, conclusion]
    names = {name for fact in facts for name in fact.split()[1:]}
    affine_frame = len(model.get("anchor") or ()) == 3 and all(
        fact.split()[0] in _AFFINE_PREDICATES for fact in facts
    )
    candidates = [
        guard
        for guard in propose()
        if guard not in facts and set(guard.split()[1:]) <= names
    ]
    optional, guards = _optional_factors(
        candidates,
        coordinates,
        variables,
        renaming,
        required_factors,
        affine_frame,
    )
    factors = required_factors + optional
    status, fields = search(
        inputs, variables, goal, factors, len(required_factors), basis, seconds
    )
    if status != "certified":
        timings["proof"] = perf_counter() - started - timings["model"]
        return status, (), "singular"

    used = [int(i) - 1 for i in fields["USED"].split(",") if int(i) > 0]
    powers = list(map(int, fields["POWERS"].split(","))) if used else []
    exponent = int(fields["EXPONENT"])
    if not basis:
        basis = tuple(
            parse_polynomial(fields[f"GB_{i + 1}"], variables)
            for i in range(int(fields["GB_SIZE"]))
        )
        verified = check_membership(
            inputs, basis, goal, factors, used, powers, exponent, variables
        )
        if verified:
            _remember(key, basis, _reuse_verified_basis(basis, variables))
    else:
        verified = _check_reduction(native, goal, factors, used, powers, exponent)
    timings["proof"] = perf_counter() - started - timings["model"]
    if not verified:
        return "rejected_certificate", (), "singular"
    selected = [
        guards[index - len(required_factors)]
        for index in used
        if index >= len(required_factors)
    ]
    return "certified", (*selected, *_strict_domains(selected)), "singular"


def _remember(key, basis, native):
    if basis:
        _verified_bases[key] = (basis, native)
        if len(_verified_bases) > 8:
            _verified_bases.popitem(last=False)

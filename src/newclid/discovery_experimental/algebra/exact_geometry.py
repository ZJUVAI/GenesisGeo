"""Exact coordinate parsing, substitutions, and linear polynomial elimination."""

from functools import lru_cache

import sympy as sp


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def parse_statement(statement):
    predicate, *names = statement.split()
    ratio = None
    if predicate == "aconst":
        value = names.pop()
        if value.endswith("o"):
            ratio = sp.Rational(value[:-1]) / 180
        else:
            ratio = sp.Rational(value.replace("pi/", "/"))
    elif predicate == "rconst":
        ratio = sp.Rational(names.pop())
        if ratio <= 0:
            raise ValueError("distance ratio must be positive")
    return predicate, names, ratio


@lru_cache(maxsize=2)
def _composed_coordinates(substitutions):
    # Build the triangular map backwards so replacement expressions already
    # include later eliminations. xreplace applies the map simultaneously.
    mapping = {}
    for variable, replacement in reversed(substitutions):
        mapping[variable] = sp.sympify(replacement).xreplace(mapping)
    return mapping


def substitute_coordinates(expression, substitutions):
    """Apply triangular coordinate assignments in elimination order."""
    return sp.expand(
        sp.sympify(expression).xreplace(_composed_coordinates(tuple(substitutions)))
    )


def _eliminate_affine_polynomials(equations, variables):
    """Use native rational composition while preserving symbolic pivot order."""
    import flint

    from .native_basis import rational_terms

    active = list(variables)
    if not active or not any(
        all(term == 1 or term.is_Symbol for term in eq.as_coefficients_dict())
        and eq.free_symbols.intersection(active)
        for eq in equations
    ):
        return equations, active, []
    variables = tuple(active)
    indexes = {v: i for i, v in enumerate(variables)}
    context = flint.fmpq_mpoly_ctx.get(tuple(map(str, variables)), "degrevlex")
    generators = list(context.gens())

    def encode(expr):
        return context.from_dict(
            {
                monomial: flint.fmpq(int(coefficient.p), int(coefficient.q))
                for monomial, coefficient in rational_terms(expr, variables).items()
                if coefficient
            }
        )

    def decode(poly):
        return sp.Add(
            *(
                sp.Rational(int(coefficient.p), int(coefficient.q))
                * sp.Mul(*(v**power for v, power in zip(variables, monomial) if power))
                for monomial, coefficient in poly.to_dict().items()
            )
        )

    def inspect_polynomial(poly):
        equation = decode(poly) if poly.total_degree() == 1 else None
        return (
            poly,
            poly.degrees(),
            equation,
            sp.count_ops(equation) if equation is not None else None,
        )

    try:
        polys = [inspect_polynomial(poly) for eq in equations if (poly := encode(eq))]
    except (sp.PolynomialError, sp.CoercionFailed):
        return None

    substitutions = []
    while polys and active:
        pivot = None
        for poly, degrees, equation, cost in polys:
            if equation is None:
                continue
            variable = next((v for v in active if degrees[indexes[v]]), None)
            if variable is None:
                continue
            if pivot is None or cost < pivot[0]:
                pivot = cost, poly, equation, variable
        if pivot is None:
            break
        _, selected, equation, variable = pivot
        coefficient = equation.coeff(variable)
        value = -(equation - coefficient * variable) / coefficient
        active.remove(variable)
        substitutions.append((variable, value))
        index = indexes[variable]
        replacements = generators.copy()
        replacements[index] = encode(sp.expand(value))
        remaining = []
        for item in polys:
            poly, degrees, affine, _ = item
            if affine is not None and poly == selected:
                continue
            if degrees[index] > 0:
                poly = poly.compose(*replacements)
                if poly:
                    remaining.append(inspect_polynomial(poly))
            else:
                remaining.append(item)
        polys = remaining
    return (
        [
            equation if equation is not None else decode(poly)
            for poly, _, equation, _ in polys
        ],
        active,
        substitutions,
    )


def eliminate_linear_variables(equations, variables):
    """Eliminate coordinates using equations with a nonzero rational pivot."""
    equations = [sp.expand(eq) for eq in equations if eq != 0]
    variables = list(variables)
    native = _eliminate_affine_polynomials(equations, variables)
    if native is not None:
        return native
    substitutions = []

    @lru_cache(maxsize=None)
    def is_affine(equation):
        return all(
            term == 1 or term.is_Symbol for term in equation.as_coefficients_dict()
        )

    operation_count = lru_cache(maxsize=None)(sp.count_ops)
    while equations and variables:
        eligible = (eq for eq in equations if is_affine(eq))
        pivot = None
        for equation in sorted(eligible, key=operation_count):
            active = equation.free_symbols
            for variable in variables:
                if variable not in active:
                    continue
                coefficient = equation.coeff(variable)
                if coefficient.is_Rational and coefficient != 0:
                    offset = equation - coefficient * variable
                    if not offset.has(variable):
                        pivot = (equation, variable, -offset / coefficient)
                        break
            if pivot:
                break
        if pivot is None:
            break
        equation, variable, value = pivot
        substitutions.append((variable, value))
        variables.remove(variable)
        replacement = {variable: value}
        remaining = []
        for other in equations:
            if other == equation:
                continue
            updated = (
                sp.expand(other.xreplace(replacement)) if other.has(variable) else other
            )
            if updated != 0:
                remaining.append(updated)
        equations = remaining
    return equations, variables, substitutions

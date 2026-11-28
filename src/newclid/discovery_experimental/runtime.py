"""Source checks and application through the public DDAR interface."""

from itertools import combinations
import math
from types import SimpleNamespace

from .algebra.predicate_encodings import domain_guards
from .compression import simplify_collinear_parallel_premises
from .rules import parse_predicate, split_rule_text, to_pipe_format


def check_statements(row, statements):
    from newclid.api import DDAR

    points = [(p["name"], p["x"], p["y"]) for p in row["points"]]
    parsed = [(f.split()[0], f.split()[1:]) for f in statements]
    batch = getattr(DDAR, "check_numerical_statements", None)
    return batch(points, parsed) if batch is not None else _check_scalar(points, parsed)


def _check_scalar(points, statements):
    from newclid.numerical.geometries import PointNum
    from newclid.predicates import NAME_TO_PREDICATE

    lookup = {name: SimpleNamespace(num=PointNum(x, y)) for name, x, y in points}
    if len(lookup) != len(points) or any(
        not math.isfinite(x) or not math.isfinite(y) for _, x, y in points
    ):
        return [False] * len(statements)
    checked = []
    for kind, names in statements:
        try:
            args = tuple(lookup[name] for name in names)
            result = NAME_TO_PREDICATE[kind].check_numerical(SimpleNamespace(args=args))
        except (ArithmeticError, AttributeError, IndexError, KeyError, ValueError):
            result = False
        checked.append(bool(result))
    return checked


def supports_extended_guards():
    from newclid.api import DDAR

    return hasattr(DDAR, "check_numerical_statements")


def stock_guard(guard):
    parts = guard.split()
    if parts[0] == "diff":
        _, a, b = parts
        # A segment's dot product with itself is its squared length.
        return f"nperp {a} {b} {a} {b}"
    return guard


def _stock_guards(row, required, runtime):
    result = [stock_guard(guard) for guard in required]
    angles = []
    for guard in runtime:
        points = guard.split()[1:]
        segments = list(zip(points[::2], points[1::2]))
        options = ["npara " + " ".join(a + b) for a, b in combinations(segments, 2)]
        chosen = next(
            (p for p, ok in zip(options, check_statements(row, options)) if ok), None
        )
        if chosen is None:
            raise ValueError(f"No stock matcher guard supports {guard}")
        angles.append(chosen)
    return list(dict.fromkeys(result)), list(dict.fromkeys(angles))


def prepare(row):
    simplification = simplify_collinear_parallel_premises(row["rule_text"])
    text = simplification["rule_text"]
    premises, conclusion = text.split(" => ")
    facts = premises.split(", ")
    goals = conclusion.split(", ")
    previous = [
        item.strip() for item in row.get("guards", "").split(",") if item.strip()
    ]
    guards = [
        guard for fact in facts + previous + goals for guard in domain_guards(fact)
    ]
    for fact in facts + goals:
        kind, *args = fact.split()
        if kind in {"simtri", "simtrir"}:
            pairs = [f"diff {a} {b}" for a, b in zip(args[:3], args[3:]) if a != b]
            chosen = next(
                (p for p, ok in zip(pairs, check_statements(row, pairs)) if ok), None
            )
            if chosen is None:
                raise ValueError("similarity triangles overlap pointwise")
            guards.append(chosen)
    guards = list(
        dict.fromkeys(
            g
            for g in previous + guards + simplification["extra_guards"]
            if g not in facts
        )
    )
    runtime = list(
        dict.fromkeys(
            "angle_defined " + f.split(" ", 1)[1]
            for f in facts + goals
            if f.startswith("eqangle ")
        )
    )
    if not supports_extended_guards():
        guards, runtime = _stock_guards(row, guards, runtime)
    return {"runtime_rule_text": text, "guards": guards, "runtime_guards": runtime}


def check_runtime(row):
    """Check compatibility in the C++ matcher; exact proof remains mandatory."""
    from newclid.api import DDAR

    premises, goal = split_rule_text(row["rule_text"])
    premises = [(name, list(args)) for name, args in map(parse_predicate, premises)]
    goals = [(name, list(args)) for name, args in map(parse_predicate, goal.split(","))]
    points = [(p["name"], p["x"], p["y"]) for p in row["points"]]
    settings = {f"r{i:02d}": False for i in range(1000)}
    settings.update(
        using_ar=False,
        targeted_ar=False,
        using_log=False,
        using_exp=False,
        verbose=False,
        allow_duplicate_points=False,
    )
    try:
        solved, _ = DDAR.run_ddar_with_custom_theorems(
            "exact_guarded",
            points,
            premises,
            goals,
            [to_pipe_format("exact_rule", row["rule_text"], row["guards"])],
            3,
            settings,
        )
        return {"passed": bool(solved)}
    except Exception as error:
        return {"passed": False, "error": str(error)}

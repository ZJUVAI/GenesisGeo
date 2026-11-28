"""Conservative exact certificates: no implicit distinct-point assumptions.

Only explicit diff/ncoll/negative facts may supply nonzero assumptions. Positive
premises are relaxed to their equations; their domains are never silently added.
This can return unproved for valid geometry, but cannot certify by discarding a
degenerate branch. Guarded proposals are different, visibly strengthened rules.
"""

from collections import Counter
from itertools import combinations
from functools import lru_cache
import sympy as sp

from .exact_geometry import (
    eliminate_linear_variables,
    parse_statement,
)
from .predicate_encodings import encode, EXPLICIT_NONZERO
from .polynomial_precondition import cancel_guarded_factors


@lru_cache(maxsize=4096)
def _premise_equations(statement, point_coordinates):
    equations, obligations = encode(statement, dict(point_coordinates), premise=True)
    return tuple(equations), tuple(obligations)


def _linear_anchor(facts):
    """Prefer an explicitly distinct pair that makes premise equations affine."""
    allowed = set()
    score = Counter()
    frequency = Counter()
    for statement in facts:
        kind, points, _ = parse_statement(statement)
        pairs = list(combinations(sorted(set(points)), 2))
        frequency.update(pairs)
        if kind in {"diff", "ncoll"}:
            allowed.update(pairs)
        elif (
            kind == "nperp" and len(points) == 4 and set(points[:2]) == set(points[2:])
        ):
            allowed.update(pairs)
        elif kind == "coll":
            score.update(pairs)
        elif kind in {"perp", "para"}:
            score.update(tuple(sorted(pair)) for pair in zip(points[::2], points[1::2]))
        elif kind == "cong":
            left, right = set(points[:2]), set(points[2:])
            endpoints = left ^ right
            if len(endpoints) == 2 and left & right:
                score[tuple(sorted(endpoints))] += 1
        elif kind == "circle":
            score.update(combinations(sorted(set(points[1:])), 2))
    if allowed:
        return list(
            min(allowed, key=lambda pair: (-score[pair], -frequency[pair], pair))
        )
    return None


def build_model(rule):
    """Compile the explicit premises and domain guards into exact equations."""
    name, premise_text, goal_text, *extra = rule.split("|")
    premises = [part.strip() for part in premise_text.split(",") if part.strip()]
    guards = (
        [part.strip() for part in extra[0].split(",") if part.strip()] if extra else []
    )
    goals = [part.strip() for part in goal_text.split(",") if part.strip()]
    facts = premises + guards
    names = sorted(
        {
            point
            for statement in facts + goals
            for point in parse_statement(statement)[1]
        }
    )
    if not facts or not goals or not names:
        raise ValueError("nonempty premises, goals and point names are required")
    coordinates = {point: sp.symbols(f"{point}_x {point}_y") for point in names}
    # Translation is always safe. Scale/affine gauges require explicit witnesses.
    anchor = next(
        (
            statement.split()[1:4]
            for statement in facts
            if statement.split()[0] == "ncoll"
        ),
        None,
    )
    if anchor is None:
        anchor = next(
            (
                statement.split()[1:3]
                for statement in facts
                if statement.split()[0] == "diff"
            ),
            None,
        )
    anchor = _linear_anchor(facts) or anchor
    affine = all(
        statement.split()[0] in {"coll", "ncoll", "para", "midp", "diff"}
        for statement in facts + goals
    )
    if anchor:
        if len(set(anchor)) != len(anchor):
            return {
                "rule": name,
                "certified": False,
                "reason": "inconsistent_explicit_guard",
            }
        coordinates[anchor[0]] = (sp.Integer(0), sp.Integer(0))
        coordinates[anchor[1]] = (sp.Integer(1), sp.Integer(0))
        if len(anchor) == 3 and affine:
            coordinates[anchor[2]] = (sp.Integer(0), sp.Integer(1))
    else:
        coordinates[names[0]] = (sp.Integer(0), sp.Integer(0))
    explicit_triangles = {
        frozenset(s.split()[1:]) for s in facts if s.split()[0] == "ncoll"
    }
    for statement in facts:
        kind, *args = statement.split()
        if kind == "sameclock" and any(
            frozenset(t) not in explicit_triangles for t in (args[:3], args[3:])
        ):
            return {
                "rule": name,
                "certified": False,
                "reason": "orientation_requires_explicit_triangle_domains",
            }
    equations, nonzero = [], []
    for statement in facts:
        _, points, _ = parse_statement(statement)
        zero, obligations = _premise_equations(
            statement, tuple((point, coordinates[point]) for point in points)
        )
        equations.extend(zero)
        if statement.split()[0] in EXPLICIT_NONZERO:
            nonzero.extend(obligations)
    if not nonzero:
        origin = {
            value: sp.Integer(0)
            for pair in coordinates.values()
            for value in pair
            if getattr(value, "is_Symbol", False)
        }
        collapsed = all(expression.subs(origin) == 0 for expression in equations)
        if collapsed and any(encode(goal, coordinates)[1] for goal in goals):
            return {
                "rule": name,
                "certified": False,
                "assumption_policy": "explicit_guards_only",
                "reason": "degenerate_branch_not_excluded_by_explicit_guards",
                "relaxed_premise_model": {point: [0, 0] for point in names},
            }
    variables = [
        value
        for point in coordinates.values()
        for value in point
        if getattr(value, "is_Symbol", False)
    ]
    equations, variables, substitutions = eliminate_linear_variables(
        equations, variables
    )
    equations, variables, substitutions, factors = cancel_guarded_factors(
        equations,
        variables,
        nonzero,
        substitutions,
    )
    if factors is None:
        return {"reason": "inconsistent_explicit_guard"}
    return {
        "coordinates": coordinates,
        "variables": variables,
        "substitutions": substitutions,
        "equations": equations,
        "explicit_nonzero_factors": sorted(factors, key=sp.default_sort_key),
        "anchor": anchor,
    }

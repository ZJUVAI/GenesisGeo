"""Remove guards already provided by another check."""

from itertools import combinations

from ..rules import parse_predicate
from .model import Rule

SEGMENT_ARITIES = {
    "cong": 4,
    "para": 4,
    "perp": 4,
    "npara": 4,
    "nperp": 4,
    "eqangle": 8,
    "eqratio": 8,
    "angle_defined": 8,
    "neqangle": 8,
}


def distinct_pairs(rule: Rule) -> set[tuple[str, ...]]:
    pairs = set()
    for literal in (*rule.premises, *rule.guards):
        kind, points = parse_predicate(literal)
        if kind in {"coll", "ncoll"} and len(points) == 3:
            pairs.update(tuple(sorted(pair)) for pair in combinations(points, 2))
        elif SEGMENT_ARITIES.get(kind) == len(points):
            pairs.update(tuple(sorted(pair)) for pair in zip(points[::2], points[1::2]))
    return pairs


def trim_guards(rule: Rule, bounded_coordinates: bool) -> Rule:
    pairs = distinct_pairs(rule) if bounded_coordinates else set()
    premises = set(rule.premises)
    # DDAR's cyclic check explicitly tests the first three points using the
    # same collinearity equation as ncoll, and checks all involved segments.
    cyclic_triples = {
        points[:3]
        for kind, points in map(parse_predicate, rule.premises)
        if kind == "cyclic" and len(points) == 4
    }
    kept = []
    seen = set()
    for guard in rule.guards:
        kind, points = parse_predicate(guard)
        redundant = (
            guard in seen
            or (kind == "angle_defined" and f"eqangle {' '.join(points)}" in premises)
            or (kind == "ncoll" and points in cyclic_triples)
            or (kind == "diff" and tuple(sorted(points)) in pairs)
        )
        if not redundant:
            kept.append(guard)
            seen.add(guard)
    return rule._replace(guards=tuple(kept))

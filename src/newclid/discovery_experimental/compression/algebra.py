"""Remove premises using exact geometric identities over existing points."""

from collections import Counter
from fractions import Fraction
from itertools import combinations

from ..rules import parse_predicate
from .guards import distinct_pairs
from .model import Rule

_LENGTH = {"cong", "eqratio"}
_DIRECTION = {"para", "perp", "eqangle"}
_HALF_TURN = ("", "")


def _pair(a, b):
    return tuple(sorted((a, b)))


def _row(terms):
    coefficients = Counter()
    for key, value in terms:
        coefficients[key] += value
    return {key: value for key, value in coefficients.items() if value}


class _Basis:
    def __init__(self, *, modular=False):
        self.rows = {}
        self.modular = modular

    @staticmethod
    def _subtract(row, pivot, scale):
        for key, value in pivot.items():
            remaining = row.get(key, 0) - scale * value
            if remaining:
                row[key] = remaining
            else:
                row.pop(key, None)

    def reduce(self, equation):
        row = dict(equation)
        for key in sorted(self.rows, reverse=self.modular):
            if key in row:
                self._subtract(row, self.rows[key], row[key])
        if self.modular and _HALF_TURN in row:
            row[_HALF_TURN] %= 2
            if not row[_HALF_TURN]:
                del row[_HALF_TURN]
        return row

    def add(self, equation):
        row = self.reduce(equation)
        if not row:
            return
        pivot = max(row) if self.modular else min(row)
        if self.modular:
            # Direction equations are modulo pi. Dividing by 2 would pick an
            # unjustified branch, so eliminate only with unit pivots.
            if pivot == _HALF_TURN or abs(row[pivot]) != 1:
                return
            multiplier = row[pivot]
        else:
            multiplier = Fraction(1, row[pivot])
        self.rows[pivot] = {key: value * multiplier for key, value in row.items()}


def _segment_equation(kind, args):
    if kind not in _LENGTH | _DIRECTION:
        return None
    segments = tuple(_pair(a, b) for a, b in zip(args[::2], args[1::2]))
    if any(a == b for a, b in segments):
        return None
    weights = (1, -1) if len(segments) == 2 else (1, -1, -1, 1)
    equation = _row(zip(segments, weights, strict=True))
    if kind == "perp":
        equation[_HALF_TURN] = -1
    return segments, equation


def _point_equation(kind, args):
    if kind == "cong":
        a, b, c, d = args
        return "square", _row(((_pair(a, b), 1), (_pair(c, d), -1)))
    if kind == "perp":
        a, b, c, d = args
        return "square", _row(
            (_pair(x, y), weight)
            for x, y, weight in ((b, c, 1), (a, d, 1), (b, d, -1), (a, c, -1))
            if x != y
        )
    if kind in {"coll", "para"}:
        if kind == "coll":
            a, b, d = args
            c = a
        else:
            a, b, c, d = args
        return "area", _row(
            (_pair(x, y), weight if x < y else -weight)
            for x, y, weight in ((b, d, 1), (b, c, -1), (a, d, -1), (a, c, 1))
            if x != y
        )
    return None


def _geometry(kind, args):
    if len(set(args)) != len(args):
        return
    if kind in {"coll", "midp"}:
        a, b, c = args
        yield "para", (a, b, a, c)
        yield "para", (a, b, b, c)
        if kind == "midp":
            yield "cong", (a, b, a, c)
            yield "coll", args
    elif kind == "cyclic":
        for a, b in combinations(args, 2):
            p, q = (point for point in args if point not in (a, b))
            yield "eqangle", (p, a, p, b, q, a, q, b)
    elif kind == "circle":
        a, b, c, d = args
        yield "cong", (a, b, a, c)
        yield "cong", (a, b, a, d)


def _equations(kind, args):
    equation = _segment_equation(kind, args)
    if equation is not None:
        yield ("length" if kind in _LENGTH else "direction"), equation[1]
    equation = _point_equation(kind, args)
    if equation is not None:
        yield equation


def _triangle_guards(rule):
    triangles = {
        args
        for kind, args in map(parse_predicate, rule.guards)
        if kind == "ncoll" and len(set(args)) == 3
    }
    triangles.update(
        args[:3]
        for kind, args in map(parse_predicate, rule.premises)
        if kind == "cyclic" and len(set(args)) == 4
    )
    return triangles


def _sine_equations(triangles, direction):
    """On a noncollinear triangle, each side is proportional to the sine opposite."""

    def angle(a, b):
        row = direction.reduce(_row(((a, 1), (b, -1))))
        constant = row.pop(_HALF_TURN, 0)
        forward = tuple(sorted(row.items()))
        reverse = tuple((key, -value) for key, value in forward)
        return min(forward, reverse), constant

    for a, b, c in sorted(triangles):
        sides = (_pair(b, c), _pair(a, c), _pair(a, b))
        angles = [angle(sides[(i + 1) % 3], sides[(i + 2) % 3]) for i in range(3)]
        for i in (1, 2):
            yield _row(
                (
                    ((0, sides[0]), 1),
                    ((0, sides[i]), -1),
                    ((1, angles[0]), -1),
                    ((1, angles[i]), 1),
                )
            )


def _priority(fact):
    kind, args = parse_predicate(fact)
    return kind in _LENGTH | _DIRECTION | {"coll"}, len(set(args))


def reduce_premises(rule: Rule) -> Rule:
    bases = {
        "direction": _Basis(modular=True),
        "length": _Basis(),
        "square": _Basis(),
        "area": _Basis(),
    }
    defined = set(rule.guards)
    kept, bound = set(), set()
    domain = distinct_pairs(rule._replace(premises=()))
    for guard in rule.guards:
        kind, args = parse_predicate(guard)
        if kind == "diff" and len(args) == 2:
            domain.add(_pair(*args))

    for literal in sorted(rule.premises, key=_priority):
        kind, args = parse_predicate(literal)
        equations = list(_equations(kind, args))
        required = (
            {_pair(a, b) for a, b in combinations(args, 2)}
            if kind in {"coll", "para"}
            else {_pair(a, b) for a, b in zip(args[::2], args[1::2])}
        )
        can_remove = (
            set(args) <= bound
            and required <= domain
            and (kind != "eqangle" or f"angle_defined {' '.join(args)}" in defined)
            and (kind != "para" or f"ncoll {' '.join(args[:3])}" in defined)
        )
        if can_remove and any(not bases[name].reduce(row) for name, row in equations):
            continue
        kept.add(literal)
        bound.update(args)
        domain.update(distinct_pairs(rule._replace(premises=(literal,), guards=())))
        for name, row in equations:
            bases[name].add(row)
        for derived in _geometry(kind, args):
            for name, row in _equations(*derived):
                if name != "length" or row.keys() <= domain:
                    bases[name].add(row)

    # Direction premises are now fixed. Sine equations can consequently
    # witness length deletions without circular proofs across the two bases.
    reduced = rule._replace(premises=tuple(p for p in rule.premises if p in kept))
    triangles = _triangle_guards(reduced)
    if not triangles:
        return reduced
    lengths = _Basis()
    for row in _sine_equations(triangles, bases["direction"]):
        lengths.add(row)
    domain = distinct_pairs(
        reduced._replace(
            premises=tuple(
                p for p in reduced.premises if parse_predicate(p)[0] not in _LENGTH
            )
        )
    )
    bound = {
        point
        for p in reduced.premises
        if parse_predicate(p)[0] not in _LENGTH
        for point in parse_predicate(p)[1]
    }
    for literal in reduced.premises:
        kind, args = parse_predicate(literal)
        if kind not in _LENGTH:
            continue
        encoded = _segment_equation(kind, args)
        if encoded is None:
            continue
        segments, coefficients = encoded
        row = {(0, key): value for key, value in coefficients.items()}
        if set(segments) <= domain and set(args) <= bound and not lengths.reduce(row):
            kept.remove(literal)
        else:
            lengths.add(row)
            domain.update(segments)
            bound.update(args)
    return rule._replace(premises=tuple(p for p in rule.premises if p in kept))

"""Propose relative orientation guards and sound nonzero consequences.

SameClock(a,b) means both signed triangle areas have the same nonzero sign.
Their sum, and positive squared-edge-weighted sums, therefore cannot vanish.
These are necessary polynomial consequences, not an algebraic replacement for
the full orientation predicate. The actual predicate remains a runtime guard.
"""

from itertools import combinations, permutations
import math


def clock_candidates(row, rule_text, max_pairs=64):
    from ..runtime import check_statements

    facts = rule_text.replace(" => ", ",").split(",")
    names = sorted({n for f in facts for n in f.split()[1:]})
    triangles = list(combinations(names, 3))
    valid = check_statements(row, ["ncoll " + " ".join(t) for t in triangles])
    triangles = [t for t, ok in zip(triangles, valid) if ok]
    coordinates = {p["name"]: (p["x"], p["y"]) for p in row["points"]}

    def squared(a, b):
        return sum((x - y) ** 2 for x, y in zip(coordinates[a], coordinates[b]))

    pairs = []
    buckets = {}
    for triangle in triangles:
        for t in permutations(triangle):
            ab = squared(t[0], t[1])
            ac = squared(t[0], t[2])
            bc = squared(t[1], t[2])
            if not ab or not math.isfinite(ab + ac + bc):
                continue
            key = (round(ac / ab, 6), round(bc / ab, 6))
            if key in buckets and set(buckets[key]) != set(t):
                pairs.append((buckets[key], t))
            else:
                buckets.setdefault(key, t)

    # Pairs explicitly described by adjacent-segment ratios / angles.
    def triangle(left, right):
        common = set(left) & set(right)
        if len(common) != 1:
            return None
        vertex = next(iter(common))
        a = next(n for n in left if n != vertex)
        b = next(n for n in right if n != vertex)
        return (vertex, a, b) if a != b else None

    for fact in facts:
        kind, *args = fact.split()
        if kind not in {"eqangle", "eqratio"} or len(args) != 8:
            continue
        edges = list(zip(args[::2], args[1::2]))
        for i, j, k, last in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
            a, b = triangle(edges[i], edges[j]), triangle(edges[k], edges[last])
            if a and b and set(a) != set(b):
                pairs.append((a, b))
    raw = []
    for a, b in pairs:
        raw.extend(
            [
                "sameclock " + " ".join(a + b),
                "sameclock " + " ".join(a + (b[0], b[2], b[1])),
            ]
        )

    def rotate(t):
        return min(t, t[1:] + t[:1], t[2:] + t[:2])

    def canonical(tokens):
        a, b = tuple(tokens[:3]), tuple(tokens[3:])
        direct = tuple(sorted((rotate(a), rotate(b))))
        inverse = tuple(
            sorted((rotate((a[0], a[2], a[1])), rotate((b[0], b[2], b[1]))))
        )
        return "sameclock " + " ".join(sum(min(direct, inverse), ()))

    options = {
        canonical(f.split()[1:]) for f, ok in zip(raw, check_statements(row, raw)) if ok
    }
    ordered = sorted(options)
    return [g for g, ok in zip(ordered, check_statements(row, ordered)) if ok][
        :max_pairs
    ]


def nonzero_consequences(guard, coordinates):
    kind, *args = guard.split()
    if kind != "sameclock":
        raise ValueError("expected relative orientation guard")
    left, right = args[:3], args[3:]

    def area(names):
        a, b, c = [coordinates[n] for n in names]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def squares(names):
        return [
            sum((a - b) ** 2 for a, b in zip(coordinates[x], coordinates[y]))
            for x, y in combinations(names, 2)
        ]

    a, b = area(left), area(right)
    return [a, b, a + b] + [
        a * y + b * x for x in squares(left) for y in squares(right)
    ]


def side_candidates(row, rule_text, max_pairs=96):
    """Relative order on collinear triples; strictness is exported separately.

    CPP SameSide compares positive vs nonpositive dot products. Its negative
    branch permits zero, so both explicit nperp guards are essential before a
    sum of the two dot products can be treated as nonzero.
    """
    from ..runtime import check_statements

    names = sorted(
        {n for f in rule_text.replace(" => ", ",").split(",") for n in f.split()[1:]}
    )
    triples = list(combinations(names, 3))
    coll = [
        t
        for t, ok in zip(
            triples, check_statements(row, ["coll " + " ".join(t) for t in triples])
        )
        if ok
    ]
    # Strictness is a property of one side of the predicate. Check each
    # directed triple once before pairing it with other collinear triples.
    orientations = {
        triple: [(a, *(p for p in triple if p != a)) for a in triple] for triple in coll
    }
    strict_checks = {
        (a, b, c): f"nperp {a} {b} {a} {c}"
        for group in orientations.values()
        for a, b, c in group
    }
    strict = {
        branch
        for branch, ok in zip(
            strict_checks, check_statements(row, list(strict_checks.values()))
        )
        if ok
    }
    options = []
    for left, right in combinations(coll, 2):
        options.extend(
            "sameside " + " ".join(a + b)
            for a in orientations[left]
            if a in strict
            for b in orientations[right]
            if b in strict
        )
    return [g for g, ok in zip(options, check_statements(row, options)) if ok][
        :max_pairs
    ]


def side_strict_guards(guard):
    _, a, b, c, d, e, f = guard.split()
    return ["nperp " + " ".join((a, b, a, c)), "nperp " + " ".join((d, e, d, f))]


def side_nonzero_consequences(guard, coordinates):
    _, a, b, c, d, e, f = guard.split()

    def dot(o, x, y):
        return sum(
            (i - k) * (j - k)
            for i, j, k in zip(coordinates[x], coordinates[y], coordinates[o])
        )

    def squared(x, y):
        return sum((i - j) ** 2 for i, j in zip(coordinates[x], coordinates[y]))

    left, right = dot(a, b, c), dot(d, e, f)
    return [left + right] + [
        left * squared(u, v) + right * squared(x, y)
        for x, y in combinations((a, b, c), 2)
        for u, v in combinations((d, e, f), 2)
    ]

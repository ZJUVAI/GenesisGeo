"""Exact predicate equations plus separately declared nondegeneracy obligations."""

from itertools import combinations

import sympy as sp

from .exact_geometry import parse_statement, dot, cross

EXPLICIT_NONZERO = frozenset(
    {"diff", "ncoll", "nperp", "npara", "neqangle", "sameclock", "angle_defined"}
)


def encode(statement, points, *, premise=False):
    predicate, names, ratio = parse_statement(statement)
    args = [points[name] for name in names]

    def segment(i, j):
        return tuple(b - a for a, b in zip(args[i], args[j]))

    def squared(i, j):
        vector = segment(i, j)
        return dot(vector, vector)

    if predicate == "sameclock" and len(args) == 6:
        from .orientation_conditions import nonzero_consequences

        return [], nonzero_consequences(statement, points)
    if predicate == "eqpoint" and len(args) == 2:
        return list(segment(0, 1)), []
    if predicate == "diff" and len(args) == 2:
        return [], [segment(0, 1)]
    if predicate == "circle" and len(args) >= 3:
        radius = squared(0, 1)
        return [radius - squared(0, i) for i in range(2, len(args))], [segment(0, 1)]
    if predicate in {"simtri", "simtrir"} and len(args) == 6:
        u, v, U, V = segment(0, 1), segment(0, 2), segment(3, 4), segment(3, 5)
        # Complex cross multiplication; conjugate for reflected similarity.
        sign = 1 if predicate == "simtri" else -1

        def mul(a, b):
            return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

        left = mul(U, (v[0], sign * v[1]))
        right = mul(V, (u[0], sign * u[1]))
        return [left[i] - right[i] for i in range(2)], [
            cross(u, v),
            cross(U, V),
            tuple(c for i in range(3) for c in segment(i, i + 3)),
        ]
    if predicate in {"contri", "contrir"} and len(args) == 6:
        first = cross(segment(0, 1), segment(0, 2))
        second = cross(segment(3, 4), segment(3, 5))
        equations = [
            squared(i, j) - squared(i + 3, j + 3) for i, j in combinations(range(3), 2)
        ]
        equations.append(first - second if predicate == "contri" else first + second)
        return equations, [first, second]
    if predicate in {"coll", "ncoll"} and len(args) == 3:
        area = cross(segment(0, 1), segment(0, 2))
        if predicate == "ncoll":
            return [], [area]
        edges = [segment(i, j) for i, j in combinations(range(3), 2)]
        return [area], edges
    if predicate == "midp" and len(args) == 3:
        u, v = segment(1, 0), segment(0, 2)
        return [a - b for a, b in zip(u, v)], [u]
    if predicate == "aconst" and len(args) == 4:
        u, v = segment(0, 1), segment(2, 3)
        lengths = [u, v]
        cosine, sine = dot(u, v), cross(u, v)
        angle = sp.pi * ratio
        if sp.cos(angle) == 0:
            return [cosine], lengths
        tangent = sp.tan(angle)
        if tangent.is_Rational:
            return [sine - tangent * cosine], lengths
        # A squared cosine is a sound relaxation of a directed premise.
        cos_squared = sp.simplify(sp.cos(angle) ** 2)
        if premise and cos_squared.is_Rational:
            return [cosine**2 - cos_squared * dot(u, u) * dot(v, v)], lengths
        raise ValueError(
            f"constant angle lacks a supported rational encoding: {statement}"
        )
    if predicate == "cyclic" and len(args) == 4:
        u, v, w = (segment(0, i) for i in range(1, 4))
        determinant = sp.expand(
            dot(u, u) * cross(v, w) - dot(v, v) * cross(u, w) + dot(w, w) * cross(u, v)
        )
        areas = [
            cross(segment(i, j), segment(i, k)) for i, j, k in combinations(range(4), 3)
        ]
        edges = [segment(i, j) for i, j in combinations(range(4), 2)]
        return [determinant], areas + edges
    if predicate in {"eqratio", "rconst"} and len(args) == (8 if ratio is None else 4):
        segments = [segment(i, i + 1) for i in range(0, len(args), 2)]
        squares = [dot(edge, edge) for edge in segments]
        if ratio is not None:
            return [squares[0] - ratio**2 * squares[1]], segments
        a, b, c, d = squares
        return [a * d - b * c], segments
    if predicate in {"eqangle", "neqangle", "angle_defined"} and len(args) == 8:
        u, v, w, z = (segment(i, i + 1) for i in range(0, 8, 2))
        difference = cross(u, v) * dot(w, z) - dot(u, v) * cross(w, z)
        nondegenerate = [u, v, w, z, tuple(cross(u, t) for t in (v, w, z))]
        if predicate == "angle_defined":
            return [], nondegenerate
        if predicate == "neqangle":
            return [], [*nondegenerate, difference]
        return [difference], [u, v, w, z]
    if predicate in {"perp", "para", "cong", "nperp", "npara"} and len(args) == 4:
        u, v = segment(0, 1), segment(2, 3)
        lengths = [u, v]
        if predicate == "nperp":
            return [], lengths + [dot(u, v)]
        if predicate == "npara":
            return [], lengths + [cross(u, v)]
        equations = {
            "perp": dot(u, v),
            "para": cross(u, v),
            "cong": dot(u, u) - dot(v, v),
        }
        return [equations[predicate]], lengths
    raise ValueError(f"unsupported predicate or arity: {statement}")


def domain_guards(statement):
    """Propose visible guards; never add the actual negative conclusion itself."""
    predicate, names, _ = parse_statement(statement)
    pairs, triangles = [], []
    if predicate in {"ncoll", "diff"}:
        return []
    if predicate in {"contri", "contrir", "simtri", "simtrir"}:
        triangles = [names[:3], names[3:6]]
    elif predicate == "circle":
        pairs = [(names[0], names[1])]
        if len(names) == 4:
            triangles = [names[1:4]]
    elif predicate == "sameclock":
        triangles = [names[:3], names[3:6]]
    elif predicate == "midp":
        pairs = [(names[1], names[2])]
    elif predicate == "coll":
        pairs = list(combinations(names, 2))
    elif predicate == "cyclic":
        triangles = [names[:3]]
        pairs = list(combinations(names, 2))
    elif predicate in {
        "para",
        "perp",
        "cong",
        "npara",
        "nperp",
        "aconst",
        "eqangle",
        "neqangle",
        "eqratio",
        "rconst",
    }:
        pairs = list(zip(names[::2], names[1::2]))
    guards = ["ncoll " + " ".join(triangle) for triangle in triangles]
    guards.extend("diff " + " ".join(sorted(pair)) for pair in pairs)
    return list(dict.fromkeys(guards))

from __future__ import annotations
from fractions import Fraction
from typing import Any, Union
from numpy.random import Generator

from newclid.numerical import close_enough
from newclid.numerical.angles import ang_between, ang_of
from newclid.numerical.geometries import (
    CircleNum,
    LineNum,
    ObjNum,
    PointNum,
    RayNum,
    SegmentNum,
    circle_circle_intersection,
    line_circle_intersection,
    line_line_intersection,
)
from newclid.tools import str_to_fraction
import numpy as np


def sketch(
    name: str, args: tuple[Union[PointNum, str], ...], rng: Generator
) -> list[ObjNum]:
    fun = globals()["sketch_" + name]
    res = fun(args, rng=rng)
    if isinstance(res, list) or isinstance(res, tuple):
        return list(res)  # type: ignore
    return [res]


def sketch_aline(args: tuple[PointNum, ...], **kwargs: Any) -> LineNum:
    """Sketch the construction aline."""
    A, B, C, D, E = args
    ab = A - B
    cb = C - B
    de = D - E

    ang_ab = np.arctan2(ab.y, ab.x)
    ang_bc = np.arctan2(cb.y, cb.x)
    ang_de = np.arctan2(de.y, de.x)

    ang_ex = ang_de + ang_bc - ang_ab
    X = E + PointNum(np.cos(ang_ex), np.sin(ang_ex))
    return LineNum(E, X)


def sketch_aline0(args: tuple[PointNum, ...], **kwargs: Any) -> LineNum:
    """Sketch the construction aline0."""
    A, B, C, D, E, F, G = args
    ab = A - B
    cd = C - D
    ef = E - F

    ang_ab = np.arctan2(ab.y, ab.x)
    ang_cd = np.arctan2(cd.y, cd.x)
    ang_ef = np.arctan2(ef.y, ef.x)

    ang_ex = ang_ef + ang_cd - ang_ab
    X = G + PointNum(np.cos(ang_ex), np.sin(ang_ex))
    return LineNum(G, X)


def sketch_acircle(args: tuple[PointNum, ...], **kwargs: Any) -> CircleNum:
    a, b, c, d, f = args
    de = sketch_aline((c, a, b, f, d))
    fe = sketch_aline((a, c, b, d, f))
    (e,) = line_line_intersection(de, fe)
    return CircleNum(p1=d, p2=e, p3=f)


def sketch_amirror(args: tuple[PointNum, PointNum, PointNum], **kwargs: Any) -> LineNum:
    """Sketch the angle mirror."""
    A, B, C = args
    ab = A - B
    cb = C - B

    dab = A.distance(B)
    ang_ab = np.arctan2(ab.y / dab, ab.x / dab)
    dcb = C.distance(B)
    ang_bc = np.arctan2(cb.y / dcb, cb.x / dcb)

    ang_bx = 2 * ang_bc - ang_ab
    X = B + PointNum(np.cos(ang_bx), np.sin(ang_bx))
    return LineNum(B, X)


def sketch_bisect(args: tuple[PointNum, PointNum, PointNum], **kwargs: Any) -> LineNum:
    a, b, c = args
    ab = a.distance(b)
    bc = b.distance(c)
    x = b + (c - b) * (ab / bc)
    m = (a + x) * 0.5
    return LineNum(b, m)


def sketch_exbisect(
    args: tuple[PointNum, PointNum, PointNum], **kwargs: Any
) -> LineNum:
    _, b, _ = args
    return sketch_bisect(args).perpendicular_line(b)


def sketch_bline(args: tuple[PointNum, PointNum], **kwargs: Any) -> LineNum:
    a, b = args
    m = (a + b) * 0.5
    return m.perpendicular_line(LineNum(a, b))


def sketch_dia(args: tuple[PointNum, PointNum], **kwargs: Any) -> CircleNum:
    a, b = args
    return CircleNum((a + b) * 0.5, p1=a)


def sketch_tangent(
    args: tuple[PointNum, PointNum, PointNum], **kwargs: Any
) -> tuple[PointNum, ...]:
    a, o, b = args
    dia = sketch_dia((a, o))
    return circle_circle_intersection(CircleNum(o, p1=b), dia)


def sketch_circle(
    args: tuple[PointNum, PointNum, PointNum], **kwargs: Any
) -> CircleNum:
    a, b, c = args
    return CircleNum(center=a, radius=b.distance(c))


def sketch_cc_tangent(
    args: tuple[PointNum, PointNum, PointNum, PointNum], **kwargs: Any
) -> tuple[PointNum, ...]:
    """Sketch tangents to two circles."""
    o, a, w, b = args
    ra, rb = o.distance(a), w.distance(b)

    ow = LineNum(o, w)
    if close_enough(ra, rb):
        oo = ow.perpendicular_line(o)
        oa = CircleNum(o, ra)
        x, z = line_circle_intersection(oo, oa)
        y = x + w - o
        t = z + w - o
        return x, y, z, t

    swap = rb > ra
    if swap:
        o, a, w, b = w, b, o, a
        ra, rb = rb, ra

    oa = CircleNum(o, ra)
    q = o + (w - o) * ra / (ra - rb)

    x, z = circle_circle_intersection(sketch_dia((o, q)), oa)
    y = w.foot(LineNum(x, q))
    t = w.foot(LineNum(z, q))

    if swap:
        x, y, z, t = y, x, t, z

    return x, y, z, t


def sketch_e5128(
    args: tuple[PointNum, ...], **kwargs: Any
) -> tuple[PointNum, PointNum]:
    a, b, c, d = args

    g = (a + b) * 0.5
    de = LineNum(d, g)

    e, f = line_circle_intersection(de, CircleNum(c, p1=b))

    if e.distance(d) < f.distance(d):
        e = f
    return e, g


def random_rfss(*points: PointNum, rng: Generator) -> list[PointNum]:
    """Random rotate-flip-scale-shift a PointNum cloud."""
    # center point cloud.
    average = sum(points, PointNum(0.0, 0.0)) / len(points)
    points = tuple(p - average for p in points)

    # rotate
    ang = rng.uniform(0.0, 2 * np.pi)
    sin, cos = np.sin(ang), np.cos(ang)
    # scale and shift
    scale = rng.uniform(0.5, 2.0)
    shift = PointNum(rng.uniform(-1, 1), rng.uniform(-1, 1))
    points = tuple(p.rotate(sin, cos) * scale + shift for p in points)

    # randomly flip
    if rng.random() < 0.5:
        points = tuple(p.flip() for p in points)

    return list(points)


def head_from(tail: PointNum, ang: float, length: float = 1) -> PointNum:
    vector = PointNum(np.cos(ang) * length, np.sin(ang) * length)
    return tail + vector


def sketch_eq_quadrangle(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    """Sketch quadrangle with two equal opposite sides."""
    a = PointNum(0.0, 0.0)
    b = PointNum(1.0, 0.0)

    length = rng.uniform(0.5, 2.0)
    ang = rng.uniform(np.pi / 3, np.pi * 2 / 3)
    d = head_from(a, ang, length)

    ang = ang_of(b, d)
    ang = rng.uniform(ang / 10, ang / 9)
    c = head_from(b, ang, length)
    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def sketch_iso_trapezoid(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 0.0)
    b = PointNum(1.0, 0.0)
    lenght = rng.uniform(0.5, 2.0)
    height = rng.uniform(0.5, 2.0)
    c = PointNum(0.5 + lenght / 2.0, height)
    d = PointNum(0.5 - lenght / 2.0, height)

    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def sketch_eqangle2(args: tuple[PointNum, ...], rng: Generator) -> PointNum:
    """Sketch the def eqangle2."""
    a, b, c = args

    ba = b.distance(a)
    bc = b.distance(c)
    length = ba * ba / bc

    if rng.uniform(0.0, 1.0) < 0.5:
        be = min(length, bc)
        be = rng.uniform(be * 0.1, be * 0.9)
    else:
        be = max(length, bc)
        be = rng.uniform(be * 1.1, be * 1.5)

    e = b + (c - b) * (be / bc)
    y = b + (a - b) * (be / length)
    return line_line_intersection(LineNum(c, y), LineNum(a, e))[0]


def sketch_eqangle3(args: tuple[PointNum, ...], **kwargs: Any) -> CircleNum:
    a, b, d, e, f = args
    de = d.distance(e)
    ef = e.distance(f)
    ab = b.distance(a)
    ang_ax = ang_of(a, b) + ang_between(e, f, d)
    x = head_from(a, ang_ax, length=de / ef * ab)
    return CircleNum(p1=a, p2=b, p3=x)


def sketch_eqdia_quadrangle(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    """Sketch quadrangle with two equal diagonals."""
    m = rng.uniform(0.3, 0.7)
    n = rng.uniform(0.3, 0.7)
    a = PointNum(-m, 0.0)
    c = PointNum(1 - m, 0.0)
    b = PointNum(0.0, -n)
    d = PointNum(0.0, 1 - n)

    ang = rng.uniform(-0.25 * np.pi, 0.25 * np.pi)
    sin, cos = np.sin(ang), np.cos(ang)
    b = b.rotate(sin, cos)
    d = d.rotate(sin, cos)
    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def random_points(n: int, rng: Generator) -> list[PointNum]:
    return [PointNum(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]


def sketch_free(args: tuple[PointNum, ...], rng: Generator) -> PointNum:
    return random_points(1, rng)[0]


def sketch_isos(args: tuple[PointNum, ...], rng: Generator) -> tuple[PointNum, ...]:
    base = rng.uniform(0.5, 1.5)
    height = rng.uniform(0.5, 1.5)

    b = PointNum(-base / 2, 0.0)
    c = PointNum(base / 2, 0.0)
    a = PointNum(0.0, height)
    a, b, c = random_rfss(a, b, c, rng=rng)
    return a, b, c


def sketch_line(args: tuple[PointNum, ...], **kwargs: Any) -> LineNum:
    a, b = args
    return LineNum(a, b)


def sketch_ray(args: tuple[PointNum, ...], **kwargs: Any) -> RayNum:
    a, b = args
    return RayNum(a, b)


def sketch_segline(args: tuple[PointNum, ...], **kwargs: Any) -> SegmentNum:
    a, b = args
    return SegmentNum(a, b)


def sketch_cyclic(args: tuple[PointNum, ...], **kwargs: Any) -> CircleNum:
    a, b, c = args
    return CircleNum(p1=a, p2=b, p3=c)


def sketch_midp(args: tuple[PointNum, ...], **kwargs: Any) -> PointNum:
    a, b = args
    return (a + b) * 0.5


def sketch_pentagon(args: tuple[PointNum, ...], rng: Generator) -> tuple[PointNum, ...]:
    points = [PointNum(1.0, 0.0)]
    ang = 0.0

    for i in range(4):
        ang += (2 * np.pi - ang) / (5 - i) * rng.uniform(0.5, 1.5)
        point = PointNum(np.cos(ang), np.sin(ang))
        points.append(point)

    a, b, c, d, e = points
    a, b, c, d, e = random_rfss(a, b, c, d, e, rng=rng)
    return a, b, c, d, e


def sketch_pline(args: tuple[PointNum, ...], **kwargs: Any) -> LineNum:
    a, b, c = args
    return a.parallel_line(LineNum(b, c))


def sketch_pmirror(args: tuple[PointNum, ...], **kwargs: Any) -> PointNum:
    a, b = args
    return b * 2 - a


def sketch_quadrangle(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    """Sketch a random quadrangle."""
    m = rng.uniform(0.3, 0.7)

    a = PointNum(-m, 0.0)
    c = PointNum(1 - m, 0.0)
    b = PointNum(0.0, -rng.uniform(0.25, 0.75))
    d = PointNum(0.0, rng.uniform(0.25, 0.75))

    ang = rng.uniform(-0.25 * np.pi, 0.25 * np.pi)
    sin, cos = np.sin(ang), np.cos(ang)
    b = b.rotate(sin, cos)
    d = d.rotate(sin, cos)
    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def sketch_r_trapezoid(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 1.0)
    d = PointNum(0.0, 0.0)
    b = PointNum(rng.uniform(0.5, 1.5), 1.0)
    c = PointNum(rng.uniform(0.5, 1.5), 0.0)
    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def sketch_r_triangle(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 0.0)
    b = PointNum(0.0, rng.uniform(0.5, 2.0))
    c = PointNum(rng.uniform(0.5, 2.0), 0.0)
    a, b, c = random_rfss(a, b, c, rng=rng)
    return a, b, c


def sketch_rectangle(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 0.0)
    b = PointNum(0.0, 1.0)
    lenght = rng.uniform(0.5, 2.0)
    c = PointNum(lenght, 1.0)
    d = PointNum(lenght, 0.0)
    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def sketch_reflect(args: tuple[PointNum, ...], **kwargs: Any) -> PointNum:
    a, b, c = args
    m = a.foot(LineNum(b, c))
    return m * 2 - a


def sketch_risos(args: tuple[PointNum, ...], rng: Generator) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 0.0)
    b = PointNum(0.0, 1.0)
    c = PointNum(1.0, 0.0)
    a, b, c = random_rfss(a, b, c, rng=rng)
    return a, b, c


def sketch_rotaten90(args: tuple[PointNum, ...], **kwargs: Any) -> PointNum:
    a, b = args
    ang = -np.pi / 2
    return a + (b - a).rotate(np.sin(ang), np.cos(ang))


def sketch_rotatep90(args: tuple[PointNum, ...], **kwargs: Any) -> PointNum:
    a, b = args
    ang = np.pi / 2
    return a + (b - a).rotate(np.sin(ang), np.cos(ang))


def sketch_s_angle(args: tuple[PointNum, PointNum, str], **kwargs: Any) -> LineNum:
    a, b, angle = args
    f = str_to_fraction(angle)
    ang = float(f) * np.pi
    x = b + (a - b).rotatea(ang)
    return LineNum(b, x)


def sketch_aconst(
    args: tuple[PointNum, PointNum, PointNum, str], **kwargs: Any
) -> LineNum:
    a, b, c, angle = args
    f = str_to_fraction(angle)
    ang = float(f) * np.pi
    x = c + (a - b).rotatea(ang)
    return LineNum(c, x)


def sketch_segment(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, PointNum]:
    a, b = random_points(2, rng)
    return a, b


def sketch_shift(args: tuple[PointNum, ...], **kwargs: Any) -> PointNum:
    a, b, c = args
    return c + (b - a)


def sketch_square(
    args: tuple[PointNum, ...], **kwargs: Any
) -> tuple[PointNum, PointNum]:
    a, b = args
    c = b + (a - b).rotatea(-np.pi / 2)
    d = a + (b - a).rotatea(np.pi / 2)
    return c, d


def sketch_isquare(args: tuple[PointNum, ...], rng: Generator) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 0.0)
    b = PointNum(1.0, 0.0)
    c = PointNum(1.0, 1.0)
    d = PointNum(0.0, 1.0)
    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def sketch_tline(args: tuple[PointNum, ...], **kwargs: Any) -> LineNum:
    a, b, c = args
    return a.perpendicular_line(LineNum(b, c))


def sketch_trapezoid(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    d = PointNum(0.0, 0.0)
    c = PointNum(1.0, 0.0)

    base = rng.uniform(0.5, 2.0)
    height = rng.uniform(0.5, 2.0)
    a = PointNum(rng.uniform(-0.5, 1.5), height)
    b = PointNum(a.x + base, height)
    a, b, c, d = random_rfss(a, b, c, d, rng=rng)
    return a, b, c, d


def sketch_triangle(args: tuple[PointNum, ...], rng: Generator) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 0.0)
    b = PointNum(1.0, 0.0)
    ac = rng.uniform(0.5, 2.0)
    ang = rng.uniform(0.2, 0.8) * np.pi
    c = head_from(a, ang, ac)
    a, b, c = random_rfss(a, b, c, rng=rng)
    return a, b, c


def sketch_triangle12(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    b = PointNum(0.0, 0.0)
    c = PointNum(rng.uniform(1.5, 2.5), 0.0)
    a, _ = circle_circle_intersection(CircleNum(b, 1.0), CircleNum(c, 2.0))
    a, b, c = random_rfss(a, b, c, rng=rng)
    return a, b, c


def sketch_trisect(
    args: tuple[PointNum, ...], **kwargs: Any
) -> tuple[PointNum, PointNum]:
    """Sketch two trisectors of an angle."""
    a, b, c = args
    ang1 = ang_of(b, a)
    ang2 = ang_of(b, c)

    swap = 0
    if ang1 > ang2:
        ang1, ang2 = ang2, ang1
        swap += 1

    if ang2 - ang1 > np.pi:
        ang1, ang2 = ang2, ang1 + 2 * np.pi
        swap += 1

    angx = ang1 + (ang2 - ang1) / 3
    angy = ang2 - (ang2 - ang1) / 3

    x = b + PointNum(np.cos(angx), np.sin(angx))
    y = b + PointNum(np.cos(angy), np.sin(angy))

    ac = LineNum(a, c)
    (x,) = line_line_intersection(LineNum(b, x), ac)
    (y,) = line_line_intersection(LineNum(b, y), ac)

    if swap == 1:
        return y, x
    return x, y


def sketch_trisegment(
    args: tuple[PointNum, ...], **kwargs: Any
) -> tuple[PointNum, PointNum]:
    a, b = args
    x, y = a + (b - a) * (1.0 / 3), a + (b - a) * (2.0 / 3)
    return x, y


def sketch_ieq_triangle(
    args: tuple[PointNum, ...], rng: Generator
) -> tuple[PointNum, ...]:
    a = PointNum(0.0, 0.0)
    b = PointNum(1.0, 0.0)

    c, _ = circle_circle_intersection(CircleNum(a, p1=b), (CircleNum(b, p1=a)))
    a, b, c = random_rfss(a, b, c, rng=rng)
    return a, b, c


def sketch_incenter2(args: tuple[PointNum, ...], **kwargs: Any) -> tuple[PointNum, ...]:
    a, b, c = args
    l1 = sketch_bisect((b, a, c))
    l2 = sketch_bisect((a, b, c))
    (i,) = line_line_intersection(l1, l2)
    x = i.foot(LineNum(b, c))
    y = i.foot(LineNum(c, a))
    z = i.foot(LineNum(a, b))
    return x, y, z, i


def sketch_excenter2(args: tuple[PointNum, ...], **kwargs: Any) -> tuple[PointNum, ...]:
    a, b, c = args
    l1 = sketch_bisect((b, a, c))
    l2 = sketch_exbisect((a, b, c))
    (i,) = line_line_intersection(l1, l2)
    x = i.foot(LineNum(b, c))
    y = i.foot(LineNum(c, a))
    z = i.foot(LineNum(a, b))
    return x, y, z, i


def sketch_centroid(args: tuple[PointNum, ...], **kwargs: Any) -> tuple[PointNum, ...]:
    a, b, c = args
    x = (b + c) * 0.5
    y = (c + a) * 0.5
    z = (a + b) * 0.5
    (i,) = line_line_intersection(LineNum(a, x), LineNum(b, y))
    return x, y, z, i


def sketch_ninepoints(
    args: tuple[PointNum, ...], **kwargs: Any
) -> tuple[PointNum, ...]:
    a, b, c = args
    x = (b + c) * 0.5
    y = (c + a) * 0.5
    z = (a + b) * 0.5
    c = CircleNum(p1=x, p2=y, p3=z)
    return x, y, z, c.center


def sketch_2l1c(args: tuple[PointNum, ...], **kwargs: Any) -> tuple[PointNum, ...]:
    """Sketch a circle touching two lines and another circle."""
    a, b, c, p = args
    bc, ac = LineNum(b, c), LineNum(a, c)
    circle = CircleNum(p, p1=a)

    d, d_ = line_circle_intersection(p.perpendicular_line(bc), circle)
    if bc.diff_side(d_, a):
        d = d_

    e, e_ = line_circle_intersection(p.perpendicular_line(ac), circle)
    if ac.diff_side(e_, b):
        e = e_

    df = d.perpendicular_line(LineNum(p, d))
    ef = e.perpendicular_line(LineNum(p, e))
    (f,) = line_line_intersection(df, ef)

    g, g_ = line_circle_intersection(LineNum(c, f), circle)
    if bc.same_side(g_, a):
        g = g_

    b_ = c + (b - c) / b.distance(c)
    a_ = c + (a - c) / a.distance(c)
    m = (a_ + b_) * 0.5
    (x,) = line_line_intersection(LineNum(c, m), LineNum(p, g))
    return x.foot(ac), x.foot(bc), g, x


def sketch_3peq(args: tuple[PointNum, ...], rng: Generator) -> tuple[PointNum, ...]:
    a, b, c = args
    ab, _, ca = LineNum(a, b), LineNum(b, c), LineNum(c, a)

    z = b + (c - b) * rng.uniform(-0.5, 1.5)

    z_ = z * 2 - c
    ca_parallel_line = z_.parallel_line(ca)
    (x,) = line_line_intersection(ca_parallel_line, ab)
    y = z * 2 - x
    return x, y, z


###### NEW FUNCTIONS FOR NEW DEFINITIONS ---- V. S.


def sketch_isosvertex(args: tuple[PointNum, ...], **kwargs: Any) -> LineNum:
    b, c = args
    m = (b + c) / 2.0

    return m.perpendicular_line(LineNum(b, c))


def sketch_eqratio(args: tuple[PointNum, ...], **kwargs: Any) -> CircleNum:
    A, B, C, D, E, F, G = args

    dab = A.distance(B)
    dcd = C.distance(D)
    d_ef = E.distance(F)

    dgx = d_ef * dcd / dab
    return CircleNum(center=G, radius=dgx)


def sketch_rconst(
    args: tuple[PointNum, PointNum, PointNum, str], **kwargs: Any
) -> CircleNum:
    """Sketches point x such that ab/cx=r"""
    A, B, C, r = args
    dab = A.distance(B)
    length = float(dab / float(str_to_fraction(r)))
    return CircleNum(center=C, radius=length)


def sketch_eqratio6(
    args: tuple[PointNum, ...], **kwargs: Any
) -> Union[CircleNum, LineNum]:
    """Sketches a point x such that ax/cx=ef/gh"""
    A, C, E, F, G, H = args
    d_ef = E.distance(F)
    dgh = G.distance(H)

    if dgh == d_ef:
        M = (A + C) * 0.5
        return M.perpendicular_line(LineNum(A, C))

    else:
        ratio = d_ef / dgh
        extremum_1 = (1 / (1 - ratio)) * (A - ratio * C)
        extremum_2 = (1 / (1 + ratio)) * (ratio * C + A)
        center = (extremum_1 + extremum_2) * 0.5
        radius = 0.5 * extremum_1.distance(extremum_2)
        return CircleNum(center=center, radius=radius)


def sketch_lconst(args: tuple[PointNum, str], **kwargs: Any) -> CircleNum:
    """Sketches point x such that x in at lenght l of point a"""

    a, length = args
    return CircleNum(center=a, radius=float(str_to_fraction(length)))


def sketch_rconst2(
    args: tuple[PointNum, PointNum, str], **kwargs: Any
) -> Union[CircleNum, LineNum]:
    """Sketches point x such that ax/bx=r"""

    A, B, r = args
    ratio = float(str_to_fraction(r))

    if ratio == float(Fraction(1, 1)):
        M = (A + B) * 0.5
        return M.perpendicular_line(LineNum(A, B))

    extremum_1 = (1 / (1 - ratio)) * (A - ratio * B)
    extremum_2 = (1 / (1 + ratio)) * (ratio * B + A)
    center = (extremum_1 + extremum_2) * 0.5
    radius = 0.5 * extremum_1.distance(extremum_2)
    return CircleNum(center=center, radius=radius)

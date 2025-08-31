from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Sequence, Union

from numpy.random import Generator
from newclid.numerical import close_enough, nearly_zero, sign
import numpy as np


ObjNum = Union["PointNum", "LineNum", "CircleNum"]


class PointNum:
    """Numerical point."""

    def __init__(self, x: Any, y: Any):
        self.x: float = float(x)
        self.y: float = float(y)
        if nearly_zero(self.x):
            self.x = 0.0
        if nearly_zero(self.y):
            self.y = 0.0

    def __add__(self, p: "PointNum") -> "PointNum":
        return PointNum(self.x + p.x, self.y + p.y)

    def __sub__(self, p: "PointNum") -> "PointNum":
        return PointNum(self.x - p.x, self.y - p.y)

    def __mul__(self, f: Any) -> "PointNum":
        f = float(f)
        return PointNum(self.x * f, self.y * f)

    def __rmul__(self, f: Any) -> "PointNum":
        return self * float(f)

    def __truediv__(self, f: Any) -> "PointNum":
        f = float(f)
        return PointNum(self.x / f, self.y / f)

    def __str__(self) -> str:
        return "PointNum({},{})".format(self.x, self.y)

    def __abs__(self) -> str:
        return np.sqrt(self.dot(self))

    def angle(self) -> float:
        return np.arctan2(self.y, self.x)

    def close_enough(self, point: "PointNum") -> bool:
        return close_enough(self.x, point.x) and close_enough(self.y, point.y)

    def distance(self, p: Union["PointNum", "LineNum", "CircleNum"]) -> float:
        return np.sqrt(self.distance2(p))

    def distance2(self, p: Union["PointNum", "LineNum", "CircleNum"]) -> float:
        if isinstance(p, LineNum):
            return p.distance(self)
        if isinstance(p, CircleNum):
            return (p.radius - self.distance(p.center)) ** 2
        dx = self.x - p.x
        dy = self.y - p.y
        dx2 = dx * dx
        dy2 = dy * dy
        dx2 = 0.0 if nearly_zero(dx2) else dx2
        dy2 = 0.0 if nearly_zero(dy2) else dy2
        return dx2 + dy2

    def rot90(self) -> "PointNum":
        return PointNum(-self.y, self.x)

    def rotatea(self, ang: Any) -> "PointNum":
        sinb, cosb = np.sin(ang), np.cos(ang)
        return self.rotate(sinb, cosb)

    def rotate(self, sinb: Any, cosb: Any) -> "PointNum":
        x, y = self.x, self.y
        return PointNum(x * cosb - y * sinb, x * sinb + y * cosb)

    def flip(self) -> "PointNum":
        return PointNum(-self.x, self.y)

    def perpendicular_line(self, line: "LineNum") -> "LineNum":
        return line.perpendicular_line(self)

    def foot(self, line: Union["LineNum", "CircleNum"]) -> "PointNum":
        if isinstance(line, LineNum):
            perpendicular_line = line.perpendicular_line(self)
            return line_line_intersection(perpendicular_line, line)[0]
        else:
            c, r = line.center, line.radius
            return c + (self - c) * r / self.distance(c)

    def parallel_line(self, line: "LineNum") -> "LineNum":
        return line.parallel_line(self)

    def dot(self, other: "PointNum") -> float:
        res = self.x * other.x + self.y * other.y
        return 0 if nearly_zero(res) else res

    @classmethod
    def deduplicate(cls, points: Iterable["PointNum"]) -> list["PointNum"]:
        res: list["PointNum"] = []
        for p in points:
            if all(not r.close_enough(p) for r in res):
                res.append(p)
        return res

    def intersect(self, obj: ObjNum) -> list["PointNum"]:
        raise NotImplementedError()


class FormNum(ABC):
    @abstractmethod
    def sample_within(
        self, points: Sequence[PointNum], *, trials: int = 5, rng: Generator
    ) -> PointNum:
        ...


class LineNum(FormNum):
    """Numerical line."""

    def __init__(
        self,
        p1: Optional["PointNum"] = None,
        p2: Optional["PointNum"] = None,
        coefficients: Optional[tuple[float, float, float]] = None,
    ):
        a, b, c = None, None, None
        if coefficients:
            a, b, c = coefficients
        elif p1 and p2:
            if p1.close_enough(p2):
                raise ValueError(
                    "Not able to determine the line by two points two close"
                )
            a, b, c = (
                p1.y - p2.y,
                p2.x - p1.x,
                p1.x * p2.y - p2.x * p1.y,
            )
        assert a is not None and b is not None and c is not None

        # Make sure a is always positive (or always negative for that matter)
        # With a == 0, Assuming a = +epsilon > 0
        # Then b such that ax + by = 0 with y>0 should be negative.
        sa = sign(a)
        sb = sign(b)
        if sa == -1 or sa == 0 and sb == 1:
            a, b, c = -a, -b, -c
        if sa == 0:
            a = 0.0
        if sb == 0:
            b = 0.0
        if nearly_zero(c):
            c = 0.0

        d = np.sqrt(a**2 + b**2)
        self.coefficients = a / d, b / d, c / d

    def parallel_line(self, p: "PointNum") -> "LineNum":
        a, b, _ = self.coefficients
        return LineNum(coefficients=(a, b, -a * p.x - b * p.y))

    def perpendicular_line(self, p: "PointNum") -> "LineNum":
        a, b, _ = self.coefficients
        return LineNum(p, p + PointNum(a, b))

    def same(self, other: "LineNum") -> bool:
        a, b, c = self.coefficients
        x, y, z = other.coefficients
        return close_enough(a * y, b * x) and close_enough(b * z, c * y)

    def distance(self, p: "PointNum") -> float:
        return abs(self(p))

    def __call__(self, p: PointNum) -> float:
        a, b, c = self.coefficients
        res = p.x * a + p.y * b + c
        return 0 if nearly_zero(res) else res

    def is_parallel(self, other: "LineNum") -> bool:
        a, b, _ = self.coefficients
        x, y, _ = other.coefficients
        return close_enough(a * y, b * x)

    def is_perp(self, other: "LineNum") -> bool:
        a, b, _ = self.coefficients
        x, y, _ = other.coefficients
        return close_enough(a * x, -b * y)

    def point_at(
        self, x: Optional[float] = None, y: Optional[float] = None
    ) -> Optional[PointNum]:
        """Infer the point on the line by its x and/or y coordinate(s)"""
        a, b, c = self.coefficients
        # ax + by + c = 0
        if x is None and y is not None:
            if not close_enough(a, 0):
                return PointNum((-c - b * y) / a, y)
            else:
                return None
        elif x is not None and y is None:
            if not close_enough(b, 0):
                return PointNum(x, (-c - a * x) / b)
            else:
                return None
        elif x is not None and y is not None:
            if close_enough(a * x + b * y, -c):
                return PointNum(x, y)
        return None

    def diff_side(self, p1: "PointNum", p2: "PointNum") -> bool:
        d1 = self(p1)
        d2 = self(p2)
        if nearly_zero(d1) or nearly_zero(d2):
            return False
        return d1 * d2 < 0

    def same_side(self, p1: "PointNum", p2: "PointNum") -> bool:
        d1 = self(p1)
        d2 = self(p2)
        if close_enough(d1, 0) or close_enough(d2, 0):
            return False
        return d1 * d2 > 0

    def sample_within(
        self, points: Sequence[PointNum], *, trials: int = 5, rng: Generator
    ) -> PointNum:
        """Sample a point within the boundary of points."""
        center = sum(points, PointNum(0.0, 0.0)) / len(points)
        radius = max([p.distance(center) for p in points])

        if close_enough(center.distance(self), radius):
            center = center.foot(self)
        a, b = line_circle_intersection(self, CircleNum(center.foot(self), radius))

        result = None
        best = -1.0
        for _ in range(trials):
            rand = rng.uniform(0.0, 1.0)
            x = a + (b - a) * rand
            mind = min([x.distance(p) for p in points])
            if mind > best:
                best = mind
                result = x
        assert result
        return result

    def angle(self) -> float:
        if nearly_zero(self.coefficients[1]):
            return np.pi / 2
        res: Any = (self.point_at(x=1) - self.point_at(x=0)).angle() % np.pi  # type: ignore
        if close_enough(res, np.pi):
            return 0.0
        return res


class CircleNum(FormNum):
    """Numerical circle."""

    def __init__(
        self,
        center: Optional[PointNum] = None,
        radius: Optional[float] = None,
        p1: Optional[PointNum] = None,
        p2: Optional[PointNum] = None,
        p3: Optional[PointNum] = None,
    ):
        if not center:
            if not (p1 and p2 and p3):
                raise ValueError("Circle without center need p1 p2 p3")

            l12 = perpendicular_bisector(p1, p2)
            l23 = perpendicular_bisector(p2, p3)
            (self.center,) = line_line_intersection(l12, l23)
        else:
            self.center = center

        self.a, self.b = self.center.x, self.center.y

        if not radius:
            p = p1 or p2 or p3
            if p is None:
                raise ValueError("Circle needs radius or p1 or p2 or p3")
            self.r2 = (self.a - p.x) ** 2 + (self.b - p.y) ** 2
            self.radius: float = np.sqrt(self.r2)
        else:
            self.radius = radius
            self.r2 = radius * radius

    def sample_within(
        self, points: Sequence[PointNum], *, trials: int = 5, rng: Generator
    ) -> PointNum:
        """Sample a point within the boundary of points."""
        result = None
        best = -1.0

        for _ in range(trials):
            ang = rng.uniform(0.0, 2.0) * np.pi
            x = self.center + PointNum(np.cos(ang), np.sin(ang)) * self.radius
            mind = min([x.distance(p) for p in points])
            if mind > best:
                best = mind
                result = x

        assert result
        return result


def perpendicular_bisector(p1: "PointNum", p2: "PointNum") -> "LineNum":
    midpoint = (p1 + p2) * 0.5
    return LineNum(midpoint, midpoint + PointNum(p2.y - p1.y, p1.x - p2.x))


class InvalidIntersectError(Exception):
    pass


class InvalidReduceError(Exception):
    pass


def solve_quad(a: float, b: float, c: float) -> tuple[float, ...]:
    """Solve a x^2 + bx + c = 0."""
    if nearly_zero(a):
        return () if nearly_zero(b) else (-c / b,)
    a = 2 * a
    d = b * b - 2 * a * c
    sd = sign(d)
    if sd == -1:
        return ()
    if sd == 0:
        d = 0.0
    y = np.sqrt(d)
    if nearly_zero(y):
        return (-b / a,)
    return (-b - y) / a, (-b + y) / a


def intersect(a: FormNum, b: FormNum):
    if isinstance(a, CircleNum):
        if isinstance(b, CircleNum):
            return circle_circle_intersection(a, b)
        if isinstance(b, LineNum):
            return line_circle_intersection(b, a)
    if isinstance(a, LineNum):
        if isinstance(b, CircleNum):
            return line_circle_intersection(a, b)
        if isinstance(b, LineNum):
            return line_line_intersection(a, b)
    raise NotImplementedError


def circle_circle_intersection(c1: CircleNum, c2: CircleNum) -> tuple[PointNum, ...]:
    """Returns a pair of Points as intersections of c1 and c2."""
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    x0, y0, r0 = c1.a, c1.b, c1.radius
    x1, y1, r1 = c2.a, c2.b, c2.radius

    d = (x1 - x0) ** 2 + (y1 - y0) ** 2
    if nearly_zero(d):
        # raise InvalidIntersectError
        raise InvalidIntersectError("Two circles have the same center.")
    d: float = np.sqrt(d)

    if not (r0 + r1 >= d and abs(r0 - r1) <= d):
        # return ()
        raise InvalidIntersectError("Two circles are too far to intersect.")

    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = r0**2 - a**2
    biu: PointNum = (c2.center - c1.center) / d
    qiu = biu.rot90()
    p = c1.center + a * biu
    if nearly_zero(h):
        return (p,)
    qiu = np.sqrt(h) * qiu
    return (p + qiu, p - qiu)


def line_circle_intersection(line: LineNum, circle: CircleNum) -> tuple[PointNum, ...]:
    # Line: ax + by + c = 0
    # Circle: (x - p)^2 + (y - q)^2 = r^2
    a, b, c = line.coefficients
    r = circle.radius
    center = circle.center
    p, q = center.x, center.y

    # Line is almost vertical (b=0, equation degenerates to a x + c = 0), solve for y
    if nearly_zero(b):
        x = -c / a
        x_p = x - p
        x_p2 = x_p * x_p
        y = solve_quad(1, -2 * q, q * q + x_p2 - r * r)
        if len(y) == 0:
            raise InvalidIntersectError("Line & circle are too far to intersect.")
        return tuple(PointNum(x, y1) for y1 in y)

     # Line is almost horizontal (a=0, equation degenerates to b y + c = 0), solve for x
    if close_enough(a, 0):
        y = -c / b
        y_q = y - q
        y_q2 = y_q * y_q
        x = solve_quad(1, -2 * p, p * p + y_q2 - r * r)
        if len(y) == 0:
            raise InvalidIntersectError("Line & circle are too far to intersect.")
        return tuple(PointNum(x1, y) for x1 in x)

    # General case: x = -(b y + c) / a
    c_ap = c + a * p
    a2 = a * a
    y = solve_quad(
        a2 + b * b, 2 * (b * c_ap - a2 * q), c_ap * c_ap + a2 * (q * q - r * r)
    )
    if len(y) == 0:
        raise InvalidIntersectError("Line & circle are too far to intersect.")
    return tuple(PointNum(-(b * y1 + c) / a, y1) for y1 in y)


def _check_between(a: PointNum, b: PointNum, c: PointNum) -> bool:
    """Whether a is between b & c."""
    # return (a - b).dot(c - b) > 0 and (a - c).dot(b - c) > 0
    return sign((a - b).dot(c - b)) > 0 and sign((a - c).dot(b - c)) > 0


def circle_segment_intersect(
    circle: CircleNum, p1: PointNum, p2: PointNum
) -> tuple[PointNum, ...]:
    line = LineNum(p1, p2)
    return tuple(
        p for p in line_circle_intersection(line, circle) if _check_between(p, p1, p2)
    )


def line_line_intersection(line_1: LineNum, line_2: LineNum) -> tuple[PointNum, ...]:
    a1, b1, c1 = line_1.coefficients
    a2, b2, c2 = line_2.coefficients
    # a1x + b1y + c1 = 0
    # a2x + b2y + c2 = 0
    d = a1 * b2 - a2 * b1
    if nearly_zero(d):
        return ()
    return (PointNum((c2 * b1 - c1 * b2) / d, (c1 * a2 - c2 * a1) / d),)


def reduce(
    objs: Sequence[ObjNum],
    existing_points: Sequence[PointNum],
    rng: Generator,
) -> tuple[PointNum, ...]:
    """
    If all PointNum, then no touch.
    Else reduce intersecting objects into one point of intersections other than the existing points.
    """
    if all(isinstance(o, PointNum) for o in objs):
        return tuple(objs)  # type: ignore
    elif len(objs) == 1:
        obj = objs[0]
        assert isinstance(obj, FormNum)
        return (obj.sample_within(existing_points, rng=rng),)

    elif len(objs) == 2:
        u, v = objs
        assert isinstance(u, FormNum)
        assert isinstance(v, FormNum)
        result = np.array(intersect(u, v))
        rng.shuffle(result)
        if len(result) == 0:
            raise InvalidIntersectError("No intersection")
        # may have multiple points, choose the one that is not close to existing points
        for p in result:
            if all(not p.close_enough(x) for x in existing_points):
                return (p,)
        raise InvalidReduceError("All possible intersections are too close to existing points.")
    else:
        raise NotImplementedError

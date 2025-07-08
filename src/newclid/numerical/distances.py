from typing import Sequence
from newclid.numerical.geometries import PointNum


class PointTooCloseError(Exception):
    pass


class PointTooFarError(Exception):
    pass


def check_too_close_numerical(
    newpoints: Sequence[PointNum], points: Sequence[PointNum], tol: float = 0.1
) -> bool:
    if len(points) < 2:
        return False
    avg = sum(points, PointNum(0.0, 0.0)) * 1.0 / len(points)
    mindist = min(p.distance(avg) for p in points)
    for p0 in newpoints:
        for p1 in points:
            if p0.distance(p1) < tol * mindist:
                return True
    return False


def check_too_far_numerical(
    newpoints: Sequence[PointNum], points: Sequence[PointNum], tol: float = 4.0
) -> bool:
    if len(points) < 2:
        return False
    avg = sum(points, PointNum(0.0, 0.0)) * 1.0 / len(points)
    maxdist = max([p.distance(avg) for p in points])
    for p0 in newpoints:
        for p1 in points:
            if p0.distance(p1) > tol * maxdist:
                return True
    return False

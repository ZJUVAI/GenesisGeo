"""Numerical counterexample search for geometry rule validation.

For each candidate rule (P => C), searches for point configurations where
premises P hold but conclusions C do not, using multi-start numerical optimization.

Approach:
  1. Represent points as (dx, dy) offsets from nominal coordinates.
  2. Define a continuous "violation" for each geometric predicate (0 = satisfied).
  3. Minimize: sum(premise_violation^2) + lambda * sum(max(0, margin - concl_violation)^2)
  4. Multi-start with random perturbations + L-BFGS-B local optimization.
  5. Report any configuration where premises hold (~satisfied) but conclusions fail.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable

import numpy as np

from newclid.numerical import close_enough, nearly_zero
from newclid.numerical.geometries import PointNum, LineNum, CircleNum

# ============================================================================
# Constants
# ============================================================================

SATISFY_EPS = 1e-8       # threshold for "exact" satisfaction (legacy, unused)
SATISFY_MARGIN = 1e-6    # margin for conclusion penalty in optimization
DEFAULT_LAMBDA = 0.1     # weight of conclusion penalty relative to premise loss
MAX_RESTARTS = 100       # number of random restarts (60% local + 40% global)
LBFGS_MAXITER = 800      # max iterations for L-BFGS-B per restart

# DDAR matching tolerances (from DDAR numerical.cpp and matcher.hpp)
_DDAR_ATOM = 1e-9
_DDAR_REL_TOL = 0.001

# ============================================================================
# DDAR-style numerical helpers
# ============================================================================


def _ddar_close(a: float, b: float) -> bool:
    """Same tolerance as DDAR Numerical::close_enough.

    abs(a-b) < 4*ATOM  OR  abs(a-b)/max(|a|,|b|) < _DDAR_REL_TOL
    """
    return (abs(a - b) < 4 * _DDAR_ATOM or
            abs(a - b) / max(abs(a), abs(b), 1e-32) < _DDAR_REL_TOL)


def _relative_violation(a: float, b: float) -> float:
    """Continuous violation matching DDAR's close_enough tolerance.

    Returns 0 when DDAR would consider values equal.
    Returns relative difference when they differ.
    """
    abs_diff = abs(a - b)
    if abs_diff < 4 * _DDAR_ATOM:
        return 0.0
    denom = max(abs(a), abs(b), 1e-32)
    rel_diff = abs_diff / denom
    if rel_diff < _DDAR_REL_TOL:
        return 0.0
    return rel_diff


def _cross(v1: PointNum, v2: PointNum) -> float:
    """2D cross product."""
    return v1.x * v2.y - v1.y * v2.x


def _dot(v1: PointNum, v2: PointNum) -> float:
    """2D dot product."""
    return v1.x * v2.x + v1.y * v2.y


def _angle_equal_ddar(
    a: PointNum, b: PointNum,  # slope 1: from a to b
    c: PointNum, d: PointNum,  # slope 2: from c to d
    e: PointNum, f: PointNum,  # slope 3: from e to f
    g: PointNum, h: PointNum,  # slope 4: from g to h
) -> tuple[bool, float]:
    """Check if angle(slope1, slope2) == angle(slope3, slope4) using DDAR's method.

    DDAR EqAngle::check_equations():
      cross1/dot1 = cross2/dot2  ⟺  cross1 * dot2 = cross2 * dot1

    Returns (is_satisfied, violation).
    """
    v1 = b - a  # slope 1 direction
    v2 = d - c  # slope 2 direction
    v3 = f - e  # slope 3 direction
    v4 = h - g  # slope 4 direction

    cross1 = _cross(v1, v2)
    dot1 = _dot(v1, v2)
    cross2 = _cross(v3, v4)
    dot2 = _dot(v3, v4)

    lhs = cross1 * dot2
    rhs = cross2 * dot1

    satisfied = _ddar_close(lhs, rhs)
    viol = _relative_violation(lhs, rhs)
    return satisfied, viol

# ============================================================================
# Numerical predicate evaluation — returns (is_satisfied: bool, violation: float)
# violation = 0 means predicate is perfectly satisfied
# ============================================================================


def _eval_coll(args: list[PointNum]) -> tuple[bool, float]:
    """coll A B C: points are collinear (DDAR-style cross-product check).

    DDAR Coll::check_equations():
      Pick the median-x point as reference, check cross product == 0.
    """
    if len(args) < 3:
        return True, 0.0

    # Sort by x, use median as reference (DDAR approach)
    sorted_pts = sorted(args[:3], key=lambda p: p.x)
    ref = sorted_pts[1]      # median x
    a, c = sorted_pts[0], sorted_pts[2]

    # Cross product: (a-ref) × (c-ref)
    lhs = (a.x - ref.x) * (c.y - ref.y)
    rhs = (c.x - ref.x) * (a.y - ref.y)

    satisfied = _ddar_close(lhs, rhs)
    viol = _relative_violation(lhs, rhs)

    # Multi-point: check additional points against first line
    for p in args[3:]:
        lhs2 = (a.x - ref.x) * (p.y - ref.y)
        rhs2 = (p.x - ref.x) * (a.y - ref.y)
        v = _relative_violation(lhs2, rhs2)
        viol = max(viol, v)
        if not _ddar_close(lhs2, rhs2):
            satisfied = False

    return satisfied, viol


def _eval_ncoll(args: list[PointNum]) -> tuple[bool, float]:
    """ncoll A B C: NOT collinear. Hinge based on collinearity violation."""
    sat_coll, viol_coll = _eval_coll(args[:3])
    satisfied = not sat_coll
    viol = max(0.0, _DDAR_REL_TOL - viol_coll)
    return satisfied, viol


def _eval_npara(args: list[PointNum]) -> tuple[bool, float]:
    """npara A B C D: AB NOT parallel to CD."""
    sat_para, viol_para = _eval_para(args[:4])
    satisfied = not sat_para
    viol = max(0.0, _DDAR_REL_TOL - viol_para)
    return satisfied, viol


def _eval_nperp(args: list[PointNum]) -> tuple[bool, float]:
    """nperp A B C D: AB NOT perpendicular to CD."""
    sat_perp, viol_perp = _eval_perp(args[:4])
    satisfied = not sat_perp
    viol = max(0.0, _DDAR_REL_TOL - viol_perp)
    return satisfied, viol


def _eval_cong(args: list[PointNum]) -> tuple[bool, float]:
    """cong A B C D...: all segment distances equal (DDAR: close_enough on lengths)."""
    if len(args) < 4:
        return True, 0.0
    length = None
    all_ok, max_viol = True, 0.0
    for i in range(0, len(args), 2):
        a, b = args[i], args[i + 1]
        l = math.sqrt(max(a.distance2(b), 1e-16))
        if length is not None:
            ok = _ddar_close(length, l)
            v = _relative_violation(length, l)
            all_ok = all_ok and ok
            max_viol = max(max_viol, v)
        length = l
    return all_ok, max_viol


def _eval_para(args: list[PointNum]) -> tuple[bool, float]:
    """para A B C D...: all lines parallel (DDAR: close_enough on cross product)."""
    if len(args) < 4:
        return True, 0.0
    all_ok, max_viol = True, 0.0
    dir_vec = None
    for i in range(0, len(args), 2):
        a, b = args[i], args[i + 1]
        d = b - a
        d_len = math.sqrt(max(d.x * d.x + d.y * d.y, 1e-16))
        ux, uy = d.x / d_len, d.y / d_len
        if dir_vec is not None:
            cross = abs(ux * dir_vec[1] - uy * dir_vec[0])
            # sin(angle) close to 0 → parallel. Use REL_TOL directly.
            ok = cross < _DDAR_REL_TOL
            v = cross
            all_ok = all_ok and ok
            max_viol = max(max_viol, v)
        dir_vec = (ux, uy)
    return all_ok, max_viol


def _eval_perp(args: list[PointNum]) -> tuple[bool, float]:
    """perp A B C D...: all pairs perpendicular (DDAR: close_enough on dot product)."""
    if len(args) < 4:
        return True, 0.0
    all_ok, max_viol = True, 0.0
    for i in range(0, len(args), 4):
        a, b, c, d = args[i], args[i + 1], args[i + 2], args[i + 3]
        ab, cd = b - a, d - c
        ab2 = max(ab.x * ab.x + ab.y * ab.y, 1e-16)
        cd2 = max(cd.x * cd.x + cd.y * cd.y, 1e-16)
        cos = (ab.x * cd.x + ab.y * cd.y) / math.sqrt(ab2 * cd2)
        ok = abs(cos) < _DDAR_REL_TOL
        v = abs(cos)
        all_ok = all_ok and ok
        max_viol = max(max_viol, v)
    return all_ok, max_viol


def _eval_eqratio(args: list[PointNum]) -> tuple[bool, float]:
    """eqratio A B C D E F G H...: AB/CD = EF/GH (DDAR: close_enough on ratios)."""
    if len(args) < 8:
        return True, 0.0
    all_ok, max_viol = True, 0.0
    ref_ratio = None
    for i in range(0, len(args), 4):
        a, b, c, d = args[i], args[i + 1], args[i + 2], args[i + 3]
        num = math.sqrt(max(a.distance2(b), 1e-16))
        den = math.sqrt(max(c.distance2(d), 1e-16))
        r = num / max(den, 1e-16)
        if ref_ratio is not None:
            ok = _ddar_close(ref_ratio, r)
            v = _relative_violation(ref_ratio, r)
            all_ok = all_ok and ok
            max_viol = max(max_viol, v)
        ref_ratio = r
    return all_ok, max_viol


def _eval_midp(args: list[PointNum]) -> tuple[bool, float]:
    """midp M A B: M is midpoint of AB (DDAR tolerance)."""
    m, a, b = args[0], args[1], args[2]
    expected = (a + b) * 0.5
    dist = abs(m - expected)
    ab_len = math.sqrt(max(a.distance2(b), 1e-16))
    v = _relative_violation(dist, ab_len)  # relative to segment length
    ok = v < _DDAR_REL_TOL  # relative violation already computed
    return ok, v


def _eval_eqangle(args: list[PointNum]) -> tuple[bool, float]:
    """eqangle A B C D E F G H...: ∠(AB,CD) = ∠(EF,GH) (DDAR method).

    DDAR EqAngle::check_equations():
      cross(s1,s2) * dot(s3,s4) ≈ cross(s3,s4) * dot(s1,s2)
      where s1=(A,B), s2=(C,D), s3=(E,F), s4=(G,H)
      This avoids atan2 and is numerically stable.
    """
    if len(args) < 8 or len(args) % 4 != 0:
        return True, 0.0

    all_satisfied = True
    max_viol = 0.0

    # Compare each angle group to the first
    a1, b1, c1, d1 = args[0], args[1], args[2], args[3]
    for i in range(0, len(args), 4):
        a2, b2, c2, d2 = args[i], args[i+1], args[i+2], args[i+3]
        sat, viol = _angle_equal_ddar(a1, b1, c1, d1, a2, b2, c2, d2)
        max_viol = max(max_viol, viol)
        if not sat:
            all_satisfied = False

    return all_satisfied, max_viol


def _eval_eqratio(args: list[PointNum]) -> tuple[bool, float]:
    """eqratio A B C D E F G H...: AB/CD = EF/GH = ..."""
    if len(args) < 8:
        return True, 0.0
    ratios = []
    for i in range(0, len(args), 4):
        a, b, c, d = args[i], args[i + 1], args[i + 2], args[i + 3]
        num = math.sqrt(max(a.distance2(b), 1e-16))
        den = math.sqrt(max(c.distance2(d), 1e-16))
        ratios.append(num / max(den, 1e-16))
    if not ratios:
        return True, 0.0
    ref = ratios[0]
    max_viol = 0.0
    for r in ratios[1:]:
        v = abs(r - ref)
        max_viol = max(max_viol, v)
    return max_viol < SATISFY_EPS, max_viol


def _det4x4(m: list[list[float]]) -> float:
    """Compute 4x4 determinant via Laplace expansion on first row."""
    a, b, c, d = m[0]
    e, f, g, h = m[1]
    i, j, k, l = m[2]
    M, n, o, p = m[3]

    # 3x3 minors (cofactor expansion, first row)
    m0 = f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n)
    m1 = e * (k * p - l * o) - g * (i * p - l * M) + h * (i * o - k * M)
    m2 = e * (j * p - l * n) - f * (i * p - l * M) + h * (i * n - j * M)
    m3 = e * (j * o - k * n) - f * (i * o - k * M) + g * (i * n - j * M)

    return a * m0 - b * m1 + c * m2 - d * m3


def _eval_cyclic(args: list[PointNum]) -> tuple[bool, float]:
    """cyclic A B C D: points are concyclic (4x4 determinant method).

    Points (x_i, y_i) are concyclic iff:
      | x1^2+y1^2  x1  y1  1 |
      | x2^2+y2^2  x2  y2  1 | = 0
      | x3^2+y3^2  x3  y3  1 |
      | x4^2+y4^2  x4  y4  1 |

    This is invariant to point ordering and handles all degenerate sign cases
    that the oriented angle method doesn't.
    """
    if len(args) < 4:
        return True, 0.0

    max_viol = 0.0
    all_satisfied = True

    for d_idx in range(3, len(args)):
        a, b, c, d = args[0], args[1], args[2], args[d_idx]

        # Build 4x4 matrix rows
        pts = [a, b, c, d]
        m = []
        for p in pts:
            r2 = p.x * p.x + p.y * p.y
            m.append([r2, p.x, p.y, 1.0])

        det = _det4x4(m)

        # Normalize: determinant scales as O(L^3) where L is coordinate magnitude.
        # Divide by max_pairwise_distance^3 to get a dimensionless measure.
        max_dist = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                d2 = pts[i].distance2(pts[j])
                max_dist = max(max_dist, d2)
        scale = max(math.sqrt(max_dist), 1e-8) ** 3

        norm_det = abs(det) / max(scale, 1e-16)

        # Use normalized determinant: scale-invariant, always well-defined
        satisfied = norm_det < _DDAR_REL_TOL
        viol = norm_det
        max_viol = max(max_viol, viol)
        if not satisfied:
            all_satisfied = False

    return all_satisfied, max_viol


def _eval_midp(args: list[PointNum]) -> tuple[bool, float]:
    """midp M A B: M is midpoint of AB."""
    m, a, b = args[0], args[1], args[2]
    expected = (a + b) * 0.5
    viol = abs(m - expected)
    return viol < SATISFY_EPS, viol


def _eval_diff(args: list[PointNum]) -> tuple[bool, float]:
    """diff A B: A and B are different points. Hinge loss."""
    a, b = args[0], args[1]
    dist = abs(a - b)
    satisfied = dist > 4 * _DDAR_ATOM  # DDAR: close_enough(x,0) ≈ abs(x) < 4*ATOM
    viol = max(0.0, _DDAR_REL_TOL - _relative_violation(dist, _DDAR_REL_TOL))
    return satisfied, viol


def _eval_simtri(args: list[PointNum]) -> tuple[bool, float]:
    """simtri A B C D E F: triangle ABC similar to DEF (DDAR tolerance on ratios)."""
    if len(args) < 6:
        return True, 0.0
    a, b, c, d, e, f = args[0], args[1], args[2], args[3], args[4], args[5]
    r1 = math.sqrt(max(a.distance2(b), 1e-16)) / max(math.sqrt(max(d.distance2(e), 1e-16)), 1e-16)
    r2 = math.sqrt(max(b.distance2(c), 1e-16)) / max(math.sqrt(max(e.distance2(f), 1e-16)), 1e-16)
    r3 = math.sqrt(max(c.distance2(a), 1e-16)) / max(math.sqrt(max(f.distance2(d), 1e-16)), 1e-16)
    v12 = _relative_violation(r1, r2)
    v23 = _relative_violation(r2, r3)
    viol = max(v12, v23)
    ok = _ddar_close(r1, r2) and _ddar_close(r2, r3)
    return ok, viol


def _eval_sameclock(args: list[PointNum]) -> tuple[bool, float]:
    """sameclock A B C D E F: same orientation of triangles ABC and DEF."""
    from newclid.numerical.check import same_clock

    a, b, c, d, e, f = args[0], args[1], args[2], args[3], args[4], args[5]
    ok = same_clock(a, b, c, d, e, f)
    # Continuous: product of clock signs (positive = same orientation)
    clock1 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    clock2 = (e.x - d.x) * (f.y - d.y) - (e.y - d.y) * (f.x - d.x)
    viol = 0.0 if ok else _relative_violation(abs(clock1), abs(clock2))
    return ok, viol


def _eval_sameside(args: list[PointNum]) -> tuple[bool, float]:
    """sameside A B C X Y Z: A is to the same side of B->C as X is to Y->Z.

    Mirrors SameSide.check_numerical: sign((B-A).(C-A)) == sign((Y-X).(Z-X)).
    Continuous violation: 0 when signs agree, otherwise the smaller of the
    two |dot| magnitudes (hinge — a near-zero dot on either side means the
    point sits right on the boundary line, so this is a weak/marginal call).
    """
    a, b, c, x, y, z = args[0], args[1], args[2], args[3], args[4], args[5]
    d1 = (b - a).dot(c - a)
    d2 = (y - x).dot(z - x)
    ok = (d1 > 0) == (d2 > 0)
    viol = 0.0 if ok else min(abs(d1), abs(d2))
    return ok, viol


def _eval_nsameside(args: list[PointNum]) -> tuple[bool, float]:
    """nsameside A B C X Y Z: negation of sameside."""
    sat_same, viol_same = _eval_sameside(args)
    return not sat_same, viol_same


def _eval_obtuse(args: list[PointNum]) -> tuple[bool, float]:
    """obtuse A B C: angle ABC > 90 degrees (numerical check only)."""
    a, b, c = args[0], args[1], args[2]
    ba = a - b
    bc = c - b
    dot = ba.x * bc.x + ba.y * bc.y
    # obtuse: dot < 0 (cos < 0, angle > 90)
    satisfied = dot < -1e-9
    # violation: 0 when obtuse, positive when not. Use hinge: max(0, dot)
    viol = max(0.0, dot)
    return satisfied, viol


def _eval_acute(args: list[PointNum]) -> tuple[bool, float]:
    """acute A B C: angle ABC < 90 degrees (numerical check only)."""
    a, b, c = args[0], args[1], args[2]
    ba = a - b
    bc = c - b
    dot = ba.x * bc.x + ba.y * bc.y
    # acute: dot > 0 AND not near zero
    satisfied = dot > 1e-9
    viol = max(0.0, -dot)  # 0 when acute, positive when obtuse/right
    return satisfied, viol


# Registry
_PRED_EVAL: dict[str, Callable] = {
    "coll": _eval_coll,
    "ncoll": _eval_ncoll,
    "cong": _eval_cong,
    "para": _eval_para,
    "npara": _eval_npara,
    "perp": _eval_perp,
    "nperp": _eval_nperp,
    "eqangle": _eval_eqangle,
    "eqratio": _eval_eqratio,
    "cyclic": _eval_cyclic,
    "midp": _eval_midp,
    "diff": _eval_diff,
    "simtri": _eval_simtri,
    "contri": _eval_simtri,  # same numerical check as simtri
    "sameclock": _eval_sameclock,
    "obtuse": _eval_obtuse,
    "acute": _eval_acute,
    "sameside": _eval_sameside,
    "nsameside": _eval_nsameside,
}

# Predicates we can handle
_SUPPORTED = set(_PRED_EVAL.keys())


def supported_predicate(name: str) -> bool:
    """Check if a predicate is supported for numerical evaluation."""
    return name in _SUPPORTED


def evaluate_predicate(
    name: str, args: list[str],
    points: dict[str, PointNum],
) -> tuple[bool, float]:
    """Evaluate a geometric predicate numerically.

    Args:
        name: predicate name (coll, cong, cyclic, etc.)
        args: point argument names
        points: name -> PointNum mapping

    Returns:
        (is_satisfied: bool, violation: float)
        violation is 0.0 when the predicate is perfectly satisfied.
    """
    fn = _PRED_EVAL.get(name)
    if fn is None:
        return True, 0.0

    pt_args = []
    for a in args:
        if a not in points:
            return True, 0.0
        pt_args.append(points[a])

    return fn(pt_args)


# ============================================================================
# Counterexample search via multi-start optimization
# ============================================================================


def _softplus_hinge(v: float, margin: float) -> float:
    """max(0, margin - v)^2 — penalty when v < margin (predicate too satisfied)."""
    d = margin - v
    return d * d if d > 0 else 0.0


class CounterexampleFinder:
    """Search for counterexamples to a geometric rule.

    A rule is: premises => conclusions
    A counterexample is a point configuration where premises hold but at least
    one conclusion fails.

    Uses multi-start L-BFGS-B optimization starting from randomly perturbed
    nominal coordinates.
    """

    def __init__(
        self,
        premises: list[tuple[str, list[str]]],
        conclusions: list[tuple[str, list[str]]],
        nominal_points: dict[str, PointNum],
        lambda_conclusion: float = DEFAULT_LAMBDA,
        margin: float = SATISFY_MARGIN,
    ):
        self.premises = premises
        self.conclusions = conclusions
        self.nominal = nominal_points
        self.lambda_conclusion = lambda_conclusion
        self.margin = margin

        # Build ordered point list for optimization vector
        self.point_names = sorted(nominal_points.keys())
        self.n_vars = 2 * len(self.point_names)

        # Nominal offsets (= 0 initially)
        self.x0 = np.zeros(self.n_vars)

    def _offsets_to_points(self, x: np.ndarray) -> dict[str, PointNum]:
        """Convert (dx, dy) offset vector to PointNum dict."""
        pts = {}
        for i, name in enumerate(self.point_names):
            nom = self.nominal[name]
            pts[name] = PointNum(nom.x + x[2 * i], nom.y + x[2 * i + 1])
        return pts

    def _compute_loss(self, x: np.ndarray) -> float:
        """Compute combined loss: premise violations + conclusion penalty."""
        pts = self._offsets_to_points(x)

        # Premise loss: sum of squared violations
        premise_loss = 0.0
        for pred_name, args in self.premises:
            _sat, viol = evaluate_predicate(pred_name, args, pts)
            premise_loss += viol * viol

        # Conclusion penalty: penalize when conclusions ARE satisfied
        concl_penalty = 0.0
        for pred_name, args in self.conclusions:
            _sat, viol = evaluate_predicate(pred_name, args, pts)
            concl_penalty += _softplus_hinge(viol, self.margin)

        return premise_loss + self.lambda_conclusion * concl_penalty

    def _check_counterexample(self, x: np.ndarray) -> tuple[bool, dict]:
        """Check if x constitutes a valid counterexample."""
        pts = self._offsets_to_points(x)

        # Must satisfy ALL premises
        premise_results = {}
        all_premises_ok = True
        for pred_name, args in self.premises:
            sat, viol = evaluate_predicate(pred_name, args, pts)
            premise_results[f"{pred_name} {' '.join(args)}"] = (sat, viol)
            if not sat:
                all_premises_ok = False

        if not all_premises_ok:
            return False, {"premises": premise_results}

        # At least one conclusion must FAIL
        conclusion_results = {}
        any_conclusion_fails = False
        for pred_name, args in self.conclusions:
            sat, viol = evaluate_predicate(pred_name, args, pts)
            conclusion_results[f"{pred_name} {' '.join(args)}"] = (sat, viol)
            if not sat:
                any_conclusion_fails = True

        return any_conclusion_fails, {
            "premises": premise_results,
            "conclusions": conclusion_results,
        }

    def _compute_perturbation_scale(self, restart_idx: int) -> float:
        """Perturbation scale with two phases: local + global exploration.

        First 60%: geometric progression 0.001 → 0.5 (local search)
        Last 40%:  large jumps 1.0 → 5.0 (global exploration)
        """
        total = MAX_RESTARTS
        if restart_idx < total * 0.6:
            # Local phase
            frac = restart_idx / (total * 0.6)
            return 0.001 * (500 ** frac)
        else:
            # Global phase: jump to entirely different configurations
            frac = (restart_idx - total * 0.6) / (total * 0.4)
            return 1.0 + frac * 4.0  # 1.0 → 5.0

    def search(
        self,
        max_restarts: int = MAX_RESTARTS,
        verbose: bool = False,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Run multi-start counterexample search.

        Returns:
            dict with keys:
              - counterexample_found: bool
              - counterexample_points: dict name->(x,y) if found
              - details: dict with premise/conclusion truth values at counterexample
              - n_restarts: number of restarts tried
              - runtime: seconds
        """
        t0 = time.time()
        rng = np.random.RandomState(random_seed)

        for restart in range(max_restarts):
            scale = self._compute_perturbation_scale(restart)

            # Generate perturbation: uniform in [-scale, scale]
            x_init = rng.uniform(-scale, scale, self.n_vars)

            try:
                from scipy.optimize import minimize

                res = minimize(
                    self._compute_loss,
                    x_init,
                    method="L-BFGS-B",
                    options={"maxiter": LBFGS_MAXITER, "ftol": 1e-14, "gtol": 1e-8},
                )

                is_ce, details = self._check_counterexample(res.x)
                if is_ce:
                    pts = self._offsets_to_points(res.x)
                    return {
                        "counterexample_found": True,
                        "counterexample_points": {
                            name: (p.x, p.y) for name, p in pts.items()
                        },
                        "details": details,
                        "n_restarts": restart + 1,
                        "runtime": time.time() - t0,
                        "final_loss": float(res.fun),
                        "start_scale": float(scale),
                    }
            except Exception:
                continue

        return {
            "counterexample_found": False,
            "n_restarts": max_restarts,
            "runtime": time.time() - t0,
        }


# ============================================================================
# Public API
# ============================================================================


def validate_rule(
    rule_text: str,
    points: list[dict],
    label: str = "",
) -> dict[str, Any]:
    """Validate a single geometry rule by searching for counterexamples.

    Args:
        rule_text: "prem1 arg..., prem2 arg... => concl1 arg..., concl2 arg..."
        points: [{"name": "A", "x": 0.5, "y": -1.2}, ...]
        label: optional identifier for logging

    Returns:
        Validation result dict (see CounterexampleFinder.search())
    """
    from newclid.discovery.utils.rule_parser import parse_predicate, split_rule_text

    # Parse rule text
    prem_strs, concl_str = split_rule_text(rule_text)
    premises = [
        (name, list(args))
        for name, args in (parse_predicate(p) for p in prem_strs if p.strip())
    ]

    # Handle multi-conclusion: split RHS by comma
    conclusions = []
    for c in concl_str.split(","):
        c = c.strip()
        if not c:
            continue
        name, args = parse_predicate(c)
        conclusions.append((name, list(args)))

    # Build nominal points
    nominal = {}
    for p in points:
        nominal[p["name"]] = PointNum(p["x"], p["y"])

    # Check for unsupported predicates
    unsupported_prem = [n for n, _ in premises if not supported_predicate(n)]
    unsupported_concl = [n for n, _ in conclusions if not supported_predicate(n)]
    if unsupported_prem or unsupported_concl:
        return {
            "counterexample_found": None,  # unknown
            "skipped": True,
            "reason": f"unsupported predicates: premises={unsupported_prem}, "
                      f"conclusions={unsupported_concl}",
        }

    finder = CounterexampleFinder(premises, conclusions, nominal)
    return finder.search(verbose=False)


def validate_rules(
    rules: list[dict],
    label: str = "",
) -> list[dict]:
    """Validate a batch of rules. Each dict must have 'rule_text' and 'points'."""
    results = []
    for i, rule in enumerate(rules):
        label_i = f"{label}[{i}]" if label else f"rule[{i}]"
        result = validate_rule(
            rule["rule_text"],
            rule["points"],
            label=label_i,
        )
        result["rule_text"] = rule["rule_text"]
        result["rule_id"] = rule.get("rule_id", label_i)
        results.append(result)
    return results

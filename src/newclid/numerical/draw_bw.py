"""Black-and-white geometry figure drawer.

Each construction type is handled explicitly, mapping to the correct
geometric primitives (segments, circles) without color annotations.
All strokes use a uniform line width. Infinite lines/rays are replaced
by segments between known points wherever possible.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
from matplotlib.figure import Figure
import numpy as np
from adjustText import adjust_text

from newclid.dependencies.symbols import Point
from newclid.formulations.problem import ProblemJGEX
from newclid.numerical.geometries import CircleNum, PointNum


# ---------------------------------------------------------------------------
# Style constants — single source of truth for line width
# ---------------------------------------------------------------------------

LW = 0.8          # uniform stroke width for all elements
SOLID  = {"color": "black", "lw": LW}
DASHED = {"color": "black", "lw": LW, "ls": "dashed", "alpha": 0.55}
DOTTED = {"color": "black", "lw": LW, "ls": "dotted", "alpha": 0.55}


def _seg(ax, p0: PointNum, p1: PointNum, **kw):
    style = {**SOLID, **kw}
    ax.plot([p0.x, p1.x], [p0.y, p1.y], **style)


def _circ(ax, c: CircleNum, **kw):
    style = {"color": "black", "fill": False, "lw": LW, **kw}
    ax.add_patch(patches.Circle((c.center.x, c.center.y), c.radius, **style))


def _num(sg, name: str) -> Optional[PointNum]:
    p = sg.name2node.get(name)
    return p.num if p is not None else None


def _pt_dist_to_line(p: PointNum, a: PointNum, b: PointNum) -> float:
    ab = np.array([b.x - a.x, b.y - a.y])
    ap = np.array([p.x - a.x, p.y - a.y])
    denom = np.linalg.norm(ab)
    return abs(float(np.cross(ab, ap))) / denom if denom > 1e-10 else 0.0


def _triangle(ax, a, b, c, **kw):
    for p, q in [(a, b), (b, c), (c, a)]:
        if p and q:
            _seg(ax, p, q, **kw)


def _angle_arc(ax, vertex: PointNum, p1: PointNum, p2: PointNum):
    """Draw a small arc at vertex spanning the angle p1-vertex-p2."""
    d1 = np.array([p1.x - vertex.x, p1.y - vertex.y])
    d2 = np.array([p2.x - vertex.x, p2.y - vertex.y])
    r = min(np.linalg.norm(d1), np.linalg.norm(d2)) * 0.25
    a1 = np.degrees(np.arctan2(d1[1], d1[0]))
    a2 = np.degrees(np.arctan2(d2[1], d2[0]))
    if (a2 - a1) % 360 > 180:
        a1, a2 = a2, a1
    ax.add_patch(patches.Arc(
        (vertex.x, vertex.y), 2*r, 2*r,
        angle=0, theta1=a1, theta2=a2,
        color="black", lw=LW))


# ---------------------------------------------------------------------------
# Per-construction drawing dispatch
# ---------------------------------------------------------------------------

def _handle_construction(ax, cname: str, args: list[str], sg, color: str = "black") -> None:
    pts = [_num(sg, a) for a in args]

    def p(i):
        return pts[i] if i < len(pts) else None

    def seg(p0, p1, **extra):
        _seg(ax, p0, p1, color=color, **extra)

    def circ(c, **extra):
        _circ(ax, c, color=color, **extra)

    def tri(a, b, c, **extra):
        _triangle(ax, a, b, c, color=color, **extra)

    if cname in ("free", "point"):
        pass

    elif cname == "segment":
        seg(p(0), p(1))

    elif cname in ("triangle", "triangle12", "r_triangle", "iso_triangle",
                   "iso_triangle0", "ieq_triangle", "eq_triangle", "risos"):
        tri(p(0), p(1), p(2))

    elif cname in ("quadrangle", "eq_quadrangle",
                   "trapezoid", "iso_trapezoid", "r_trapezoid",
                   "rectangle", "parallelogram", "square", "isquare", "eqangle2"):
        for i in range(4):
            seg(pts[i], pts[(i+1) % 4])

    elif cname == "eqdia_quadrangle":
        for i in range(4):
            seg(pts[i], pts[(i+1) % 4])
        seg(p(0), p(2))
        seg(p(1), p(3))

    elif cname in ("nsquare", "psquare"):
        seg(p(1), p(2))
        seg(p(0), p(1))

    elif cname == "pentagon":
        for i in range(5):
            seg(pts[i], pts[(i+1) % 5])

    elif cname in ("circle", "circumcenter"):
        circ(CircleNum(p1=p(1), p2=p(2), p3=p(3)))

    elif cname == "ninepoints":
        tri(p(4), p(5), p(6))
        circ(CircleNum(p1=p(0), p2=p(1), p3=p(2)))

    elif cname == "on_circle":
        circ(CircleNum(center=p(1), p1=p(2)))

    elif cname == "on_circum":
        circ(CircleNum(p1=p(1), p2=p(2), p3=p(3)))

    elif cname == "on_dia":
        tri(p(0), p(1), p(2))
        circ(CircleNum(p1=p(0), p2=p(1), p3=p(2)))

    elif cname == "midpoint":
        seg(p(1), p(2))

    elif cname == "foot":
        seg(p(2), p(0))
        seg(p(2), p(3))
        seg(p(0), p(1))

    elif cname == "on_line":
        seg(p(0), p(1))
        seg(p(0), p(2))

    elif cname in ("on_pline", "on_pline0", "on_tline"):
        seg(p(2), p(3))
        seg(p(0), p(1))

    elif cname == "on_bline":
        tri(p(0), p(1), p(2))

    elif cname == "on_aline":
        seg(p(0), p(1))
        seg(p(1), p(2))
        seg(p(3), p(4))
        seg(p(4), p(5))

    elif cname in ("on_aline0", "eqratio"):
        seg(p(1), p(2))
        seg(p(3), p(4))
        seg(p(5), p(6))
        seg(p(0), p(7))

    elif cname == "eqratio6":
        seg(p(0), p(1))
        seg(p(0), p(2))
        seg(p(3), p(4))
        seg(p(5), p(6))

    elif cname in ("angle_bisector", "external_bisector", "angle_mirror"):
        seg(p(2), p(1))
        seg(p(2), p(3))
        seg(p(2), p(0))

    elif cname in ("incenter", "excenter", "orthocenter"):
        tri(p(1), p(2), p(3))
        seg(p(0), p(1))
        seg(p(0), p(2))
        seg(p(0), p(3))

    elif cname in ("incenter2", "excenter2"):
        tri(p(4), p(5), p(6))
        seg(p(3), p(0))
        seg(p(3), p(1))
        seg(p(3), p(2))
        seg(p(3), p(4))
        seg(p(3), p(5))
        seg(p(3), p(6))
        seg(p(0), p(5))
        seg(p(1), p(6))
        seg(p(2), p(4))

    elif cname == "centroid":
        tri(p(4), p(5), p(6))
        seg(p(4), p(0))
        seg(p(5), p(1))
        seg(p(6), p(2))

    elif cname == "mirror":
        seg(p(0), p(1))

    elif cname == "reflect":
        seg(p(2), p(3))

    elif cname == "tangent":
        circ(CircleNum(center=p(3), p1=p(4)))
        seg(p(2), p(0))
        seg(p(2), p(1))

    elif cname == "lc_tangent":
        circ(CircleNum(center=p(2), p1=p(1)))
        seg(p(0), p(1))

    elif cname == "cc_tangent":
        circ(CircleNum(center=p(4), p1=p(5)))
        circ(CircleNum(center=p(6), p1=p(7)))
        seg(p(0), p(1))
        seg(p(2), p(3))

    elif cname == "intersection_cc":
        circ(CircleNum(center=p(1), p1=p(3)))
        circ(CircleNum(center=p(2), p1=p(3)))

    elif cname == "intersection_lc":
        circ(CircleNum(center=p(2), p1=p(3)))
        seg(p(1), p(0))
        seg(p(0), p(3))

    elif cname == "intersection_ll":
        seg(p(1), p(0))
        seg(p(0), p(2))
        seg(p(3), p(0))
        seg(p(0), p(4))

    elif cname in ("intersection_lp", "intersection_lt"):
        seg(p(1), p(0))
        seg(p(0), p(2))
        seg(p(3), p(0))
        seg(p(4), p(5))

    elif cname in ("intersection_pp", "intersection_tt"):
        seg(p(0), p(1))
        seg(p(2), p(3))
        seg(p(0), p(4))
        seg(p(5), p(6))

    elif cname == "shift":
        seg(p(0), p(1))
        seg(p(1), p(3))
        seg(p(2), p(3))
        seg(p(0), p(2))
        
    elif cname == "trisect":
        tri(p(2), p(3), p(4))
        seg(p(0), p(3))
        seg(p(1), p(3))

    elif cname == "trisegment":
        seg(p(2), p(3))

    elif cname == "eqdistance":
        seg(p(0), p(1))
        seg(p(2), p(3))

    elif cname in ("iso_triangle_vertex", "iso_triangle_vertex_angle"):
        tri(p(0), p(1), p(2))
    
    elif cname == "eqangle3":
        seg(p(0), p(1))
        seg(p(0), p(2))
        seg(p(3), p(4))
        seg(p(3), p(5))
    
    elif cname in ("rconst", "aconst"):
        seg(p(0), p(1))
        seg(p(2), p(3))
    
    elif cname == "rconst2":
        seg(p(0), p(1))
        seg(p(0), p(2))
    
    elif cname == "s_angle":
        seg(p(0), p(1))
        seg(p(1), p(2))
    
    elif cname == "lconst":
        seg(p(0), p(1))

    elif cname == "between_bound":
        seg(p(1), p(2))
        


    else:
        pass  # silently skip unrecognized constructions


# ---------------------------------------------------------------------------
# Main drawing entry point
# ---------------------------------------------------------------------------

def _handle_predicate(ax, pred_str: str, sg, color: str = "red") -> bool:
    """Draw geometric elements for a concrete predicate instance.

    pred_str: e.g. "cong a b c d", "para a b c d", "cyclic a b c d"
    Returns True if successful, False if any point is missing.
    """
    tokens = pred_str.strip().split()
    if not tokens:
        return False
    pname = tokens[0].lower()
    pt_names = tokens[1:]

    # Verify all points exist
    pts = []
    for name in pt_names:
        n = _num(sg, name)
        if n is None:
            return False
        pts.append(n)

    def p(i):
        return pts[i] if i < len(pts) else None

    def seg(i, j):
        if p(i) and p(j):
            _seg(ax, p(i), p(j), color=color)

    def circ_3(i, j, k):
        if p(i) and p(j) and p(k):
            _circ(ax, CircleNum(p1=p(i), p2=p(j), p3=p(k)), color=color)

    if pname == "cong":
        # cong a b c d  ->  segments ab, cd
        for i in range(0, len(pts) - 1, 2):
            seg(i, i + 1)

    elif pname in ("para", "perp"):
        # para/perp a b c d  ->  segments ab, cd
        seg(0, 1)
        seg(2, 3)

    elif pname == "coll":
        # coll a b c ...  ->  chain of segments
        for i in range(len(pts) - 1):
            seg(i, i + 1)

    elif pname == "midp":
        # midp m a b  ->  segment ab (m is midpoint)
        seg(1, 2)

    elif pname == "cyclic":
        # cyclic a b c d  ->  circumcircle of first three + all points
        circ_3(0, 1, 2)

    elif pname in ("eqangle",):
        # eqangle a b c d e f g h  ->  segments ab, cd, ef, gh
        for i in range(0, min(len(pts), 8), 2):
            seg(i, i + 1)

    elif pname in ("eqratio",):
        # eqratio a b c d e f g h  ->  segments ab, cd, ef, gh
        for i in range(0, min(len(pts), 8), 2):
            seg(i, i + 1)

    elif pname in ("simtri", "contri", "simtrir", "contrir"):
        # simtri/contri a b c d e f  ->  triangles abc and def
        _triangle(ax, p(0), p(1), p(2), color=color)
        _triangle(ax, p(3), p(4), p(5), color=color)

    elif pname == "circle":
        # circle o a b c  ->  circumcircle of abc
        circ_3(1, 2, 3)

    else:
        # Fallback: just draw segments between consecutive point pairs
        for i in range(0, len(pts) - 1, 2):
            seg(i, i + 1)

    return True


def draw_bw(
    proof,
    problem: ProblemJGEX,
    save_to: Union[str, Path, None] = None,
    dpi: float = 150,
    highlight_clauses: Optional[Union[int, list[int]]] = None,
    highlight_predicates: Optional[list[str]] = None,
):
    """Draw geometry figure.

    highlight_clauses: index or list of indices into problem.constructions to draw in red.
    highlight_predicates: list of predicate strings like ["cong a b c d", "para a b e f"].
        Each string is parsed into a predicate name + point names. Points are validated
        against the current figure; if any point is missing the predicate is skipped.
        Matching geometric elements are drawn on top in red.
    """
    imsize = 512 / 100
    fig = Figure(figsize=(imsize, imsize))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    sg = proof.symbols_graph
    defs = proof.defs

    if highlight_clauses is None:
        red_indices: set[int] = set()
    elif isinstance(highlight_clauses, int):
        red_indices = {highlight_clauses}
    else:
        red_indices = set(highlight_clauses)

    def _draw_clause(clause, color: str):
        clause_pts = tuple(pt.split("@")[0] for pt in clause.points)
        for sentence in clause.sentences:
            cname = sentence[0]
            if cname in defs:
                cdef = defs[cname]
                if len(sentence) == len(cdef.declare):
                    arg_names = list(sentence[1:])
                else:
                    arg_names = list(clause_pts) + list(sentence[1:])
            else:
                arg_names = list(clause_pts) + list(sentence[1:])
            _handle_construction(ax, cname, arg_names, sg, color=color)

    # collect all point names involved in highlighted clauses
    red_points: set[str] = set()
    for i, clause in enumerate(problem.constructions):
        if i in red_indices:
            clause_pts = tuple(pt.split("@")[0] for pt in clause.points)
            for pt in clause_pts:
                red_points.add(pt)
            for sentence in clause.sentences:
                cname = sentence[0]
                if cname in defs:
                    cdef = defs[cname]
                    if len(sentence) == len(cdef.declare):
                        arg_names = list(sentence[1:])
                    else:
                        arg_names = list(clause_pts) + list(sentence[1:])
                else:
                    arg_names = list(clause_pts) + list(sentence[1:])
                for pt in arg_names:
                    red_points.add(pt.split("@")[0])

    # black pass first, then red on top
    for i, clause in enumerate(problem.constructions):
        if i not in red_indices:
            _draw_clause(clause, "black")
    for i, clause in enumerate(problem.constructions):
        if i in red_indices:
            _draw_clause(clause, "red")

    # draw highlighted predicates on top
    pred_red_points: set[str] = set()
    if highlight_predicates:
        for pred_str in highlight_predicates:
            tokens = pred_str.strip().split()
            if len(tokens) >= 2:
                ok = _handle_predicate(ax, pred_str, sg, color="red")
                if ok:
                    for name in tokens[1:]:
                        pred_red_points.add(name)

    all_red_points = red_points | pred_red_points

    points: list[Point] = list(sg.nodes_of_type(Point))

    # -----------------------------------------------------------------------
    # Pixel-based label placement
    # 1. Render all geometry to an offscreen canvas
    # 2. For each point, sample candidate positions in pixel space
    # 3. Pick the position whose text bounding box overlaps fewest dark pixels
    # -----------------------------------------------------------------------

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")

    # Render geometry (no labels yet) to pixel array for occupancy detection.
    import io
    buf_io = io.BytesIO()
    fig.savefig(buf_io, format="png", dpi=dpi, facecolor="white", bbox_inches=None)
    buf_io.seek(0)
    from PIL import Image as _PILImage
    pil_img = _PILImage.open(buf_io).convert("RGB")
    render = np.array(pil_img)  # (H, W, 3) uint8 RGB
    h_px, w_px = render.shape[:2]
    dark = (255.0 - render.mean(axis=2)) / 255.0  # 0=white, positive=ink

    # Linear mapping: data coords → pixel (col, row-from-top).
    # ax fills the whole figure (subplots_adjust left=0,right=1,top=1,bottom=0)
    # so ax.get_position() == (0,0,1,1) and the image spans exactly xlim/ylim.
    pos = ax.get_position()
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    ax_left_px   = pos.x0 * w_px
    ax_right_px  = pos.x1 * w_px
    ax_bottom_px = (1.0 - pos.y0) * h_px   # row (top-origin)
    ax_top_px    = (1.0 - pos.y1) * h_px

    def _pixel_xy(dx: float, dy: float):
        col = ax_left_px  + (dx - xl) / (xr - xl) * (ax_right_px - ax_left_px)
        row = ax_bottom_px + (dy - yb) / (yt - yb) * (ax_top_px - ax_bottom_px)
        return int(round(col)), int(round(row))

    def _pixel_to_data(col: float, row: float):
        dx = xl + (col - ax_left_px) / (ax_right_px - ax_left_px) * (xr - xl)
        dy = yb + (row - ax_bottom_px) / (ax_top_px - ax_bottom_px) * (yt - yb)
        return dx, dy

    # Approximate text box size in pixels for fontsize=9 bold at given dpi
    char_w = max(5, int(6.5 * dpi / 100))
    char_h = max(8, int(12 * dpi / 100))

    # darkness is mutable: we paint placed labels into it so later points avoid them
    dark = dark.copy()

    def _box_darkness(col: int, row: int, label_len: int) -> float:
        """Dark-pixel sum in text bounding box. col/row = top-left corner."""
        bw = char_w * label_len
        r0, r1 = max(0, row), min(h_px, row + char_h)
        c0, c1 = max(0, col), min(w_px, col + bw)
        if r1 <= r0 or c1 <= c0:
            return 1e9
        return float(dark[r0:r1, c0:c1].sum())

    def _paint_label(col: int, row: int, label_len: int):
        """Mark placed label area as occupied in dark map."""
        bw = char_w * label_len
        r0, r1 = max(0, row), min(h_px, row + char_h)
        c0, c1 = max(0, col), min(w_px, col + bw)
        dark[r0:r1, c0:c1] = 1.0

    def _point_crowding(pcol: int, prow: int) -> float:
        """Sum of dark pixels in a radius around the point (measures line density)."""
        r = base_px
        r0, r1 = max(0, prow - r), min(h_px, prow + r)
        c0, c1 = max(0, pcol - r), min(w_px, pcol + r)
        return float(dark[r0:r1, c0:c1].sum())

    N_DIR = 24
    angles = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
    base_px = max(10, int(13 * dpi / 100))
    dist_px_options = [base_px, int(base_px * 1.5), int(base_px * 2.2)]

    # Compute crowding before placing any labels, then sort: most crowded first
    point_info = []
    for pt in points:
        pcol, prow = _pixel_xy(pt.num.x, pt.num.y)
        crowding = _point_crowding(pcol, prow)
        point_info.append((crowding, pt, pcol, prow))
    point_info.sort(key=lambda x: -x[0])  # most crowded first

    placed: dict[str, tuple] = {}  # name → (box_col, box_row, llen)

    for crowding, pt in [(c, p) for c, p, _, _ in point_info]:
        color = "red" if pt.name in all_red_points else "black"
        ax.scatter(pt.num.x, pt.num.y, color=color, s=8, zorder=5)

        pcol, prow = _pixel_xy(pt.num.x, pt.num.y)
        label = pt.name.upper()
        llen = len(label)

        best_col, best_row = pcol + base_px, prow - base_px
        best_score = 1e18
        for dist_px in dist_px_options:
            dist_cost = (dist_px - base_px) * char_w * 0.4
            for angle in angles:
                cand_col = pcol + int(round(dist_px * np.cos(angle)))
                cand_row = prow - int(round(dist_px * np.sin(angle)))
                box_col = cand_col - (char_w * llen) // 2
                box_row = cand_row - char_h // 2
                score = _box_darkness(box_col, box_row, llen) + dist_cost
                if score < best_score:
                    best_score = score
                    best_col, best_row = box_col, box_row

        # Paint this label into dark map so subsequent labels avoid it
        _paint_label(best_col, best_row, llen)
        placed[pt.name] = (best_col, best_row, llen)

        text_cx_px = best_col + (char_w * llen) / 2.0
        text_cy_px = best_row + char_h / 2.0
        ox, oy = _pixel_to_data(text_cx_px, text_cy_px)

        ax.annotate(
            label, (pt.num.x, pt.num.y),
            xytext=(ox, oy),
            textcoords="data",
            fontsize=9, color=color, fontweight="bold",
            annotation_clip=False,
            ha="center", va="center",
        )

    if save_to is not None:
        path = Path(save_to)
        fmt = path.suffix.lstrip(".").lower() or "svg"
        fig.savefig(path, format=fmt, dpi=dpi,
                    facecolor="white", edgecolor="none", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from newclid.api import GeometricSolverBuilder
    problem_str = "a b c = triangle a b c; o = circle o a b c"
    out = Path("out_bw.svg")
    if len(sys.argv) >= 2:
        problem_str = sys.argv[1]
    if len(sys.argv) >= 3:
        out = Path(sys.argv[2])
    builder = GeometricSolverBuilder(seed=998244353)
    builder.load_problem_from_txt(problem_str)
    solver = builder.build(max_attempts=100)
    draw_bw(solver.proof, builder.problemJGEX, save_to=out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

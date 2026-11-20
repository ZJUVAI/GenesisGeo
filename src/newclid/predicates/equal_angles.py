from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from matplotlib.axes import Axes
import numpy as np
import logging

from newclid.numerical.geometries import segment_angle
from newclid.numerical import close_enough
from newclid.numerical.draw_figure import draw_angle, draw_segment, get_figure_theme
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Angle_Chase
from newclid.tools import reshape
from newclid.dependencies.dependency import Dependency
from numpy.random import Generator

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement
    from newclid.dependencies.symbols import Point


class EqAngle(Predicate):
    """eqangle AB CD EF GH -
    Represent that one can rigidly move the crossing of lines AB and CD
    to get on top of the crossing of EF and GH, respectively (no reflections allowed).

    In particular, eqangle AB CD CD AB is only true if AB is perpendicular to CD.
    """

    NAME = "eqangle"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        return cls._normalize(tuple(args))

    @classmethod
    @lru_cache(maxsize=4096)
    def _ranked_segment(cls, left: str, right: str):
        low, high = cls.custom_key(left), cls.custom_key(right)
        if low <= high:
            return (low, high), (left, right)
        return (high, low), (right, left)

    @classmethod
    @lru_cache(maxsize=32_768)
    def _normalize(cls, args: tuple[str, ...]):
        # The same angle and ratio tokens recur across proof states.
        a, b, c, d, e, f, g, h = args
        if a == b or c == d or e == f or g == h:
            return
        # Compare segment ranks separately from their names so ties keep
        # the original permutation order.
        ranked_segment = cls._ranked_segment
        ab, cd, ef, gh = (
            ranked_segment(a, b),
            ranked_segment(c, d),
            ranked_segment(e, f),
            ranked_segment(g, h),
        )
        ra, rc, re, rg = ab[0], cd[0], ef[0], gh[0]
        order = min(
            (ra, rc, re, rg, 0),
            (re, rg, ra, rc, 1),
            (rc, ra, rg, re, 2),
            (rg, re, rc, ra, 3),
        )[-1]

        if order == 1:
            ab, cd, ef, gh = ef, gh, ab, cd
        elif order == 2:
            ab, cd, ef, gh = cd, ab, gh, ef
        elif order == 3:
            ab, cd, ef, gh = gh, ef, cd, ab
        if ef[0] < cd[0]:
            cd, ef = ef, cd
        return ab[1] + cd[1] + ef[1] + gh[1]

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        preparse = cls.preparse(args)
        return (
            tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        angle = None
        for a, b, c, d in reshape(list(args), 4):
            _angle = (
                segment_angle(d.num.x - c.num.x, d.num.y - c.num.y)
                - segment_angle(b.num.x - a.num.x, b.num.y - a.num.y)
            ) % np.pi
            if close_enough(_angle, np.pi):
                _angle = 0.0
            if angle is not None and not close_enough(angle, _angle):
                return False
            angle = _angle
        return True

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        points: tuple[Point, ...] = statement.args
        table = statement.dep_graph.ar.atable
        symbols_graph = statement.dep_graph.symbols_graph
        eqs: list[SumCV] = []
        i = 4
        while i < len(points):
            eqs.append(
                table.get_equal_difference_up_to(
                    symbols_graph.line_thru_pair(points[2], points[3], table).name,
                    symbols_graph.line_thru_pair(points[0], points[1], table).name,
                    symbols_graph.line_thru_pair(
                        points[i + 2], points[i + 3], table
                    ).name,
                    symbols_graph.line_thru_pair(points[i], points[i + 1], table).name,
                )
            )
            i += 4
        return eqs, table

    @classmethod
    def add(cls, dep: Dependency) -> None:
        eqs, table = cls._prep_ar(dep.statement)
        for eq in eqs:
            table.add_expr(eq, dep)

    @classmethod
    def check(cls, statement: Statement) -> bool:
        try:
            eqs, table = cls._prep_ar(statement)
        except Exception:
            return False
        return all(table.expr_delta(eq) for eq in eqs)

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        eqs, table = cls._prep_ar(statement)
        why: list[Dependency] = []
        for eq in eqs:
            why.extend(table.why(eq))
        return Dependency.mk(
            statement, Angle_Chase, tuple(dep.statement for dep in why)
        )

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c, d, e, f = args

        if point in [d, e, f]:
            a, b, c, d, e, f = d, e, f, a, b, c

        x, b, y, c, d = b, c, e, d, f
        if point == b:
            a, b, c, d = b, a, d, c

        if point == d and x == y:  # x p x b = x c x p
            return f"angle_bisector {point} {b} {x} {c}"

        if point == x:
            return f"eqangle3 {x} {a} {b} {y} {c} {d}"

        return f"on_aline {a} {x} {b} {c} {y} {d}"

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        return " = ".join(
            f"∠({a.pretty_name}{b.pretty_name},{c.pretty_name}{d.pretty_name})"
            for a, b, c, d in reshape(args, 4)
        )

    @classmethod
    def draw(
        cls,
        ax: Axes,
        args: tuple[Any, ...],
        dep_graph: DependencyGraph,
        rng: Generator,
        draw_annotations: bool = True,
    ):
        palette = get_figure_theme(ax)["palette"]
        setattr(ax, "angle_color", (getattr(ax, "angle_color", 0) + 1) % len(palette))
        color = palette[ax.angle_color]  # type: ignore
        if len(set(args[0:4])) > 3 or len(set(args[4:8])) > 3:
            ori_args = list(args)
            args = tuple(ori_args[0:2] + ori_args[4:6] + ori_args[2:4] + ori_args[6:8])
        if len(set(args[0:4])) > 3 or len(set(args[4:8])) > 3:
            logging.error(
                f"Cannot draw angle with more than 3 distinct points: {args[0:4]}, {args[4:8]}"
            )
        for i in range(0, len(args), 4):
            if args[i] == args[i + 2] or args[i] == args[i + 3]:
                point0 = args[i]
            else:
                point0 = args[i + 1]
            point1 = args[i + 1] if point0 == args[i] else args[i]
            point2 = args[i + 3] if point0 == args[i + 2] else args[i + 2]
            if draw_annotations:
                draw_angle(ax, point0, point1, point2, rng=rng, color=color, alpha=0.5)
            draw_segment(ax, point0, point1)
            draw_segment(ax, point0, point2)

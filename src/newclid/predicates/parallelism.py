from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from matplotlib.axes import Axes
from matplotlib.pylab import Generator
import numpy as np
from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.numerical.draw_figure import PALETTE, draw_segment, draw_segment_num
from newclid.numerical.geometries import LineNum
from newclid.predicates.congruence import Cong
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Angle_Chase
from newclid.tools import reshape
from newclid.dependencies.dependency import NUMERICAL_CHECK, Dependency

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


class Para(Predicate):
    """para A B C D -
    Represent that the line AB is parallel to the line CD.
    """

    NAME = "para"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        return Cong().preparse(args)

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        return Cong.parse(args, dep_graph)

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point] = statement.args
        angle = None
        for a, b in reshape(list(args), 2):
            _angle = (b.num - a.num).angle() % np.pi
            if angle is not None and not close_enough(angle, _angle):
                return False
            angle = _angle
        return True

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        points: tuple[Point, ...] = statement.args
        angle_table = statement.dep_graph.ar.atable
        symbols_graph = statement.dep_graph.symbols_graph
        eqs: list[SumCV] = []
        for i in range(2, len(points), 2):
            eqs.append(
                angle_table.get_equal_elements_up_to(
                    symbols_graph.line_thru_pair(
                        points[0], points[1], angle_table
                    ).name,
                    symbols_graph.line_thru_pair(
                        points[i], points[i + 1], angle_table
                    ).name,
                ),
            )
        return eqs, angle_table

    @classmethod
    def add(cls, dep: Dependency) -> None:
        eqs, table = cls._prep_ar(dep.statement)
        for eq in eqs:
            table.add_expr(eq, dep)

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
    def check(cls, statement: Statement) -> bool:
        eqs, table = cls._prep_ar(statement)
        return all(table.expr_delta(eq) for eq in eqs)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        points: tuple[Point, ...] = statement.args
        return "∥".join(a.pretty_name + b.pretty_name for a, b in reshape(points, 2))

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        return f"on_pline {a} {b} {c} {d}"

    @classmethod
    def draw(
        cls,
        ax: Axes,
        args: tuple[Any, ...],
        dep_graph: "DependencyGraph",
        rng: Generator,
    ):
        setattr(ax, "para_color", (getattr(ax, "angle_color", 0) + 1) % len(PALETTE))
        points: tuple[Point, ...] = args
        seglen = 100
        for i in range(0, len(points), 2):
            draw_segment(ax, points[i], points[i + 1], ls="dashed")
            seglen = min(seglen, points[i].num.distance(points[i + 1].num))
        seglen /= 3.0
        for i in range(0, len(points), 2):
            d = points[i + 1].num - points[i].num
            d = d / abs(d)
            d = d.rot90()
            if d.x < 0.0:
                d = -0.03 * d
            else:
                d = 0.03 * d
            p = points[i + 1].num - points[i].num
            p = p / abs(p)
            p = p * (points[i].num.distance(points[i + 1].num) - seglen) * 0.5
            draw_segment_num(
                ax,
                points[i].num + d + p,
                points[i + 1].num + d - p,
                color=PALETTE[ax.para_color],  # type: ignore
            )


class NPara(Predicate):
    """npara A B C D -
    Represent that lines AB and CD are NOT parallel.

    It can only be numerically checked
    (angular coefficient of the equations of the lines are different).
    """

    NAME = "npara"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        return Para.preparse(args)

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        return Para.parse(args, dep_graph)

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        a, b, c, d = args
        l1 = LineNum(a.num, b.num)
        l2 = LineNum(c.num, d.num)
        return not l1.is_parallel(l2)

    @classmethod
    def check(cls, statement: Statement) -> bool:
        return True

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        return Dependency.mk(statement, NUMERICAL_CHECK, tuple())

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        a, b, c, d = args
        return f"{a.pretty_name}{b.pretty_name} ∦ {c.pretty_name}{d.pretty_name}"

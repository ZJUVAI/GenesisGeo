from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from matplotlib.axes import Axes
from matplotlib.pylab import Generator

from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.numerical.draw_figure import draw_segment
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Ratio_Chase
from newclid.tools import reshape
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.statement import Statement
    from newclid.dependencies.dependency_graph import DependencyGraph


class Cong(Predicate):
    """cong A B C D -
    Represent that segments AB and CD are congruent."""

    NAME = "cong"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        segs: list[tuple[str, str]] = []
        if len(args) % 2 != 0:
            return None
        for a, b in zip(args[::2], args[1::2]):
            if cls.compare(a, b) > 0:
                a, b = b, a
            if a == b:
                return None
            segs.append((a, b))
        segs.sort(key = lambda pair: [cls.custom_key(arg) for arg in pair])
        points: list[str] = []
        for a, b in segs:
            points.append(a)
            points.append(b)
        return tuple(points)

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        preparse = cls.preparse(args)
        return (
            tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point] = statement.args
        length = None
        for a, b in reshape(list(args), 2):
            _length = a.num.distance2(b.num)
            if length is not None and not close_enough(length, _length):
                return False
            length = _length
        return True

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        points: tuple[Point, ...] = statement.args
        table = statement.dep_graph.ar.rtable
        eqs: list[SumCV] = []
        i = 2
        while i < len(points):
            eqs.append(
                table.get_equal_elements_up_to(
                    table.get_length(points[0], points[1]),
                    table.get_length(points[i], points[i + 1]),
                ),
            )
            i += 2
        return eqs, table

    @classmethod
    def add(cls, dep: Dependency) -> None:
        eqs, table = cls._prep_ar(dep.statement)
        for eq in eqs:
            table.add_expr(eq, dep)

    @classmethod
    def check(cls, statement: Statement) -> bool:
        eqs, table = cls._prep_ar(statement)
        return all(table.expr_delta(eq) for eq in eqs)

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        eqs, table = cls._prep_ar(statement)
        why: list[Dependency] = []
        for eq in eqs:
            why.extend(table.why(eq))
        return Dependency.mk(
            statement, Ratio_Chase, tuple(dep.statement for dep in why)
        )

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if a == c and a == point:
            return f"on_bline {a} {b} {d}"
        if b in [c, d]:
            if b == d:
                c, d = d, c
            return f"on_circle {a} {b} {d}"
        return f"eqdistance {a} {b} {c} {d}"

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args = statement.args
        return " = ".join(
            f"{a.pretty_name}{b.pretty_name}" for a, b in zip(args[::2], args[1::2])
        )

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def draw(
        cls,
        ax: Axes,
        args: tuple[Any, ...],
        dep_graph: "DependencyGraph",
        rng: Generator,
    ):
        for i in range(0, len(args), 2):
            draw_segment(ax, args[i], args[i + 1], ls="solid")

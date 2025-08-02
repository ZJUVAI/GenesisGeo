from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from matplotlib.axes import Axes
from newclid.dependencies.dependency import NUMERICAL_CHECK, Dependency
from newclid.dependencies.symbols import Line, Point
from newclid.numerical.draw_figure import draw_line
from newclid.numerical.geometries import LineNum
from newclid.predicates.predicate import Predicate
from newclid.tools import notNone
from numpy.random import Generator

if TYPE_CHECKING:
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


class Coll(Predicate):
    """coll A B C ... -
    Represent that the 3 (or more) points in the arguments are collinear."""

    NAME = "coll"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        if len(args) <= 2 or len(args) != len(set(args)):
            return None
        return tuple(sorted(args, key = cls.custom_key))

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        preparse = cls.preparse(args)
        return (
            tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        points: tuple[Point, ...] = tuple(statement.args)
        line = LineNum(points[0].num, points[1].num)
        return all(line.point_at(p.num.x, p.num.y) is not None for p in points[2:])

    @classmethod
    def check(cls, statement: Statement) -> bool:
        return Line.check_coll(statement.args)

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        return Line.why_coll(statement)

    @classmethod
    def add(cls, dep: Dependency):
        rep, merged = Line.make_coll(dep.statement.args, dep)
        table = dep.statement.dep_graph.ar.atable
        for line in merged:
            table.add_expr(table.get_equal_elements_up_to(rep.name, line.name), dep)

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c = args
        if point == b:
            a, b = b, a
        if point == c:
            a, b, c = c, a, b
        return f"on_line {a} {b} {c}"

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        return f"{', '.join(p.pretty_name for p in statement.args)} are collinear"

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def draw(
        cls, ax: Axes, args: tuple[Any, ...], dep_graph: DependencyGraph, rng: Generator
    ):
        symbols_graph = dep_graph.symbols_graph
        draw_line(
            ax,
            notNone(symbols_graph.container_of(set(args), Line)),
            alpha=1.0,
            ls="dashed",
        )


class NColl(Predicate):
    """ncoll A B C ... -
    Represent that any of the 3 (or mo}re) points is not aligned with the others.

    Numerical only.
    """

    NAME = "ncoll"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        return Coll.preparse(args)

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        return Coll.parse(args, dep_graph)

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        return not Coll.check_numerical(statement)

    @classmethod
    def check(cls, statement: Statement) -> bool:
        return True

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        return Dependency.mk(statement, NUMERICAL_CHECK, tuple())

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        return f"{', '.join(p.pretty_name for p in statement.args)} are not collinear"

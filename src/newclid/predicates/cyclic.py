from __future__ import annotations
from typing import TYPE_CHECKING, Any

from newclid.dependencies.dependency import Dependency
from newclid.dependencies.symbols import Circle, Point
from newclid.numerical import close_enough
from newclid.numerical.draw_figure import draw_circle
from newclid.numerical.geometries import CircleNum
from newclid.predicates.predicate import Predicate
from matplotlib.axes import Axes
from numpy.random import Generator

from newclid.tools import notNone


if TYPE_CHECKING:
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


class Cyclic(Predicate):
    """cyclic A B C D -
    Represent that 4 (or more) points lie on the same circle."""

    NAME = "cyclic"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        if len(args) <= 3 or len(args) != len(set(args)):
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
        points: tuple[Point, ...] = statement.args
        try:
            circle = CircleNum(p1=points[0].num, p2=points[1].num, p3=points[2].num)
        except ValueError:
            return False

        return all(
            close_enough(circle.radius**2, circle.center.distance2(p.num))
            for p in points[3:]
        )

    @classmethod
    def check(cls, statement: Statement) -> bool:
        return Circle.check_cyclic(statement.args)

    @classmethod
    def add(cls, dep: Dependency):
        Circle.make_cyclic(dep.statement.args, dep)

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c = [x for x in args if x != point]
        return f"on_circum {point} {a} {b} {c}"

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        return Circle.why_cyclic(statement)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        return f"{''.join(p.pretty_name for p in statement.args)} are cyclic"

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def draw(
        cls, ax: Axes, args: tuple[Any, ...], dep_graph: DependencyGraph, rng: Generator
    ):
        symbols_graph = dep_graph.symbols_graph
        draw_circle(ax, notNone(symbols_graph.container_of(set(args), Circle)))

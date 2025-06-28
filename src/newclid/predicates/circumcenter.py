from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from newclid.dependencies.dependency import Dependency
from newclid.numerical import close_enough
from newclid.numerical.geometries import CircleNum
from newclid.predicates.congruence import Cong
from newclid.predicates.cyclic import Cyclic
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Shortcut_Derivation


if TYPE_CHECKING:
    from newclid.dependencies.symbols import Point
    from newclid.statement import Statement
    from newclid.dependencies.dependency_graph import DependencyGraph


class Circumcenter(Predicate):
    """circle O A B C -
    Represent that O is the center of the circle through A, B, and C
    (circumcenter of triangle ABC).

    Can be equivalent to cong O A O B and cong O A O C,
    and equivalent pairs of congruences.
    """

    NAME = "circle"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        if len(args) <= 2 or len(args) != len(set(args)):
            return None
        return (args[0],) + tuple(sorted(args[1:]))

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
        points: tuple[Point, ...] = statement.args
        circle = CircleNum(points[0].num, points[0].num.distance(points[1].num))
        return all(
            close_enough(circle.radius, circle.center.distance(p.num))
            for p in points[2:]
        )

    @classmethod
    def check(cls, statement: Statement) -> bool:
        points: tuple[Point, ...] = statement.args
        o = points[0]
        p0 = points[1]
        for p1 in points[2:]:
            cong = statement.with_new(Cong, (o, p0, o, p1))
            if not cong.check():
                return False
        return True

    @classmethod
    def add(cls, dep: Dependency) -> None:
        points: tuple[Point, ...] = dep.statement.args
        o = points[0]
        p0 = points[1]
        for p1 in points[2:]:
            cong = dep.with_new(dep.statement.with_new(Cong, (o, p0, o, p1)))
            cong.add()
        if len(points) > 4:
            dep.with_new(dep.statement.with_new(Cyclic, points[1:])).add()

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        points: tuple[Point, ...] = statement.args
        o = points[0]
        p0 = points[1]
        return Dependency.mk(
            statement,
            Shortcut_Derivation,
            tuple(statement.with_new(Cong, (o, p0, o, p1)) for p1 in points[2:]),
        )

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        points: tuple[Point, ...] = statement.args
        o = points[0]
        return f"{o.pretty_name} is the circumcenter of the circle {''.join(p.pretty_name for p in points[1:])}"

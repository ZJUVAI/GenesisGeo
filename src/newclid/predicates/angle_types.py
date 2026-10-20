"""obtuse / acute : numerical-only angle predicates.

obtuse A B C : angle ABC > 90 degrees
acute  A B C : angle ABC < 90 degrees

These are numerical-check-only; they cannot be deduced symbolically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from newclid.dependencies.dependency import NUMERICAL_CHECK, Dependency
from newclid.dependencies.symbols import Point
from newclid.predicates.predicate import Predicate

if TYPE_CHECKING:
    from newclid.statement import Statement
    from newclid.dependencies.dependency_graph import DependencyGraph


def _angle_dot(a: Point, b: Point, c: Point) -> float:
    """Compute dot product (A-B)·(C-B). Positive = acute, negative = obtuse."""
    return (a.num.x - b.num.x) * (c.num.x - b.num.x) + \
           (a.num.y - b.num.y) * (c.num.y - b.num.y)


class Obtuse(Predicate):
    """obtuse A B C : angle ABC > 90 degrees."""

    NAME = "obtuse"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        if len(args) != 3:
            return None
        a, b, c = args
        if a == b or b == c:
            return None
        # Keep vertex (b) fixed; sort the other two
        others = sorted([a, c], key=cls.custom_key)
        return (others[0], b, others[1])

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        preparse = cls.preparse(args)
        return tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        return _angle_dot(args[0], args[1], args[2]) < -1e-9

    @classmethod
    def check(cls, statement: Statement) -> bool:
        return True  # numerical only

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        return Dependency.mk(statement, NUMERICAL_CHECK, tuple())

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        a, b, c = statement.args
        return f"∠({a.pretty_name}{b.pretty_name}{c.pretty_name}) > 90°"

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)


class Acute(Predicate):
    """acute A B C : angle ABC < 90 degrees."""

    NAME = "acute"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        if len(args) != 3:
            return None
        a, b, c = args
        if a == b or b == c:
            return None
        others = sorted([a, c], key=cls.custom_key)
        return (others[0], b, others[1])

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        preparse = cls.preparse(args)
        return tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        return _angle_dot(args[0], args[1], args[2]) > 1e-9

    @classmethod
    def check(cls, statement: Statement) -> bool:
        return True  # numerical only

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        return Dependency.mk(statement, NUMERICAL_CHECK, tuple())

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        a, b, c = statement.args
        return f"∠({a.pretty_name}{b.pretty_name}{c.pretty_name}) < 90°"

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

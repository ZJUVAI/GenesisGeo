from __future__ import annotations
from typing import TYPE_CHECKING

from newclid.dependencies.symbols import Point
from newclid.predicates.predicate import Predicate

if TYPE_CHECKING:
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


class MidPoint(Predicate):
    """midp M A B -
    Represent that M is the midpoint of the segment AB.

    Can be equivalent to coll M A B and cong A M B M."""

    NAME = "midp"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        if len(set(args)) != 3:
            return None
        m, a, b = args
        a, b = sorted((a, b), key = cls.custom_key)
        return (m, a, b)

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        preparse = cls.preparse(args)
        return (
            tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        m, a, b = args
        return m.num.close_enough((a.num + b.num) / 2)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        m, a, b = args
        return f"{m.pretty_name} is the midpoint of {a.pretty_name}{b.pretty_name}"

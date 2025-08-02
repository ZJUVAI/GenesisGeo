from __future__ import annotations
from typing import TYPE_CHECKING

from newclid.dependencies.dependency import NUMERICAL_CHECK, Dependency
from newclid.dependencies.symbols import Point
from newclid.predicates.predicate import Predicate


if TYPE_CHECKING:
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


class Diff(Predicate):
    """diff a b -

    Represent that a is not equal to b.

    Numerical only.
    """

    NAME = "diff"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        return tuple(sorted(args, key = cls.custom_key))

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        return tuple(dep_graph.symbols_graph.names2points(cls.preparse(args)))

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        p: Point
        q: Point
        p, q = statement.args
        return not p.num.close_enough(q.num)

    @classmethod
    def check(cls, statement: Statement) -> bool:
        return True

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        return Dependency.mk(statement, NUMERICAL_CHECK, ())

    @classmethod
    def to_repr(cls, statement: Statement) -> str:
        p: Point
        q: Point
        p, q = statement.args
        return f"{p.pretty_name} ≠ {q.pretty_name}"

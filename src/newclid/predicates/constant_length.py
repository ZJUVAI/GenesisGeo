from __future__ import annotations
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Optional

from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Ratio_Chase
from newclid.tools import fraction_to_len, get_quotient, str_to_fraction
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


class ConstantLength(Predicate):
    """lconst A B L -
    Represent that the length of segment AB is L

    L should be given as a float.
    """

    NAME = "lconst"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        a, b, length = args
        if a == b:
            return None
        a, b = sorted((a, b), key = lambda pair: [cls.custom_key(arg) for arg in pair])
        return (a, b, fraction_to_len(str_to_fraction(length)))

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        preparse = cls.preparse(args)
        if not preparse:
            return None
        a, b, length = preparse
        return tuple(dep_graph.symbols_graph.names2points((a, b))) + (
            str_to_fraction(length),
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, Point, Fraction] = statement.args
        a, b, length = args
        return close_enough(a.num.distance(b.num), float(length))

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        args: tuple[Point, Point, str] = statement.args
        p0, p1, _ = args
        table = statement.dep_graph.ar.rtable

        return [table.get_equal_elements(table.get_length(p0, p1))], table

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
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return (args[0].name, args[1].name, fraction_to_len(args[2]))

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        a, b, length = statement.args
        return f"{a.pretty_name}{b.pretty_name} = {fraction_to_len(length)}"


class LCompute(Predicate):
    """lcompute A B"""

    NAME = "lcompute"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        a, b = args
        if a == b:
            return None
        return tuple(sorted((a, b), key = cls.custom_key))

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
        args: tuple[Point, Point] = statement.args
        a, b = args
        get_quotient(a.num.distance(b.num))
        return True

    @classmethod
    def check(cls, statement: Statement) -> bool:
        args: tuple[Point, Point] = statement.args
        a, b = args
        length = get_quotient(a.num.distance(b.num))
        return statement.with_new(ConstantLength, (a, b, length)).check()

    @classmethod
    def why(cls, statement: Statement):
        args: tuple[Point, Point] = statement.args
        a, b = args
        length = get_quotient(a.num.distance(b.num))
        return statement.with_new(ConstantLength, (a, b, length)).why()

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        a, b = statement.args
        length = get_quotient(a.num.distance(b.num))
        return f"{a.pretty_name}{b.pretty_name} = {fraction_to_len(length)}"

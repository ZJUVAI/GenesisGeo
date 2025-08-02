from __future__ import annotations
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from newclid.dependencies.dependency import Dependency
from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.predicates.constant_angle import ACompute
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Ratio_Chase
from newclid.tools import fraction_to_ratio, get_quotient, str_to_fraction

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.statement import Statement
    from newclid.dependencies.dependency_graph import DependencyGraph


class ConstantRatio(Predicate):
    """rconst A B C D r -
    Represent that AB / CD = r

    r should be given with numerator and denominator separated by '/', as in 2/3.
    """

    NAME = "rconst"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        a, b, c, d, r = args
        if a == b or c == d:
            return None
        f = str_to_fraction(r)
        a, b = sorted((a, b), key = lambda pair: [cls.custom_key(arg) for arg in pair])
        c, d = sorted((c, d), key = lambda pair: [cls.custom_key(arg) for arg in pair])
        if cls.compare((a, b), (c, d)) > 0:
            a, b, c, d = c, d, a, b
            f = 1 / f
        return (a, b, c, d, fraction_to_ratio(f))

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        preparse = cls.preparse(args)
        if not preparse:
            return None
        a, b, c, d, f = preparse
        return tuple(dep_graph.symbols_graph.names2points((a, b, c, d))) + (
            str_to_fraction(f),
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, Point, Point, Point, Fraction] = statement.args
        a, b, c, d, r = args
        return close_enough(a.num.distance(b.num) / c.num.distance(d.num), float(r))

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        args: tuple[Point, Point, Point, Point, str] = statement.args
        p0, p1, p2, p3, _ = args
        table = statement.dep_graph.ar.rtable
        l0 = table.get_length(p0, p1)
        l1 = table.get_length(p2, p3)
        return [table.get_equal_elements_up_to(l0, l1)], table

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
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, Point, Point, Point, Fraction] = statement.args
        a, b, c, d, r = args
        return f"{a.pretty_name}{b.pretty_name}:{c.pretty_name}{d.pretty_name} = {fraction_to_ratio(r)}"

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        a, b, c, d, r = args
        return (a.name, b.name, c.name, d.name, fraction_to_ratio(r))  # type: ignore


class RCompute(Predicate):
    """rcompute A B C D"""

    NAME = "rcompute"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        return ACompute.preparse(args)

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        return ACompute.parse(args, dep_graph)

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        get_quotient(a.num.distance(b.num) / c.num.distance(d.num))
        return True

    @classmethod
    def check(cls, statement: Statement) -> bool:
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        r = get_quotient(a.num.distance(b.num) / c.num.distance(d.num))
        return statement.with_new(ConstantRatio, (a, b, c, d, r)).check()

    @classmethod
    def why(cls, statement: Statement):
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        r = get_quotient(a.num.distance(b.num) / c.num.distance(d.num))
        return statement.with_new(ConstantRatio, (a, b, c, d, r)).why()

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        r = get_quotient(a.num.distance(b.num) / c.num.distance(d.num))
        return f"{a.pretty_name}{b.pretty_name}:{c.pretty_name}{d.pretty_name} = {fraction_to_ratio(r)}"

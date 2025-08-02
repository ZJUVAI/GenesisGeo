from __future__ import annotations
from fractions import Fraction
from numpy import pi
from typing import TYPE_CHECKING, Any, Optional

from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Angle_Chase
from newclid.tools import fraction_to_angle, get_quotient, str_to_fraction
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


class ConstantAngle(Predicate):
    """aconst AB CD Y -
    Represent that the rotation needed to go from line AB to line CD,
    oriented on the clockwise direction is Y.

    The syntax of Y is either a fraction of pi like 2pi/3 for radians
    or a number followed by a 'o' like 120o for degree.
    """

    NAME = "aconst"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        a, b, c, d, y = args
        if a == b or c == d:
            return None
        a, b = sorted((a, b), key = cls.custom_key)
        c, d = sorted((c, d), key = cls.custom_key)
        f = str_to_fraction(y)
        if cls.compare((a, b), (c, d)) > 0:
            a, b, c, d = c, d, a, b
            f = -f
        f %= 1
        return (a, b, c, d, fraction_to_angle(f))

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
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
        a, b, c, d, y = args
        return close_enough(
            float(y) * pi, ((c.num - d.num).angle() - (a.num - b.num).angle()) % pi
        )

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        args: tuple[Point, Point, Point, Point, Fraction] = statement.args
        p0, p1, p2, p3, _ = args
        table = statement.dep_graph.ar.atable
        symbols_graph = statement.dep_graph.symbols_graph

        return [
            table.get_equal_elements_up_to(
                symbols_graph.line_thru_pair(p2, p3, table).name,
                symbols_graph.line_thru_pair(p0, p1, table).name,
            )
        ], table

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
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        a, b, c, d, y = args
        return (a.name, b.name, c.name, d.name, fraction_to_angle(y))  # type: ignore

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, Point, Point, Point, Fraction] = statement.args
        a, b, c, d, y = args
        return f"∠({a.pretty_name}{b.pretty_name},{c.pretty_name}{d.pretty_name}) = {fraction_to_angle(y)}"


class ACompute(Predicate):
    """acompute AB CD"""

    NAME = "acompute"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        a, b, c, d = args
        if a == b or c == d:
            return None
        a, b = sorted((a, b), key = lambda pair: [cls.custom_key(arg) for arg in pair])
        c, d = sorted((c, d), key = lambda pair: [cls.custom_key(arg) for arg in pair])
        return (a, b, c, d)

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        preparse = cls.preparse(args)
        if not preparse:
            return None
        a, b, c, d = preparse
        return tuple(dep_graph.symbols_graph.names2points((a, b, c, d)))

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        ang = ((c.num - d.num).angle() - (a.num - b.num).angle()) % pi
        if close_enough(ang, pi):
            ang = 0
        get_quotient(ang / pi)
        return True

    @classmethod
    def why(cls, statement: Statement):
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        y = get_quotient(((c.num - d.num).angle() - (a.num - b.num).angle()) % pi / pi)
        dep = statement.with_new(ConstantAngle, (a, b, c, d, y)).why()
        return dep.with_new(statement) if dep else None

    @classmethod
    def check(cls, statement: Statement) -> bool:
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        y = get_quotient(((c.num - d.num).angle() - (a.num - b.num).angle()) % pi / pi)
        return statement.with_new(ConstantAngle, (a, b, c, d, y)).check()

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, Point, Point, Point] = statement.args
        a, b, c, d = args
        y = get_quotient(((c.num - d.num).angle() - (a.num - b.num).angle()) % pi / pi)
        return f"∠({a.pretty_name}{b.pretty_name},{c.pretty_name}{d.pretty_name}) = {fraction_to_angle(y)}"

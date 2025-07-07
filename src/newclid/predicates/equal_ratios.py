from __future__ import annotations
from typing import TYPE_CHECKING, Any

from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.predicates.equal_angles import EqAngle
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Ratio_Chase
from newclid.tools import reshape
from newclid.dependencies.dependency import Dependency


if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.statement import Statement
    from newclid.dependencies.dependency_graph import DependencyGraph


class EqRatio(Predicate):
    """eqratio AB CD EF GH -

    Represent that AB/CD=EF/GH, as ratios between lengths of segments.
    """

    NAME = "eqratio"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        a, b, c, d, e, f, g, h = args
        if a == b or c == d or e == f or g == h:
            return 
        if a == c:
            if e == g:
                a, b, c, d, e, f, g, h = a, b, c, d, e, f, g, h
            elif e == h:
                a, b, c, d, e, f, g, h = a, b, c, d, e, f, h, g
            elif f == g:
                a, b, c, d, e, f, g, h = a, b, c, d, f, e, g, h
            elif f == h:
                a, b, c, d, e, f, g, h = a, b, c, d, f, e, h, g
        elif a == d:
            if e == g:
                a, b, c, d, e, f, g, h = a, b, d, c, e, f, g, h
            elif e == h:
                a, b, c, d, e, f, g, h = a, b, d, c, e, f, h, g
            elif f == g:
                a, b, c, d, e, f, g, h = a, b, d, c, f, e, g, h
            elif f == h:
                a, b, c, d, e, f, g, h = a, b, d, c, f, e, h, g
        elif b == c:
            if e == g:
                a, b, c, d, e, f, g, h = b, a, c, d, e, f, g, h
            elif e == h:
                a, b, c, d, e, f, g, h = b, a, c, d, e, f, h, g
            elif f == g:
                a, b, c, d, e, f, g, h = b, a, c, d, f, e, g, h
            elif f == h:
                a, b, c, d, e, f, g, h = b, a, c, d, f, e, h, g
        elif b == d:
            if e == g:
                a, b, c, d, e, f, g, h = b, a, d, c, e, f, g, h
            elif e == h:
                a, b, c, d, e, f, g, h = b, a, d, c, e, f, h, g
            elif f == g:
                a, b, c, d, e, f, g, h = b, a, d, c, f, e, g, h
            elif f == h:
                a, b, c, d, e, f, g, h = b, a, d, c, f, e, h, g
        elif a == e:
            if c == g:
                a, b, c, d, e, f, g, h = a, b, e, f, c, d, g, h
            elif c == h:
                a, b, c, d, e, f, g, h = a, b, e, f, c, d, h, g
            elif d == g:
                a, b, c, d, e, f, g, h = a, b, e, f, d, c, g, h
            elif d == h:
                a, b, c, d, e, f, g, h = a, b, e, f, d, c, h, g
        elif a == f:
            if c == g:
                a, b, c, d, e, f, g, h = a, b, f, e, c, d, g, h
            elif c == h:
                a, b, c, d, e, f, g, h = a, b, f, e, c, d, h, g
            elif d == g:
                a, b, c, d, e, f, g, h = a, b, f, e, d, c, g, h
            elif d == h:
                a, b, c, d, e, f, g, h = a, b, f, e, d, c, h, g
        elif b == e:
            if c == g:
                a, b, c, d, e, f, g, h = b, a, e, f, c, d, g, h
            elif c == h:
                a, b, c, d, e, f, g, h = b, a, e, f, c, d, h, g
            elif d == g:
                a, b, c, d, e, f, g, h = b, a, e, f, d, c, g, h
            elif d == h:
                a, b, c, d, e, f, g, h = b, a, e, f, d, c, h, g
        elif b == f:
            if c == g:
                a, b, c, d, e, f, g, h = b, a, f, e, c, d, g, h
            elif c == h:
                a, b, c, d, e, f, g, h = b, a, f, e, c, d, h, g
            elif d == g:
                a, b, c, d, e, f, g, h = b, a, f, e, d, c, g, h
            elif d == h:
                a, b, c, d, e, f, g, h = b, a, f, e, d, c, h, g
        if a != c or e != g:
            if a > b: a, b = b, a
            if c > d: c, d = d, c
            if e > f: e, f = f, e
            if g > h: g, h = h, g
            g1a = (a, b, c, d)
            g1b = (e, f, g, h)
            g2a = (c, d, a, b)
            g2b = (g, h, e, f)
            if g1a <= g1b:
                groups1 = g1a + g1b
            else:
                groups1 = g1b + g1a
            if g2a <= g2b:
                groups2 = g2a + g2b
            else:
                groups2 = g2b + g2a
            return groups1 if groups1 <= groups2 else groups2

        if a == e:
            x = min(b, d, f, h)
            if b == x:
                if d < f:
                    return (a, b, c, d, e, f, g, h)
                else:
                    return (a, b, e, f, c, d, g, h)
            elif d == x:
                if b < h:
                    return (c, d, a, b, g, h, e, f)
                else:
                    return (c, d, g, h, a, b, e, f)
            elif f == x:
                if b < h:
                    return (e, f, a, b, g, h, c, d)
                else:
                    return (e, f, g, h, a, b, c, d)
            else:
                if d < f:
                    return (g, h, c, d, e, f, a, b)
                else:
                    return (g, h, e, f, c, d, a, b)
        elif a < e:
            if b < d:
                return (a, b, c, d, e, f, g, h)
            else:
                return (c, d, a, b, g, h, e, f)
        else:
            if f < h:
                return (e, f, g, h, a, b, c, d)
            else:
                return (g, h, e, f, c, d, a, b)
        return EqAngle.preparse(args)    

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        return EqAngle.parse(args, dep_graph)

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        ratio = None
        for a, b, c, d in reshape(statement.args, 4):
            a: Point
            b: Point
            c: Point
            d: Point
            _ratio = a.num.distance(b.num) / c.num.distance(d.num)
            if ratio is not None and not close_enough(ratio, _ratio):
                return False
            ratio = _ratio
        return True

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        points: tuple[Point, ...] = statement.args
        table = statement.dep_graph.ar.rtable
        eqs: list[SumCV] = []
        i = 4
        while i < len(points):
            eqs.append(
                table.get_equal_difference_up_to(
                    table.get_length(points[0], points[1]),
                    table.get_length(points[2], points[3]),
                    table.get_length(points[i], points[i + 1]),
                    table.get_length(points[i + 2], points[i + 3]),
                )
            )
            i += 4
        return eqs, table

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
            statement, Ratio_Chase, tuple(dep.statement for dep in why)
        )

    @classmethod
    def check(cls, statement: Statement) -> bool:
        eqs, table = cls._prep_ar(statement)
        return all(table.expr_delta(eq) for eq in eqs)

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        return " = ".join(
            f"{a.pretty_name}{b.pretty_name}:{c.pretty_name}{d.pretty_name}"
            for a, b, c, d in reshape(args, 4)
        )


class EqRatio3(Predicate):
    """eqratio AB CD MN -

    Represent three eqratios through a list of 6 points (due to parallel lines).
    It can be viewed as in an instance of Thales theorem which has AB // MN // CD.

    It thus represent the corresponding eqratios:
    MA / MC = NB / ND and AM / AC = BN / BD and MC / AC = ND / BD

    ::

          a -- b
         m ---- n
        c ------ d


    """

    NAME = "eqratio3"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        a, b, c, d, m, n = args
        if len(set((a, c, m))) < 3 or len(set((b, d, n))) < 3:
            return None
        groups = ((a, b), (c, d), (m, n))
        groups1 = ((b, a), (d, c), (n, m))
        return sum(min(sorted(groups), sorted(groups1)), ())

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        preparse = cls.preparse(args)
        return (
            tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        a, b, c, d, m, n = statement.args
        eqr1 = statement.with_new(EqRatio, (m, a, m, c, n, b, n, d))
        eqr2 = statement.with_new(EqRatio, (m, a, a, c, b, n, b, d))
        eqr3 = statement.with_new(EqRatio, (m, c, a, c, n, d, b, d))
        return (
            eqr1.check_numerical() and eqr2.check_numerical() and eqr3.check_numerical()
        )

    @classmethod
    def check(cls, statement: Statement) -> bool:
        a, b, c, d, m, n = statement.args
        eqr1 = statement.with_new(EqRatio, (m, a, m, c, n, b, n, d))
        eqr2 = statement.with_new(EqRatio, (m, a, a, c, b, n, b, d))
        eqr3 = statement.with_new(EqRatio, (m, c, a, c, n, d, b, d))
        return eqr1.check() and eqr2.check() and eqr3.check()

    @classmethod
    def add(cls, dep: Dependency):
        statement = dep.statement
        a, b, c, d, m, n = statement.args
        eqr1 = statement.with_new(EqRatio, (m, a, m, c, n, b, n, d))
        eqr2 = statement.with_new(EqRatio, (m, a, a, c, b, n, b, d))
        eqr3 = statement.with_new(EqRatio, (m, c, a, c, n, d, b, d))
        dep.with_new(eqr1).add()
        dep.with_new(eqr2).add()
        dep.with_new(eqr3).add()

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
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c, d, e, f, g, h = args

        if point == h:
            return f"eqratio {h} {a} {b} {c} {d} {e} {f} {g}"
        if point == g:
            return f"eqratio {g} {a} {b} {c} {d} {e} {f} {h}"
        if point == f:
            return f"eqratio {f} {c} {d} {a} {b} {g} {h} {e}"
        if point == e:
            return f"eqratio {e} {c} {d} {a} {b} {g} {h} {f}"
        if point == d:
            return f"eqratio {d} {e} {f} {g} {h} {a} {b} {c}"
        if point == c:
            return f"eqratio {c} {e} {f} {g} {h} {a} {b} {d}"
        if point == b:
            return f"eqratio {b} {g} {h} {e} {f} {c} {d} {a}"
        if point == a:
            return f"eqratio {a} {g} {h} {e} {f} {c} {d} {b}"

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
        sorted_groups = sorted(groups, key=lambda pair: [cls.custom_key(arg) for arg in pair])
        sorted_groups1 = sorted(groups1, key=lambda pair: [cls.custom_key(arg) for arg in pair])
        return sum(min(sorted_groups, sorted_groups1, key = lambda pair: [[cls.custom_key(arg[0]), cls.custom_key(arg[1])] for arg in pair]), ())

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

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        a, b, c, d, m, n = statement.args
        eqr1 = statement.with_new(EqRatio, (m, a, m, c, n, b, n, d))
        eqr2 = statement.with_new(EqRatio, (m, a, a, c, b, n, b, d))
        eqr3 = statement.with_new(EqRatio, (m, c, a, c, n, d, b, d))
        return Dependency.mk(statement, Ratio_Chase, [eqr1, eqr2, eqr3])

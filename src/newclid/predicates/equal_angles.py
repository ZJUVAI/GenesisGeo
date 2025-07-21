from __future__ import annotations
from typing import TYPE_CHECKING, Any

from matplotlib.axes import Axes
import numpy as np

from newclid.dependencies.symbols import Line
from newclid.numerical import close_enough
from newclid.numerical.draw_figure import PALETTE, draw_angle, draw_line
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Angle_Chase
from newclid.tools import reshape
from newclid.dependencies.dependency import Dependency
from numpy.random import Generator

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement
    from newclid.dependencies.symbols import Point


class EqAngle(Predicate):
    """eqangle AB CD EF GH -
    Represent that one can rigidly move the crossing of lines AB and CD
    to get on top of the crossing of EF and GH, respectively (no reflections allowed).

    In particular, eqangle AB CD CD AB is only true if AB is perpendicular to CD.
    """

    NAME = "eqangle"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        a, b, c, d, e, f, g, h = args
        if a == b or c == d or e == f or g == h:
            return 
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
        groups1 = groups1 if groups1 <= groups2 else groups2  
        a, b, c, d, e, f, g, h = groups1
        groups2 = (a, b, e, f, c, d, g, h)
        return groups1 if groups1 <= groups2 else groups2  
        
        # groups: list[tuple[str, str, str, str]] = []
        # groups1: list[tuple[str, str, str, str]] = []
        # if len(args) % 4:
        #     return None
        # for a, b, c, d in reshape(args, 4):
        #     if a == b or c == d:
        #         return None
        #     a, b = sorted((a, b))
        #     c, d = sorted((c, d))
        #     groups.append((a, b, c, d))
        #     groups1.append((c, d, a, b))
        # assert data == sum(min(sorted(groups), sorted(groups1)), ())
        # return sum(min(sorted(groups), sorted(groups1)), ())

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        preparse = cls.preparse(args)
        return (
            tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        angle = None
        for a, b, c, d in reshape(list(args), 4):
            _angle = ((d.num - c.num).angle() - (b.num - a.num).angle()) % np.pi
            if close_enough(_angle, np.pi):
                _angle = 0.0
            if angle is not None and not close_enough(angle, _angle):
                return False
            angle = _angle
        return True

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        points: tuple[Point, ...] = statement.args
        table = statement.dep_graph.ar.atable
        symbols_graph = statement.dep_graph.symbols_graph
        eqs: list[SumCV] = []
        i = 4
        while i < len(points):
            eqs.append(
                table.get_equal_difference_up_to(
                    symbols_graph.line_thru_pair(points[2], points[3], table).name,
                    symbols_graph.line_thru_pair(points[0], points[1], table).name,
                    symbols_graph.line_thru_pair(
                        points[i + 2], points[i + 3], table
                    ).name,
                    symbols_graph.line_thru_pair(points[i], points[i + 1], table).name,
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
            statement, Angle_Chase, tuple(dep.statement for dep in why)
        )

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c, d, e, f = args

        if point in [d, e, f]:
            a, b, c, d, e, f = d, e, f, a, b, c

        x, b, y, c, d = b, c, e, d, f
        if point == b:
            a, b, c, d = b, a, d, c

        if point == d and x == y:  # x p x b = x c x p
            return f"angle_bisector {point} {b} {x} {c}"

        if point == x:
            return f"eqangle3 {x} {a} {b} {y} {c} {d}"

        return f"on_aline {a} {x} {b} {c} {y} {d}"

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        return " = ".join(
            f"∠({a.pretty_name}{b.pretty_name},{c.pretty_name}{d.pretty_name})"
            for a, b, c, d in reshape(args, 4)
        )

    @classmethod
    def draw(
        cls, ax: Axes, args: tuple[Any, ...], dep_graph: DependencyGraph, rng: Generator
    ):
        setattr(ax, "angle_color", (getattr(ax, "angle_color", 0) + 1) % len(PALETTE))
        color = PALETTE[ax.angle_color]  # type: ignore
        r = rng.random()
        width = r * 0.1
        symbols_graph = dep_graph.symbols_graph
        for i in range(0, len(args), 4):
            line0 = symbols_graph.container_of({args[i], args[i + 1]}, Line)
            line1 = symbols_graph.container_of({args[i + 2], args[i + 3]}, Line)
            assert line0 and line1
            draw_angle(ax, line0, line1, color=color, alpha=0.5, width=width, r=r)
            draw_line(ax, line0, ls=":")
            draw_line(ax, line1, ls=":")

# from __future__ import annotations
# from typing import TYPE_CHECKING, Any

# from matplotlib.axes import Axes
# import numpy as np

# from newclid.dependencies.symbols import Line
# from newclid.numerical import close_enough
# from newclid.numerical.draw_figure import PALETTE, draw_angle, draw_line
# from newclid.predicates.predicate import Predicate
# from newclid.algebraic_reasoning.tables import Angle_Chase
# from newclid.tools import reshape
# from newclid.dependencies.dependency import Dependency
# from numpy.random import Generator

# if TYPE_CHECKING:
#     from newclid.algebraic_reasoning.tables import Table
#     from newclid.algebraic_reasoning.tables import SumCV
#     from newclid.dependencies.dependency_graph import DependencyGraph
#     from newclid.statement import Statement
#     from newclid.dependencies.symbols import Point


# class EqAngle(Predicate):
#     """eqangle AB CD EF GH -
#     Represent that one can rigidly move the crossing of lines AB and CD
#     to get on top of the crossing of EF and GH, respectively (no reflections allowed).

#     In particular, eqangle AB CD CD AB is only true if AB is perpendicular to CD.
#     """

#     NAME = "eqangle"

#     @classmethod
#     def preparse(cls, args: tuple[str, ...]):
#         a, b, c, d, e, f, g, h = args
#         if a == b or c == d or e == f or g == h:
#             return None
#         if a > b: a, b = b, a
#         if c > d: c, d = d, c
#         if e > f: e, f = f, e
#         if g > h: g, h = h, g
#         g1a = (a, b, c, d)
#         g1b = (e, f, g, h)
#         g2a = (c, d, a, b)
#         g2b = (g, h, e, f)
#         if g1a <= g1b:
#             groups1 = g1a + g1b
#         else:
#             groups1 = g1b + g1a
#         if g2a <= g2b:
#             groups2 = g2a + g2b
#         else:
#             groups2 = g2b + g2a
#         return groups1 if groups1 <= groups2 else groups2
    
#         groups: list[tuple[str, str, str, str]] = []
#         groups1: list[tuple[str, str, str, str]] = []
#         if len(args) % 4:
#             return None
#         for a, b, c, d in reshape(args, 4):
#             if a == b or c == d:
#                 return None
#             a, b = sorted((a, b))
#             c, d = sorted((c, d))
#             groups.append((a, b, c, d))
#             groups1.append((c, d, a, b))
#         assert data == sum(min(sorted(groups), sorted(groups1)), ())
#         return sum(min(sorted(groups), sorted(groups1)), ())

#     @classmethod
#     def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
#         preparse = cls.preparse(args)
#         return (
#             tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
#         )

#     @classmethod
#     def check_numerical(cls, statement: Statement) -> bool:
#         args: tuple[Point, ...] = statement.args
#         angle = None
#         for a, b, c, d in reshape(list(args), 4):
#             _angle = ((d.num - c.num).angle() - (b.num - a.num).angle()) % np.pi
#             if angle is not None and not close_enough(angle, _angle):
#                 return False
#             angle = _angle
#         return True

#     @classmethod
#     def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
#         points: tuple[Point, ...] = statement.args
#         table = statement.dep_graph.ar.atable
#         symbols_graph = statement.dep_graph.symbols_graph
#         eqs: list[SumCV] = []
#         i = 4
#         while i < len(points):
#             eqs.append(
#                 table.get_equal_difference_up_to(
#                     symbols_graph.line_thru_pair(points[2], points[3], table).name,
#                     symbols_graph.line_thru_pair(points[0], points[1], table).name,
#                     symbols_graph.line_thru_pair(
#                         points[i + 2], points[i + 3], table
#                     ).name,
#                     symbols_graph.line_thru_pair(points[i], points[i + 1], table).name,
#                 )
#             )
#             i += 4
#         return eqs, table

#     @classmethod
#     def add(cls, dep: Dependency) -> None:
#         eqs, table = cls._prep_ar(dep.statement)
#         for eq in eqs:
#             table.add_expr(eq, dep)

#     @classmethod
#     def check(cls, statement: Statement) -> bool:
#         eqs, table = cls._prep_ar(statement)
#         return all(table.expr_delta(eq) for eq in eqs)

#     @classmethod
#     def why(cls, statement: Statement) -> Dependency:
#         eqs, table = cls._prep_ar(statement)
#         why: list[Dependency] = []
#         for eq in eqs:
#             why.extend(table.why(eq))
#         return Dependency.mk(
#             statement, Angle_Chase, tuple(dep.statement for dep in why)
#         )

#     @classmethod
#     def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
#         a, b, c, d, e, f = args

#         if point in [d, e, f]:
#             a, b, c, d, e, f = d, e, f, a, b, c

#         x, b, y, c, d = b, c, e, d, f
#         if point == b:
#             a, b, c, d = b, a, d, c

#         if point == d and x == y:  # x p x b = x c x p
#             return f"angle_bisector {point} {b} {x} {c}"

#         if point == x:
#             return f"eqangle3 {x} {a} {b} {y} {c} {d}"

#         return f"on_aline {a} {x} {b} {c} {y} {d}"

#     @classmethod
#     def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
#         return tuple(p.name for p in args)

#     @classmethod
#     def pretty(cls, statement: Statement) -> str:
#         args: tuple[Point, ...] = statement.args
#         return " = ".join(
#             f"∠({a.pretty_name}{b.pretty_name},{c.pretty_name}{d.pretty_name})"
#             for a, b, c, d in reshape(args, 4)
#         )

#     @classmethod
#     def draw(
#         cls, ax: Axes, args: tuple[Any, ...], dep_graph: DependencyGraph, rng: Generator
#     ):
#         setattr(ax, "angle_color", (getattr(ax, "angle_color", 0) + 1) % len(PALETTE))
#         color = PALETTE[ax.angle_color]  # type: ignore
#         r = rng.random()
#         width = r * 0.1
#         symbols_graph = dep_graph.symbols_graph
#         for i in range(0, len(args), 4):
#             line0 = symbols_graph.container_of({args[i], args[i + 1]}, Line)
#             line1 = symbols_graph.container_of({args[i + 2], args[i + 3]}, Line)
#             assert line0 and line1
#             draw_angle(ax, line0, line1, color=color, alpha=0.5, width=width, r=r)
#             draw_line(ax, line0, ls=":")
#             draw_line(ax, line1, ls=":")
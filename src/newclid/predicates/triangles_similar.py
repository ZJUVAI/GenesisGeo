from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from matplotlib.axes import Axes
from matplotlib.pylab import Generator
from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.numerical.check import same_clock
from newclid.numerical.draw_figure import draw_segment
from newclid.predicates.predicate import Predicate


if TYPE_CHECKING:
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement


def two_triangles(
    a: str, b: str, c: str, p: str, q: str, r: str
) -> Optional[tuple[str, str, str, str, str, str]]:
    if a == b or a == c or b == c or p == q or p == r or q == r:
        return None
    (a0, p0), (b0, q0), (c0, r0) = sorted(((a, p), (b, q), (c, r)), key = lambda pair: [Predicate.custom_key(arg) for arg in pair])
    (a1, p1), (b1, q1), (c1, r1) = sorted(((p, a), (q, b), (r, c)), key = lambda pair: [Predicate.custom_key(arg) for arg in pair])
    return min((a0, b0, c0, p0, q0, r0), (a1, b1, c1, p1, q1, r1), key = lambda pair: [Predicate.custom_key(arg) for arg in pair])


class SimtriClock(Predicate):
    """simtri A B C P Q R -

    Represent that triangles ABC and PQR are similar under orientation-preserving
    transformations taking A to P, B to Q and C to R.

    It is equivalent to the three eqangle and eqratio predicates
    on the corresponding angles and sides.
    """

    NAME = "simtri"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        return two_triangles(*args)

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        twot = two_triangles(*args)
        return tuple(dep_graph.symbols_graph.names2points(twot)) if twot else None

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        a, b, c, p, q, r = args
        k = p.num.distance(q.num) / a.num.distance(b.num)
        return (
            close_enough(a.num.distance(c.num) * k, p.num.distance(r.num))
            and close_enough(b.num.distance(c.num) * k, q.num.distance(r.num))
            and same_clock(a.num, b.num, c.num, p.num, q.num, r.num)
        )

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        a, b, c, p, q, r = args
        return f"▲{a.pretty_name}{b.pretty_name}{c.pretty_name} ≅ ▲{p.pretty_name}{q.pretty_name}{r.pretty_name}"

    @classmethod
    def draw(
        cls, ax: Axes, args: tuple[Any, ...], dep_graph: DependencyGraph, rng: Generator
    ):
        draw_segment(ax, args[0], args[1], ls="dashed")
        draw_segment(ax, args[1], args[2], ls="dashed")
        draw_segment(ax, args[0], args[2], ls="dashed")
        draw_segment(ax, args[0 + 3], args[1 + 3], ls="dashed")
        draw_segment(ax, args[1 + 3], args[2 + 3], ls="dashed")
        draw_segment(ax, args[0 + 3], args[2 + 3], ls="dashed")


class SimtriReflect(Predicate):
    """simtrir A B C P Q R -

    Represent that triangles ABC and PQR are similar under orientation-preserving
    transformations taking A to P, B to Q and C to R.

    It is equivalent to the three eqangle and eqratio predicates
    on the corresponding angles and sides.
    """

    NAME = "simtrir"

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        return two_triangles(*args)

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        twot = two_triangles(*args)
        return tuple(dep_graph.symbols_graph.names2points(twot)) if twot else None

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        args: tuple[Point, ...] = statement.args
        a, b, c, p, q, r = args
        k = p.num.distance(q.num) / a.num.distance(b.num)
        return (
            close_enough(a.num.distance(c.num) * k, p.num.distance(r.num))
            and close_enough(b.num.distance(c.num) * k, q.num.distance(r.num))
            and same_clock(a.num, b.num, c.num, p.num, r.num, q.num)
        )

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        a, b, c, p, q, r = args
        return f"▲{a.pretty_name}{b.pretty_name}{c.pretty_name} ≅ ▲{p.pretty_name}{q.pretty_name}{r.pretty_name}"

    @classmethod
    def draw(
        cls,
        ax: Axes,
        args: tuple[Any, ...],
        dep_graph: "DependencyGraph",
        rng: Generator,
    ):
        SimtriClock.draw(ax, args, dep_graph, rng)

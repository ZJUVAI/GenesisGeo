from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from matplotlib.axes import Axes
from matplotlib.pylab import Generator

from newclid.dependencies.symbols import Point
from newclid.numerical import close_enough
from newclid.numerical.draw_figure import (
    draw_segment,
    draw_segment_num,
    get_figure_theme,
)
from newclid.predicates.predicate import Predicate
from newclid.algebraic_reasoning.tables import Ratio_Chase
from newclid.tools import reshape
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
    from newclid.algebraic_reasoning.tables import Table
    from newclid.algebraic_reasoning.tables import SumCV
    from newclid.statement import Statement
    from newclid.dependencies.dependency_graph import DependencyGraph


class Cong(Predicate):
    """cong A B C D -
    Represent that segments AB and CD are congruent."""

    NAME = "cong"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        segs: list[tuple[str, str]] = []
        if len(args) % 2 != 0:
            return None
        for a, b in zip(args[::2], args[1::2]):
            if cls.compare(a, b) > 0:
                a, b = b, a
            if a == b:
                return None
            segs.append((a, b))
        segs.sort(key=lambda pair: [cls.custom_key(arg) for arg in pair])
        points: list[str] = []
        for a, b in segs:
            points.append(a)
            points.append(b)
        return tuple(points)

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
        args: tuple[Point] = statement.args
        length = None
        for a, b in reshape(list(args), 2):
            _length = a.num.distance2(b.num)
            if length is not None and not close_enough(length, _length):
                return False
            length = _length
        return True

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        points: tuple[Point, ...] = statement.args
        table = statement.dep_graph.ar.rtable
        eqs: list[SumCV] = []
        i = 2
        while i < len(points):
            eqs.append(
                table.get_equal_elements_up_to(
                    table.get_length(points[0], points[1]),
                    table.get_length(points[i], points[i + 1]),
                ),
            )
            i += 2
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
            statement, Ratio_Chase, tuple(dep.statement for dep in why)
        )

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if a == c and a == point:
            return f"on_bline {a} {b} {d}"
        if b in [c, d]:
            if b == d:
                c, d = d, c
            return f"on_circle {a} {b} {d}"
        return f"eqdistance {a} {b} {c} {d}"

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args = statement.args
        return " = ".join(
            f"{a.pretty_name}{b.pretty_name}" for a, b in zip(args[::2], args[1::2])
        )

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def draw(
        cls,
        ax: Axes,
        args: tuple[Any, ...],
        dep_graph: "DependencyGraph",
        rng: Generator,
        segment_parent: Optional[dict[tuple[str, str], tuple[str, str]]] = None,
        segment_colors: Optional[dict[tuple[str, str], int]] = None,
        figure_sizes: list = [],
        draw_annotations: bool = True,
    ):
        # Draw original segments
        for i in range(0, len(args), 2):
            draw_segment(ax, args[i], args[i + 1], ls="solid")

        if not draw_annotations:
            return

        # Union-Find operations within draw function
        graph = dep_graph.symbols_graph
        if segment_parent is None:
            segment_parent = graph.segment_parent
        if segment_colors is None:
            segment_colors = graph.segment_colors

        def find_root(segment: tuple[str, str]) -> tuple[str, str]:
            """Find root with path compression"""
            if segment not in segment_parent:
                segment_parent[segment] = segment
                return segment
            if segment_parent[segment] != segment:
                segment_parent[segment] = find_root(segment_parent[segment])
            return segment_parent[segment]

        def union_segments(seg1: tuple[str, str], seg2: tuple[str, str]):
            """Union two segments"""
            root1 = find_root(seg1)
            root2 = find_root(seg2)
            if root1 != root2:
                segment_parent[root2] = root1

        # Collect current segments and normalize them
        current_segments = []
        for i in range(0, len(args), 2):
            p1, p2 = args[i], args[i + 1]
            current_segments.append(
                (p1.name, p2.name) if p1.name <= p2.name else (p2.name, p1.name)
            )

        # Union all current segments together
        for i in range(1, len(current_segments)):
            union_segments(current_segments[0], current_segments[i])

        # Find the root and assign color if needed
        root = find_root(current_segments[0])
        if root not in segment_colors:
            # Assign new color
            palette = get_figure_theme(ax)["palette"]
            setattr(ax, "cong_color", (getattr(ax, "cong_color", 0) + 1) % len(palette))
            color_index = getattr(ax, "cong_color")
            segment_colors[root] = color_index
        color_index = segment_colors[root]
        palette = get_figure_theme(ax)["palette"]

        # Find all segments in the same equivalence class and draw marks
        root = find_root(current_segments[0])
        equivalent_segments = []
        for seg in segment_parent:
            if find_root(seg) == root:
                equivalent_segments.append(seg)

        # Draw congruence marks for all equivalent segments
        for seg_key in equivalent_segments:
            p1_name, p2_name = seg_key
            p1 = graph.name2node[p1_name]
            p2 = graph.name2node[p2_name]

            # Calculate segment midpoint
            mid = (p1.num + p2.num) * 0.5

            # Calculate segment direction and perpendicular vectors
            direction = p2.num - p1.num
            direction = direction / abs(direction)
            perpendicular = direction.rot90()

            # Get current axis limits to determine figure size
            if len(figure_sizes) > 0:
                figure_size = figure_sizes[0]
            else:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                figure_width = xlim[1] - xlim[0]
                figure_height = ylim[1] - ylim[0]
                figure_size = max(figure_width, figure_height)
                figure_sizes.append(figure_size)

            # Set slash parameters proportional to figure size
            slash_length = figure_size * 0.015  # 1.5% of figure size
            slash_gap = figure_size * 0.006  # 0.6% of figure size

            # Calculate slash positions
            slash1_start = mid - direction * slash_gap - perpendicular * slash_length
            slash1_end = mid - direction * slash_gap + perpendicular * slash_length
            slash2_start = mid + direction * slash_gap - perpendicular * slash_length
            slash2_end = mid + direction * slash_gap + perpendicular * slash_length

            # Draw slashes
            draw_segment_num(
                ax,
                slash1_start,
                slash1_end,
                color=palette[color_index % len(palette)],
                lw=1.2,
            )
            draw_segment_num(
                ax,
                slash2_start,
                slash2_end,
                color=palette[color_index % len(palette)],
                lw=1.2,
            )

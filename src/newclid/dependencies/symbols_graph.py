from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Optional, Type, TypeVar

from newclid.algebraic_reasoning.tables import Table
import newclid.numerical.geometries as num_geo
from newclid.dependencies.symbols import Circle, Line, Point, Symbol
from newclid.tools import add_edge

if TYPE_CHECKING:
    from newclid.dependencies.dependency import Dependency

S = TypeVar("S", bound="Symbol")
CircL = TypeVar("CircL", "Circle", "Line")


class SymbolsGraph:
    def __init__(self) -> None:
        self._type2nodes: dict[Type[Symbol], list[Symbol]] = {
            Point: list(),
            Line: list(),
            Circle: list(),
        }
        self.name2node: dict[str, Symbol] = {}
        # Union-Find for segment congruence classes
        self.segment_parent: dict[tuple[str, str], tuple[str, str]] = {}
        self.segment_colors: dict[tuple[str, str], int] = {}

    def nodes_of_type(self, t: Type[S]) -> list[S]:
        return self._type2nodes[t]  # type: ignore

    def names2points(
        self, pnames: Collection[str], create_new_point: bool = True
    ) -> list[Point]:
        """Return Point objects given names."""
        result: list[Point] = []
        lookup = self.name2node.get
        for name in pnames:
            obj = lookup(name)
            if obj is None:
                if not create_new_point:
                    raise ValueError(f"Cannot find point {name} in graph")
                obj = self.new_node(Point, name, None)
            else:
                assert isinstance(obj, Point)
            result.append(obj)

        return result

    def container_of(self, pnames: set[Point], t: Type[CircL]) -> Optional[CircL]:
        for container in self.nodes_of_type(t):
            if pnames <= container.points:
                return container
        return None

    def new_node(
        self, oftype: Type[S], name: str, dep: Optional[Dependency] = None
    ) -> S:
        if name in self.name2node:
            raise ValueError(f"Node {name} already present")
        node = oftype(name, self, dep)
        self._type2nodes[type(node)].append(node)
        self.name2node[node.name] = node
        return node

    def _get_new_line_thru_pair(self, p1: Point, p2: Point) -> Line:
        name = p1.name + p2.name
        line = self.new_node(Line, name)
        line.num = num_geo.LineNum(p1.num, p2.num)
        line.points = {p1, p2}
        return line

    def line_thru_pair(self, p1: Point, p2: Point, table: Table) -> Line:
        for line in self.nodes_of_type(Line):
            if {p1, p2} <= line.points:
                for line1 in line.fellows:
                    if {p1, p2} == line1.points:
                        return line1
                res = self._get_new_line_thru_pair(p1, p2)
                assert line.dep
                table.add_expr(
                    table.get_equal_elements_up_to(res.name, line.name),
                    line.dep,
                )
                line.merge([res])
                return res
        return self._get_new_line_thru_pair(p1, p2)

    def save_pyvis(self, path: Path):
        from pyvis.network import Network  # type: ignore

        net = Network("1080px", directed=True)
        linecolor = "red"
        circlecolor = "green"
        pointcolor = "#97c2fc"
        for line in self.nodes_of_type(Line):
            net.add_node(  # type: ignore
                line.pretty_name,
                color=linecolor,
                title=line.dep.reason if line.dep else "",
                size=10,
            )
        for circle in self.nodes_of_type(Circle):
            net.add_node(  # type: ignore
                circle.pretty_name,
                color=circlecolor,
                title=circle.dep.reason if circle.dep else "",
                size=10,
            )
        for point in self.nodes_of_type(Point):
            net.add_node(point.pretty_name, color=pointcolor, size=10)  # type: ignore
        for line in self.nodes_of_type(Line):
            for p in line.points:
                add_edge(net, p.pretty_name, line.pretty_name)
        for circle in self.nodes_of_type(Circle):
            for p in circle.points:
                add_edge(net, p.pretty_name, circle.pretty_name)
        net.show_buttons(filter_=["physics", "layout"])  # type: ignore
        net.show(str(path), notebook=False)  # type: ignore

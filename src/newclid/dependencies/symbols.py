"""Implements geometric objects used in the graph representation."""

from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Optional, TypeVar, Union
from typing_extensions import Self

from newclid.numerical import nearly_zero
from newclid.numerical.geometries import CircleNum, LineNum, PointNum
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
    from newclid.formulations.clause import Clause
    from newclid.statement import Statement
    from newclid.dependencies.symbols_graph import SymbolsGraph

S = TypeVar("S", bound="Symbol")


class Symbol(ABC):
    r"""
    Symbol in the symbols graph.

    Can be Point, Line, Circle, etc.

    Each node maintains a merge history to
    other nodes if they are (found out to be) equivalent

    ::

        a -> b -
                \
            c -> d -> e -> f -> g


    d.merged_to = e
    d.rep = g
    d.merged_from = {a, b, c, d}
    d.equivs = {a, b, c, d, e, f, g}
    d.members = {a, b, c, d}

    """

    def __init__(
        self, name: str, symbols_graph: "SymbolsGraph", dep: Optional[Dependency]
    ):
        self.name = name
        self.symbols_graph = symbols_graph
        self.dep = dep
        self.fellows: list[Self] = [self]
        self._rep: Self = self

    def rep(self) -> Self:
        if self._rep != self:
            self._rep = self._rep.rep()
        return self._rep

    def _merge_one(self, node: Self) -> Self:
        selfrep = self.rep()
        noderep = node.rep()
        if selfrep == noderep:
            return selfrep
        noderep._rep = selfrep

        selfrep.fellows.extend(noderep.fellows)
        return selfrep

    def merge(self, nodes: list[Self]) -> None:
        """Merge all nodes."""
        for node in nodes:
            self._merge_one(node)
        reg = self.symbols_graph.nodes_of_type(type(self))
        for node in nodes + [self]:
            if node.rep() != node:
                reg.remove(node)

    def __repr__(self) -> str:
        return self.name


class Point(Symbol):
    num: PointNum
    clause: Clause
    rely_on: set[Point]

    @property
    def pretty_name(self) -> str:
        p = self.name[0].upper()
        for c in self.name[1:]:
            if c.isdigit():
                p += chr(ord("₀") + ord(c) - ord("0"))
            else:
                p += f"_{c}"
        return p


class Line(Symbol):
    """Symbol of type Line."""

    points: set[Point]
    num: LineNum

    @classmethod
    def check_coll(cls, points: Union[list[Point], tuple[Point, ...]]) -> bool:
        symbols_graph = points[0].symbols_graph
        s = set(points)
        for line in symbols_graph.nodes_of_type(Line):
            if s <= line.points:
                return True
        return False

    @classmethod
    def make_coll(
        cls, points: Union[list[Point], tuple[Point, ...]], dep: Dependency
    ) -> tuple[Line, list[Line]]:
        symbols_graph = points[0].symbols_graph
        s = set(points)
        merge: list[Line] = []
        pending = list(symbols_graph.nodes_of_type(Line))
        # A line visited early may overlap in only one point until another
        # line is merged.  Keep scanning until the point set stops growing,
        # otherwise an existing two-point line can remain disconnected from
        # the representative containing both of its points.
        while True:
            remaining = []
            for line in pending:
                if len(s & line.points) >= 2:
                    merge.append(line)
                    s.update(line.points)
                else:
                    remaining.append(line)
            if len(remaining) == len(pending):
                break
            pending = remaining
        for line in merge:
            if s <= line.points:
                # Reuse an existing full representative, but still connect
                # every overlapping line discovered during the closure.
                others = [other for other in merge if other is not line]
                line.merge(others)
                return line, others
        line = symbols_graph.new_node(
            Line, f"line/{'-'.join(p.name for p in points)}/", dep
        )
        line.points = s
        points = list(line.points)
        p1 = points[0]
        p2 = None
        for p in points[1:]:
            if not p1.num.close_enough(p.num):
                p2 = p
                break
        if p2 is None:
            p2 = points[-1]
        line.num = LineNum(p1=p1.num, p2=p2.num)
        # line.num = LineNum(p1=points[0].num, p2=points[1].num)
        line.merge(merge)
        return line, merge

    @classmethod
    def why_coll(cls, statement: Statement) -> Dependency:
        points: tuple[Point, ...] = statement.args
        symbols_graph = points[0].symbols_graph
        s = set(points)
        for line in symbols_graph.nodes_of_type(Line):
            if s <= line.points:
                fellows = list(line.fellows)
                fellows.sort(key=lambda x: len(x.points))
                why = []
                lines = []
                for fellow in fellows:
                    if len(fellow.points) < 3:
                        continue
                    why.append(fellow.dep.statement)
                    current_points = set(fellow.points)
                    while True:
                        merged = False
                        for line in lines:
                            if (
                                not line <= current_points
                                and len(line & current_points) >= 2
                            ):
                                current_points |= line
                                merged = True
                        if not merged:
                            break
                    if s <= current_points:
                        break
                    lines = [line for line in lines if not line <= current_points]
                    lines.append(current_points)
                return Dependency.mk(statement, "Same Line", why)
                # return Dependency.mk(statement, "Same Line", [])
        raise Exception("why_coll failed")

    @property
    def pretty_name(self) -> str:
        return "Line(" + "-".join(p.pretty_name for p in self.points) + ")"


class Circle(Symbol):
    """Symbol of type Circle."""

    points: set[Point]
    num: CircleNum

    @classmethod
    def check_cyclic(cls, points: Union[list[Point], tuple[Point, ...]]) -> bool:
        symbols_graph = points[0].symbols_graph
        s = set(points)
        for c in symbols_graph.nodes_of_type(Circle):
            if s <= c.points:
                return True
        return False

    @classmethod
    def make_cyclic(
        cls, points: Union[list[Point], tuple[Point, ...]], dep: Dependency
    ):
        symbols_graph = points[0].symbols_graph
        s = set(points)
        merge: list[Circle] = []
        # A later merge can supply the third shared point needed to merge an
        # earlier circle, so a single pass does not compute the closure.
        pending = list(symbols_graph.nodes_of_type(Circle))
        while True:
            remaining = []
            for circle in pending:
                if len(s & circle.points) >= 3:
                    merge.append(circle)
                    s.update(circle.points)
                else:
                    remaining.append(circle)
            if len(remaining) == len(pending):
                break
            pending = remaining
        for circle in merge:
            if s <= circle.points:
                circle.merge([other for other in merge if other is not circle])
                return
        c = symbols_graph.new_node(
            Circle, f"circle({''.join(p.name for p in points)})", dep
        )
        c.points = s
        points = list(c.points)

        # 选择三个彼此距离合适的点，避免数值误差
        # 策略：贪心选择，确保每对点之间都有足够距离
        p1 = points[0]
        p2 = None
        for p in points[1:]:
            if not p1.num.close_enough(p.num):
                p2 = p
                break
        if p2 is None:
            p2 = points[-1]

        p3 = None
        for p in points:
            if p == p1 or p == p2:
                continue
            # 检查与前两个点的距离都足够远
            if not p1.num.close_enough(p.num) and not p2.num.close_enough(p.num):
                # 检查不共线：计算叉积
                v1 = p2.num - p1.num
                v2 = p.num - p1.num
                cross = v1.x * v2.y - v1.y * v2.x
                if not nearly_zero(cross):
                    p3 = p
                    break
        if p3 is None:
            # 回退到默认选择
            for p in points:
                if p != p1 and p != p2:
                    p3 = p
                    break
        if p3 is None:
            p3 = points[-1]

        c.num = CircleNum(p1=p1.num, p2=p2.num, p3=p3.num)
        c.merge(merge)

    @classmethod
    def why_cyclic(cls, statement: Statement) -> Dependency:
        points: tuple[Point, ...] = statement.args
        symbols_graph = points[0].symbols_graph
        s = set(points)
        for line in symbols_graph.nodes_of_type(Circle):
            if s <= line.points:
                fellows = list(line.fellows)
                fellows.sort(key=lambda x: len(x.points))
                why = []
                circles = []
                for fellow in fellows:
                    why.append(fellow.dep.statement)
                    current_points = set(fellow.points)
                    while True:
                        merged = False
                        for circle in circles:
                            if (
                                not circle <= current_points
                                and len(circle & current_points) >= 3
                            ):
                                current_points |= circle
                                merged = True
                        if not merged:
                            break
                    if s <= current_points:
                        break
                    circles = [
                        circle for circle in circles if not circle <= current_points
                    ]
                    circles.append(current_points)
                return Dependency.mk(statement, "Same Circle", why)
                # return target.dep.with_new(statement)
        raise Exception("why_concyclic failed")

    @property
    def pretty_name(self) -> str:
        return "Circle(" + "-".join(p.pretty_name for p in self.points) + ")"

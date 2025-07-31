"""Implements geometric objects used in the graph representation."""

from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Optional, TypeVar, Union
from typing_extensions import Self

from newclid.formulations.clause import Clause
from newclid.numerical.geometries import CircleNum, LineNum, PointNum
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
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
        for line in symbols_graph.nodes_of_type(Line):
            if s <= line.points:
                return line, []
            if len(s & line.points) >= 2:
                merge.append(line)
                s.update(line.points)
        line = symbols_graph.new_node(
            Line, f"line/{'-'.join(p.name for p in points)}/", dep
        )
        line.points = s
        points = list(line.points)
        line.num = LineNum(p1=points[0].num, p2=points[1].num)
        line.merge(merge)
        return line, merge

    @classmethod
    def why_coll(cls, statement: Statement) -> Dependency:
        points: tuple[Point, ...] = statement.args
        symbols_graph = points[0].symbols_graph
        s = set(points)
        for line in symbols_graph.nodes_of_type(Line):
            if s <= line.points:
                target = line
                for _target in line.fellows:
                    if s <= _target.points and len(_target.points) < len(target.points):
                        target = _target
                assert target.dep
                return Dependency.mk(statement, "points on same line", [target.dep.statement])
                # return target.dep.with_new(statement)
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
        for c in symbols_graph.nodes_of_type(Circle):
            if s <= c.points:
                return
            if len(s & c.points) >= 3:
                merge.append(c)
                s.update(c.points)
        c = symbols_graph.new_node(
            Circle, f"circle({''.join(p.name for p in points)})", dep
        )
        c.points = s
        points = list(c.points)
        c.num = CircleNum(p1=points[0].num, p2=points[1].num, p3=points[2].num)
        c.merge(merge)

    @classmethod
    def why_cyclic(cls, statement: Statement) -> Dependency:
        points: tuple[Point, ...] = statement.args
        symbols_graph = points[0].symbols_graph
        s = set(points)
        for line in symbols_graph.nodes_of_type(Circle):
            if s <= line.points:
                target = line
                for _target in line.fellows:
                    if s <= _target.points and len(_target.points) < len(target.points):
                        target = _target
                if target.dep:
                    return Dependency.mk(statement, "points on same circle", [target.dep.statement])
                    # return target.dep.with_new(statement)
                else:
                    assert False
        raise Exception("why_concyclic failed")

    @property
    def pretty_name(self) -> str:
        return "Circle(" + "-".join(p.pretty_name for p in self.points) + ")"

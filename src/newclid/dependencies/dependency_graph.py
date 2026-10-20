from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Optional, Set
from newclid.dependencies.dependency import IN_PREMISES, NUMERICAL_CHECK, TRIVIAL
from newclid.dependencies.symbols import Point
from newclid.dependencies.symbols_graph import SymbolsGraph
from pyvis.network import Network  # type: ignore
from newclid.statement import Statement
from . import geometry
from collections import deque

from newclid.tools import add_edge, boring_statement  # type: ignore

if TYPE_CHECKING:
    from newclid.dependencies.dependency import Dependency
    from newclid.statement import Statement
    from newclid.algebraic_reasoning.algebraic_manipulator import (
        AlgebraicManipulator,
    )


def goal_filter(name: str, args: tuple[str]) -> bool:
    if name == "eqratio" or name == "eqangle":
        seg_1 = {args[0], args[1]}
        seg_2 = {args[2], args[3]}
        seg_3 = {args[4], args[5]}
        seg_4 = {args[6], args[7]}
        # case: eqratio AB/CD = DC/BA
        if seg_1 == seg_3 and seg_2 == seg_4:
            return False
        if seg_1 == seg_4 and seg_2 == seg_3:
            return False
        # AB/AB = CD/EF => cong CD = EF
        if seg_1 == seg_2 or seg_3 == seg_4:
            return False
    else:
        raise ValueError(f"Unknown goal type: {name}")
    return True


class DependencyGraph:
    """Hyper graph linking statements by dependencies as hyper-edges."""

    def __init__(self, ar: "AlgebraicManipulator") -> None:
        self.symbols_graph = SymbolsGraph()
        self.hyper_graph: dict[Statement, Dependency] = {}
        self.ar = ar
        self.check_numerical: dict[Statement, bool] = {}
        self.token_statement: dict[tuple[str, ...], Optional[Statement]] = {}
        self.numerical_checked_eqangle: list[tuple[str, ...]] = []
        self.numerical_checked_eqratio: list[tuple[str, ...]] = []
        self.numerical_checked_midp: list[tuple[str, ...]] = []
        self.numerical_checked_simtri: list[tuple[str, ...]] = []
        self.numerical_checked_simtrir: list[tuple[str, ...]] = []
        self.numerical_checked_cong: list[tuple[str, ...]] = []
        self.numerical_checked_para: list[tuple[str, ...]] = []
        self.numerical_checked_perp: list[tuple[str, ...]] = []

    def obtain_numerical_checked_premises(self):
        points = self.symbols_graph.nodes_of_type(Point)
        point_coords = [(point.num.x, point.num.y) for point in points]

        angle_ids, ratio_ids = geometry.process_points(point_coords)

        eqangles = geometry.findeq(point_coords, angle_ids)
        for eqangle in eqangles:
            self.numerical_checked_eqangle.append(eqangle)

        midpoints = geometry.findmidp(point_coords, ratio_ids)
        for midp in midpoints:
            tokens = (points[midp[0]].name, points[midp[1]].name, points[midp[2]].name)
            self.numerical_checked_midp.append(tokens)

        eqratios = geometry.findeq(point_coords, ratio_ids)
        for eqratio in eqratios:
            self.numerical_checked_eqratio.append(eqratio)

        simtris, simtrirs = geometry.findsimitri(point_coords, eqratios)
        for simtri in simtris:
            tokens = (
                points[simtri[0]].name,
                points[simtri[1]].name,
                points[simtri[2]].name,
                points[simtri[3]].name,
                points[simtri[4]].name,
                points[simtri[5]].name,
            )
            self.numerical_checked_simtri.append(tokens)
        for simtrir in simtrirs:
            tokens = (
                points[simtrir[0]].name,
                points[simtrir[1]].name,
                points[simtrir[2]].name,
                points[simtrir[3]].name,
                points[simtrir[4]].name,
                points[simtrir[5]].name,
            )
            self.numerical_checked_simtrir.append(tokens)

        self.numerical_checked_eqangle = list(set(self.numerical_checked_eqangle))
        self.numerical_checked_eqratio = list(set(self.numerical_checked_eqratio))
        self.numerical_checked_midp = list(set(self.numerical_checked_midp))
        self.numerical_checked_simtri = list(set(self.numerical_checked_simtri))
        self.numerical_checked_simtrir = list(set(self.numerical_checked_simtrir))

    def has_edge(self, dep: Dependency):
        return (
            dep.statement in self.hyper_graph and dep in self.hyper_graph[dep.statement]
        )

    def checked(self):
        return list(self.hyper_graph.keys())

    def premises(self):
        res: list[Dependency] = []
        for _, dep in self.hyper_graph.items():
            if dep.reason == IN_PREMISES:
                res.append(dep)
        return res

    def conclusions(self):
        res: list[Statement] = []
        for statement, dep in self.hyper_graph.items():
            if dep.reason != IN_PREMISES:
                res.append(statement)
        return res

    def _proof_text(
        self,
        statement: Statement,
        sub_proof: dict[Statement, tuple[Dependency, ...]],
    ) -> tuple[Dependency, ...]:
        if statement in sub_proof:
            return sub_proof[statement]
        dep = self.hyper_graph[statement]
        cur_proof: tuple[Dependency, ...] = ()
        for premise in dep.why:
            cur_proof += self._proof_text(premise, sub_proof)
        sub_proof[statement] = cur_proof
        return cur_proof + (dep,)

    def proof_deps(self, goals: list[Statement]) -> tuple[Dependency, ...]:
        sub_proof: dict[Statement, tuple[Dependency, ...]] = {}
        seen: set[Dependency] = set()
        res: list[Dependency] = []
        for goal in goals:
            proof_of_goal = self._proof_text(goal, sub_proof)
            for s in proof_of_goal:
                if s not in seen:
                    seen.add(s)
                    res.append(s)
        return tuple(res)

    def get_proof_steps(
        self, goals: list[Statement]
    ) -> tuple[
        set[Point],
        list[Dependency],
        list[Dependency],
        list[Dependency],
        set[Point],
        list[Dependency],
        list[Dependency],
        list[Dependency],
        list[Dependency],
    ]:
        proof_deps = self.proof_deps(goals)

        possible_points: set[Point] = set()
        queue = deque()

        for goal in goals:
            for arg in goal.args:
                if isinstance(arg, Point):
                    if arg not in possible_points:
                        possible_points.add(arg)
                        queue.append(arg)

        while queue:
            q = queue.popleft()
            for p in q.clause.points:
                p_name = p.split("@")[0]
                p_point = self.symbols_graph.names2points([p_name])[0]
                if p_point not in possible_points:
                    possible_points.add(p_point)
                    queue.append(p_point)
            for p in q.rely_on:
                if p not in possible_points:
                    possible_points.add(p)
                    queue.append(p)

        points: set[Point] = set()
        premises: list[Dependency] = []
        numerical_checked_premises: list[Dependency] = []
        trivial_premises: list[Dependency] = []
        aux_points: set[Point] = set()
        aux: list[Dependency] = []
        numerical_checked_aux: list[Dependency] = []
        trivial_aux: list[Dependency] = []
        proof_steps: list[Dependency] = []

        for line in proof_deps:
            is_aux = False
            for p in line.statement.args:
                if isinstance(p, Point):
                    if p in possible_points:
                        points.add(p)
                    else:
                        aux_points.add(p)
                        is_aux = True

            reason = line.reason
            if reason == IN_PREMISES:
                (aux if is_aux else premises).append(line)
            elif reason == NUMERICAL_CHECK:
                (
                    numerical_checked_aux if is_aux else numerical_checked_premises
                ).append(line)
            elif reason == TRIVIAL:
                (trivial_aux if is_aux else trivial_premises).append(line)
            else:
                proof_steps.append(line)
                continue

        return (
            points,
            premises,
            numerical_checked_premises,
            trivial_premises,
            aux_points,
            aux,
            numerical_checked_aux,
            trivial_aux,
            proof_steps,
        )

    def get_premises_and_aux(self, goals):
        visited: Set[Statement] = set()
        top_premises: Set[Dependency] = set()

        def dfs(stmt):
            if stmt in visited:
                return
            visited.add(stmt)

            if stmt not in self.hyper_graph:
                return

            dep = self.hyper_graph[stmt]

            if not dep.why:  # 顶层
                if dep.reason == IN_PREMISES:
                    top_premises.add(dep)
                return

            for pre in dep.why:
                dfs(pre)

        for goal in goals:
            dfs(goal)

        points: Set[Point] = set()
        queue = deque()

        for goal in goals:
            for arg in goal.args:
                if isinstance(arg, Point):
                    if arg not in points:
                        points.add(arg)
                        queue.append(arg)

        while queue:
            q = queue.popleft()
            for p in q.clause.points:
                p_name = p.split("@")[0]
                p_point = self.symbols_graph.names2points([p_name])[0]
                if p_point not in points:
                    points.add(p_point)
                    queue.append(p_point)
            for p in q.rely_on:
                if p not in points:
                    points.add(p)
                    queue.append(p)

        premises = []
        aux = []

        for dep in top_premises:
            uses_aux = any(
                isinstance(p, Point) and p not in points for p in dep.statement.args
            )

            if not uses_aux:
                premises.append(dep)
            else:
                aux.append(dep)

        return premises, aux

    def points_dependency_closure(self, points: Collection[Point]) -> Set[Point]:
        """
        从 points 出发，沿 clause.points / rely_on 正向扩展，返回这些点及其
        全部（可传递）依赖所构成的闭包。和 get_proof_steps/get_premises_and_aux
        里内联的 BFS 是同一套边，抽成公共方法以便复用。
        """
        closure: set[Point] = set()
        queue = deque(points)
        for p in points:
            closure.add(p)
        while queue:
            q = queue.popleft()
            for p in q.clause.points:
                p_name = p.split("@")[0]
                p_point = self.symbols_graph.names2points([p_name])[0]
                if p_point not in closure:
                    closure.add(p_point)
                    queue.append(p_point)
            for p in q.rely_on:
                if p not in closure:
                    closure.add(p)
                    queue.append(p)
        return closure

    def point_has_dependents_among(
        self, candidate: Point, others: Collection[Point]
    ) -> bool:
        """
        如果 others 中任意一点（通过 clause.points 或 rely_on，可传递）依赖于
        candidate，返回 True。用来判断 candidate 相对于 others 是否"下游/自由"——
        即能否安全地把它当作可移除/可重构的点。

        和 get_premises_and_aux/get_proof_steps 里的 BFS 用的是同一套边
        （clause.points、rely_on），只是从 others 出发正向跑，看闭包是否会碰到
        candidate。
        """
        for start in others:
            if start is candidate:
                continue
            seen: set[Point] = set()
            queue = deque([start])
            while queue:
                q = queue.popleft()
                if q in seen:
                    continue
                seen.add(q)
                if q is candidate:
                    return True
                for p in q.clause.points:
                    p_name = p.split("@")[0]
                    p_point = self.symbols_graph.names2points([p_name])[0]
                    if p_point not in seen:
                        queue.append(p_point)
                for p in q.rely_on:
                    if p not in seen:
                        queue.append(p)
        return False

    def save_pyvis(self, *, path: Path, stars: Collection[Statement] = []):
        if stars:
            deps = self.proof_deps(list(stars))
        else:
            deps = tuple(dep for _, dep in self.hyper_graph.items())
        net = Network("1080px", directed=True)
        for dep in deps:
            if boring_statement(dep.statement):
                continue
            shape = "dot"
            color = "#97c2fc"
            if dep.statement in stars:
                shape = "star"
                color = "gold"
            net.add_node(  # type: ignore
                dep.statement.pretty(),
                title=f"{dep.reason}",
                shape=shape,
                color=color,
                size=10,
            )
        for dep in deps:
            if boring_statement(dep.statement):
                continue
            for premise in dep.why:
                add_edge(net, premise.pretty(), dep.statement.pretty())  # type: ignore
        net.options.layout = {  # type: ignore
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
            },
        }
        net.show_buttons(filter_=["physics", "layout"])  # type: ignore
        net.show(str(path), notebook=False)  # type: ignore

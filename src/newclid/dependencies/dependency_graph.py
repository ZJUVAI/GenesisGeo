from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Optional
import numpy as np
from newclid.dependencies.dependency import IN_PREMISES, NUMERICAL_CHECK
from newclid.dependencies.symbols import Point
from newclid.dependencies.symbols_graph import SymbolsGraph
from pyvis.network import Network  # type: ignore
from newclid.statement import Statement
from newclid.predicates import NAME_TO_PREDICATE
from . import geometry

from newclid.tools import add_edge, boring_statement  # type: ignore

if TYPE_CHECKING:
    from newclid.dependencies.dependency import Dependency
    from newclid.statement import Statement
    from newclid.algebraic_reasoning.algebraic_manipulator import (
        AlgebraicManipulator,
    )

def goal_filter(name: str, args: tuple[str]) -> bool:
    if name == 'eqratio' or name == 'eqangle':
        seg_1 = {args[0], args[1]}
        seg_2 = {args[2], args[3]}
        seg_3 = {args[4], args[5]}
        seg_4 = {args[6], args[7]}
        #case: eqratio AB/CD = DC/BA
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
            tokens = (points[simtri[0]].name, points[simtri[1]].name, points[simtri[2]].name, points[simtri[3]].name, points[simtri[4]].name, points[simtri[5]].name)
            self.numerical_checked_simtri.append(tokens)   
        for simtrir in simtrirs:
            tokens = (points[simtrir[0]].name, points[simtrir[1]].name, points[simtrir[2]].name, points[simtrir[3]].name, points[simtrir[4]].name, points[simtrir[5]].name)
            self.numerical_checked_simtrir.append(tokens)

        # congs = geometry.findcong(point_coords, ratio_ids)
        # for cong in congs:
        #     tokens = (points[cong[0]].name, points[cong[1]].name, points[cong[2]].name, points[cong[3]].name)
        #     self.numerical_checked_cong.append(tokens)

        # perps = geometry.findperp(point_coords, angle_ids)
        # for perp in perps:
        #     tokens = (points[perp[0]].name, points[perp[1]].name, points[perp[2]].name, points[perp[3]].name)
        #     self.numerical_checked_perp.append(tokens)

        # paras = geometry.findpara(point_coords, angle_ids)
        # for para in paras:
        #     tokens = (points[para[0]].name, points[para[1]].name, points[para[2]].name, points[para[3]].name)
        #     self.numerical_checked_para.append(tokens)
                        
        self.numerical_checked_eqangle = list(set(self.numerical_checked_eqangle))
        self.numerical_checked_eqratio = list(set(self.numerical_checked_eqratio))
        self.numerical_checked_midp = list(set(self.numerical_checked_midp))
        self.numerical_checked_simtri = list(set(self.numerical_checked_simtri))
        self.numerical_checked_simtrir = list(set(self.numerical_checked_simtrir))
        # self.numerical_checked_cong = list(set(self.numerical_checked_cong))
        # self.numerical_checked_para = list(set(self.numerical_checked_para))
        # self.numerical_checked_perp = list(set(self.numerical_checked_perp))

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
        res: list[Dependency] = []
        for goal in goals:
            proof_of_goal = self._proof_text(goal, sub_proof)
            for s in proof_of_goal:
                if s not in res:
                    res.append(s)
        return tuple(res)

    def get_proof_steps(
        self, goals: list[Statement]
    ) -> tuple[
        set[Point],
        list[Dependency],
        list[Dependency],
        set[Point],
        list[Dependency],
        list[Dependency],
        list[Dependency],
    ]:
        proof_deps = self.proof_deps(goals)

        points: set[Point] = set()
        for goal in goals:
            points.update(goal.args)
        queue: list[Point] = list(points)
        i = 0
        while i < len(queue):
            q = queue[i]
            i += 1
            if not isinstance(q, Point):
                continue
            for p in q.rely_on:
                if p not in points:
                    points.add(p)
                    queue.append(p)

        premises: list[Dependency] = []
        numercial_checked_premises: list[Dependency] = []
        aux_points: set[Point] = set()
        aux: list[Dependency] = []
        numercial_checked_aux: list[Dependency] = []
        proof_steps: list[Dependency] = []

        for line in proof_deps:
            is_aux = any(
                [p not in points for p in line.statement.args if isinstance(p, Point)]
            )
            if IN_PREMISES == line.reason:
                if is_aux:
                    aux.append(line)
                    aux_points.update(
                        [
                            p
                            for p in line.statement.args
                            if isinstance(p, Point) and p not in points
                        ]
                    )
                else:
                    premises.append(line)
            elif NUMERICAL_CHECK == line.reason:
                if is_aux:
                    numercial_checked_aux.append(line)
                    aux_points.update(
                        [
                            p
                            for p in line.statement.args
                            if isinstance(p, Point) and p not in points
                        ]
                    )
                else:
                    numercial_checked_premises.append(line)
            else:
                proof_steps.append(line)

        return (
            points,
            premises,
            numercial_checked_premises,
            aux_points,
            aux,
            numercial_checked_aux,
            proof_steps,
        )

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
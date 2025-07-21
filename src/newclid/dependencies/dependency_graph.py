from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Optional
import numpy as np
from newclid.dependencies.dependency import IN_PREMISES, NUMERICAL_CHECK
from newclid.dependencies.symbols import Point
from newclid.dependencies.symbols_graph import SymbolsGraph
from pyvis.network import Network  # type: ignore
from newclid.numerical import close_enough, nearly_zero
from newclid.numerical.check import same_clock
from newclid.statement import Statement
from newclid.predicates import NAME_TO_PREDICATE
import itertools
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
        self.numerical_checked_eqangle: list[Statement] = []
        self.numerical_checked_eqratio: list[Statement] = []
        self.numerical_checked_midp: list[Statement] = []
        self.numerical_checked_simtri: list[Statement] = []
        self.numerical_checked_simtrir: list[Statement] = []
        # self.numerical_checked_cong: list[Statement] = []
        # self.numerical_checked_para: list[Statement] = []

    def args_rearrange(self, args: list[Point]) -> list[Point]:
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
        
        return [a,b,c,d,e,f,g,h]

    def obtain_numerical_checked_premises(self):
        points = self.symbols_graph.nodes_of_type(Point)
        point_coords = [(point.num.x, point.num.y) for point in points]

        angle_ids, ratio_ids = geometry.process_points(point_coords)
        
        angles = [(angle[0], points[angle[1]], points[angle[2]], points[angle[3]], points[angle[4]])for angle in angle_ids]
        ratios = [(ratio[0], points[ratio[1]], points[ratio[2]], points[ratio[3]], points[ratio[4]])for ratio in ratio_ids]

        # for (i, A) in enumerate(angles):
        #     if close_enough(A[0], 0) or close_enough(A[0], np.pi):
        #         continue
        #     for B in angles[i + 1:]:
        #         if not close_enough(A[0], B[0]):
        #             break
        #         if goal_filter('eqangle', [item.name for item in list(A[1:] + B[1:])]):
        #             tokens = tuple(['eqangle']+[item.name for item in list(A[1:] + B[1:])])
        #             pred = NAME_TO_PREDICATE[tokens[0]]
        #             preparsed = pred.preparse(tokens[1:])
        #             if preparsed[0] != preparsed[2] or preparsed[4] != preparsed[6] or preparsed[1] == preparsed[3] or preparsed[5] == preparsed[7]:
        #                 continue
        #             goal = Statement.from_tokens(tokens, self)
        #             if goal:
        #                 self.numerical_checked_eqangle.append(goal)
        eqangles = geometry.findeqangle(point_coords, angle_ids)
        for eqangle in eqangles:
            tokens = ('eqangle', points[eqangle[0]].name, points[eqangle[1]].name, points[eqangle[2]].name, points[eqangle[3]].name, points[eqangle[4]].name, points[eqangle[5]].name, points[eqangle[6]].name, points[eqangle[7]].name)
            goal = Statement.from_tokens(tokens, self)
            if goal:
                self.numerical_checked_eqangle.append(goal)
            # tokens = [points[eqangle[0]].name, points[eqangle[1]].name, points[eqangle[2]].name, points[eqangle[3]].name, points[eqangle[4]].name, points[eqangle[5]].name, points[eqangle[6]].name, points[eqangle[7]].name]
            # self.numerical_checked_eqangle.append(goal)

        midpoints = geometry.findmidp(point_coords, ratio_ids)
        for midp in midpoints:
            tokens = ('midp', points[midp[0]].name, points[midp[1]].name, points[midp[2]].name)
            goal = Statement.from_tokens(tokens, self)
            if goal:
                self.numerical_checked_midp.append(goal)
            # tokens = [ points[midp[0]].name, points[midp[1]].name, points[midp[2]].name]
            # self.numerical_checked_midp.append(tokens)


        # eqratios = geometry.findeqratio(point_coords, ratio_ids)
        # for eqratio in eqratios:
        #     tokens = ('eqratio', points[eqratio[0]].name, points[eqratio[1]].name, points[eqratio[2]].name, points[eqratio[3]].name, points[eqratio[4]].name, points[eqratio[5]].name, points[eqratio[6]].name, points[eqratio[7]].name)
        #     goal = Statement.from_tokens(tokens, self)
        #     if goal:
        #         self.numerical_checked_eqratio.append(goal)

        for (i, A) in enumerate(ratios):
            for B in ratios[i + 1:]:
                if not close_enough(A[0], B[0]):
                    break
                if goal_filter('eqratio', [item.name for item in list(A[1:] + B[1:])]):
                    tokens = tuple(['eqratio']+[item.name for item in list(A[1:] + B[1:])])
                    preparsed = self.args_rearrange(tokens[1:])

                    if preparsed[0] != preparsed[2] or preparsed[4] != preparsed[6]:
                        continue

                    goal = Statement.from_tokens(tokens, self)
                    if goal:
                        self.numerical_checked_eqratio.append(goal)
                    
                    if preparsed[0] != preparsed[4] and (preparsed[1] == preparsed[3] or preparsed[5] == preparsed[7]):
                        continue
                    
                    name2point = {item.name: item for item in list(A[1:] + B[1:])}
                    parsedpoints = [name2point[item] for item in preparsed]
                    
                    if preparsed[5] != preparsed[7]:    
                        ratio = parsedpoints[0].num.distance(parsedpoints[1].num)/parsedpoints[4].num.distance(parsedpoints[5].num)
                        if close_enough(parsedpoints[1].num.distance(parsedpoints[3].num), ratio * parsedpoints[5].num.distance(parsedpoints[7].num)):
                            if same_clock(parsedpoints[0].num, parsedpoints[1].num, parsedpoints[3].num, parsedpoints[4].num, parsedpoints[5].num, parsedpoints[7].num):
                                tokens = ('simtri', parsedpoints[0].name, parsedpoints[1].name, parsedpoints[3].name, parsedpoints[4].name, parsedpoints[5].name, parsedpoints[7].name)
                                goal = Statement.from_tokens(tokens, self)
                                if goal:
                                    self.numerical_checked_simtri.append(goal)
                            if same_clock(parsedpoints[0].num, parsedpoints[1].num, parsedpoints[3].num, parsedpoints[4].num, parsedpoints[7].num, parsedpoints[5].num):
                                tokens = ('simtrir', parsedpoints[0].name, parsedpoints[1].name, parsedpoints[3].name, parsedpoints[4].name, parsedpoints[5].name, parsedpoints[7].name)
                                goal = Statement.from_tokens(tokens, self)
                                if goal:
                                    self.numerical_checked_simtrir.append(goal)

                    if preparsed[0] == preparsed[4] and preparsed[3] != preparsed[7]: 
                        ratio = parsedpoints[0].num.distance(parsedpoints[1].num)/parsedpoints[2].num.distance(parsedpoints[3].num)
                        if close_enough(parsedpoints[1].num.distance(parsedpoints[5].num),ratio * parsedpoints[3].num.distance(parsedpoints[7].num)):
                            if same_clock(parsedpoints[0].num, parsedpoints[1].num, parsedpoints[5].num, parsedpoints[2].num, parsedpoints[3].num, parsedpoints[7].num):
                                tokens = ('simtri', parsedpoints[0].name, parsedpoints[1].name, parsedpoints[5].name, parsedpoints[2].name, parsedpoints[3].name, parsedpoints[7].name)
                                goal = Statement.from_tokens(tokens, self)
                                if goal:
                                    self.numerical_checked_simtri.append(goal)
                            if same_clock(parsedpoints[0].num, parsedpoints[1].num, parsedpoints[5].num, parsedpoints[2].num, parsedpoints[7].num, parsedpoints[3].num):
                                tokens = ('simtrir', parsedpoints[0].name, parsedpoints[1].name, parsedpoints[5].name, parsedpoints[2].name, parsedpoints[3].name, parsedpoints[7].name)
                                goal = Statement.from_tokens(tokens, self)
                                if goal:
                                    self.numerical_checked_simtrir.append(goal)


                        
        self.numerical_checked_eqangle = list(set(self.numerical_checked_eqangle))
        self.numerical_checked_eqratio = list(set(self.numerical_checked_eqratio))
        self.numerical_checked_midp = list(set(self.numerical_checked_midp))
        self.numerical_checked_simtri = list(set(self.numerical_checked_simtri))
        self.numerical_checked_simtrir = list(set(self.numerical_checked_simtrir))
        # self.numerical_checked_cong = list(set(self.numerical_checked_cong))
        # self.numerical_checked_para = list(set(self.numerical_checked_para))
        
        # print("eqangle",len(self.numerical_checked_eqangle))
        # print("eqratio",len(self.numerical_checked_eqratio))
        # print("midp",len(self.numerical_checked_midp))
        # print("simtri",len(self.numerical_checked_simtri))
        # print("simtrir",len(self.numerical_checked_simtrir))

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

# from __future__ import annotations
# from pathlib import Path
# from typing import TYPE_CHECKING, Collection, Optional
# import numpy as np
# from newclid.dependencies.dependency import IN_PREMISES, NUMERICAL_CHECK
# from newclid.dependencies.symbols import Point
# from newclid.dependencies.symbols_graph import SymbolsGraph
# from pyvis.network import Network  # type: ignore
# from newclid.numerical import close_enough
# from newclid.statement import Statement

# from newclid.tools import add_edge, boring_statement  # type: ignore

# if TYPE_CHECKING:
#     from newclid.dependencies.dependency import Dependency
#     from newclid.statement import Statement
#     from newclid.algebraic_reasoning.algebraic_manipulator import (
#         AlgebraicManipulator,
#     )

# def goal_filter(name: str, args: tuple[str]) -> bool:
#     if name == 'eqratio':
#         seg_1 = {args[0], args[1]}
#         seg_2 = {args[2], args[3]}
#         seg_3 = {args[4], args[5]}
#         seg_4 = {args[6], args[7]}
#         #case: eqratio AB/CD = DC/BA
#         if seg_1 == seg_3 and seg_2 == seg_4:
#             return False
#         if seg_1 == seg_4 and seg_2 == seg_3:
#             return False
#         # AB/AB = CD/EF => cong CD = EF
#         if seg_1 == seg_2 or seg_3 == seg_4: 
#             return False
#     elif name == 'eqangle':
#         #case: eqangle ∠AB CD = ∠DC/BA
#         seg_1 = {args[0], args[1]}
#         seg_2 = {args[2], args[3]}
#         seg_3 = {args[4], args[5]}
#         seg_4 = {args[6], args[7]}
#         if seg_1 == seg_3 and seg_2 == seg_4:
#             return False
#         if seg_1 == seg_4 and seg_2 == seg_3:
#             return False
#         if seg_1 == seg_2 or seg_3 == seg_4:
#             return False
#     else:
#         raise ValueError(f"Unknown goal type: {name}")
#     return True

# class DependencyGraph:
#     """Hyper graph linking statements by dependencies as hyper-edges."""

#     def __init__(self, ar: "AlgebraicManipulator") -> None:
#         self.symbols_graph = SymbolsGraph()
#         self.hyper_graph: dict[Statement, Dependency] = {}
#         self.ar = ar
#         self.check_numerical: dict[Statement, bool] = {}
#         self.token_statement: dict[tuple[str, ...], Optional[Statement]] = {}
#         self.numerical_checked_eqangle: list[Statement] = []
#         self.numerical_checked_eqratio: list[Statement] = []

#     def obtain_numerical_checked_premises(self):
#         self.numerical_checked_eqangle, self.numerical_checked_eqratio = self.get_numerical_checked_eqangle_and_eqratio()

#     def get_numerical_checked_eqangle_and_eqratio(self) -> tuple[list[Statement], list[Statement]]:
#         points = self.symbols_graph.nodes_of_type(Point)
#         angles: list[tuple[float, str, str, str, str]] = list()
#         ratios: list[tuple[float, str, str, str, str]] = list()

#         for (i, a) in enumerate(points):
#             for b in points[i + 1:]: 
#                 angle1 = (a.num - b.num).angle()
#                 dis = a.num.distance(b.num)
#                 for (k, c) in enumerate(points):
#                     for d in points[k + 1:]:
#                         angle = ((c.num - d.num).angle() - angle1) % np.pi
#                         ratio = dis / c.num.distance(d.num)
#                         angles.append((angle, a.name, b.name, c.name, d.name))
#                         ratios.append((ratio, a.name, b.name, c.name, d.name))
#                         ratios.append((1 / ratio, c.name, d.name, a.name, b.name))
        
#         angles.sort(key=lambda x: x[0])
#         ratios.sort(key=lambda x: x[0])
#         for (i, A) in enumerate(angles):
#             for B in angles[i + 1:]:
#                 if not close_enough(A[0], B[0]):
#                     break
#                 if goal_filter('eqangle', A[1:] + B[1:]):
#                     tokens = tuple(['eqangle'] + list(A[1:] + B[1:]))
#                     goal = Statement.from_tokens(tokens, self)
#                     if goal:
#                         self.numerical_checked_eqangle.append(goal)

#         for (i, A) in enumerate(ratios):
#             for B in ratios[i + 1:]:
#                 if not close_enough(A[0], B[0]):
#                     break
#                 if goal_filter('eqratio', A[1:] + B[1:]):
#                     tokens = tuple(['eqratio'] + list(A[1:] + B[1:]))
#                     goal = Statement.from_tokens(tokens, self)
#                     if goal:
#                         self.numerical_checked_eqratio.append(goal)
                
#         return list(set(self.numerical_checked_eqangle)), list(set(self.numerical_checked_eqratio))

#     def has_edge(self, dep: Dependency):
#         return (
#             dep.statement in self.hyper_graph and dep in self.hyper_graph[dep.statement]
#         )

#     def checked(self):
#         return list(self.hyper_graph.keys())

#     def premises(self):
#         res: list[Dependency] = []
#         for _, dep in self.hyper_graph.items():
#             if dep.reason == IN_PREMISES:
#                 res.append(dep)
#         return res

#     def conclusions(self):
#         res: list[Statement] = []
#         for statement, dep in self.hyper_graph.items():
#             if dep.reason != IN_PREMISES:
#                 res.append(statement)
#         return res

#     def _proof_text(
#         self,
#         statement: Statement,
#         sub_proof: dict[Statement, tuple[Dependency, ...]],
#     ) -> tuple[Dependency, ...]:
#         if statement in sub_proof:
#             return sub_proof[statement]
#         dep = self.hyper_graph[statement]
#         cur_proof: tuple[Dependency, ...] = ()
#         for premise in dep.why:
#             cur_proof += self._proof_text(premise, sub_proof)
#         sub_proof[statement] = cur_proof
#         return cur_proof + (dep,)

#     def proof_deps(self, goals: list[Statement]) -> tuple[Dependency, ...]:
#         sub_proof: dict[Statement, tuple[Dependency, ...]] = {}
#         res: list[Dependency] = []
#         for goal in goals:
#             proof_of_goal = self._proof_text(goal, sub_proof)
#             for s in proof_of_goal:
#                 if s not in res:
#                     res.append(s)
#         return tuple(res)

#     def get_proof_steps(
#         self, goals: list[Statement]
#     ) -> tuple[
#         set[Point],
#         list[Dependency],
#         list[Dependency],
#         set[Point],
#         list[Dependency],
#         list[Dependency],
#         list[Dependency],
#     ]:
#         proof_deps = self.proof_deps(goals)

#         points: set[Point] = set()
#         for goal in goals:
#             points.update(goal.args)
#         queue: list[Point] = list(points)
#         i = 0
#         while i < len(queue):
#             q = queue[i]
#             i += 1
#             if not isinstance(q, Point):
#                 continue
#             for p in q.rely_on:
#                 if p not in points:
#                     points.add(p)
#                     queue.append(p)

#         premises: list[Dependency] = []
#         numercial_checked_premises: list[Dependency] = []
#         aux_points: set[Point] = set()
#         aux: list[Dependency] = []
#         numercial_checked_aux: list[Dependency] = []
#         proof_steps: list[Dependency] = []

#         for line in proof_deps:
#             is_aux = any(
#                 [p not in points for p in line.statement.args if isinstance(p, Point)]
#             )
#             if IN_PREMISES == line.reason:
#                 if is_aux:
#                     aux.append(line)
#                     aux_points.update(
#                         [
#                             p
#                             for p in line.statement.args
#                             if isinstance(p, Point) and p not in points
#                         ]
#                     )
#                 else:
#                     premises.append(line)
#             elif NUMERICAL_CHECK == line.reason:
#                 if is_aux:
#                     numercial_checked_aux.append(line)
#                     aux_points.update(
#                         [
#                             p
#                             for p in line.statement.args
#                             if isinstance(p, Point) and p not in points
#                         ]
#                     )
#                 else:
#                     numercial_checked_premises.append(line)
#             else:
#                 proof_steps.append(line)

#         return (
#             points,
#             premises,
#             numercial_checked_premises,
#             aux_points,
#             aux,
#             numercial_checked_aux,
#             proof_steps,
#         )

#     def get_essential_clauses(
#         self,
#         goals: list[Statement],
#     ) -> tuple[set[str], set[str]]:
#         essential_clauses: set[str] = set()
#         essential_clauses_aux: set[str] = set()

#         points, _, _, aux_points, _, _, _ = self.get_proof_steps(goals)
#         for p in points:
#             essential_clauses.add(str(p.clause))
#         for p in aux_points:
#             essential_clauses_aux.add(str(p.clause))
#         return essential_clauses, essential_clauses_aux

#     def save_pyvis(self, *, path: Path, stars: Collection[Statement] = []):
#         if stars:
#             deps = self.proof_deps(list(stars))
#         else:
#             deps = tuple(dep for _, dep in self.hyper_graph.items())
#         net = Network("1080px", directed=True)
#         for dep in deps:
#             if boring_statement(dep.statement):
#                 continue
#             shape = "dot"
#             color = "#97c2fc"
#             if dep.statement in stars:
#                 shape = "star"
#                 color = "gold"
#             net.add_node(  # type: ignore
#                 dep.statement.pretty(),
#                 title=f"{dep.reason}",
#                 shape=shape,
#                 color=color,
#                 size=10,
#             )
#         for dep in deps:
#             if boring_statement(dep.statement):
#                 continue
#             for premise in dep.why:
#                 add_edge(net, premise.pretty(), dep.statement.pretty())  # type: ignore
#         net.options.layout = {  # type: ignore
#             "hierarchical": {
#                 "enabled": True,
#                 "direction": "LR",
#                 "sortMethod": "directed",
#             },
#         }
#         net.show_buttons(filter_=["physics", "layout"])  # type: ignore
#         net.show(str(path), notebook=False)  # type: ignore
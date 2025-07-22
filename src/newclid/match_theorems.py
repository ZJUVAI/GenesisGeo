"""Implements theorem matching functions for the Deductive Database (DD)."""

import itertools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Optional
import json
import time

from newclid.formulations.clause import translate_sentence
from newclid.dependencies.symbols import Point
from newclid.statement import Statement
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
    import numpy as np
    from newclid.formulations.rule import Rule
    from newclid.dependencies.dependency_graph import DependencyGraph

from newclid import matchinC

class Matcher:
    def __init__(
        self,
        dep_graph: "DependencyGraph",
        runtime_cache_path: Optional[Path],
        rng: "np.random.Generator",
    ) -> None:
        self.dep_graph = dep_graph
        self.rng = rng
        self.runtime_cache_path: Optional[Path] = None
        self.update(runtime_cache_path)
        self.cache: dict["Rule", tuple[Dependency, ...]] = {}
        self.rule_match_time: dict[str, float] = {}

    def update(self, runtime_cache_path: Optional[Path] = None):
        self.runtime_cache_path = runtime_cache_path
        if self.runtime_cache_path is not None and not self.runtime_cache_path.exists():
            os.makedirs(os.path.dirname(self.runtime_cache_path), exist_ok=True)
            self.runtime_cache_path.touch()
            with open(self.runtime_cache_path, "w") as f:
                json.dump({}, f)
        self.cache = {}
        self.rule_match_time = {}
        
    def apply_theorem(self, theorem: "Rule", mapping: dict[str, str]) -> Optional[set[Dependency]]:
        res: set[Dependency] = set()
        why: list[Statement] = []
        reason = theorem.descrption
        applicable = True
        for premise in theorem.premises:
            s = Statement.from_tokens(
                translate_sentence(mapping, premise), self.dep_graph
            )
            if s is None:
                applicable = False
                break
            if not s.check_numerical():
                applicable = False
                break
            why.append(s)
        if not applicable:
            return None
        for conclusion in theorem.conclusions:
            conclusion_statement = Statement.from_tokens(
                translate_sentence(mapping, conclusion), self.dep_graph
            )
            # assert conclusion_statement.check_numerical()
            if conclusion_statement is None or not conclusion_statement.check_numerical():
                continue
            dep = Dependency.mk(conclusion_statement, reason, tuple(why))
            res.add(dep)
        return res
    
    def rearrange(self, args: list[str]) -> set[tuple[str]]:
        assert len(args) == 8
        permutations = set()
        for i in range(64):
            perm = args.copy()
            for j in range(4):
                if (i >> j) & 1:
                    perm[2 * j], perm[2 * j + 1] = perm[2 * j + 1], perm[2 * j]
            if (i >> 4) & 1:
                perm[0], perm[1], perm[2], perm[3] = perm[2], perm[3], perm[0], perm[1]
                perm[4], perm[5], perm[6], perm[7] = perm[6], perm[7], perm[4], perm[5]
            if (i >> 5) & 1:
                perm[0], perm[1], perm[4], perm[5] = perm[4], perm[5], perm[0], perm[1]
                perm[2], perm[3], perm[6], perm[7] = perm[6], perm[7], perm[2], perm[3]
            permutations.add(tuple(perm))
        return permutations

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



    def match_theorem(self, theorem: "Rule") -> Generator["Dependency", None, None]:
        if theorem not in self.cache:
            self.cache_theorem(theorem)
        for dep in self.cache[theorem]:
            if dep.statement in dep.statement.dep_graph.hyper_graph:
                continue
            applicable = True
            for premise in dep.why:
                if not premise.check():
                    applicable = False
            if applicable:
                yield dep

    def map_premises_to_points(self, premises, points):
        result = []
        premise_points = []
        for premise in premises:
            # 获取premise中这些点的新点（在point中没有出现的点）
            new_points = []
            for p in premise[1:]:
                if p not in premise_points:
                    new_points.append(p)
                    premise_points.append(p)
            if new_points:
                mappings =  [{v: p for v, p in zip(new_points, point_list)}
                        for point_list in itertools.product(points, repeat=len(new_points))]
                result.append(mappings)
            else:
                # 如果没有新点，返回空的映射
                result.append([{}])

        return result
    
    def cache_eq_premise_theorem(self, theorem, points):
        statement_list = None
        res: set[Dependency] = set()
        premise = theorem.premises[0]
        variables = theorem.variables()
        if premise[0] == "eqangle":
            statement_list = self.dep_graph.numerical_checked_eqangle
        if premise[0] == "eqratio":
            statement_list = self.dep_graph.numerical_checked_eqratio
        variables_in_premise = set(premise[1:])
        variables_not_in_premise= list(set(variables) - variables_in_premise)
        for statement in statement_list:
            # args = [p.name for p in statement.args]
            args = statement
            assert len(args) == 8
            # args = self.args_rearrange(args)
            if args[0] != args[4]:
                args_permutation = {
                    (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                    (args[2], args[3], args[0], args[1], args[6], args[7], args[4], args[5]),
                    (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]),
                    (args[6], args[7], args[4], args[5], args[2], args[3], args[0], args[1])
                }
            else:
                args_permutation = {
                    (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                    (args[2], args[3], args[0], args[1], args[6], args[7], args[4], args[5]),
                    (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]),
                    (args[6], args[7], args[4], args[5], args[2], args[3], args[0], args[1]),
                    (args[0], args[1], args[4], args[5], args[2], args[3], args[6], args[7]),
                    (args[4], args[5], args[0], args[1], args[6], args[7], args[2], args[3]),
                    (args[6], args[7], args[2], args[3], args[4], args[5], args[0], args[1]),
                    (args[2], args[3], args[6], args[7], args[0], args[1], args[4], args[5]),
                }

            for args in args_permutation:
                mapping = {}
                flag = True
                for v, p in zip(premise[1:], args):
                    if v not in mapping:
                        mapping[v] = p
                    elif mapping[v] != p:
                        flag = False
                        break
                if not flag:
                    continue
                    
                for extra_mapping in (
                    {v: p for v, p in zip(variables_not_in_premise, point_list)} 
                    for point_list in itertools.product(points, repeat=len(variables_not_in_premise))
                ):
                    mapping.update(extra_mapping)
                    # logging.info(f"{theorem} mapping {mapping=}")
                    new_conclusions = self.apply_theorem(theorem, mapping)
                    if new_conclusions:
                        res.update(new_conclusions)
        return res
    
    def cache_midp_premise_theorem(self, theorem, points):
        statement_list = self.dep_graph.numerical_checked_midp
        res: set[Dependency] = set()
        premise = theorem.premises[0]
        variables = theorem.variables()
        variables_in_premise = set(premise[1:])
        variables_not_in_premise= list(set(variables) - variables_in_premise)
        for statement in statement_list:
            # args = [p.name for p in statement.args]
            args = statement
            assert len(args) == 3
            args_permutation = {
                (args[0], args[1], args[2]),
                (args[0], args[2], args[1])
            }
            for args in args_permutation:
                mapping = {}
                flag = True
                for v, p in zip(premise[1:], args):
                    if v not in mapping:
                        mapping[v] = p
                    elif mapping[v] != p:
                        flag = False
                        break
                if not flag:
                    continue
                    
                for extra_mapping in (
                    {v: p for v, p in zip(variables_not_in_premise, point_list)} 
                    for point_list in itertools.product(points, repeat=len(variables_not_in_premise))
                ):
                    mapping.update(extra_mapping)
                    # logging.info(f"{theorem} mapping {mapping=}")
                    new_conclusions = self.apply_theorem(theorem, mapping)
                    if new_conclusions:
                        res.update(new_conclusions)
        return res
    
    def cache_simtri_premise_theorem(self, theorem, points):
        statement_list = None
        res: set[Dependency] = set()
        premise = theorem.premises[0]
        variables = theorem.variables()
        if premise[0] == "simtri":
            statement_list = self.dep_graph.numerical_checked_simtri
        elif premise[0] == "simtrir":
            statement_list = self.dep_graph.numerical_checked_simtrir
        variables_in_premise = set(premise[1:])
        variables_not_in_premise= list(set(variables) - variables_in_premise)
        for statement in statement_list:
            # args = [p.name for p in statement.args]
            args = statement
            assert len(args) == 6
            args_permutation = {
                (args[0], args[1], args[2], args[3], args[4], args[5]),
                (args[0], args[2], args[1], args[3], args[5], args[4]),
                (args[1], args[0], args[2], args[4], args[3], args[5]),
                (args[1], args[2], args[0], args[4], args[5], args[3]),
                (args[2], args[0], args[1], args[5], args[3], args[4]),
                (args[2], args[1], args[0], args[5], args[4], args[3]),
                (args[3], args[4], args[5], args[0], args[1], args[2]),
                (args[3], args[5], args[4], args[0], args[2], args[1]),
                (args[4], args[3], args[5], args[1], args[0], args[2]),
                (args[4], args[5], args[3], args[1], args[2], args[0]),
                (args[5], args[3], args[4], args[2], args[0], args[1]),
                (args[5], args[4], args[3], args[2], args[1], args[0])
            }
            for args in args_permutation:
                mapping = {}
                flag = True
                for v, p in zip(premise[1:], args):
                    if v not in mapping:
                        mapping[v] = p
                    elif mapping[v] != p:
                        flag = False
                        break
                if not flag:
                    continue
                    
                for extra_mapping in (
                    {v: p for v, p in zip(variables_not_in_premise, point_list)} 
                    for point_list in itertools.product(points, repeat=len(variables_not_in_premise))
                ):
                    mapping.update(extra_mapping)
                    # logging.info(f"{theorem} mapping {mapping=}")
                    new_conclusions = self.apply_theorem(theorem, mapping)
                    if new_conclusions:
                        res.update(new_conclusions)
        return res

    def cache_normal_theorem(self, theorem : "Rule", points):
        premises = theorem.premises
        point_ids = [(p.num.x, p.num.y) for p in self.dep_graph.symbols_graph.nodes_of_type(Point)]
        list_premises = [list(premise) for premise in premises]
        mappings = matchinC.mapping_normal_theorem(list_premises, point_ids)
        res: set[Dependency] = set()

        for mapping in mappings:
            why = []
            flag = False
            for key in mapping:
                mapping[key] = points[mapping[key]]
            for premise in premises:
                s = Statement.from_tokens(
                        translate_sentence(mapping, premise), self.dep_graph
                    )
                if s is None:
                    flag = True
                    break
                why.append(s)
            if flag:
                continue
            for conclusion in theorem.conclusions:
                s = Statement.from_tokens(
                        translate_sentence(mapping, conclusion), self.dep_graph
                    )
                if s is None or not s.check_numerical():
                    continue
                dep = Dependency.mk(s, theorem.descrption, tuple(why))
                res.add(dep)
                
        return res
    
    def cache_simtri_conclusion_theorem(self, theorem: "Rule", points):
        statement_list = None
        res: set[Dependency] = set()
        conclusion = theorem.conclusions[0]
        variables = theorem.variables()
        if conclusion[0] == "simtri":
            statement_list = self.dep_graph.numerical_checked_simtri
        elif conclusion[0] == "simtrir":
            statement_list = self.dep_graph.numerical_checked_simtrir
        variables_in_premise = set(conclusion[1:])
        variables_not_in_premise= list(set(variables) - variables_in_premise)
        for statement in statement_list:
            # args = [p.name for p in statement.args]
            args = statement
            assert len(args) == 6
            args_permutation = {
                (args[0], args[1], args[2], args[3], args[4], args[5]),
                (args[0], args[2], args[1], args[3], args[5], args[4]),
                (args[1], args[0], args[2], args[4], args[3], args[5]),
                (args[1], args[2], args[0], args[4], args[5], args[3]),
                (args[2], args[0], args[1], args[5], args[3], args[4]),
                (args[2], args[1], args[0], args[5], args[4], args[3]),
                (args[3], args[4], args[5], args[0], args[1], args[2]),
                (args[3], args[5], args[4], args[0], args[2], args[1]),
                (args[4], args[3], args[5], args[1], args[0], args[2]),
                (args[4], args[5], args[3], args[1], args[2], args[0]),
                (args[5], args[3], args[4], args[2], args[0], args[1]),
                (args[5], args[4], args[3], args[2], args[1], args[0])
            }
            for args in args_permutation:
                mapping = {}
                flag = True
                for v, p in zip(conclusion[1:], args):
                    if v not in mapping:
                        mapping[v] = p
                    elif mapping[v] != p:
                        flag = False
                        break
                if not flag:
                    continue
                    
                for extra_mapping in (
                    {v: p for v, p in zip(variables_not_in_premise, point_list)} 
                    for point_list in itertools.product(points, repeat=len(variables_not_in_premise))
                ):
                    mapping.update(extra_mapping)
                    # logging.info(f"{theorem} mapping {mapping=}")
                    new_conclusions = self.apply_theorem(theorem, mapping)
                    if new_conclusions:
                        res.update(new_conclusions)
        return res
    
    def cache_eq_conclusion_theorem(self, theorem: "Rule", points):
        statement_list = None
        res: set[Dependency] = set()
        premise = theorem.conclusions[0]
        variables = theorem.variables()
        if premise[0] == "eqangle":
            statement_list = self.dep_graph.numerical_checked_eqangle
        if premise[0] == "eqratio":
            statement_list = self.dep_graph.numerical_checked_eqratio
        variables_in_premise = set(premise[1:])
        variables_not_in_premise= list(set(variables) - variables_in_premise)
        for statement in statement_list:
            # args = [p.name for p in statement.args]
            args = statement
            assert len(args) == 8
            # args = self.args_rearrange(args)
            if args[0] != args[4]:
                args_permutation = {
                    (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                    (args[2], args[3], args[0], args[1], args[6], args[7], args[4], args[5]),
                    (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]),
                    (args[6], args[7], args[4], args[5], args[2], args[3], args[0], args[1])
                }
            else:
                args_permutation = {
                    (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                    (args[2], args[3], args[0], args[1], args[6], args[7], args[4], args[5]),
                    (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]),
                    (args[6], args[7], args[4], args[5], args[2], args[3], args[0], args[1]),
                    (args[0], args[1], args[4], args[5], args[2], args[3], args[6], args[7]),
                    (args[4], args[5], args[0], args[1], args[6], args[7], args[2], args[3]),
                    (args[6], args[7], args[2], args[3], args[4], args[5], args[0], args[1]),
                    (args[2], args[3], args[6], args[7], args[0], args[1], args[4], args[5]),
                }

            for args in args_permutation:
                mapping = {}
                flag = True
                for v, p in zip(premise[1:], args):
                    if v not in mapping:
                        mapping[v] = p
                    elif mapping[v] != p:
                        flag = False
                        break
                if not flag:
                    continue
                    
                for extra_mapping in (
                    {v: p for v, p in zip(variables_not_in_premise, point_list)} 
                    for point_list in itertools.product(points, repeat=len(variables_not_in_premise))
                ):
                    mapping.update(extra_mapping)
                    # logging.info(f"{theorem} mapping {mapping=}")
                    new_conclusions = self.apply_theorem(theorem, mapping)
                    if new_conclusions:
                        res.update(new_conclusions)
        return res

    def cache_theorem(self, theorem: "Rule"):
        start_time = time.time()
        
        # file_cache = None
        # write = False
        # read = False
        # mappings: list[dict[str, str]] = []
        
        # if self.runtime_cache_path is not None:
        #     with open(self.runtime_cache_path) as f:
        #         file_cache = json.load(f)
        #     if "matcher" not in file_cache:
        #         file_cache["matcher"] = {}
        #     if str(theorem) in file_cache["matcher"]:
        #         mappings = file_cache["matcher"][str(theorem)]
        #         read = True
        #     else:
        #         file_cache["matcher"][str(theorem)] = mappings
        #         write = True
        
        self.cache[theorem] = ()
        points = [p.name for p in self.dep_graph.symbols_graph.nodes_of_type(Point)]
        
        # logging.debug(
        #     f"{theorem} matching cache : before {len(self.cache[theorem])=} {read=} {write=} {len(mappings)=}"
        # )
        # if read:
        #     for mapping in mappings:
        #         new_conclusions = self.apply_theorem(theorem, mapping)
        #         if new_conclusions:
        #             res.update(new_conclusions)
        if theorem.conclusions[0][0] == 'simtri' or theorem.conclusions[0][0] == 'simtrir':
            res = self.cache_simtri_conclusion_theorem(theorem, points)
        elif theorem.premises[0][0] == 'simtri' or theorem.premises[0][0] == 'simtrir':
            res = self.cache_simtri_premise_theorem(theorem, points)
        elif theorem.premises[0][0] == 'eqangle' or theorem.premises[0][0] == 'eqratio':
            res = self.cache_eq_premise_theorem(theorem, points)
        elif theorem.conclusions[0][0] == 'eqangle' or theorem.conclusions[0][0] == 'eqratio':
            res = self.cache_eq_conclusion_theorem(theorem, points)
        elif theorem.premises[0][0] == 'midp':
            res = self.cache_midp_premise_theorem(theorem, points)
        else:
            res = self.cache_normal_theorem(theorem, points)

        self.cache[theorem] = tuple(
            sorted(res, key=lambda x: repr(x))
        )  # to maintain determinism

        # if self.runtime_cache_path is not None and write:
        #     with open(self.runtime_cache_path, "w") as f:
        #         json.dump(file_cache, f)
        # logging.debug(
        #     f"{theorem} matching cache : now {len(self.cache[theorem])=} {read=} {write=} {len(mappings)=}"
        # )
        
        # record cache time
        elapsed_time = time.time() - start_time
        rule_name = theorem.descrption
        if rule_name not in self.rule_match_time:
            self.rule_match_time[rule_name] = 0
        self.rule_match_time[rule_name] += elapsed_time

    def get_rule_match_time_stats(self) -> dict[str, float]:
        """get the statistics of rule match time"""
        return self.rule_match_time.copy()
    
    def reset_rule_match_time_stats(self):
        """Reset the rule match time statistics."""
        self.rule_match_time = {}


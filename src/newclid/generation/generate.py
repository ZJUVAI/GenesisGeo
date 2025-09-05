import logging
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import re
import argparse
import itertools
import json
import random
import string
import time
from collections import defaultdict
from datetime import timedelta
import ray
from millify import millify
import numpy as np

from newclid.agent.ddarn import DDARN
from newclid.api import GeometricSolver, GeometricSolverBuilder
from newclid.configs import default_defs_path
from newclid.dependencies.dependency import Dependency, IN_PREMISES, NUMERICAL_CHECK
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.dependencies.symbols import Point
from newclid.formulations.clause import translate_sentence
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.problem import ProblemJGEX
from newclid.generation.clause_generation import CompoundClauseGen
from newclid.generation.output_summary import Summary, get_first_predicate
from newclid.proof import ProofState
from newclid.proof_writing import get_structured_proof, write_proof_steps
from newclid.statement import Statement

class GeometryGoalFilter:
    def goal_valid_check(self, tokens, dep_graph):
        name = tokens[0]
        args = tokens[1:]
        if args[-1] == '':
            args = args[:-1]
        # AG1 do not support aconst and rconst
        if name in ('aconst', 'rconst'):  # rconst AB:AB=1, aconst ∠AB AB=0
            if name == 'rconst' and args[-1] == '1':
                return False
        # case: cong AB = AB,
        if name == 'cong':
            left = {args[0], args[1]}
            right = {args[2], args[3]}
            if left == right:
                return False
        # para AB ∥ AB, AB ∥ AC
        if name == 'para':
            if len({args[0], args[1], args[2], args[3]}) < 4:
                return False
        if name == 'eqratio':
            seg_1, seg_2, seg_3, seg_4 = {args[0], args[1]}, {
                args[2], args[3]}, {args[4], args[5]}, {args[6], args[7]}
            # case: eqratio AB/CD = AB/CD
            # case: eqratio AB/CD = CD/AB => cong AB = CD
            # case: eqratio AB/AB = CD/EF => cong CD = EF
            if seg_1 == seg_2 or seg_1 == seg_3 or seg_3 == seg_4 or seg_2 == seg_4:
                return False
            # case: exist two segments with the same length
            sm1 = Statement.from_tokens(['cong'] + list(args[:4]), dep_graph)
            sm2 = Statement.from_tokens(
                ['cong'] + list(args[0:2])+list(args[4:6]), dep_graph)
            if sm1.check() or sm2.check():
                return False
        if name == 'eqangle':
            seg_1, seg_2, seg_3, seg_4 = {args[0], args[1]}, {
                args[2], args[3]}, {args[4], args[5]}, {args[6], args[7]}
            # case: eqangle ∠(AB,CD) = ∠(AB,CD)
            # case: eqangle ∠(AB,CD) = ∠(CD,AB) => perp AB⊥CD
            # case: eqangle ∠(AB,AB) = ∠(CD,EF) => para CD∥EF
            if seg_1 == seg_2 or seg_1 == seg_3 or seg_3 == seg_4 or seg_2 == seg_4:
                return False
            # case: two parallels or perp
            parallel_sets = [
                [args[0], args[1], args[2], args[3]],
                [args[0], args[1], args[4], args[5]],
                [args[0], args[1], args[6], args[7]],
                [args[4], args[5], args[6], args[7]],
                [args[2], args[3], args[4], args[5]],
                [args[2], args[3], args[6], args[7]]
            ]
            for arg_set in parallel_sets:
                sm = Statement.from_tokens(['para'] + arg_set, dep_graph)
                if sm.check():
                    return False
            sm = Statement.from_tokens(['perp'] + list(args[:4]), dep_graph)
            if sm.check():
                return False
            # case: simtri
            a1_args = list(set(args[:4]))
            a2_args = list(set(args[4:]))
            if len(a1_args) == 3 and len(a2_args) == 3:
                simtri_sets = [
                    [*a1_args, *a2_args],
                    [*a1_args, a2_args[0], a2_args[2], a2_args[1]],
                    [*a1_args, a2_args[1], a2_args[0], a2_args[2]],
                    [*a1_args, a2_args[1], a2_args[2], a2_args[0]],
                    [*a1_args, a2_args[2], a2_args[0], a2_args[1]],
                    [*a1_args, a2_args[2], a2_args[1], a2_args[0]]
                ]
                for simtri_set in simtri_sets:
                    sm = Statement.from_tokens(
                        ['simtri']+simtri_set, dep_graph)
                    if sm.check():
                        return False
                for simtri_set in simtri_sets:
                    sm = Statement.from_tokens(
                        ['simtrir']+simtri_set, dep_graph)
                    if sm.check():
                        return False
        if name in ('simtri', 'simtrir', 'contri', 'contrir'):
            # case: simtri △ABC ≅ △ABC
            tri_1 = {args[0], args[1], args[2]}
            tri_2 = {args[3], args[4], args[5]}
            if tri_1 == tri_2:
                return False
        if name == 'sameclock':
            return False

        return True

    def goal_filter(self, possible_goals, dep_graph):
        """filter the equivalent eq goals"""

        def check_equivalence(p1, p2, token_type):
            args1 = [arg.name for arg in p1.args]
            args2 = [arg.name for arg in p2.args]
            statements = [
                Statement.from_tokens(
                    [token_type, args1[i], args1[i + 1], args2[i], args2[i + 1]], dep_graph)
                for i in range(0, len(args1), 2)
            ]
            return all(sm.check() for sm in statements)

        def remove_duplicates(goals, equivalence_fn, token_type):
            unique_goals = []
            for goal in goals:
                if not any(equivalence_fn(existing_goal, goal, token_type) for existing_goal in unique_goals):
                    unique_goals.append(goal)
            return unique_goals

        eqangle_goals = [
            goal for goal in possible_goals 
            if goal.predicate.NAME == 'eqangle'
        ]
        eqratio_goals = [
            goal for goal in possible_goals 
            if goal.predicate.NAME == 'eqratio'
        ]
        other_goals = [
            goal for goal in possible_goals 
            if goal.predicate.NAME not in ('eqangle', 'eqratio')
        ]

        eqangle_goals = remove_duplicates(eqangle_goals, check_equivalence, 'para')
        eqratio_goals = remove_duplicates(eqratio_goals, check_equivalence, 'cong')

        return other_goals + eqangle_goals + eqratio_goals

    def aux_predicates_valid_check(self, llm_output: str) -> bool:

        def is_valid(statement: str, valid_predicates: set) -> bool:
            prefix_match = re.match(r"(x00 \w+)\s*:\s*(.*)", statement)
            if prefix_match:
                # coll a c e [002] coll b d e [003]
                rest = prefix_match.group(2)
                segments = re.split(r"\s*\[\d+\]", rest)
                # 'coll a c e' , 'coll b d e'
                segments = [seg.strip() for seg in segments if seg.strip()]
                for segment in segments:
                    parts = segment.split()
                    if parts and parts[0] not in valid_predicates:
                        logging.debug(
                            f"Invalid auxiliary predicate: {parts[0]}")
                        return False
            return True

        # <aux> x00 c : perp k n n s [024] cong k n n s [025]; x00 h : ; x00 i : ; x00 j : perp h i h j [009] cong h i h j [010] ; </aux> <proof> cong a k c k [002] r19 [000] [001] ; cong b k c k [003] r19 [000] [001] ; cong a k b k [004] a00 [002] [003] ; </proof>
        valid_aux_predicates = {'perp', 'para',
                                'cong', 'coll', 'eqangle', 'cyclic', 'midp'}
        aux_match = re.match(r"<aux>\s*(.*)\s*</aux>", llm_output)
        # c : perp a c b c [001] ; c : perp a c b c [001] ;
        if aux_match:
            aux_content = aux_match.group(1)
            for content_item in aux_content.split(';'):
                content_item = content_item.strip()
                if content_item:
                    if not is_valid(content_item, valid_aux_predicates):
                        return False
        return True

class GeometryGenerator:
    def __init__(self, n_clauses=5, n_threads=1, output_dir="dataset", min_proof_steps=5, min_clauses_num=3, n_samples=100, timeout=3600, filteration_rate=0.6):
        self.n_clauses = n_clauses
        self.min_proof_steps = min_proof_steps
        self.min_clauses_num = min_clauses_num
        self.n_samples = n_samples
        self.n_threads = n_threads
        self.timeout = timeout
        self.output_dir = output_dir
        self.filteration_rate = filteration_rate
        self.path_prefix = os.path.join(
            self.output_dir, f"geometry_clauses{self.n_clauses}_samples{millify(self.n_samples)}")
        self.write_buffer = []
        self.writer_hash = set()
        self.filter = GeometryGoalFilter()
        self.defs = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))
        self.clauses_generator = CompoundClauseGen(seed=int(time.time())+os.getpid(), defs=self.defs)

    def all_possible_goals_by_ar(self, dep_graph: DependencyGraph) -> list[Statement]:
        def extract_points(s):
            return re.findall(r'[a-z][\d]*', s)

        def goal_from_tokens(tokens):
            if self.filter.goal_valid_check(tokens, dep_graph):
                goal = Statement.from_tokens(tokens, dep_graph)
                if goal and goal.check():
                    return [goal]
            return []

        points_name = sorted(
            [p.name for p in dep_graph.symbols_graph.nodes_of_type(Point)])
        for i, p in enumerate(points_name):
            for q in points_name[i + 1:]:
                ar = dep_graph.ar
                if (p + q) not in ar.atable.v2e:
                    ar.atable.add_free(p + q)
                if f"l({p},{q})" not in ar.rtable.v2e:
                    ar.rtable.add_free(f"l({p},{q})")

        ar = dep_graph.ar

        e2v, e2v_pairs2, e2v_pairs4 = ar.atable.possible_pairs()
        for e in e2v_pairs2.keys():
            for v1, v2 in e2v_pairs2[e]:
                try:
                    v1, v2 = extract_points(v1), extract_points(v2)
                    goal_from_tokens(tuple(['para'] + list(v1 + v2)))
                    goal_from_tokens(tuple(['perp'] + list(v1 + v2)))
                except Exception as e:
                    logging.warning(
                        f"Error in goal_from_tokens: {e} para/perp for {v1}, {v2}")
                    continue
        for v1, v2, v3, v4 in e2v_pairs4:
            try:
                v1, v2, v3, v4 = extract_points(v1), extract_points(v2), extract_points(v3), extract_points(v4)
                goal_from_tokens(tuple(['eqangle'] + list(v1 + v2 + v3 + v4)))
            except Exception as e:
                logging.warning(
                    f"Error in goal_from_tokens: {e} for eqangle {v1}, {v2}, {v3}, {v4}")
                continue

        e2v, e2v_pairs2, e2v_pairs4 = ar.rtable.possible_pairs()
        for e in e2v_pairs2.keys():
            for v1, v2 in e2v_pairs2[e]:
                try:
                    goal_from_tokens(
                        tuple(['cong'] + v1[2:-1].split(',') + v2[2:-1].split(',')))
                except Exception as e:
                    logging.warning(
                        f"Error in goal_from_tokens: {e} cong for {v1}, {v2}")
                    continue
        for v1, v2, v3, v4 in e2v_pairs4:
            try:
                tokens = tuple(['eqratio'] + list(v1[2:-1].split(',') +
                               v2[2:-1].split(',') + v3[2:-1].split(',') + v4[2:-1].split(',')))
                goal_from_tokens(tokens)
            except Exception as e:
                logging.warning(
                    f"Error in goal_from_tokens: {e} for eqratio {v1}, {v2}, {v3}, {v4}")
                continue

    def llm_solution(self, problem: ProblemJGEX, aux_points: list[str], proof_state: ProofState) -> str:
        dep_idx: dict[Statement, str] = {}

        data_tmp = defaultdict(list)
        for construction in problem.constructions:
            group = {}
            p2deps = defaultdict(list)
            for constr_sentence in construction.sentences:
                cdef = self.defs[constr_sentence[0]]
                if len(constr_sentence) == len(cdef.declare):
                    mapping = dict(zip(cdef.declare[1:], constr_sentence[1:]))
                else:
                    assert len(constr_sentence) + \
                        len(construction.points) == len(cdef.declare)
                    mapping = dict(
                        zip(cdef.declare[1:], construction.points + constr_sentence[1:]))
                for points, bs in cdef.basics:
                    points = tuple([mapping[x] for x in points])
                    for p in points:
                        group[p] = points
                    for b in bs:
                        statement = Statement.from_tokens(
                            translate_sentence(mapping, b), proof_state.dep_graph)
                        p2deps[points].append(statement)

            points = construction.points
            while points:
                p = points[0]
                gr = group[p]
                points = [x for x in points if x not in gr]

                deps = []
                for dep in p2deps[gr]:
                    deps.append(dep)
                data_tmp[' '.join(gr)] = deps

        # <problem> </problem>
        data_problem = '<problem> '
        string_premise = []
        for k, v in data_tmp.items():
            if not all(p in aux_points for p in k.split(' ')):
                tmp_string = k + ' : '
                for dep in v:
                    if dep not in dep_idx:
                        dep_idx[dep] = f"{len(dep_idx):03d}"
                    tmp_string += dep.to_str() + f' [{dep_idx[dep]}] '
                string_premise.append(tmp_string)
        data_problem += ' ; '.join([s.strip() for s in string_premise]) + ' ? '
        data_problem += ' ;'.join([
            (goal[0] + ' ' + ' '.join(goal[1:]))
            for goal in problem.goals
        ])
        data_problem += ' </problem>'

        # <aux> </aux>
        data_aux = ''
        string_aux = []
        for k, v in data_tmp.items():
            if all(p in aux_points for p in k.split(' ')):
                tmp_string = 'x00 ' + k + ' : '
                for dep in v:
                    if dep not in dep_idx:
                        dep_idx[dep] = f"{len(dep_idx):03d}"
                    tmp_string += dep.to_str() + f' [{dep_idx[dep]}] '
                string_aux.append(tmp_string)
        if len(string_aux) > 0:
            data_aux += '<aux> '
            data_aux += ' ; '.join([s.strip() for s in string_aux])
            data_aux += ' ; </aux> '

        # get analysis, numerical_check and proof
        data_analysis, data_numerical_check, data_proof = get_structured_proof(
            proof_state, dep_idx)

        # <numerical_check> </numerical_check>
        if data_numerical_check != '':
            data_numerical_check += ' '

        return {
            "llm_data": data_problem + ' ' + data_aux + data_numerical_check + data_proof,
            "llm_input": data_problem,
            # "llm_output": data_aux + data_analysis + ' ' + data_numerical_check + data_proof,
            "llm_output": data_aux + data_numerical_check + data_proof,
        }
    
    def llm_solution_renamed(self, problem: ProblemJGEX, aux_points: list[str], proof_state: ProofState) -> str:
        def get_apha_geo_solver_var(va_idx):
            letter_part = string.ascii_lowercase[va_idx % 26]
            number_part = va_idx // 26

            # Prepare the point name
            if number_part == 0:
                # For the first cycle (A-Z), we don't add a number part
                point_name = letter_part
            else:
                # For subsequent cycles, add the number part (reduced by 1 to start from 0)
                point_name = f"{letter_part}{number_part - 1}"

            return point_name

        def statement2str_with_mapping(statement: Statement, mp):
            res = [statement.predicate.NAME] + [mp[arg.name]
                                                if isinstance(arg, Point) else str(arg) for arg in statement.args]
            return " ".join(res)

        def get_all_premise(problem):
            data_tmp = defaultdict(list)
            for construction in problem.constructions:
                group = {}
                p2deps = defaultdict(list)
                points_in_basic_order = []
                for constr_sentence in construction.sentences:
                    cdef = self.defs[constr_sentence[0]]
                    if len(constr_sentence) == len(cdef.declare):
                        mapping = dict(
                            zip(cdef.declare[1:], constr_sentence[1:]))
                    else:
                        assert len(constr_sentence) + \
                            len(construction.points) == len(cdef.declare)
                        mapping = dict(
                            zip(cdef.declare[1:], construction.points + constr_sentence[1:]))
                    for points, bs in cdef.basics:
                        points = tuple([mapping[x] for x in points])
                        for p in points:
                            points_in_basic_order.append(p)
                            group[p] = points
                        for b in bs:
                            statement = Statement.from_tokens(
                                translate_sentence(mapping, b), proof_state.dep_graph)
                            p2deps[points].append(statement)

                points = points_in_basic_order
                while points:
                    p = points[0]
                    gr = group[p]
                    points = [x for x in points if x not in gr]

                    deps = []
                    for dep in p2deps[gr]:
                        deps.append(dep)
                    data_tmp[' '.join(gr)] = deps
            return data_tmp

        def get_essential_points_and_premise(premises, proof_deps, points, aux_points):
            # essential_premises
            essential_premises = []
            for line in premises:
                essential_premises.append(line.statement)
            # essential_points points
            essential_points = set()
            for line in proof_deps:
                for arg in line.statement.args:
                    if isinstance(arg, Point):
                        essential_points.add(arg.name)
            points = set([p.name for p in points]) & essential_points
            aux_points = set([p.name for p in aux_points]) & essential_points
            return points, aux_points, essential_premises

        def rediger_new_format(dep, mp, dep_idx) -> str:
            """Generate proof step in new format: statement [id] rule_id [required_statement_ids]"""
            for statement in (dep.statement,) + dep.why:
                statemtn_str = statement2str_with_mapping(statement, mp)
                if statemtn_str not in dep_idx:
                    dep_idx[statemtn_str] = f"{len(dep_idx):03d}"

            # Extract rule ID from reason string and handle special cases
            reason = dep.reason
            if "Ratio Chasing" in reason:
                rule_id = "a00"
            elif "Angle Chasing" in reason:
                rule_id = "a01"
            elif "Shortcut Derivation" in reason:
                rule_id = "r99"
            elif "Same Circle" in reason:
                rule_id = "r98"
            elif "Same Line" in reason:
                rule_id = "r97"
            elif reason and ' ' in reason:
                rule_id = reason.split()[0]
            else:
                rule_id = reason if reason else "unknown"

            # Generate new format: statement [statement_id] rule_id [premise_ids]
            premise_ids = ' '.join(
                f"[{dep_idx[statement2str_with_mapping(premise, mp)]}]" for premise in dep.why)
            conclusion_str = statement2str_with_mapping(dep.statement, mp)
            return f"{conclusion_str} [{dep_idx[conclusion_str]}] {rule_id} {premise_ids}".strip()

        dep_idx: dict[str, str] = {}
        # get proof info
        goals = [goal for goal in proof_state.goals if goal.check()]
        (
            points,
            premises,
            numercial_checked_premises,
            aux_points,
            aux,
            numercial_checked_aux,
            proof_steps,
        ) = proof_state.dep_graph.get_proof_steps(goals)

        # find essential_premises
        all_premise = get_all_premise(problem)  # all premises from problem
        essential_points, essential_aux_points, essential_premises = get_essential_points_and_premise(
            premises+aux, proof_state.dep_graph.proof_deps(goals), points, aux_points)  # only included in proof steps
        # mapping
        mp: dict[str, str] = {}
        for k, v in all_premise.items():
            kps = k.split(' ')
            if any(p in essential_points for p in kps):
                for dep in v:
                    if dep in essential_premises:
                        for arg in dep.args:
                            if isinstance(arg, Point) and arg.name not in mp:
                                mp[arg.name] = get_apha_geo_solver_var(len(mp))
                for p in kps:
                    if p not in mp:
                        mp[p] = get_apha_geo_solver_var(len(mp))

        for k, v in all_premise.items():
            ps = k.split(' ')
            if any(p in essential_aux_points for p in ps):
                for p in ps:
                    if p not in mp:
                        mp[p] = get_apha_geo_solver_var(len(mp))
        # import pdb; pdb.set_trace()
        # <problem> </problem>
        try:
            string_premise = []
            for k, v in all_premise.items():
                # if not all(p in essential_aux_points for p in k.split(' ')):
                if any(p in essential_points for p in k.split(' ')):
                    tmp_string = ""
                    for dep in v:
                        if dep in essential_premises:  # only select useful premise and free points withou useful premises
                            dep_str_renamed = statement2str_with_mapping(
                                dep, mp)
                            if dep_str_renamed not in dep_idx:
                                dep_idx[dep_str_renamed] = f"{len(dep_idx):03d}"
                            tmp_string += dep_str_renamed + \
                                f' [{dep_idx[dep_str_renamed]}] '
                    if tmp_string == "":
                        # if this premise is useless, free all points in it
                        for p in k.split(' '):
                            string_premise.append(mp[p] + " : ")
                    else:
                        k_renamed = " ".join(mp[p] for p in k.split(' '))
                        tmp_string = k_renamed + ' : ' + tmp_string
                        string_premise.append(tmp_string)
            data_problem = '<problem> '
            data_problem += ' ; '.join([s.strip()
                                       for s in string_premise]) + ' ? '
            data_problem += ' ;'.join([statement2str_with_mapping(goal, mp)
                                      for goal in proof_state.goals])
            data_problem += ' </problem>'
            # import pdb; pdb.set_trace()
            # <aux> </aux>
            data_aux = ''
            string_aux = []
            for k, v in all_premise.items():
                # if all(p in aux_points for p in k.split(' ')):
                if all(p in essential_aux_points for p in k.split(' ')):
                    k_renamed = " ".join(mp[p] for p in k.split(' '))
                    tmp_string = 'x00 ' + k_renamed + ' : '
                    for dep in v:
                        if dep in essential_premises:  # free points withou useful premises
                            dep_str_renamed = statement2str_with_mapping(
                                dep, mp)
                            if dep_str_renamed not in dep_idx:
                                dep_idx[dep_str_renamed] = f"{len(dep_idx):03d}"
                            tmp_string += dep_str_renamed + \
                                f' [{dep_idx[dep_str_renamed]}] '
                    string_aux.append(tmp_string)
            if len(string_aux) > 0:
                data_aux += '<aux> '
                data_aux += ' ; '.join([s.strip() for s in string_aux])
                data_aux += ' ; </aux> '
            # import pdb; pdb.set_trace()
            # <numerical_check> </numerical_check>
            numerical_check_items = []
            # numercial_checked_premises
            for line in numercial_checked_premises:
                statemtn_str = statement2str_with_mapping(line.statement, mp)
                if statemtn_str not in dep_idx:
                    dep_idx[statemtn_str] = f"{len(dep_idx):03d}"
            sorted_numercial_checked_premises = sorted(
                numercial_checked_premises, key=lambda line: dep_idx[statement2str_with_mapping(line.statement, mp)])
            for line in sorted_numercial_checked_premises:
                statemtn_str = statement2str_with_mapping(line.statement, mp)
                numerical_check_items.append(
                    f"{statemtn_str} [{dep_idx[statemtn_str]}]")
            # numercial_checked_premises
            for line in numercial_checked_aux:
                statemtn_str = statement2str_with_mapping(line.statement, mp)
                if statemtn_str not in dep_idx:
                    dep_idx[statemtn_str] = f"{len(dep_idx):03d}"
            sorted_numercial_checked_aux = sorted(
                numercial_checked_aux, key=lambda line: dep_idx[statement2str_with_mapping(line.statement, mp)])
            for line in sorted_numercial_checked_aux:
                statemtn_str = statement2str_with_mapping(line.statement, mp)
                numerical_check_items.append(
                    f"{statemtn_str} [{dep_idx[statemtn_str]}]")
            if len(numerical_check_items) > 0:
                numerical_check = "<numerical_check> " + \
                    " ; ".join(numerical_check_items) + \
                    " ; </numerical_check> "
            else:
                numerical_check = ""

            # <proof> </proof>
            proof = "<proof> "
            proof_steps_formatted = []
            for k, line in enumerate(proof_steps):
                if NUMERICAL_CHECK not in line.reason and IN_PREMISES not in line:
                    proof_steps_formatted.append(
                        rediger_new_format(line, mp, dep_idx))
            proof += " ; ".join(proof_steps_formatted) + " ; </proof>"
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(essential_points)
            print(essential_aux_points)
            print(essential_premises)
            print(mp)
            print(all_premise)
            # import pdb; pdb.set_trace()
        return {
            "llm_input": data_problem,
            "llm_output": data_aux + numerical_check + proof,
        }

    def process_single_problem(self, args: tuple) -> tuple[list, dict]:
        def find_minimal_aux_clauses(all_constructions, goal_str, essential_clauses, essential_clauses_aux):
            # Iterate through all possible subsets to find the minimal necessary auxiliary clause set
            minimal_aux_set = set()
            # Search through subsets from size 0 to len-1 (excluding full set)
            for r in range(len(essential_clauses_aux)):
                for aux_subset in itertools.combinations(essential_clauses_aux, r):
                    aux_subset_set = set(aux_subset)
                    statements_test = []
                    for clause in all_constructions:
                        clause_str = str(clause)
                        if clause_str in essential_clauses or clause_str in aux_subset_set:
                            statements_test.append(clause_str)
                    fl_problem_test = '; '.join(
                        statements_test) + ' ? ' + goal_str

                    solver_builder_test = GeometricSolverBuilder(
                        seed=random.randint(0, 998244353))
                    solver_builder_test.with_deductive_agent(DDARN())
                    solver_builder_test.load_problem_from_txt(fl_problem_test)
                    try:
                        solver_test = solver_builder_test.build(
                            max_attempts=100)
                    except Exception as e:
                        logging.debug(f"Error: {e}")
                        continue
                    if solver_test.run(timeout=self.timeout):
                        minimal_aux_set = aux_subset_set
                        return {
                            "info": 'success',
                            "aux_clauses": minimal_aux_set,
                            "solver": solver_test,
                            "problem": solver_builder_test.problemJGEX
                        }
            return {"info": 'failed'}
        
        try:
            """Process a single geometry problem."""
            pid, fl_statement = args
            start_time = time.time()

            # t = time.time()
            # # clauses_generator = CompoundClauseGen(seed=os.getpid(), defs=self.defs)
            # # fl_statement = clauses_generator.generate(self.n_clauses)
            # prob_gen_time = time.time() - t
            # print(f'problem statement generated in {prob_gen_time:.1f} seconds')

            solver_builder = GeometricSolverBuilder(seed=998244353)
            solver_builder.with_deductive_agent(DDARN())
            solver_builder.load_problem_from_txt(fl_statement)
            try:
                solver = solver_builder.build(max_attempts=1)
            except Exception as e:
                logging.info(f"Error: {e}")
                return [], {}
            solver.run(timeout=self.timeout)

            t = time.time()
            self.all_possible_goals_by_ar(solver.proof.dep_graph)
            possible_goals = [goal for goal in solver.proof.dep_graph.conclusions() if self.filter.goal_valid_check(goal.to_str().split(" "), solver.proof.dep_graph)]
            possible_goals = self.filter.goal_filter(possible_goals, solver.proof.dep_graph)
            checkgoals_runtime = time.time() - t

            n_filtered_samples = 0
            proofs_of_used_rules = {}
            generated_data = []
            for goal in possible_goals:
                # find minimal aux clauses
                solver.proof.goals = [goal]
                solver_new = solver
                problem_new = str(solver_builder.problemJGEX) + goal.to_str()
                problem_new = ProblemJGEX.from_text(problem_new)

                last_essential_clauses_len = float('inf')
                last_essential_clauses_aux_len = float('inf')
                while True:
                    # get proof and essential_clauses
                    points, _, _, aux_points, _, _, proof_steps = solver_new.proof.dep_graph.get_proof_steps(solver_new.proof.goals)
                    essential_clauses: set[str] = set()
                    essential_clauses_aux: set[str] = set()
                    for p in aux_points:
                        essential_clauses_aux.add(str(p.clause))
                    for p in points:
                        if str(p.clause) not in essential_clauses_aux:
                            essential_clauses.add(str(p.clause))
                    if last_essential_clauses_len == len(essential_clauses) and last_essential_clauses_aux_len == len(essential_clauses_aux):
                        break
                    last_essential_clauses_len = len(essential_clauses)
                    last_essential_clauses_aux_len = len(essential_clauses_aux)
                    res = find_minimal_aux_clauses(
                        [str(cons) for cons in solver_builder.problemJGEX.constructions], 
                        goal.to_str(), 
                        essential_clauses, 
                        essential_clauses_aux
                    )
                    if res['info'] == 'success':
                        essential_clauses_aux = res['aux_clauses']
                        solver_new = res['solver']
                        problem_new = res['problem']

                # filter clauses
                n_clauses = len(essential_clauses | essential_clauses_aux)
                if n_clauses < self.min_clauses_num:
                    logging.debug(f"Too few clauses: {n_clauses}")
                    continue

                # get new proof
                points, _, _, aux_points, _, _, proof_steps = solver_new.proof.dep_graph.get_proof_steps(
                    solver_new.proof.goals)

                #  filter proof
                n_proof_steps = len(proof_steps)
                if n_proof_steps < self.min_proof_steps:
                    logging.debug(f"Naive proof with length {n_proof_steps}")
                    continue
                #  llm data generation
                aux_points = [p.name for p in aux_points]
                llm_renamed = self.llm_solution_renamed(problem_new, aux_points, solver_new.proof)

                if len(aux_points) > 0 and not self.filter.aux_predicates_valid_check(llm_renamed['llm_output']):
                    continue

                generated_data.append({
                    # "fl_statement_src": fl_statement,
                    "n_clauses": n_clauses,
                    "fl_problem": str(problem_new),
                    "nl_problem": "",
                    "n_proof_steps": n_proof_steps,
                    # "nl_solution": nl_solution,
                    "llm_input_renamed": llm_renamed['llm_input'],
                    "llm_output_renamed": llm_renamed['llm_output'],
                })

            summary = {
                'total_time': time.time() - start_time,
                'runtime': solver.run_infos['runtime'],
                'checkgoals_runtime': checkgoals_runtime,
                'n_samples': len(generated_data),
                'goals': [re.search(r'\?\s*(\w+)', d['fl_problem']).group(1) for d in generated_data],
                'first_predicate': [get_first_predicate(d['fl_problem']) for d in generated_data],
                'n_clauses': [d['n_clauses'] for d in generated_data],
                'n_proof_steps': [d['n_proof_steps'] for d in generated_data],
                'n_filtered_samples': n_filtered_samples,
            }
            return generated_data, summary
        except Exception as e:
            logging.info(f"Error generating problem: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def write_data(self, all_data: list, force: bool = False):
        """Append a single JSON object to a .jsonl file."""
        self.write_buffer.extend(all_data)
        if len(self.write_buffer) > 10000 or force:
            filename = self.path_prefix + ".jsonl"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'a', encoding='utf-8') as f:
                for data_item in self.write_buffer:
                    data_item['fl_problem'] = ''
                    json.dump(data_item, f, ensure_ascii=False)
                    f.write('\n')
            self.write_buffer.clear()

    def generate_problems(self): 
        import signal
        class TimeoutError(Exception):
            pass
        def handler(signum, frame):
            raise TimeoutError()
        def task_generator():
            for i in range(10**9):
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(10)
                try:
                    clauses = self.clauses_generator.generate(
                        np.clip(
                            np.random.binomial(n=self.n_clauses * 2, p=0.5), 
                            max(1, self.n_clauses - 10), 
                            self.n_clauses + 10
                        )
                    )
                except TimeoutError:
                    continue
                signal.alarm(0)
                yield i, clauses
        @ray.remote(num_cpus=1, max_retries=0)
        def ray_process_single_problem(args):
            return self.process_single_problem(args)
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=self.n_threads)
        task_iterator = task_generator()
        max_pending = int(self.n_threads * 1.5)
        summary_reporter = Summary(prefix = self.path_prefix)
        
        start_time = time.time()
        all_data_len = 0
        pending_tasks = {}
        while all_data_len < self.n_samples:
            done, _ = ray.wait(list(pending_tasks.keys()), num_returns=1, timeout=10)
            if not done:
                now = time.time()
                for task, s_time in list(pending_tasks.items()):
                    if now - s_time > self.timeout:
                        print(f"⚠️ Task {task} timeout. Canceling")
                        ray.cancel(task, force=True)
                        del pending_tasks[task]
            else:
                try:         
                    result = ray.get(done[0])
                    data, summary = result
                except Exception as e:
                    print(f"⚠️ Task {task} Error. {e}")
                    data, summary = [], {}
                del pending_tasks[done[0]]
            
                if data:
                    self.write_data(data)
                    all_data_len += len(data)
                    summary_reporter.add(summary)
                    elapsed_time = time.time() - start_time
                    logging.info(
                        f"Progress: [{all_data_len}/{self.n_samples}] ({len(data):4d} new) in {elapsed_time:.0f}s. "
                        f"Total: {summary['total_time']:2.0f}s. DDAR: {summary['runtime']:3.0f}s. Checkgoals: {summary['checkgoals_runtime']:2.0f}s. "
                        f"Speed: {all_data_len / (elapsed_time):2.0f} samples/s. "
                        f"ETA: {timedelta(seconds=int(self.n_samples/all_data_len*(elapsed_time)-elapsed_time))}"
                    )

            while len(pending_tasks) < max_pending:
                pending_tasks[ray_process_single_problem.remote(next(task_iterator))] = time.time()

        # Cancel any remaining tasks
        for task in pending_tasks.keys():
            ray.cancel(task, force=True)
        ray.shutdown()

        self.write_data([], force=True)
        final_elapsed_time = time.time() - start_time
        summary_reporter.total_elapsed_time = final_elapsed_time
        summary_reporter.total_samples_generated = all_data_len
        logging.info(f"Generated {all_data_len} samples successfully in {final_elapsed_time:.2f}s.")
        summary_reporter.output_report()

def main():
    parser = argparse.ArgumentParser(description="Create problem fl - nl dataset")
    parser.add_argument("--n_clauses", required=False, type=int, default=10)
    parser.add_argument("--min_proof_steps", required=False, type=int, default=3)
    parser.add_argument("--min_clauses_num", required=False, type=int, default=2)
    parser.add_argument("--n_threads", required=False, type=int, default=1)
    parser.add_argument("--n_samples", required=False, type=int, default=1000)
    parser.add_argument("--dir", required=False, default="dataset")
    parser.add_argument("--log_level", required=False, default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--timeout", required=False, type=int, default=3600)
    parser.add_argument("--filteration_rate", required=False, type=float, default=0.6)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    generator = GeometryGenerator(
        n_clauses=args.n_clauses,
        n_threads=args.n_threads,
        output_dir=args.dir,
        min_proof_steps=args.min_proof_steps,
        min_clauses_num=args.min_clauses_num,
        n_samples=args.n_samples,
        timeout=args.timeout,
        filteration_rate=args.filteration_rate
    )

    generator.generate_problems()

if __name__ == "__main__":
    main()

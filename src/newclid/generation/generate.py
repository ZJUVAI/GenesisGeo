import logging
import os
import re
import argparse
import json
import random
import time
from datetime import timedelta
from pathlib import Path
import itertools
from collections import defaultdict
import numpy as np
from millify import millify
import multiprocessing

from newclid.configs import default_defs_path, default_rules_path
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.rule import Rule
from newclid.dependencies.symbols import Point
from newclid.formulations.problem import ProblemJGEX
from newclid.statement import Statement
from newclid.api import GeometricSolverBuilder, GeometricSolver
from newclid.agent.ddarn import DDARN
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.generation.clause_generation import CompoundClauseGen
from newclid.proof import ProofState
from newclid.proof_writing import get_structured_proof, write_proof_steps
from newclid.formulations.clause import translate_sentence
from newclid.numerical import close_enough
from newclid.generation.output_summary import Summary, get_first_predicate


class GeometryGenerator: 
    def __init__(self, max_clauses=5, n_threads=1, output_dir="dataset", min_proof_steps=5, min_clauses_num=3, n_samples=100, timeout=3600, filteration_rate=0.6):
        self.max_clauses = max_clauses
        self.min_proof_steps = min_proof_steps
        self.min_clauses_num = min_clauses_num
        self.n_samples = n_samples
        self.n_threads = n_threads
        self.timeout = timeout
        self.output_dir = output_dir
        self.filteration_rate = filteration_rate
        self.path_prefix = os.path.join(self.output_dir, f"geometry_clauses{self.max_clauses}_samples{millify(self.n_samples)}")
        self.write_buffer = []
        self.clauses_generator = CompoundClauseGen(
            max_comma_sep_clause=2,
            max_single_clause=1,
            max_sets=self.max_clauses,
            seed=0,
            shuffle_var_names=False,
        ) 
    
    def all_possible_goals_by_goals(self, dep_graph: DependencyGraph):
        def load_predicates(rule_path: Path = default_rules_path()) -> set[tuple[str, int]]:
            predicates: set[tuple[str, int]] = set()
            rules = list(Rule.parse_txt_file(rule_path))

            for theorem in rules:
                for conclusion in theorem.conclusions:
                    if conclusion[0] in ['PythagoreanConclusions', 'rconst', 'aconst', 'eqratio3']:
                        continue
                    if conclusion[0] in ['eqangle', 'eqratio']:
                        continue
                    new_predicate = (conclusion[0], len(conclusion) - 1)
                    predicates.add(new_predicate)
            return predicates
        
        predicates = load_predicates()
        points_name = [p.name for p in dep_graph.symbols_graph.nodes_of_type(Point)]
        for name, num_args in predicates:
            for point_list in itertools.product(points_name, repeat=num_args):
                tokens = tuple([name] + list(point_list))
                if self.goal_filter(name, point_list):
                    goal = Statement.from_tokens(tokens, dep_graph)
                    if goal: goal.check()
    
    def get_numerical_checked_eqangle_and_eqratio(self, dep_graph: DependencyGraph) -> tuple[list[Statement], list[Statement]]:
        points = dep_graph.symbols_graph.nodes_of_type(Point)
        angles: list[tuple[float, str, str, str, str]] = list()
        ratios: list[tuple[float, str, str, str, str]] = list()

        for (i, a) in enumerate(points):
            for b in points[i + 1:]: 
                angle1 = (a.num - b.num).angle()
                dis = a.num.distance(b.num)
                for (k, c) in enumerate(points):
                    for d in points[k + 1:]: 
                        if a.name == c.name and b.name == d.name:
                            continue
                        angle = ((c.num - d.num).angle() - angle1) % np.pi
                        ratio = dis / c.num.distance(d.num)
                        angles.append((angle, a.name, b.name, c.name, d.name))
                        ratios.append((ratio, a.name, b.name, c.name, d.name))
                        ratios.append((1 / ratio, c.name, d.name, a.name, b.name))
        
        angles.sort(key=lambda x: x[0])
        ratios.sort(key=lambda x: x[0])
        for (i, A) in enumerate(angles):
            for B in angles[i + 1:]:
                if not close_enough(A[0], B[0]):
                    break
                if self.goal_filter('eqangle', A[1:] + B[1:]):
                    tokens = tuple(['eqangle'] + list(A[1:] + B[1:]))
                    goal = Statement.from_tokens(tokens, dep_graph)
                    if goal: goal.check()

        for (i, A) in enumerate(ratios):
            for B in ratios[i + 1:]:
                if not close_enough(A[0], B[0]):
                    break
                if self.goal_filter('eqratio', A[1:] + B[1:]):
                    tokens = tuple(['eqratio'] + list(A[1:] + B[1:]))
                    goal = Statement.from_tokens(tokens, dep_graph)
                    if goal: goal.check()

    def all_possible_goals_by_ar(self, dep_graph: DependencyGraph) -> list[Statement]:
        def extract_points(s):
            return re.findall(r'[a-z][\d]*', s)

        def goal_from_tokens(tokens):
            if self.goal_filter(tokens[0], tokens[1:], dep_graph):
                goal = Statement.from_tokens(tokens, dep_graph)
                if goal and goal.check():
                    return [goal]
            return []
        
        points_name = sorted([p.name for p in dep_graph.symbols_graph.nodes_of_type(Point)])
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
                    logging.warning(f"Error in goal_from_tokens: {e} para/perp for {v1}, {v2}")
                    continue
        for v1, v2, v3, v4 in e2v_pairs4:
            try:
                v1, v2, v3, v4 = extract_points(v1), extract_points(v2), extract_points(v3), extract_points(v4)
                goal_from_tokens(tuple(['eqangle'] + list(v1 + v2 + v3 + v4)))
            except Exception as e:
                logging.warning(f"Error in goal_from_tokens: {e} for eqangle {v1}, {v2}, {v3}, {v4}")
                continue

        e2v, e2v_pairs2, e2v_pairs4 = ar.rtable.possible_pairs()
        for e in e2v_pairs2.keys():
            for v1, v2 in e2v_pairs2[e]:
                try:
                    goal_from_tokens(tuple(['cong'] + v1[2:-1].split(',') + v2[2:-1].split(',')))
                except Exception as e:
                    logging.warning(f"Error in goal_from_tokens: {e} cong for {v1}, {v2}")
                    continue
        for v1, v2, v3, v4 in e2v_pairs4:
            try:
                tokens = tuple(['eqratio'] + list(v1[2:-1].split(',') + v2[2:-1].split(',') + v3[2:-1].split(',') + v4[2:-1].split(',')))
                goal_from_tokens(tokens)
            except Exception as e:
                logging.warning(f"Error in goal_from_tokens: {e} for eqratio {v1}, {v2}, {v3}, {v4}")
                continue

    def clauses_num_filter(self, problemJGEX: ProblemJGEX) -> bool:
        if len(problemJGEX.constructions) < self.min_clauses_num:
            logging.debug(f"Too few clauses: {len(problemJGEX.constructions)}")
            return False
        else:
            return True
    
    def proof_filter(self, solver: GeometricSolver, goal: Statement) -> bool:
        try:
            _, _, _, _, _, _, proof_steps, = solver.proof.dep_graph.get_proof_steps([goal])
            if len(proof_steps) < self.min_proof_steps:
                logging.debug(f"Naive proof: {goal}")
                return False
            else:
                return True
                # connot detect proof of the goal
                # rconst[b,c,c,e,Fraction(2, 1),]
        except Exception as e:
            logging.warning(f"error in get_proof_steps {goal}: {e}. Why?")
            return False

    def goal_filter(self, name, args, dep_graph):
        if args[-1] == '':
            args = args[:-1]
        # AG1 do not support aconst and rconst
        if name == 'aconst' or name == 'rconst': # rconst AB:AB=1, aconst ∠AB AB=0
            return False
        # case: cong AB = AB, 
        if name == 'cong': 
            left = {args[0], args[1]}
            right = {args[2], args[3]}
            if left == right:
                return False
        # para AB ∥∥ AB, AB ∥∥ AC
        if name == 'para':
            if len({args[0], args[1], args[2], args[3]}) < 4:
                return False
        if name == 'eqratio':
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
            # case: exist two segments with the same length
            arg_set1 = [args[0], args[1], args[2], args[3]]
            arg_set2 = [args[0], args[1], args[4], args[5]]
            arg_set3 = [args[2], args[3], args[6], args[7]]
            arg_set4 = [args[4], args[5], args[6], args[7]]
            sm11 = Statement.from_tokens(['cong']+ arg_set1, dep_graph)
            sm22 = Statement.from_tokens(['cong']+ arg_set2, dep_graph)
            sm33 = Statement.from_tokens(['cong']+ arg_set3, dep_graph)
            sm44 = Statement.from_tokens(['cong']+ arg_set4, dep_graph)
            if sm11.check() or sm22.check() or sm33.check() or sm44.check():
                return False
        if name == 'eqangle':
            #case: eqangle ∠AB CD = ∠DC/BA
            seg_1 = {args[0], args[1]}
            seg_2 = {args[2], args[3]}
            seg_3 = {args[4], args[5]}
            seg_4 = {args[6], args[7]}
            if seg_1 == seg_3 and seg_2 == seg_4:
                return False
            if seg_1 == seg_4 and seg_2 == seg_3:
                return False
            if seg_1 == seg_2 or seg_3 == seg_4:
                return False
            # case: two parallels or perp
            arg_set1 = [args[0], args[1], args[2], args[3]]
            arg_set2 = [args[4], args[5], args[6], args[7]]
            arg_set3 = [args[0], args[1], args[4], args[5]]
            arg_set4 = [args[0], args[1], args[6], args[7]]
            arg_set5 = [args[2], args[3], args[4], args[5]]
            arg_set6 = [args[2], args[3], args[6], args[7]]
            arg_set7 = [args[0], args[1], args[2], args[3]]
            arg_set8 = [args[4], args[5], args[6], args[7]]
            sm1 = Statement.from_tokens(['para']+ arg_set1, dep_graph)
            sm2 = Statement.from_tokens(['para']+ arg_set2, dep_graph)
            sm3 = Statement.from_tokens(['para']+ arg_set3, dep_graph)
            sm4 = Statement.from_tokens(['para']+ arg_set4, dep_graph)
            sm5 = Statement.from_tokens(['para']+ arg_set5, dep_graph)
            sm6 = Statement.from_tokens(['para']+ arg_set6, dep_graph)
            sm7 = Statement.from_tokens(['perp']+ arg_set7, dep_graph)
            sm8 = Statement.from_tokens(['perp']+ arg_set8, dep_graph)
            if sm1.check() or sm2.check() or sm3.check() or sm4.check() or sm5.check() or sm6.check() or sm7.check() or sm8.check():
                return False
            # case: simtri
            a1_args = list(set(args[:4]))
            a2_args = list(set(args[4:]))
            if len(a1_args) == 3 and len(a2_args) == 3:
                sm1 = Statement.from_tokens(['simtri']+a1_args+[a2_args[0], a2_args[1], a2_args[2]], dep_graph)
                sm2 = Statement.from_tokens(['simtri']+a1_args+[a2_args[0], a2_args[2], a2_args[1]], dep_graph)
                sm3 = Statement.from_tokens(['simtri']+a1_args+[a2_args[1], a2_args[0], a2_args[2]], dep_graph)
                sm4 = Statement.from_tokens(['simtri']+a1_args+[a2_args[1], a2_args[2], a2_args[0]], dep_graph)
                sm5 = Statement.from_tokens(['simtri']+a1_args+[a2_args[2], a2_args[0], a2_args[1]], dep_graph)
                sm6 = Statement.from_tokens(['simtri']+a1_args+[a2_args[2], a2_args[1], a2_args[0]], dep_graph)
                if sm1.check() or sm2.check() or sm3.check() or sm4.check() or sm5.check() or sm6.check():
                    return False
                sm1 = Statement.from_tokens(['simtrir']+a1_args+[a2_args[0], a2_args[1], a2_args[2]], dep_graph)
                sm2 = Statement.from_tokens(['simtrir']+a1_args+[a2_args[0], a2_args[2], a2_args[1]], dep_graph)
                sm3 = Statement.from_tokens(['simtrir']+a1_args+[a2_args[1], a2_args[0], a2_args[2]], dep_graph)
                sm4 = Statement.from_tokens(['simtrir']+a1_args+[a2_args[1], a2_args[2], a2_args[0]], dep_graph)
                sm5 = Statement.from_tokens(['simtrir']+a1_args+[a2_args[2], a2_args[0], a2_args[1]], dep_graph)
                sm6 = Statement.from_tokens(['simtrir']+a1_args+[a2_args[2], a2_args[1], a2_args[0]], dep_graph)
                if sm1.check() or sm2.check() or sm3.check() or sm4.check() or sm5.check() or sm6.check():
                    return False
        if name == 'simtri' or name == 'simtrir' or name == 'contri' or name == 'contrir':
            #case: simtri △ABC ≅ △ABC
            tri_1 = {args[0], args[1], args[2]}
            tri_2 = {args[3], args[4], args[5]}
            if tri_1 == tri_2:
                return False
        if name == 'sameclock':
            return False

        return True
    
    def eqangle_goals_filter(self, eqangle_goals, dep_graph):
        def eqangle_equiv(p1, p2):
            args1 = [arg.name for arg in p1.args]
            args2 = [arg.name for arg in p2.args]
            sm1 = Statement.from_tokens(['para', args1[0], args1[1], args2[0], args2[1]], dep_graph)
            sm2 = Statement.from_tokens(['para', args1[2], args1[3], args2[2], args2[3]], dep_graph)
            sm3 = Statement.from_tokens(['para', args1[4], args1[5], args2[4], args2[5]], dep_graph)
            sm4 = Statement.from_tokens(['para', args1[6], args1[7], args2[6], args2[7]], dep_graph)
            if sm1.check() and sm2.check() and sm3.check() and sm4.check():
                return True
        res = []
        for eqangle_goal in eqangle_goals:
            exist = False
            for p_goals in res:
                if eqangle_equiv(p_goals, eqangle_goal):
                    exist = True
                    break
            if not exist:
                res.append(eqangle_goal)
        return res
    
    def eqratio_goals_filter(self, eqratio_goals, dep_graph):
        def eqratio_equiv(p1, p2):
            args1 = [arg.name for arg in p1.args]
            args2 = [arg.name for arg in p2.args]
            sm1 = Statement.from_tokens(['cong', args1[0], args1[1], args2[0], args2[1]], dep_graph)
            sm2 = Statement.from_tokens(['cong', args1[2], args1[3], args2[2], args2[3]], dep_graph)
            sm3 = Statement.from_tokens(['cong', args1[4], args1[5], args2[4], args2[5]], dep_graph)
            sm4 = Statement.from_tokens(['cong', args1[6], args1[7], args2[6], args2[7]], dep_graph)
            if sm1.check() and sm2.check() and sm3.check() and sm4.check():
                return True
        res = []
        for eqratio_goal in eqratio_goals:
            exist = False
            for p_goals in res:
                if eqratio_equiv(p_goals, eqratio_goal):
                    exist = True
                    break
            if not exist:
                res.append(eqratio_goal)
        # print(f'{len(res)} / {len(eqratio_goals)}')
        return res
    
    def llm_solution(self, problem: ProblemJGEX, aux_points: list[str], proof_state: ProofState) -> str:
        dep_idx: dict[Statement, str] = {}
        defs = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))
        
        data_tmp = defaultdict(list)
        for construction in problem.constructions:
            group = {}
            p2deps = defaultdict(list)
            for constr_sentence in construction.sentences:
                cdef = defs[constr_sentence[0]]
                if len(constr_sentence) == len(cdef.declare):
                    mapping = dict(zip(cdef.declare[1:], constr_sentence[1:]))
                else:
                    assert len(constr_sentence) + len(construction.points) == len(cdef.declare)
                    mapping = dict(zip(cdef.declare[1:], construction.points + constr_sentence[1:]))
                for points, bs in cdef.basics:
                    points = tuple([mapping[x] for x in points])
                    for p in points:
                        group[p] = points
                    for b in bs:
                        statement = Statement.from_tokens(translate_sentence(mapping, b), proof_state.dep_graph)
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
        data_analysis, data_numerical_check, data_proof = get_structured_proof(proof_state, dep_idx)
        
        # <numerical_check> </numerical_check>
        if data_numerical_check != '':
            data_numerical_check += ' '

        return {
            "llm_data": data_problem + ' ' + data_aux + data_numerical_check + data_proof,
            "llm_input": data_problem,
            # "llm_output": data_aux + data_analysis + ' ' + data_numerical_check + data_proof,
            "llm_output": data_aux + data_numerical_check + data_proof,
        }

    def llm_nat_solution(self, problem: ProblemJGEX, aux_points: list[str], proof_state: ProofState) -> str:
        defs = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))
        data_tmp = defaultdict(list)
        for construction in problem.constructions:
            group = {}
            p2deps = defaultdict(list)
            for constr_sentence in construction.sentences:
                cdef = defs[constr_sentence[0]]
                if len(constr_sentence) == len(cdef.declare):
                    mapping = dict(zip(cdef.declare[1:], constr_sentence[1:]))
                else:
                    assert len(constr_sentence) + len(construction.points) == len(cdef.declare)
                    mapping = dict(zip(cdef.declare[1:], construction.points + constr_sentence[1:]))
                for points, bs in cdef.basics:
                    points = tuple([mapping[x] for x in points])
                    for p in points:
                        group[p] = points
                    for b in bs:
                        statement = Statement.from_tokens(translate_sentence(mapping, b), proof_state.dep_graph)
                        p2deps[points].append(statement)
            points = construction.points
            while points:
                p = points[0]
                gr = group[p]
                points = [x for x in points if x not in gr]
                data_tmp[' '.join(gr)] = p2deps[gr]

        # <problem_nl> </problem_nl>
        data = '* Problem\n'
        string_premise_nl = []
        for k, v in data_tmp.items():
            if not all(p in aux_points for p in k.split(' ')):
                tmp_string_nl = ' '.join(dep.pretty() for dep in v)
                if tmp_string_nl.strip():
                    string_premise_nl.append(tmp_string_nl)
        data += '; '.join([s.strip() for s in string_premise_nl]) + ' ? '

        goal_statements = [
            Statement.from_tokens(goal, proof_state.dep_graph)
            for goal in problem.goals
            if Statement.from_tokens(goal, proof_state.dep_graph)
        ]
        data += '; '.join([goal.pretty() for goal in goal_statements])
        data += '\n'

        data += write_proof_steps(proof_state, print_output=False)
        return data
    
    @staticmethod
    def proofs_similarity(proof_lines1: tuple[str], proof_lines2: tuple[str]) -> float:
        """Calculate the similarity between two proofs based on their lines."""
        set1 = set(proof_lines1)
        set2 = set(proof_lines2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 0.0
        # print(f"Intersection: {intersection}, Union: {union}")
        return len(intersection) / len(union)

    @staticmethod
    def get_rules_and_lines_from_output(llm_output: str) -> tuple[str, tuple[str]]:
        """Extract rules from the proof string."""
        proof = llm_output.split('<proof>')[1].split('</proof>')[0].strip()
        lines = [line.strip() for line in proof.split(';')][:-1]
        rules = [line.split(']')[1].strip().split()[0] for line in lines]

        # print(f"Proof lines: {lines}")
        # print(f"Extracted rules: {rules}")
        
        return ('_'.join(rules), tuple(lines))
    
    def similarity_check(self, llm_output: str, proofs_of_used_rules: dict[str, list[tuple]]) -> bool:
        """Check if the LLM output is similar to any of the existing proofs."""
        rules, proof_lines = GeometryGenerator.get_rules_and_lines_from_output(llm_output)
        for proofs in proofs_of_used_rules.get(rules, []):
            similarity = GeometryGenerator.proofs_similarity(proof_lines, proofs)
            # print(f"Similarity between {proof_lines} and {proofs}: {similarity:.2f}")
            if similarity > self.filteration_rate:
                return True
                
        if rules not in proofs_of_used_rules:
            proofs_of_used_rules[rules] = []
        proofs_of_used_rules[rules].append(proof_lines)
        return False
    
    def process_single_problem(self, args: tuple) -> tuple[list, dict]:
        try:
            """Process a single geometry problem."""
            pid, fl_statement = args
            # fl_statement = "a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c"
            
            solver_builder = GeometricSolverBuilder(seed=998244353)
            solver_builder.with_deductive_agent(DDARN())
            solver_builder.load_problem_from_txt(fl_statement)
            try:
                solver = solver_builder.build(max_attempts=100)
            except Exception as e:
                logging.debug(f"Error: {e}")
                return [], {}
            solver.run(timeout=self.timeout)
            logging.info(f"ddar time: {solver.run_infos['runtime']}s")

            t = time.time()
            # self.all_possible_goals_by_goals(solver.proof.dep_graph)
            # self.get_numerical_checked_eqangle_and_eqratio(solver.proof.dep_graph)
            self.all_possible_goals_by_ar(solver.proof.dep_graph)
            possible_goals = [goal for goal in solver.proof.dep_graph.conclusions() if self.goal_filter(goal.predicate.NAME, [arg.name if hasattr(arg, "name") else arg for arg in goal.args], solver.proof.dep_graph)]
            possible_goals_others = [goal for goal in possible_goals if goal.predicate.NAME != 'eqangle' and goal.predicate.NAME != 'eqratio'] 
            possible_goals_eqangle = self.eqangle_goals_filter([goal for goal in possible_goals if goal.predicate.NAME == 'eqangle'], solver.proof.dep_graph) 
            possible_goals_eqratio = self.eqratio_goals_filter([goal for goal in possible_goals if goal.predicate.NAME == 'eqratio'], solver.proof.dep_graph)
            possible_goals = possible_goals_others + possible_goals_eqangle + possible_goals_eqratio
            logging.info(f"check goals time: {time.time() - t:.2f}s")
            logging.info(f"{len(possible_goals)=}")

            n_filtered_samples = 0
            proofs_of_used_rules = {}
            generated_data = []
            for goal in possible_goals:
                # essential fl_problem
                essential_clauses, essential_aux_clauses = solver.proof.dep_graph.get_essential_clauses([goal])
                statements = []
                for clause in solver_builder.problemJGEX.constructions:
                    if str(clause) in essential_clauses or str(clause) in essential_aux_clauses:
                        statements.append(str(clause))
                fl_problem = '; '.join(statements) + ' ? ' + goal.predicate.NAME + ' ' + ' '.join([arg.name for arg in goal.args])
                fl_problem = ProblemJGEX.from_text(fl_problem)

                # cluases num filter
                n_clauses = len(essential_clauses | essential_aux_clauses)
                if n_clauses < self.min_clauses_num:
                    logging.debug(f"Too few clauses: {len(essential_clauses | essential_aux_clauses)}")
                    continue

                # get and filter proof
                _, _, _, aux_points, _, _, proof_steps, = solver.proof.dep_graph.get_proof_steps([goal])
                n_proof_steps = len(proof_steps)
                if n_proof_steps < self.min_proof_steps:
                    logging.debug(f"Naive proof with length {n_proof_steps}")
                    continue
                
                # solution
                solver.proof.goals = [goal]
                aux_points = [p.name for p in aux_points]
                nl_solution = write_proof_steps(solver.proof, print_output=False)
                llm = self.llm_solution(fl_problem, aux_points, solver.proof)
                # llm_nat_solution = self.llm_nat_solution(fl_problem, aux_points, solver.proof)

                # check similarity
                # if self.similarity_check(llm['llm_output'], proofs_of_used_rules):
                #     logging.debug(f"Similar proof found for {goal.predicate.NAME} with clauses {n_clauses} and proof steps {n_proof_steps}. Skipping.")
                #     n_filtered_samples += 1
                #     continue
                
                generated_data.append({
                    "fl_statement_src": fl_statement,
                    "n_clauses": n_clauses,
                    "fl_problem": str(fl_problem),
                    "nl_problem": "",
                    "n_proof_steps": n_proof_steps,
                    # "nl_solution": nl_solution,
                    # "llm_data": llm['llm_data'],
                    "llm_input": llm['llm_input'],
                    "llm_output": llm['llm_output'],
                    # "llm_nat_solution": llm_nat_solution,
                })
            summary = {
                'runtime': solver.run_infos['runtime'],
                'n_samples': len(generated_data),
                'goals': [re.search(r'\?\s*(\w+)', d['fl_problem']).group(1) for d in generated_data],
                'first_predicate': [get_first_predicate(d['fl_problem']) for d in generated_data],
                'n_clauses': [d['n_clauses'] for d in generated_data],
                'n_proof_steps': [d['n_proof_steps'] for d in generated_data],
                'n_filtered_samples': n_filtered_samples
            }

            return generated_data, summary
        except Exception as e:
            logging.info(f"Error generating problem: {e}")
            return [], {}

    def generate_problems(self):
        """Generate geometry problems one at a time using a generator."""
        def task_generator():
            for i in range(10**9):
                clauses = self.clauses_generator.generate_clauses()
                yield (i, clauses)
        task_iterator = task_generator()
        
        all_data_len = 0
        summary_reporter = Summary(prefix=self.path_prefix)
        start_time = time.time()
        if self.n_threads == 1:
            while True:
                data, summary = self.process_single_problem(next(task_iterator))
                if data:
                    self.write_data(data)
                    all_data_len += len(data)
                    summary_reporter.add(summary)
                    elapsed_time = time.time() - start_time
                    logging.info(
                        f"Progress: [{all_data_len}/{self.n_samples}] ({len(data)} new) in {elapsed_time:.1f}s. "
                        f"Speed: {(elapsed_time)/all_data_len:.1f}s/sample. "
                        f"ETA: {timedelta(seconds=int(self.n_samples/all_data_len*(elapsed_time)-elapsed_time))}"
                    )
                if all_data_len >= self.n_samples:
                    break
        else:
            try:
                with multiprocessing.Pool(self.n_threads) as pool:
                    for data, summary in pool.imap_unordered(self.process_single_problem, task_generator()):
                        if data:
                            self.write_data(data)
                            all_data_len += len(data)
                            summary_reporter.add(summary)
                            elapsed_time = time.time() - start_time
                            logging.info(
                                f"Progress: [{all_data_len}/{self.n_samples}] ({len(data)} new) in {elapsed_time:.1f}s. "
                                f"Speed: {all_data_len / (elapsed_time):.1f} samples/s. "
                                f"ETA: {timedelta(seconds=int(self.n_samples/all_data_len*(elapsed_time)-elapsed_time))}"
                            )
                        if all_data_len >= self.n_samples:
                            pool.terminate() 
                            break
                    pool.close()
                    pool.join()
            except Exception as e:
                logging.error(f"multiprocessing Pool error: {e}")

        self.write_data([], force=True)
        final_elapsed_time = time.time() - start_time
        summary_reporter.total_elapsed_time = final_elapsed_time
        summary_reporter.total_samples_generated = all_data_len
        logging.info(f"Generated {all_data_len} samples successfully in {final_elapsed_time:.2f}s.")
        summary_reporter.output_report()

    def write_data(self, all_data: list, force: bool = False):
        """Append a single JSON object to a .jsonl file."""
        self.write_buffer.extend(all_data)
        if len(self.write_buffer) > 10000 or force:
            filename = self.path_prefix + ".jsonl"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'a', encoding='utf-8') as f:
                for data_item in self.write_buffer:
                    json.dump(data_item, f, ensure_ascii=False)
                    f.write('\n')
            self.write_buffer.clear()

def main():
    parser = argparse.ArgumentParser(description="Create problem fl - nl dataset")
    parser.add_argument("--max_clauses", required=False, type=int, default=5)
    parser.add_argument("--min_proof_steps", required=False, type=int, default=3)
    parser.add_argument("--min_clauses_num", required=False, type=int, default=3)
    parser.add_argument("--n_threads", required=False, type=int, default=1)
    parser.add_argument("--n_samples", required=False, type=int, default=100)
    parser.add_argument("--dir", required=False, default="dataset")
    parser.add_argument("--log_level", required=False, default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--timeout", required=False, type=int, default=3600)
    parser.add_argument("--filteration_rate", required=False, type=float, default=0.6)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    generator = GeometryGenerator(
        max_clauses=args.max_clauses,
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
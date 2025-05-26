import multiprocessing
import logging
import os
import argparse
import csv
import random
import time
from typing import Iterator, Dict, Any
from pathlib import Path
import itertools
from collections import defaultdict
import numpy as np
import signal

from newclid.agent.ddarn import DDARN
from newclid.api import GeometricSolverBuilder, GeometricSolver
from newclid.configs import default_defs_path, default_rules_path
from newclid.dependencies.symbols import Point
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.generation.clause_generation import CompoundClauseGen
from newclid.proof_writing import return_proof_steps, get_structured_proof
from newclid.statement import Statement
from newclid.formulations.rule import Rule
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.problem import ProblemJGEX
from newclid.formulations.clause import translate_sentence
from newclid.predicates import NAME_TO_PREDICATE
from newclid.numerical import close_enough
from newclid.proof import ProofState

def handler(signum, frame):
    raise TimeoutError("Function execution timed out")

signal.signal(signal.SIGALRM, handler)

class GeometryGenerator: 
    def __init__(self, max_clauses=5, search_depth=5, n_threads=1, output_dir="dataset", min_proof_steps=5, min_clauses_num=3, n_samples=100, time_limit=3600):
        self.max_clauses = max_clauses
        self.search_depth = search_depth
        self.n_threads = n_threads
        self.output_dir = output_dir
        self.min_proof_steps = min_proof_steps
        self.min_clauses_num = min_clauses_num
        self.n_samples = n_samples
        self.time_limit = time_limit
        self.samples_per_thread = []
        self.ddar_times = []

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
        def goal_from_tokens(tokens):
            if self.goal_filter(tokens[0], tokens[1:]):
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
                goal_from_tokens(tuple(['para'] + list(v1 + v2)))
                goal_from_tokens(tuple(['perp'] + list(v1 + v2)))
        for v1, v2, v3, v4 in e2v_pairs4:
            goal_from_tokens(tuple(['eqangle'] + list(v1 + v2 + v3 + v4)))

        e2v, e2v_pairs2, e2v_pairs4 = ar.rtable.possible_pairs()
        for e in e2v_pairs2.keys():
            for v1, v2 in e2v_pairs2[e]:
                goal_from_tokens(tuple(['cong'] + v1[2:-1].split(',') + v2[2:-1].split(',')))
        for v1, v2, v3, v4 in e2v_pairs4:
            tokens = tuple(['eqratio'] + list(v1[2:-1].split(',') + v2[2:-1].split(',') + v3[2:-1].split(',') + v4[2:-1].split(',')))
            goal_from_tokens(tokens)

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

    def goal_filter(self, name, args):
        if args[-1] == '':
            args = args[:-1]
        # AG1 do not support aconst and rconst
        if name == 'aconst' or name == 'rconst':
            return False
        # case: cong AB = AB, para AB ∥∥ AB, rconst AB:AB=1, aconst ∠AB AB=0
        if name == 'cong' or name == 'para': # or predicate == 'rconst' or predicate == 'aconst':
            left = {args[0], args[1]}
            right = {args[2], args[3]}
            if left == right:
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
        if name == 'simtri':
            #case: simtri △ABC ≅ △ABC
            tri_1 = {args[0], args[1], args[2]}
            tri_2 = {args[3], args[4], args[5]}
            if tri_1 == tri_2:
                return False
        if name == 'sameclock':
            return False

        return True
 
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
        data = '\n<problem>\n'
        string_premise = []
        for k, v in data_tmp.items():
            if not all(p in aux_points for p in k.split(' ')):
                tmp_string = k + ': '
                for dep in v:
                    if dep not in dep_idx:
                        dep_idx[dep] = f"{len(dep_idx):03d}"
                    tmp_string += dep.to_str() + f' [{dep_idx[dep]}] '
                string_premise.append(tmp_string)
        data += '; '.join([s.strip() for s in string_premise]) + ' ? '
        data += ';'.join([
            (goal[0] + ' ' + ' '.join(goal[1:])) 
            for goal in problem.goals
            ])
        data += '\n</problem>\n'

        # <aux> </aux>
        string_aux = []
        for k, v in data_tmp.items():
            if all(p in aux_points for p in k.split(' ')):
                tmp_string = k + ': '
                for dep in v:
                    if dep not in dep_idx:
                        dep_idx[dep] = f"{len(dep_idx):03d}"
                    tmp_string += dep.to_str() + f' [{dep_idx[dep]}] '
                string_aux.append(tmp_string)
        if len(string_aux) > 0:
            data += '<aux>\n'
            data += '\n'.join([s.strip() for s in string_aux])
            data += '\n</aux>\n'

        # get analysis and proof
        analysis, numerical_check, proof = get_structured_proof(proof_state, dep_idx)
        
        # <analysis> </analysis>
        data += analysis

        # <numerical_check> </numerical_check>
        data += numerical_check        
        
        # <proof> </proof>
        data += proof
        return data

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

        data += return_proof_steps(proof_state)
        return data

    def process_single_problem(self, args: tuple) -> tuple[list, float]:
        """Process a single geometry problem."""
        pid, fl_statement = args
        
        solver_builder = GeometricSolverBuilder(seed=998244353)
        solver_builder.with_deductive_agent(DDARN())
        solver_builder.load_problem_from_txt(fl_statement)

        if not self.clauses_num_filter(solver_builder.problemJGEX):
            return [], 0.0
        
        try:
            solver = solver_builder.build(max_attempts=100)
        except Exception as e:
            logging.debug(f"Error: {e}")
            return [], 0.0

        t = time.time()        
        try:
            signal.alarm(self.time_limit*2)
            #set time limit for the solver, when the solver meets time out will stop run.
            solver.run(max_level=self.search_depth, time_limit=self.time_limit)
            ddar_time = time.time() - t
            signal.alarm(0)
        except Exception as e:
            logging.info(f"Problem couldn't be solved. {e}.")
            return [], 0.0
        logging.info(f"ddar time: {time.time() - t:.2f}s")

        t = time.time()
        # self.all_possible_goals_by_goals(solver.proof.dep_graph)
        # self.get_numerical_checked_eqangle_and_eqratio(solver.proof.dep_graph)
        self.all_possible_goals_by_ar(solver.proof.dep_graph)
        possible_goals = [goal for goal in solver.proof.dep_graph.conclusions() if self.goal_filter(goal.predicate.NAME, goal.args)]
        logging.info(f"check goals time: {time.time() - t:.2f}s")
        logging.info(f"{len(possible_goals)=}")

        t = time.time()
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
            llm_solution = self.llm_solution(fl_problem, aux_points, solver.proof)
            llm_nat_solution = self.llm_nat_solution(fl_problem, aux_points, solver.proof)

            # output
            generated_data.append({
                "n_clauses": n_clauses,
                "fl_problem": fl_problem,
                "nl_problem": "",
                "n_proof_steps": n_proof_steps,
                "llm_solution": llm_solution,
                "llm_nat_solution": llm_nat_solution,
            })
        logging.info(f"problem essential_clauses search time: {time.time() - t:.2f}s")
        return generated_data, ddar_time

    def generate_problems(self):# -> Iterator[Dict[str, Any]]:
        """Generate geometry problems one at a time using a generator."""
        def task_generator():
            for i in range(10**9):
                clauses = self.clauses_generator.generate_clauses()
                yield (i, clauses)

        data = []
        start_time = time.time()

        if self.n_threads == 1:
            task_iterator = task_generator()
            while True:
                results = self.process_single_problem(next(task_iterator))
                result, ddar_time = results
                if result:
                    data += result
                    self.ddar_times.append(ddar_time)
                    self.samples_per_thread.append(len(result))
                    logging.info(
                        f"Generated {len(data)} samples ({len(result)} new) in {time.time() - start_time:.1f}s "
                        f"({(time.time() - start_time)/len(data):.1f}s/sample)")
                if len(data) >= self.n_samples:
                    break
        else:
            try:
                with multiprocessing.Pool(self.n_threads) as pool:
                    for results in pool.imap_unordered(self.process_single_problem, task_generator()):
                        result, ddar_time = results
                        if result:
                            data += result
                            self.ddar_times.append(ddar_time)
                            self.samples_per_thread.append(len(result))
                            logging.info(
                                f"Generated {len(data)} samples ({len(result)} new) in {time.time() - start_time:.1f}s "
                                f"({(time.time() - start_time)/len(data):.1f}s/sample)")
                        if len(data) >= self.n_samples:
                            pool.terminate() 
                            break
                pool.close()
                pool.join()
            except Exception as e:
                logging.error(f"multiprocessing Pool error: {e}")
        
        self.write_data(data)
        self.write_data_summary()
        logging.info(f"Generated {len(data)} samples successfully")

    def write_data(self, all_data: list) -> int:
        """Write all generated data to output files."""
        filename = os.path.join(self.output_dir, f"geometry_clauses_{self.max_clauses}_depth{self.search_depth}_raw_samples{self.n_samples}.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        nl_filename = os.path.join(self.output_dir, f"geometry_clauses_{self.max_clauses}_depth{self.search_depth}_nl_samples{self.n_samples}.txt")

        with open(filename, "w", newline="", encoding="utf-8") as csvfile, \
            open(nl_filename, "w", encoding="utf-8") as nlf:
            field_names = [
                "id",
                "n_clauses",
                "fl_problem",
                "nl_problem",
                # "nl_solution",
                "n_proof_steps",
                # "fl_problem_renamed",
                # "dsl_problem_renamed",
                "llm_solution",
                # "llm_nat_solution",
            ]
            writer = csv.DictWriter(
                csvfile, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL, quotechar='"'
            )
            writer.writeheader()

            for i, row in enumerate(all_data):
                row["id"] = i
                if "llm_nat_solution" in row:
                    nlf.write(f"=== Sample {i} ===\n")
                    nlf.write(row["llm_nat_solution"])
                    nlf.write("\n\n")
                    # 写入csv前去除llm_nat_solution字段
                    row = dict(row)
                    row.pop("llm_nat_solution", None)
                writer.writerow(row)

        return len(all_data)

    def write_data_summary(self):
        """Write all generated data to output files."""
        filename = os.path.join(self.output_dir, f"geometry_clauses{self.max_clauses}_depth{self.search_depth}_ddar_times.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            field_names = [
                "id",
                "ddar_time",
                "samples_per_thread",
            ]
            writer = csv.DictWriter(
                csvfile, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL, quotechar='"'
            )
            writer.writeheader()

            for i, time in enumerate(self.ddar_times):
                writer.writerow({"id": i,
                                 "ddar_time": f"{time:.2f}",
                                 "samples_per_thread": self.samples_per_thread[i]}
                                )


def main():
    parser = argparse.ArgumentParser(description="Create problem fl - nl dataset")
    parser.add_argument("--max_clauses", required=False, type=int, default=10)
    parser.add_argument("--search_depth", required=False, type=int, default=1000)
    parser.add_argument("--min_proof_steps", required=False, type=int, default=2)
    parser.add_argument("--min_clauses_num", required=False, type=int, default=2)
    parser.add_argument("--n_threads", required=False, type=int, default=1)
    parser.add_argument("--n_samples", required=False, type=int, default=100)
    parser.add_argument("--dir", required=False, default="dataset")
    parser.add_argument("--log_level", required=False, default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--time_limit", required=False, type=int, default=3600)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    generator = GeometryGenerator(
        max_clauses=args.max_clauses,
        search_depth=args.search_depth,
        n_threads=args.n_threads,
        output_dir=args.dir,
        min_proof_steps=args.min_proof_steps,
        min_clauses_num=args.min_clauses_num,
        n_samples=args.n_samples,
        time_limit=args.time_limit,
    )
    
    generator.generate_problems()

if __name__ == "__main__":
    main()
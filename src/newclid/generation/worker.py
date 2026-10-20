from newclid.agent.ddarn import DDARN
from newclid.algebraic_reasoning.algebraic_manipulator import AlgebraicManipulator
from newclid.generation.filter import GoalFilter
from newclid.generation.statistics import get_first_predicate
from newclid.proof import ProofState
from newclid.statement import Statement
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.clause import Clause, translate_sentence
from newclid.dependencies.symbols import Point, Symbol
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.dependencies.dependency import Dependency, IN_PREMISES, NUMERICAL_CHECK
from newclid.configs import default_defs_path
from newclid.api import GeometricSolver, GeometricSolverBuilder, CSolver
import logging
import re
import string
import time
import signal
from contextlib import contextmanager
from collections import defaultdict
import ray
import numpy as np

from newclid.generation.sampler import ProblemSampler

import json
import os

logger = logging.getLogger(__name__)

# --- 反向重构失败埋点（诊断用，默认关闭） ---
# 设置环境变量 NEWCLID_RECON_FAIL_LOG=<目录> 后，每个 worker 进程会把
# rejected/solved_without_aux/unsolved_with_aux 三种丢弃结果的结构化 key
# 追加写入 <目录>/recon_fail_<pid>.jsonl（每个进程独立文件，避免多进程写
# 同一文件的锁竞争）。只加一次字符串拼接 + 文件 append，不改变任何求解
# 逻辑，对现有生产跑无影响（未设置该环境变量时是纯 no-op）。
_RECON_FAIL_LOG_DIR = os.environ.get("NEWCLID_RECON_FAIL_LOG")
_recon_fail_fp = None
if _RECON_FAIL_LOG_DIR:
    os.makedirs(_RECON_FAIL_LOG_DIR, exist_ok=True)
    _recon_fail_fp = open(
        os.path.join(_RECON_FAIL_LOG_DIR, f"recon_fail_{os.getpid()}.jsonl"),
        "a",
        buffering=1,
    )


def _log_recon_fail(outcome: str, construction_name: str, removed_pred: str,
                     kept_pred: str, goal_pred: str) -> None:
    """记录一次 reconstruction 失败/丢弃的结构化 key（不含具体点名/坐标）。"""
    if _recon_fail_fp is None:
        return
    try:
        _recon_fail_fp.write(json.dumps({
            "outcome": outcome,
            "construction": construction_name,
            "removed_pred": removed_pred,
            "kept_pred": kept_pred,
            "goal_pred": goal_pred,
        }, ensure_ascii=False) + "\n")
    except Exception:
        pass


# eqangle 类型的可逆构造很多（angle_bisector/external_bisector/angle_mirror/
# eqangle2/on_aline/eqangle3/iso_triangle_vertex_angle/on_aline0/...），其中
# on_aline 派生的 eqangle 目标数量巨大，会让反向重构大部分时间都花在 eqangle
# 上。只保留角平分线（含外角平分线）和角的 mirror 这两种，其余 eqangle 构造
# 类型不再作为反向重构候选。
EQANGLE_RECONSTRUCTIBLE_CONSTRUCTIONS = {
    "angle_bisector",
    "external_bisector",
    "angle_mirror",
}


def _build_reversible_constructions(
    defs: dict[str, DefinitionJGEX],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """
    映射：构造名 -> (谓词名, 谓词参数模板)。

    只保留恰好一个 basics Clause、Clause 里恰好一条句子、且句子参数正好等于
    {cdef.args} | {cdef.points} 的构造类型——即"一点一谓词"类型（新点不带
    premise-only / rely-only 参数）。谓词为 eqangle 的构造再按
    EQANGLE_RECONSTRUCTIBLE_CONSTRUCTIONS 收窄。
    """
    result = {}
    for name, cdef in defs.items():
        if len(cdef.basics) != 1:
            continue
        basic_clause = cdef.basics[0]
        if len(basic_clause.sentences) != 1:
            continue
        sentence = basic_clause.sentences[0]
        sentence_args = set(sentence[1:])
        declared = set(cdef.args) | set(cdef.points)
        if sentence_args != declared:
            continue
        if sentence[0] == "eqangle" and name not in EQANGLE_RECONSTRUCTIBLE_CONSTRUCTIONS:
            continue
        result[name] = (sentence[0], sentence)
    return result


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutError("Timed out")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ProblemWorker:
    """
    Worker class to process individual geometry problems.
    This class is designed to be used with Ray for parallel processing.
    """

    filter = GoalFilter()
    defs = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))
    REVERSIBLE_CONSTRUCTIONS: dict[str, tuple[str, tuple[str, ...]]] = (
        _build_reversible_constructions(defs)
    )

    # --- 反向重构功能开关 ---
    ENABLE_REVERSE_RECONSTRUCTION = True
    # 不限制总次数/deadline：每个 possible_goal 都必须至少被尝试一次，不能
    # 因为某种谓词类型（比如 on_aline 派生出的一大批 eqangle）数量多就把
    # 后面的目标类型饿死。闭包截断已经让单次尝试足够轻量。
    # 只考虑点数不超过这个值的目标作为重构候选（匹配"一点一谓词"构造的元数范围，
    # 让匹配过程保持轻量）。
    REVERSE_RECON_MAX_GOAL_POINTS = 8
    # 候选题目没有坐标信息，数值坐标搜索可能需要重试；使用比默认 1 更宽松的
    # max_attempts，但远小于 remove_coords 场景下的 10，因为这些候选题目
    # 通常结构简单、约束不多。
    REVERSE_RECON_BUILD_MAX_ATTEMPTS = 5
    # 每题最多尝试的重构候选数：候选先打乱（按 seed），再取前 N 个，避免
    # possible_goals 很大时（如 on_aline 派生出的大量 eqangle）尝试次数暴涨。
    REVERSE_RECON_MAX_ATTEMPTS_PER_PROBLEM = 20

    @ray.remote(num_cpus=1, max_retries=0)
    def ray_process_single_problem(args):
        try:
            return ProblemWorker._process_single_problem(args)
        except MemoryError as e:
            logging.error(f"⚠️ Worker OOM killed: {e}")
            return [], {"error": "oom"}
        except Exception as e:
            logging.error(f"Worker error: {e}")
            return [], {"error": str(e)}

    @staticmethod
    def _process_single_problem(args: tuple) -> tuple[list, dict]:
        """Process a single geometry problem with unique seed."""
        try:
            (
                pid,
                seed,
                n_clauses,
                max_level,
                using_log,
                using_exp,
                img,
                aux_only,
                add_auxiliary,
                max_auxiliary_points,
                prune,
                remove_coords,
                construction_config,
                fl_statement,
                *optional,
            ) = args
            # Trailing fields are optional: `custom_rules` was added after the
            # tuple was introduced and `triple_symmetry_enabled` after that, so
            # older callers pass a shorter tuple. Triple symmetry is on by
            # default, so a caller that omits the field still gets it.
            custom_rules = optional[0] if len(optional) > 0 else None
            triple_symmetry_enabled = bool(optional[1]) if len(optional) > 1 else True
            start_time = time.time()

            TIMELIMIT = 180  # 30 seconds per problem
            DEADLINE = start_time + TIMELIMIT

            # geneate fl_statement
            generation_start = time.time()
            sampling_timings = None
            if fl_statement is None:
                clauses_generator = ProblemSampler(
                    seed=seed,
                    construction_config=construction_config,
                )
                try:
                    with time_limit(10):
                        fl_statement, sampling_timings = clauses_generator.generate(
                            length=n_clauses,
                            add_auxiliary=add_auxiliary,
                            max_auxiliary_points=max_auxiliary_points,
                            prune=prune,
                            with_coords=not remove_coords,
                            return_timings=True,
                            triple_symmetry_enabled=triple_symmetry_enabled,
                        )
                except TimeoutError:
                    return [], {}
            generation_time = time.time() - generation_start

            # Build solver
            build_solver_start = time.time()
            solver, solver_builder = ProblemWorker._build_solver(
                fl_statement,
                max_attempts=10 if remove_coords else 1,
            )
            build_solver_time = time.time() - build_solver_start
            if not solver:
                return [], {}

            n_clauses = len(fl_statement.split(";"))
            csolver = CSolver(
                fl_statement,
                seed=seed,
                solver=solver,
                using_log=using_log,
                using_exp=using_exp,
                # `light=True` makes CSolver ignore the solver we just built
                # and re-derive its own numeric input via `build_ddar_input`,
                # so DDAR's closure is never merged into
                # `solver.proof.dep_graph.hyper_graph`. `_generate_possible_goals`
                # reads possible goals off that graph, so with light mode this
                # worker silently yields zero samples for every problem. Same
                # requirement `full_closure_builder.run_full_closure` documents.
                light=False,
            )

            # Run solver
            ddar_start = time.time()
            csolver.run(max_level=max_level, custom_rules=custom_rules)
            ddar_time = time.time() - ddar_start

            # Generate possible goals
            possible_goals, checkgoals_runtime = ProblemWorker._generate_possible_goals(
                solver
            )

            # Obtain mapping from clauses to basic statements
            clauses_without_coords: list[Clause] = []
            for clause in solver_builder.problemJGEX.constructions:
                clauses_without_coords.append(
                    Clause(
                        points=tuple(p.split("@")[0] for p in clause.points),
                        sentences=clause.sentences,
                    )
                )
            clause2basics, clause2args = ProblemWorker._get_all_premise(
                clauses_without_coords, solver.proof
            )
            statement_str_idxs = dict()
            pointstr2basicstrs = defaultdict(set)
            basicstr2pointstrs = defaultdict(set)
            for clause in clauses_without_coords:
                for points, basics in clause2basics[clause]:
                    for b in basics:
                        b_str = b.to_str()
                        if b_str not in statement_str_idxs:
                            statement_str_idxs[b_str] = len(statement_str_idxs)
                        for pname in points:
                            pointstr2basicstrs[pname].add(b_str)
                            basicstr2pointstrs[b_str].add(pname)

            # Process goals
            # first, group goals by problem key
            group_runtime = time.time()
            eq_predicates_goals = dict()
            for goal in possible_goals:
                # find essential_clauses
                premises, aux = solver.proof.dep_graph.get_premises_and_aux([goal])
                if aux_only > 0 and len(aux) == 0:
                    continue
                premises = [dep.statement for dep in premises]
                aux = sorted(
                    [dep.statement for dep in aux],
                    key=lambda s: statement_str_idxs[s.to_str()],
                )
                point_names = set()
                for premise in premises:
                    for arg in premise.args:
                        if isinstance(arg, Point):
                            point_names.add(arg.name)
                for arg in goal.args:
                    if isinstance(arg, Point):
                        point_names.add(arg.name)
                predicates = (
                    " ".join(sorted(point_names))
                    + " $$ "
                    + "; ".join(sorted([statement.to_str() for statement in premises]))
                    + " $$ "
                    + "; ".join(sorted([statement.to_str() for statement in aux]))
                )
                eq_predicates_goals.setdefault(predicates, []).append(
                    (goal, premises, aux)
                )
            group_runtime = time.time() - group_runtime

            # then, process goal groups
            process_goal_time = time.time()
            generated_data = []
            # 超过 DEADLINE 时下面的 break 会把当前及之后所有 goal group 都丢掉，
            # 记录一下，供上层统计「本来能出样本、却被时间截断」的任务数。
            deadline_hit = False
            for group_idx, (_, goal_list) in enumerate(eq_predicates_goals.items()):
                if time.time() > DEADLINE:
                    deadline_hit = True
                    break
                goals = [data[0] for data in goal_list]
                premises = goal_list[0][1]
                aux = goal_list[0][2]
                data = ProblemWorker._process_goals_with_same_statement(
                    clause2basics,
                    clause2args,
                    pointstr2basicstrs,
                    basicstr2pointstrs,
                    goals,
                    solver,
                    solver_builder,
                    premises,
                    aux,
                    n_clauses,
                    img,
                    aux_only,
                    seed,
                    using_log,
                    using_exp,
                    custom_rules,
                )
                generated_data.extend(data)
            process_goal_time = time.time() - process_goal_time

            reverse_recon_time = time.time()
            recon_stats = {
                "candidates": 0,
                "solved_without_aux": 0,
                "unsolved_with_aux": 0,
                "kept": 0,
            }
            if ProblemWorker.ENABLE_REVERSE_RECONSTRUCTION:
                recon_data, recon_stats = ProblemWorker._try_reverse_reconstruction(
                    possible_goals,
                    solver,
                    solver_builder,
                    seed,
                    using_log,
                    using_exp,
                    custom_rules,
                )
                generated_data.extend(recon_data)
            reverse_recon_time = time.time() - reverse_recon_time

            has_real_aux = any(
                "<aux>" in d.get("llm_output_renamed", "") for d in generated_data
            )
            # 时间截断且已经产出了带 aux 的样本：这道题确实有 aux，但剩下的
            # goal group 全被 DEADLINE 砍掉了，aux 样本是损失掉的。
            aux_truncated = deadline_hit and has_real_aux

            # Create summary
            summary = {
                "total_time": time.time() - start_time,
                "generation_time": generation_time,
                "build_solver_time": build_solver_time,
                "ddar_time": ddar_time,
                # CSolver only fills run_infos when it owns a symbolic solver;
                # in light mode it skips that path and run_infos stays empty.
                # We pin light=False above, but keep the fallback so a config
                # change can't turn a timing number into a KeyError crash.
                # ddar_time measures the same span.
                "runtime": solver.run_infos.get("runtime", ddar_time),
                "checkgoals_runtime": checkgoals_runtime,
                "process_goal_runtime": process_goal_time,
                "reverse_recon_runtime": reverse_recon_time,
                "recon_candidates": recon_stats["candidates"],
                "recon_solved_without_aux": recon_stats["solved_without_aux"],
                "recon_unsolved_with_aux": recon_stats["unsolved_with_aux"],
                "recon_kept": recon_stats["kept"],
                "group_runtime": group_runtime,
                "n_samples_raw": len(generated_data),
                "goals_raw": [
                    re.search(r"\?\s*(\w+)", d["fl_problem"]).group(1)
                    for d in generated_data
                ],
                "first_predicate_raw": [
                    get_first_predicate(d["fl_problem"]) for d in generated_data
                ],
                "n_premises_raw": [d["n_premises"] for d in generated_data],
                "n_proof_steps_raw": [d["n_proof_steps"] for d in generated_data],
                "n_filtered_samples": 0,  # This value will be updated in generate.py
                "has_real_aux": has_real_aux,
                "deadline_hit": deadline_hit,
                "aux_truncated": aux_truncated,
                "fl_statement": fl_statement,
            }

            # Combine the detailed sampling timing info
            if "sampling_timings" in locals() and sampling_timings:
                summary["sampling_timings"] = sampling_timings

            return generated_data, summary

        except Exception as e:
            logging.info(f"Error generating problem: {e}")
            import traceback

            traceback.print_exc()
            return [], {}

    @staticmethod
    def _build_solver(fl_statement, max_attempts=1):
        """Build geometric solver"""
        solver_builder = GeometricSolverBuilder(seed=998244353)
        solver_builder.with_deductive_agent(DDARN())
        solver_builder.load_problem_from_txt(fl_statement)
        try:
            solver = solver_builder.build(max_attempts=max_attempts)
            return solver, solver_builder
        except Exception as e:
            logging.info(f"Error: {e}")
            return None, None

    @staticmethod
    def _generate_possible_goals(solver):
        """Generate possible goals"""
        t = time.time()
        # ProblemWorker.all_possible_goals_by_ar(solver.proof.dep_graph)
        possible_goals = [goal for goal in solver.proof.dep_graph.conclusions()]
        possible_goals = ProblemWorker.filter.goal_filter(
            possible_goals, solver.proof.dep_graph
        )
        checkgoals_runtime = time.time() - t
        return possible_goals, checkgoals_runtime

    @staticmethod
    def all_possible_goals_by_ar(dep_graph: DependencyGraph) -> list[Statement]:
        def extract_points(s):
            return re.findall(r"[a-z][\d]*", s)

        def goal_from_tokens(tokens):
            if ProblemWorker.filter.naive_goal_filter(tokens[0], tokens[1:], dep_graph):
                goal = Statement.from_tokens(tokens, dep_graph)
                if goal:
                    goal.check()

        points_name = sorted(
            [p.name for p in dep_graph.symbols_graph.nodes_of_type(Point)]
        )
        for i, p in enumerate(points_name):
            for q in points_name[i + 1 :]:
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
                    goal_from_tokens(tuple(["para"] + list(v1 + v2)))
                    goal_from_tokens(tuple(["perp"] + list(v1 + v2)))
                    goal_from_tokens(tuple(["acompute"] + list(v1 + v2)))
                except Exception as e:
                    logging.warning(
                        f"Error in goal_from_tokens: {e} para/perp for {v1}, {v2}"
                    )
                    continue
        for v1, v2, v3, v4 in e2v_pairs4:
            try:
                v1, v2, v3, v4 = (
                    extract_points(v1),
                    extract_points(v2),
                    extract_points(v3),
                    extract_points(v4),
                )
                goal_from_tokens(tuple(["eqangle"] + list(v1 + v2 + v3 + v4)))
            except Exception as e:
                logging.warning(
                    f"Error in goal_from_tokens: {e} for eqangle {v1}, {v2}, {v3}, {v4}"
                )
                continue

        e2v, e2v_pairs2, e2v_pairs4 = ar.rtable.possible_pairs()
        for e in e2v_pairs2.keys():
            for v1, v2 in e2v_pairs2[e]:
                try:
                    goal_from_tokens(
                        tuple(["cong"] + v1[2:-1].split(",") + v2[2:-1].split(","))
                    )
                    goal_from_tokens(
                        tuple(["rcompute"] + v1[2:-1].split(",") + v2[2:-1].split(","))
                    )
                except Exception as e:
                    logging.warning(
                        f"Error in goal_from_tokens: {e} cong for {v1}, {v2}"
                    )
                    continue
        # for v1, v2, v3, v4 in e2v_pairs4:
        #     try:
        #         tokens = tuple(['eqratio'] + list(v1[2:-1].split(',') +
        #                        v2[2:-1].split(',') + v3[2:-1].split(',') + v4[2:-1].split(',')))
        #         goal_from_tokens(tokens)
        #     except Exception as e:
        #         logging.warning(f"Error in goal_from_tokens: {e} for eqratio {v1}, {v2}, {v3}, {v4}")
        #         continue

    @staticmethod
    def _try_reverse_reconstruction(
        possible_goals: list,
        solver: "GeometricSolver",
        solver_builder: "GeometricSolverBuilder",
        seed: int,
        using_log: bool,
        using_exp: bool,
        custom_rules,
    ) -> list:
        """
        对匹配到"一点一谓词"构造类型的已证目标，尝试反向重构。
        采用随机采样策略：打乱所有候选(goal,point,split)组合，
        然后只尝试前N个（REVERSE_RECON_MAX_ATTEMPTS_PER_PROBLEM）。
        """
        stats = {
            "candidates": 0,
            "solved_without_aux": 0,
            "unsolved_with_aux": 0,
            "kept": 0,
        }
        if not ProblemWorker.ENABLE_REVERSE_RECONSTRUCTION:
            return [], stats

        results: list = []
        dep_graph = solver.proof.dep_graph
        reversible = ProblemWorker.REVERSIBLE_CONSTRUCTIONS
        rng = np.random.default_rng(seed)

        # 预先收集所有候选，然后打乱并采样
        candidates = []
        for goal in possible_goals:
            goal_points = [a for a in goal.args if isinstance(a, Point)]
            if len(goal_points) > ProblemWorker.REVERSE_RECON_MAX_GOAL_POINTS:
                continue

            goal_closure_names = {
                p.name for p in dep_graph.points_dependency_closure(goal_points)
            }

            for candidate_point in goal_points:
                other_points = [p for p in goal_points if p is not candidate_point]
                match = ProblemWorker._match_goal_to_construction(
                    goal, candidate_point, reversible, dep_graph
                )
                if match is None:
                    continue
                construction_name, matched_sentence = match

                if dep_graph.point_has_dependents_among(candidate_point, other_points):
                    continue

                orig_clause = candidate_point.clause
                if len(orig_clause.sentences) != 2:
                    continue
                splits = ProblemWorker._split_defining_clause(
                    orig_clause, candidate_point.name
                )
                if not splits:
                    continue

                for kept_sentence, removed_sentence in splits:
                    candidates.append((
                        goal, candidate_point, construction_name, matched_sentence,
                        kept_sentence, removed_sentence, orig_clause, goal_closure_names
                    ))

        # 打乱并采样前N个
        rng.shuffle(candidates)
        sampled = candidates[:ProblemWorker.REVERSE_RECON_MAX_ATTEMPTS_PER_PROBLEM]

        for (goal, candidate_point, construction_name, matched_sentence,
             kept_sentence, removed_sentence, orig_clause, goal_closure_names) in sampled:
            stats["candidates"] += 1
            record, outcome = ProblemWorker._attempt_one_reconstruction(
                goal=goal,
                candidate_point=candidate_point,
                construction_name=construction_name,
                matched_sentence=matched_sentence,
                kept_sentence=kept_sentence,
                removed_sentence=removed_sentence,
                orig_clause=orig_clause,
                goal_closure_names=goal_closure_names,
                solver=solver,
                solver_builder=solver_builder,
                seed=seed,
                using_log=using_log,
                using_exp=using_exp,
                custom_rules=custom_rules,
            )
            if outcome in stats:
                stats[outcome] += 1
            if record is not None:
                results.append(record)

        return results, stats

    @staticmethod
    def _match_goal_to_construction(
        goal: Statement,
        candidate_point: Point,
        reversible: dict,
        dep_graph: DependencyGraph,
    ):
        """
        检查 goal（把 candidate_point 当作"新点" x）是否匹配某个可逆构造类型的
        谓词形式。匹配返回 (construction_name, sentence_template)，否则 None。

        必须通过 Statement.from_tokens(...).to_str() 归一化后比较，不能用原始
        位置比较（rconst 的 preparse 会反转比例，eqratio/eqangle 会重排参数等）。
        """
        goal_pred_name = goal.predicate.NAME
        goal_str = goal.to_str()
        for cname, (pred_name, sentence_template) in reversible.items():
            if pred_name != goal_pred_name:
                continue
            template_args = sentence_template[1:]
            cdef = ProblemWorker.defs[cname]
            new_point_slots = [
                i for i, v in enumerate(template_args) if v in cdef.points
            ]
            if len(new_point_slots) != 1:
                continue  # 只考虑单新点构造
            x_slot = new_point_slots[0]

            goal_args = goal.args
            if len(goal_args) != len(template_args):
                continue
            if (
                not isinstance(goal_args[x_slot], Point)
                or goal_args[x_slot] is not candidate_point
            ):
                continue
            # candidate_point 必须只在这个谓词里出现一次（恰好是"新点"槽位）。
            # 如果它在别的槽位也出现，"删掉 x、换成以 y 为新点的构造"这个替换
            # 就不再是单点重命名能表达的，跳过以保持替换的正确性。
            occurrences = sum(1 for a in goal_args if a is candidate_point)
            if occurrences != 1:
                continue

            candidate_sentence_args = tuple(
                (a.name if isinstance(a, Point) else str(a)) for a in goal_args
            )
            candidate_statement = Statement.from_tokens(
                (pred_name,) + candidate_sentence_args, dep_graph
            )
            if candidate_statement is None:
                continue
            if candidate_statement.to_str() != goal_str:
                continue

            return cname, sentence_template
        return None

    @staticmethod
    def _split_defining_clause(clause: Clause, point_name: str):
        """
        clause 是原题目里定义 point_name 的那条子句（可能还定义了其他点）。只接受
        单点、恰好两条句子的子句。

        x 原本两条句子中，究竟该删除哪一条并没有先验依据（新匹配到的构造类型
        通常和两条原始句子都不同）——所以两种拆分都作为候选返回，分别尝试。

        每种候选是 (kept_sentence, removed_sentence)：kept_sentence 会被重命名
        x->y 并和新匹配到的构造合并进重构题目；removed_sentence 的谓词形式会
        变成新目标。removed_sentence 必须本身是一个可逆构造类型，否则代表被
        删除的构造无法转成单一谓词目标，这种拆分被跳过。
        """
        if len(clause.points) != 1 or clause.points[0].split("@")[0] != point_name:
            return []
        if len(clause.sentences) != 2:
            return []
        s0, s1 = clause.sentences
        reversible = ProblemWorker.REVERSIBLE_CONSTRUCTIONS

        splits = []
        for removed, kept in ((s0, s1), (s1, s0)):
            if removed[0] not in reversible:
                continue
            splits.append((kept, removed))
        return splits

    @staticmethod
    def _fresh_point_name(x_name: str, solver: "GeometricSolver") -> str:
        used = set(solver.proof.symbols_graph.name2node.keys())
        idx = 0
        while True:
            candidate = f"y{idx}" if idx else "y"
            if candidate not in used and candidate != x_name:
                return candidate
            idx += 1

    @staticmethod
    def _other_clauses_in_closure(
        constructions, orig_clause: Clause, x_name: str, goal_closure_names: set
    ):
        """
        返回除 orig_clause 之外、且声明的点落在目标依赖闭包 goal_closure_names
        内的子句——把题目截断到目标证明实际用到的最大点集合。闭包之外的子句
        （包括提到 x_name、但和这次重构无关的子句）直接丢弃，而不是因为
        x_name 在别处出现就拒绝整次重构。
        """
        other_clauses = []
        for clause in constructions:
            clause_no_coords = Clause(
                points=tuple(p.split("@")[0] for p in clause.points),
                sentences=clause.sentences,
            )
            if clause_no_coords == orig_clause:
                continue
            if x_name in clause_no_coords.points:
                # x 自己的其他定义子句（理论上不会出现，因为一个点只由一条
                # 子句定义），保险起见仍然跳过。
                continue
            if not all(p in goal_closure_names for p in clause_no_coords.points):
                continue
            other_clauses.append(clause_no_coords)
        return other_clauses

    @staticmethod
    def _removed_sentence_to_goal_tokens(
        removed_sentence: tuple, rename_map: dict
    ) -> tuple:
        """
        用 removed_sentence（构造的实际参数）代入该构造类型唯一 basics 句子模板，
        得到谓词 token 元组，并应用 x->y 重命名。
        """
        cdef = ProblemWorker.defs[removed_sentence[0]]
        mapping = dict(zip(cdef.declare[1:], removed_sentence[1:]))
        points, bs = cdef.basics[0]
        assert len(bs) == 1
        b = bs[0]
        tokens = translate_sentence(mapping, b)
        return translate_sentence(rename_map, tokens)

    @staticmethod
    def _goal_solved(solver: "GeometricSolver", goal_tokens: tuple) -> bool:
        stmt = Statement.from_tokens(goal_tokens, solver.proof.dep_graph)
        return stmt is not None and stmt.check()

    @staticmethod
    def _attempt_one_reconstruction(
        *,
        goal: Statement,
        candidate_point: Point,
        construction_name: str,
        matched_sentence: tuple,
        kept_sentence: tuple,
        removed_sentence: tuple,
        orig_clause: Clause,
        goal_closure_names: set,
        solver: "GeometricSolver",
        solver_builder: "GeometricSolverBuilder",
        seed: int,
        using_log: bool,
        using_exp: bool,
        custom_rules,
    ):
        """
        构建更小的候选题目，跑第一阶段（不加辅助点）；不可证的话跑第二阶段
        （加回辅助点）。只有"加辅助点才可证"这条路径返回输出字典，否则返回 None。
        """
        x_name = candidate_point.name
        y_name = ProblemWorker._fresh_point_name(x_name, solver)

        rename_map = {x_name: y_name}
        kept_sentence_renamed = translate_sentence(rename_map, kept_sentence)
        cdef = ProblemWorker.defs[construction_name]
        goal_args_by_name = {
            v: (
                goal.args[i].name
                if isinstance(goal.args[i], Point)
                else str(goal.args[i])
            )
            for i, v in enumerate(matched_sentence[1:])
        }
        x_slot_name = [v for v in matched_sentence[1:] if v in cdef.points][0]
        goal_args_by_name[x_slot_name] = y_name
        matched_construction_tokens = (construction_name,) + tuple(
            goal_args_by_name[v] for v in cdef.declare[1:] if v in goal_args_by_name
        )

        new_clause_str = (
            y_name
            + " = "
            + " ".join(kept_sentence_renamed)
            + ", "
            + " ".join(matched_construction_tokens)
        )

        other_clauses = ProblemWorker._other_clauses_in_closure(
            solver_builder.problemJGEX.constructions,
            orig_clause,
            x_name,
            goal_closure_names,
        )

        new_goal_tokens = ProblemWorker._removed_sentence_to_goal_tokens(
            removed_sentence, rename_map
        )

        candidate_fl = (
            "; ".join(str(c) for c in other_clauses)
            + "; "
            + new_clause_str
            + " ? "
            + " ".join(new_goal_tokens)
        )

        # 第一阶段：不加辅助点直接求解。
        solver_a, builder_a = ProblemWorker._build_solver(
            candidate_fl, max_attempts=ProblemWorker.REVERSE_RECON_BUILD_MAX_ATTEMPTS
        )
        if solver_a is None:
            _log_recon_fail("rejected", construction_name, removed_sentence[0],
                             kept_sentence[0], goal.predicate.NAME)
            return None, "rejected"
        csolver_a = CSolver(
            candidate_fl,
            seed=seed,
            solver=solver_a,
            using_log=using_log,
            using_exp=using_exp,
            light=False,
        )
        csolver_a.run(custom_rules=custom_rules)
        if ProblemWorker._goal_solved(solver_a, new_goal_tokens):
            _log_recon_fail("solved_without_aux", construction_name, removed_sentence[0],
                             kept_sentence[0], goal.predicate.NAME)
            return None, "solved_without_aux"  # 直接可证 -> 按纠正后的范围丢弃

        # 第二阶段：把 x 按其完整原始定义（kept + removed 两条句子）重新加回去
        # 作为辅助点。只用 removed_sentence 一条无法唯一确定 x 的位置——它在
        # 原题目里本来就是两条句子的交点，只加一条会让数值构造在自由度更高的
        # 曲线/直线上任取一点，得到的不是原来那个精确的 x。
        aux_clause_str = (
            x_name
            + " = "
            + " ".join(kept_sentence)
            + ", "
            + " ".join(removed_sentence)
        )
        candidate_fl_with_aux = (
            "; ".join(str(c) for c in other_clauses)
            + "; "
            + new_clause_str
            + "; "
            + aux_clause_str
            + " ? "
            + " ".join(new_goal_tokens)
        )
        solver_b, builder_b = ProblemWorker._build_solver(
            candidate_fl_with_aux,
            max_attempts=ProblemWorker.REVERSE_RECON_BUILD_MAX_ATTEMPTS,
        )
        if solver_b is None:
            _log_recon_fail("rejected", construction_name, removed_sentence[0],
                             kept_sentence[0], goal.predicate.NAME)
            return None, "rejected"
        csolver_b = CSolver(
            candidate_fl_with_aux,
            seed=seed,
            solver=solver_b,
            using_log=using_log,
            using_exp=using_exp,
            light=False,
        )
        csolver_b.run(custom_rules=custom_rules)
        if not ProblemWorker._goal_solved(solver_b, new_goal_tokens):
            _log_recon_fail("unsolved_with_aux", construction_name, removed_sentence[0],
                             kept_sentence[0], goal.predicate.NAME)
            return None, "unsolved_with_aux"  # 加了辅助点仍不可证 -> 丢弃

        record = ProblemWorker._build_reconstruction_output_record(
            solver_b,
            builder_b,
            new_goal_tokens,
            seed,
            using_log,
            using_exp,
            custom_rules,
        )
        return record, ("kept" if record is not None else "rejected")

    @staticmethod
    def _build_reconstruction_output_record(
        solver_b: "GeometricSolver",
        builder_b: "GeometricSolverBuilder",
        new_goal_tokens: tuple,
        seed: int,
        using_log: bool,
        using_exp: bool,
        custom_rules,
    ):
        """
        对第二阶段求解成功的题目，走现有、不做任何修改的 pipeline 阶段
        （_get_all_premise -> get_premises_and_aux -> pointstr2basicstrs/
        basicstr2pointstrs -> _process_goals_with_same_statement），产出和
        主 pipeline 完全一致形状的输出记录。
        """
        clauses_b_no_coords = [
            Clause(
                points=tuple(p.split("@")[0] for p in c.points),
                sentences=c.sentences,
            )
            for c in builder_b.problemJGEX.constructions
        ]
        clause2basics_b, clause2args_b = ProblemWorker._get_all_premise(
            clauses_b_no_coords, solver_b.proof
        )
        goal_b = Statement.from_tokens(new_goal_tokens, solver_b.proof.dep_graph)
        if goal_b is None:
            return None
        premises_b, aux_b = solver_b.proof.dep_graph.get_premises_and_aux([goal_b])
        premises_b = [dep.statement for dep in premises_b]
        aux_b = [dep.statement for dep in aux_b]

        pointstr2basicstrs_b = defaultdict(set)
        basicstr2pointstrs_b = defaultdict(set)
        for clause in clauses_b_no_coords:
            for points, basics in clause2basics_b[clause]:
                for b in basics:
                    b_str = b.to_str()
                    for pname in points:
                        pointstr2basicstrs_b[pname].add(b_str)
                        basicstr2pointstrs_b[b_str].add(pname)

        records = ProblemWorker._process_goals_with_same_statement(
            clause2basics_b,
            clause2args_b,
            pointstr2basicstrs_b,
            basicstr2pointstrs_b,
            [goal_b],
            solver_b,
            builder_b,
            premises_b,
            aux_b,
            n_clauses=len(clauses_b_no_coords),
            img=None,
            aux_only=0,
            seed=seed,
            using_log=using_log,
            using_exp=using_exp,
            custom_rules=custom_rules,
        )
        if not records:
            return None
        record = records[0]
        if "<aux>" not in record.get("llm_output_renamed", ""):
            # 兜底检查：第二阶段本来是靠加回去的 x 才成功的。如果
            # _process_goals_with_same_statement 找到的最小证明根本没用到
            # 辅助点，说明第一阶段本该已经成功了——保守地丢弃。
            return None
        return record

    @staticmethod
    def _extract_proof_info(solver, goal):
        """Extract proof information from solver for a given goal"""
        (
            points,
            premises,
            numerical_checked_premises,
            trivial_premises,
            aux_points,
            aux,
            numerical_checked_aux,
            trivial_aux,
            proof_steps,
        ) = solver.proof.dep_graph.get_proof_steps([goal])

        return {
            "goal": goal,
            "points": points,
            "premises": premises,
            "numerical_checked_premises": numerical_checked_premises,
            "trivial_premises": trivial_premises,
            "aux_points": aux_points,
            "aux": aux,
            "numerical_checked_aux": numerical_checked_aux,
            "trivial_aux": trivial_aux,
            "proof_steps": proof_steps,
        }

    @staticmethod
    def _flatten_statement_groups(
        statement_groups: list[list[Statement]],
    ) -> list[Statement]:
        """Flatten grouped statements into one list."""
        flattened: list[Statement] = []
        for group in statement_groups:
            flattened.extend(group)
        return flattened

    @staticmethod
    def _expand_final_premises_and_aux(
        clause2basics: dict[Clause, list],
        clause2args: dict[Clause, set],
        premises: list[Statement],
        aux: list[Statement],
        essential_point_names: set[str],
        aux_point_names: set[str],
    ) -> tuple[
        dict[Clause, list],
        list[Clause],
        list[Statement],
        list[tuple],
        list[list[Statement]],
    ]:
        """Expand proof-level premise/aux statements into final exported units."""
        expanded_clause2basics = clause2basics.copy()
        essential_premise_clauses = ProblemWorker._get_essential_premise_clauses(
            expanded_clause2basics,
            clause2args,
            [premise.to_str() for premise in premises],
            essential_point_names.copy(),
        )
        expanded_premises: list[Statement] = []
        for clause in essential_premise_clauses:
            for _, basics in expanded_clause2basics[clause]:
                expanded_premises.extend(basics)

        essential_aux_basics = ProblemWorker._get_aux_basics(
            expanded_clause2basics,
            [statement.to_str() for statement in aux],
            aux_point_names.copy(),
        )
        expanded_aux_groups = [
            list(basics) for _, basics in essential_aux_basics if len(basics) > 0
        ]

        return (
            expanded_clause2basics,
            essential_premise_clauses,
            expanded_premises,
            essential_aux_basics,
            expanded_aux_groups,
        )

    @staticmethod
    def _find_minimal_aux_clauses_new(
        pointstr2basicstrs,
        basicstr2pointstrs,
        solver,
        solver_builder,
        goals_str,
        expanded_premises,
        expanded_aux_groups,
        aux_only,
        rng,
        using_log=True,
        using_exp=False,
        custom_rules=None,
    ):
        """Find minimal auxiliary clause set"""
        results = []
        timings = {
            "build_predicates_time": 0.0,
            "build_solver_time": 0.0,
            "run_solver_time": 0.0,
        }

        # Step 1: First try solving without aux
        t0 = time.time()
        try:
            proof_state_no_aux = ProofState.build_predicates(
                predicates=expanded_premises,
                defsJGEX=solver_builder.defs,
                goals_str=goals_str.copy(),
                rng=np.random.default_rng(solver_builder.seed),
            )
        except (IndexError, ValueError) as e:
            logger.error(
                f"Failed to build predicates!\n"
                f"Error: {e}\n"
                f"goals_str: {goals_str}\n"
                f"expanded_premises ({len(expanded_premises)} items): "
                f"{[p.to_str() for p in expanded_premises][:10]}"  # Show first 10
            )
            raise
        timings["build_predicates_time"] = time.time() - t0

        t0 = time.time()
        solver_no_aux = GeometricSolver(
            proof_state_no_aux, solver_builder.rules, DDARN()
        )
        timings["build_solver_time"] = time.time() - t0

        t0 = time.time()
        csolver_no_aux = CSolver(
            problem="",
            solver=solver_no_aux,
            using_log=using_log,
            using_exp=using_exp,
        )
        csolver_no_aux.run(custom_rules=custom_rules)
        timings["run_solver_time"] = time.time() - t0

        for goal in solver_no_aux.goals:
            if goal.check():
                goals_str.remove(goal.to_str())
                # For aux_only==1, keep data without aux with 0.1 probability
                # For aux_only==0, always keep
                # For aux_only==2, never keep
                if aux_only == 0 or (aux_only == 1 and rng.random() < 0.1):
                    results.append(
                        ProblemWorker._extract_proof_info(solver_no_aux, goal)
                    )

        if len(expanded_aux_groups) == 1:
            # If only one aux, no need to continue
            if len(goals_str) == 0:
                return results, timings

            # For remaining goals, excute with all aux to strip extra points
            # that might be introduced in numerical checks
            t0 = time.time()
            proof_state_all_aux = ProofState.build_predicates(
                predicates=expanded_premises
                + ProblemWorker._flatten_statement_groups(expanded_aux_groups),
                defsJGEX=solver_builder.defs,
                goals_str=goals_str.copy(),
                rng=np.random.default_rng(solver_builder.seed),
            )
            timings["build_predicates_time"] += time.time() - t0

            t0 = time.time()
            solver_all_aux = GeometricSolver(
                proof_state_all_aux, solver_builder.rules, DDARN()
            )
            timings["build_solver_time"] += time.time() - t0

            csolver_all_aux = CSolver(
                problem="",
                solver=solver_all_aux,
                using_log=using_log,
                using_exp=using_exp,
            )
            t0 = time.time()
            csolver_all_aux.run(custom_rules=custom_rules)
            timings["run_solver_time"] += time.time() - t0

            for goal in solver_all_aux.goals:
                if goal.check():
                    goals_str.remove(goal.to_str())
                    results.append(
                        ProblemWorker._extract_proof_info(solver_all_aux, goal)
                    )
                else:
                    logger.warning(
                        f"Goal {goal.to_str()} cannot be solved even with all aux"
                    )
            return results, timings

        if len(goals_str) == 0:
            return results, timings

        # Step 2: For remaining goals, try removing aux one by one from back to front
        # Group goals by the minimal aux they need
        goal_groups = [
            {
                "goals": goals_str.copy(),
                "aux_groups": list(expanded_aux_groups),
                "solvers": {},
            }
        ]
        # premise_strs = set([p.to_str() for p in premises])
        # premise_pointstrs = set()
        # for p in premises:
        #     for arg in p.args:
        #         if isinstance(arg, Point):
        #             premise_pointstrs.add(arg.name)

        for i in range(len(expanded_aux_groups) - 1, -1, -1):
            new_goal_groups = []

            for group in goal_groups:
                # Try without this aux for all goals in this group
                test_aux_groups = group["aux_groups"][:i] + group["aux_groups"][i + 1 :]

                # # Check if the aux set is valid w.r.t rely_on
                # flag = True
                # exist_stmt_strs = premise_strs.copy()
                # # print(f"group[aux][:i]: {group["aux"][:i]}, group[aux][i+1:]: {group["aux"][i+1:]}")
                # for aux_stmt in group["aux"][:i]:
                #     exist_stmt_strs.add(aux_stmt.to_str())
                # for aux_stmt in group["aux"][i+1:]:
                #     aux_str = aux_stmt.to_str()
                #     exist_stmt_strs.add(aux_str)
                #     # print(f"aux: {aux_stmt}, points: {basicstr2pointstrs[aux_stmt.to_str()]}")
                #     # print(f"exist_stmt_strs: {exist_stmt_strs}")
                #     pointstrs_to_check = set()
                #     for arg in aux_stmt.args:
                #         if isinstance(arg, Point):
                #             pointstrs_to_check.add(arg.name)
                #     for pname in pointstrs_to_check:
                #         if pname in premise_pointstrs or len(pointstr2basicstrs[pname]) == 0:
                #             continue
                #         # print(f"  checking point: {pname}, basicstrs: {pointstr2basicstrs[pname]}")
                #         if pointstr2basicstrs[pname].isdisjoint(exist_stmt_strs):
                #             flag = False
                #             break
                #     if not flag:
                #         break
                # # print(f"test result for {test_aux}: {flag}")
                # if not flag:
                #     # Cannot remove this aux due to rely_on
                #     new_goal_groups.append(group)
                #     continue

                t0 = time.time()
                proof_state_test = ProofState.build_predicates(
                    predicates=expanded_premises
                    + ProblemWorker._flatten_statement_groups(test_aux_groups),
                    defsJGEX=solver_builder.defs,
                    goals_str=group["goals"],
                    rng=np.random.default_rng(solver_builder.seed),
                )
                timings["build_predicates_time"] += time.time() - t0

                t0 = time.time()
                solver_test = GeometricSolver(
                    proof_state_test, solver_builder.rules, DDARN()
                )
                timings["build_solver_time"] += time.time() - t0

                t0 = time.time()
                csolver_test = CSolver(
                    problem="",
                    solver=solver_test,
                    using_log=using_log,
                    using_exp=using_exp,
                )
                csolver_test.run(custom_rules=custom_rules)
                timings["run_solver_time"] += time.time() - t0

                # Check which goals are solved
                solved_goals = []
                unsolved_goals = []
                for goal_str in group["goals"]:
                    goal_found = False
                    for goal in solver_test.goals:
                        if goal.to_str() == goal_str:
                            if goal.check():
                                solved_goals.append(goal_str)
                                # Save solver for this goal
                                group["solvers"][goal_str] = (solver_test, goal)
                            else:
                                unsolved_goals.append(goal_str)
                            goal_found = True
                            break
                    assert goal_found, f"Goal {goal_str} not found in solver_test.goals"

                # Split into groups based on whether they can be solved without this aux
                if len(solved_goals) > 0:
                    new_goal_groups.append(
                        {
                            "goals": solved_goals,
                            "aux_groups": test_aux_groups,
                            "solvers": {
                                g: group["solvers"][g]
                                for g in solved_goals
                                if g in group["solvers"]
                            },
                        }
                    )
                if len(unsolved_goals) > 0:
                    new_goal_groups.append(
                        {
                            "goals": unsolved_goals,
                            "aux_groups": group["aux_groups"],
                            "solvers": {
                                g: group["solvers"][g]
                                for g in unsolved_goals
                                if g in group["solvers"]
                            },
                        }
                    )

            goal_groups = new_goal_groups

        # Collect results from all groups
        for group in goal_groups:
            for goal_str in group["goals"]:
                if goal_str in group["solvers"]:
                    goals_str.remove(goal_str)
                    # Use saved solver and extract proof information
                    best_solver, best_goal = group["solvers"][goal_str]
                    results.append(
                        ProblemWorker._extract_proof_info(best_solver, best_goal)
                    )

        if len(goals_str) > 0:
            # For remaining goals, excute with all aux to strip extra points
            # that might be introduced in numerical checks
            t0 = time.time()
            proof_state_all_aux = ProofState.build_predicates(
                predicates=expanded_premises
                + ProblemWorker._flatten_statement_groups(expanded_aux_groups),
                defsJGEX=solver_builder.defs,
                goals_str=goals_str.copy(),
                rng=np.random.default_rng(solver_builder.seed),
            )
            timings["build_predicates_time"] += time.time() - t0

            t0 = time.time()
            solver_all_aux = GeometricSolver(
                proof_state_all_aux, solver_builder.rules, DDARN()
            )
            timings["build_solver_time"] += time.time() - t0

            csolver_all_aux = CSolver(
                problem="",
                solver=solver_all_aux,
                using_log=using_log,
                using_exp=using_exp,
            )
            t0 = time.time()
            csolver_all_aux.run(custom_rules=custom_rules)
            timings["run_solver_time"] += time.time() - t0

            for goal in solver_all_aux.goals:
                goal_check_result = goal.check()
                if goal_check_result:
                    goals_str.remove(goal.to_str())
                    results.append(
                        ProblemWorker._extract_proof_info(solver_all_aux, goal)
                    )
                else:
                    logger.warning(
                        f"Goal {goal.to_str()} cannot be solved even with all aux"
                    )

        return results, timings

    @staticmethod
    def _process_goals_with_same_statement(
        clause2basics,
        clause2args,
        pointstr2basicstrs,
        basicstr2pointstrs,
        goals,
        solver,
        solver_builder,
        premises,
        aux,
        n_clauses,
        img,
        aux_only,
        seed,
        using_log=True,
        using_exp=False,
        custom_rules=None,
    ):
        """Process a single goal"""

        results = []
        name2node = dict()
        for k, v in solver.proof.symbols_graph.name2node.items():
            if isinstance(v, Point):
                name2node[k] = v

        # Create RNG for probabilistic filtering
        rng = np.random.default_rng(solver_builder.seed)

        # Expand proof-level premises/aux into the same clause/basic view used by
        # llm_solution_renamed() when exporting the final fl_problem and <aux>.
        # _find_minimal_aux_clauses_new() must reason over this expanded view,
        # otherwise Step 1/Step 2 would still be checking a goal against the
        # minimal proof premises instead of the final serialized sample.
        essential_point_names = set()
        for statement in premises:
            for arg in statement.args:
                if isinstance(arg, Point):
                    essential_point_names.add(arg.name)
        for goal in goals:
            for arg in goal.args:
                if isinstance(arg, Point):
                    essential_point_names.add(arg.name)

        aux_point_names = set()
        for statement in aux:
            for arg in statement.args:
                if isinstance(arg, Point):
                    aux_point_names.add(arg.name)

        _, _, expanded_premises, _, expanded_aux_groups = (
            ProblemWorker._expand_final_premises_and_aux(
                clause2basics,
                clause2args,
                premises,
                aux,
                essential_point_names,
                aux_point_names,
            )
        )

        try:
            res_list, aux_timings = ProblemWorker._find_minimal_aux_clauses_new(
                pointstr2basicstrs,
                basicstr2pointstrs,
                solver,
                solver_builder,
                [goal.to_str() for goal in goals],
                expanded_premises,
                expanded_aux_groups,
                aux_only,
                rng,
                using_log=using_log,
                using_exp=using_exp,
                custom_rules=custom_rules,
            )
        except Exception as e:
            # Extract fl_statement from solver_builder
            fl_statement_parts = []
            try:
                if hasattr(solver_builder, 'problemJGEX') and solver_builder.problemJGEX:
                    fl_statement_parts.append(str(solver_builder.problemJGEX))
            except:
                pass

            try:
                # Try to reconstruct from solver's proof state
                if hasattr(solver, 'proof') and hasattr(solver.proof, 'symbols_graph'):
                    points = [p.name for p in solver.proof.symbols_graph.nodes_of_type(Point)]
                    fl_statement_parts.append(f"Points: {', '.join(points)}")
            except:
                pass

            fl_statement = '\n'.join(fl_statement_parts) if fl_statement_parts else "<unable to extract>"

            logger.error(
                f"\n{'='*60}\n"
                f"ERROR in _find_minimal_aux_clauses_new!\n"
                f"{'='*60}\n"
                f"fl_statement:\n{fl_statement}\n"
                f"{'-'*60}\n"
                f"goals: {[goal.to_str() for goal in goals]}\n"
                f"{'-'*60}\n"
                f"n_premises (before expansion): {len(premises)}\n"
                f"n_expanded_premises: {len(expanded_premises)}\n"
                f"{'-'*60}\n"
                f"Original error: {type(e).__name__}: {e}\n"
                f"{'='*60}"
            )
            raise

        for res in res_list:
            goal_new = res["goal"]
            points = res["points"]
            premises = res["premises"]
            numerical_checked_premises = res["numerical_checked_premises"]
            trivial_premises = res["trivial_premises"]
            aux_points = res["aux_points"]
            aux = res["aux"]
            numerical_checked_aux = res["numerical_checked_aux"]
            trivial_aux = res["trivial_aux"]
            proof_steps = res["proof_steps"]

            # llm data generation
            (
                llm_renamed,
                clauses,
                mapping,
                premise_point_names,
                n_premises,
                n_proof_steps,
            ) = ProblemWorker.llm_solution_renamed(
                clause2basics.copy(),
                clause2args.copy(),
                [goal_new],
                points,
                premises,
                numerical_checked_premises,
                trivial_premises,
                aux_points,
                aux,
                numerical_checked_aux,
                trivial_aux,
                proof_steps,
                name2node,
            )

            if aux_only == 2 and "aux" not in llm_renamed["llm_output"]:
                continue

            if "aux" in llm_renamed[
                "llm_output"
            ] and not ProblemWorker.filter.aux_predicates_valid_check(
                llm_renamed["llm_output"]
            ):
                continue

            result = {
                "seed": seed,
                "n_premises": n_premises,
                "fl_problem": llm_renamed["fl_problem"],
                "nl_problem": "",
                "n_proof_steps": n_proof_steps,
                "llm_input_renamed": llm_renamed["llm_input"],
                "llm_output_renamed": llm_renamed["llm_output"],
                "_timings": {
                    **aux_timings,
                },
            }

            if img:
                # Store drawing metadata instead of matplotlib figures
                # to reduce memory usage in Ray workers
                point_coords = {
                    name: (p.num.x, p.num.y) for name, p in name2node.items()
                }
                result["draw_data"] = {
                    "clauses": [(c.points, c.sentences) for c in clauses],
                    "mapping": mapping,
                    "premise_point_names": premise_point_names,
                    "goal_tokens": goal_new.to_str().split(" "),
                    "point_coords": point_coords,
                    "seed": solver_builder.seed,
                }

            results.append(result)
        return results

    @staticmethod
    def _rediger_new_format(dep, mp, dep_idx, dep_graph: DependencyGraph) -> str:
        """Generate proof step in new format: statement [id] rule_id [required_statement_ids]"""
        for statement in (dep.statement,) + dep.why:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            if statement_str not in dep_idx:
                dep_idx[statement_str] = f"{len(dep_idx):03d}"

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
        elif reason and " " in reason:
            rule_id = reason.split()[0]
        else:
            rule_id = reason if reason else "unknown"

        # Generate new format: statement [statement_id] rule_id [premise_ids]
        premise_id_items = []
        for premise in dep.why:
            premise_str = Statement.from_tokens(
                translate_sentence(mp, premise.to_str().split(" ")),
                dep_graph,
            ).to_str()
            premise_id_items.append(f"[{dep_idx[premise_str]}]")
        premise_ids = " ".join(premise_id_items)
        conclusion_str = Statement.from_tokens(
            translate_sentence(mp, dep.statement.to_str().split(" ")),
            dep_graph,
        ).to_str()
        return f"{conclusion_str} [{dep_idx[conclusion_str]}] {rule_id} {premise_ids}".strip()

    @staticmethod
    def llm_solution_renamed(
        clause2basics: dict[Clause, list],
        clause2args: dict[Clause, set],
        goals: list[Statement],
        points: set[Point],
        premises: list[Dependency],
        numerical_checked_premises: list[Dependency],
        trivial_premises: list[Dependency],
        aux_points: set[Point],
        aux: list[Dependency],
        numerical_checked_aux: list[Dependency],
        trivial_aux: list[Dependency],
        proof_steps: list[Dependency],
        name2node: dict[str, Symbol],
    ):
        """Refactored main method to generate LLM solution with renamed points"""
        try:
            # Initialize data
            dep_idx: dict[str, str] = {}
            output_dep_graph = DependencyGraph(AlgebraicManipulator())

            # Get essential premises/points
            (
                expanded_clause2basics,
                essential_premise_clauses,
                _expanded_premises,
                essential_aux_basics,
                _expanded_aux_groups,
            ) = ProblemWorker._expand_final_premises_and_aux(
                clause2basics,
                clause2args,
                [premise.statement for premise in premises],
                [a.statement for a in aux],
                set([p.name for p in points]),
                set([p.name for p in aux_points]),
            )

            # Create point name mapping
            essential_premise_point_names: list[str] = []
            for clause in essential_premise_clauses:
                for _points, bs in expanded_clause2basics[clause]:
                    essential_premise_point_names.extend(_points)
            essential_aux_point_names: list[str] = []
            for _points, bs in essential_aux_basics:
                essential_aux_point_names.extend(_points)
            mp = ProblemWorker._create_point_mapping(
                essential_premise_point_names, essential_aux_point_names
            )
            new_points_with_coords = {}
            for k, v in mp.items():
                new_points_with_coords[mp[k]] = (name2node[k].num.x, name2node[k].num.y)
            # Generate each section
            data_problem_clauses = ProblemWorker._generate_problem_clauses_section(
                mp,
                essential_premise_clauses,
                goals,
                new_points_with_coords,
                output_dep_graph,
            )
            data_problem = ProblemWorker._generate_problem_predicates_section(
                mp,
                dep_idx,
                expanded_clause2basics,
                essential_premise_clauses,
                goals,
                output_dep_graph,
            )
            n_premises = len(dep_idx)
            data_aux = ProblemWorker._generate_aux_section(
                mp, dep_idx, essential_aux_basics, output_dep_graph
            )
            numerical_check = ProblemWorker._generate_numerical_check_section(
                mp,
                dep_idx,
                numerical_checked_premises,
                numerical_checked_aux,
                output_dep_graph,
            )
            trivial_check = ProblemWorker._generate_trivial_section(
                mp, dep_idx, trivial_premises, trivial_aux, output_dep_graph
            )
            proof = ProblemWorker._generate_proof_section(
                mp, dep_idx, proof_steps, output_dep_graph
            )

            # Assemble result
            return (
                {
                    "fl_problem": data_problem_clauses,
                    "llm_input": data_problem,
                    "llm_output": data_aux + numerical_check + trivial_check + proof,
                },
                essential_premise_clauses,
                mp,
                sorted(set(essential_premise_point_names)),
                n_premises,
                len(proof_steps),
            )

        except Exception:
            import traceback

            traceback.print_exc()
            print(f"clause2basics: {clause2basics}")
            print(f"essential_clauses: {locals().get('essential_premise_clauses')}")
            print(f"essential_aux_basics: {locals().get('essential_aux_basics')}")
            print(
                f"essential_premise_point_names: {locals().get('essential_premise_point_names')}"
            )
            print(
                f"essential_aux_point_names: {locals().get('essential_aux_point_names')}"
            )
            print(f"mp: {locals().get('mp')}")
            print(f"point_coords: {locals().get('new_points_with_coords')}")
            raise

    @staticmethod
    def _get_apha_geo_solver_var(va_idx):
        """Generate a point name using letters and numbers"""
        letter_part = string.ascii_lowercase[va_idx % 26]
        number_part = va_idx // 26
        return f"{letter_part}{number_part - 1}" if number_part else letter_part

    @staticmethod
    def _get_all_premise(
        clauses: list[Clause], proof_state: ProofState
    ) -> tuple[dict[Clause, list], dict[Clause, set]]:
        """Get all premises from the problem constructions"""
        clause2basics: dict[Clause, list] = defaultdict(list)
        clause2args: dict[Clause, set] = defaultdict(set)
        for clause in clauses:
            points2basics: dict[tuple[str, ...], list[Statement]] = defaultdict(list)
            for constr_sentence in clause.sentences:
                cdef = ProblemWorker.defs[constr_sentence[0]]
                if len(constr_sentence) == len(cdef.declare):
                    mapping = dict(zip(cdef.declare[1:], constr_sentence[1:]))
                else:
                    assert len(constr_sentence) + len(clause.points) == len(
                        cdef.declare
                    )
                    mapping = dict(
                        zip(cdef.declare[1:], clause.points + constr_sentence[1:])
                    )
                for rely_points in cdef.rely.values():
                    clause2args[clause].update(set([mapping[p] for p in rely_points]))
                for points, bs in cdef.basics:
                    points = tuple([mapping[x] for x in points])
                    bs_statements = []
                    for b in bs:
                        statement = Statement.from_tokens(
                            translate_sentence(mapping, b), proof_state.dep_graph
                        )
                        bs_statements.append(statement)
                    points2basics[points].extend(bs_statements)
            for points, bs_statements in points2basics.items():
                clause2basics[clause].append((points, bs_statements))
        return clause2basics, clause2args

    @staticmethod
    def _get_essential_premise_clauses(
        clause2basics: dict[Clause, list],
        clause2args: dict[Clause, set],
        premise_statement_strs: list[str],
        essential_point_names: set[str],
    ) -> list[Clause]:
        """
        Get essential clauses according to the premise statements and essential points.

        A clause is considered essential if one of the following conditions is met:
        1. any predicate in the clause's basics is in the premise_statements
        2. any point in the clause is in the essential points and the clause has no predicate (e.g. triangle)
        After collecting essential clauses, update the essential points set.
        For non-essential clauses, if any point in the clause is in the essential points, construct the point with a 'free' clause.
        """
        essential_premise_clauses_set: set[Clause] = set()
        new_free_points: list[tuple[Clause, str]] = []
        for clause, basics in clause2basics.items():
            num_of_predicates = sum(len(bs) for _, bs in basics)
            if any(
                statement.to_str() in premise_statement_strs
                for _, bs in basics
                for statement in bs
            ):
                essential_premise_clauses_set.add(clause)
            elif num_of_predicates == 0 and any(
                p in essential_point_names for p in clause.points
            ):
                essential_premise_clauses_set.add(clause)
        # update essential points
        for clause in essential_premise_clauses_set:
            for point_name in clause2args[clause]:
                essential_point_names.add(point_name)
        # Construct 'free' clauses for non-essential clauses
        essential_premise_clauses: list[Clause] = []
        for clause, basics in clause2basics.items():
            if clause in essential_premise_clauses_set:
                essential_premise_clauses.append(clause)
            else:
                for p in clause.points:
                    if p in essential_point_names:
                        free_clause = Clause(points=(p,), sentences=(("free", p),))
                        essential_premise_clauses.append(free_clause)
                        new_free_points.append((free_clause, p))
        for free_clause, p in new_free_points:
            clause2basics[free_clause] = [((p,), ())]
        return essential_premise_clauses

    @staticmethod
    def _get_aux_basics(
        clause2basics: dict[Clause, list],
        aux_strs: list[str],
        aux_point_names: set[str],
    ) -> list[tuple[tuple[str, ...], tuple[Statement, ...]]]:
        """
        Get auxiliary basics according to the auxiliary statements and auxiliary points.

        Only reserve basics that contain auxiliary statements or points.
        """
        essential_aux_basics: list[tuple[tuple[str, ...], tuple[Statement, ...]]] = []
        for clause, basics in clause2basics.items():
            for points, bs in basics:
                bs_filtered = [
                    statement for statement in bs if statement.to_str() in aux_strs
                ]
                if len(bs_filtered) > 0 or any(p in aux_point_names for p in points):
                    essential_aux_basics.append(
                        (
                            tuple(p for p in points if p in aux_point_names),
                            tuple(bs_filtered),
                        )
                    )
        return essential_aux_basics

    @staticmethod
    def _create_point_mapping(
        essential_premise_point_names: list[str],
        essential_aux_point_names: list[str],
    ) -> dict[str, str]:
        """Create point name mapping"""
        mp: dict[str, str] = {}
        for idx, p in enumerate(
            essential_premise_point_names + essential_aux_point_names
        ):
            assert p not in mp
            mp[p] = ProblemWorker._get_apha_geo_solver_var(idx)
        return mp

    @staticmethod
    def _generate_problem_clauses_section(
        mp: dict[str, str],
        essential_premise_clauses: list[Clause],
        goals: list[Statement],
        new_points_with_coords: dict[str, tuple[float, float]],
        dep_graph: DependencyGraph,
    ) -> str:
        """Generate problem clauses section"""
        renamed_clauses = [clause.renamed(mp) for clause in essential_premise_clauses]
        renamed_goal_strs = [
            Statement.from_tokens(
                translate_sentence(mp, goal.to_str().split(" ")),
                dep_graph,
            ).to_str()
            for goal in goals
        ]
        return (
            "; ".join(
                [
                    clause.to_str_with_coordinates(new_points_with_coords)
                    for clause in renamed_clauses
                ]
            )
            + " ? "
            + "; ".join(renamed_goal_strs)
        )

    @staticmethod
    def _generate_problem_predicates_section(
        mp: dict[str, str],
        dep_idx: dict[str, str],
        clause2basics: dict[Clause, list],
        clauses: list[Clause],
        goals: list[Statement],
        dep_graph: DependencyGraph,
    ):
        """Generate problem predicates section"""
        renamed_basic_strs_with_idx: list[str] = []
        for clause in clauses:
            basics = clause2basics[clause]
            for points, bs in basics:
                predicate_strs_with_idx: list[str] = []
                for b in bs:
                    # leverage preparsing in Statement.from_tokens to ensure format
                    statement_str = Statement.from_tokens(
                        translate_sentence(mp, b.to_str().split(" ")),
                        dep_graph,
                    ).to_str()
                    assert statement_str not in dep_idx
                    if statement_str not in dep_idx:
                        dep_idx[statement_str] = f"{len(dep_idx):03d}"
                    predicate_strs_with_idx.append(
                        f"{statement_str} [{dep_idx[statement_str]}]"
                    )
                if len(predicate_strs_with_idx) > 0:
                    renamed_basic_strs_with_idx.append(
                        " ".join([mp[p] for p in points])
                        + " : "
                        + " ".join(predicate_strs_with_idx)
                    )
                else:
                    renamed_basic_strs_with_idx.append(
                        " ".join([mp[p] for p in points]) + " :"
                    )
        renamed_goal_strs = [
            Statement.from_tokens(
                translate_sentence(mp, goal.to_str().split(" ")),
                dep_graph,
            ).to_str()
            for goal in goals
        ]
        return (
            "<problem> "
            + " ; ".join(renamed_basic_strs_with_idx)
            + " ? "
            + "; ".join(renamed_goal_strs)
            + " </problem>"
        )

    @staticmethod
    def _generate_aux_section(
        mp: dict[str, str],
        dep_idx: dict[str, str],
        aux_basics: list[tuple[tuple[str, ...], tuple[Statement, ...]]],
        dep_graph: DependencyGraph,
    ):
        """Generate auxiliary information section"""
        renamed_basic_strs_with_idx: list[str] = []
        for points, bs in aux_basics:
            predicate_strs_with_idx: list[str] = []
            for b in bs:
                # leverage preparsing in Statement.from_tokens to ensure format
                statement_str = Statement.from_tokens(
                    translate_sentence(mp, b.to_str().split(" ")),
                    dep_graph,
                ).to_str()
                assert statement_str not in dep_idx
                if statement_str not in dep_idx:
                    dep_idx[statement_str] = f"{len(dep_idx):03d}"
                predicate_strs_with_idx.append(
                    f"{statement_str} [{dep_idx[statement_str]}]"
                )
            renamed_basic_strs_with_idx.append(
                "x00 "
                + " ".join([mp[p] for p in points])
                + " : "
                + " ".join(predicate_strs_with_idx)
            )
        if len(renamed_basic_strs_with_idx) == 0:
            return ""
        return "<aux> " + " ; ".join(renamed_basic_strs_with_idx) + " ; </aux> "

    @staticmethod
    def _generate_numerical_check_section(
        mp,
        dep_idx,
        numerical_checked_premises,
        numerical_checked_aux,
        dep_graph: DependencyGraph,
    ):
        """Generate numerical check section"""
        numerical_check_items = []
        for line in numerical_checked_premises:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            if statement_str not in dep_idx:
                dep_idx[statement_str] = f"{len(dep_idx):03d}"
        sorted_numerical_checked_premises = sorted(
            numerical_checked_premises,
            key=lambda line: dep_idx[
                Statement.from_tokens(
                    translate_sentence(mp, line.statement.to_str().split(" ")),
                    dep_graph,
                ).to_str()
            ],
        )
        for line in sorted_numerical_checked_premises:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            numerical_check_items.append(f"{statement_str} [{dep_idx[statement_str]}]")
        for line in numerical_checked_aux:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            if statement_str not in dep_idx:
                dep_idx[statement_str] = f"{len(dep_idx):03d}"
        sorted_numerical_checked_aux = sorted(
            numerical_checked_aux,
            key=lambda line: dep_idx[
                Statement.from_tokens(
                    translate_sentence(mp, line.statement.to_str().split(" ")),
                    dep_graph,
                ).to_str()
            ],
        )
        for line in sorted_numerical_checked_aux:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            numerical_check_items.append(f"{statement_str} [{dep_idx[statement_str]}]")
        if len(numerical_check_items) > 0:
            numerical_check = (
                "<numerical_check> "
                + " ; ".join(numerical_check_items)
                + " ; </numerical_check> "
            )
        else:
            numerical_check = ""
        return numerical_check

    @staticmethod
    def _generate_trivial_section(
        mp, dep_idx, trivial_premises, trivial_aux, dep_graph: DependencyGraph
    ):
        """Generate trivial section"""
        trivial_items = []
        # trivial_premises
        for line in trivial_premises:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            if statement_str not in dep_idx:
                dep_idx[statement_str] = f"{len(dep_idx):03d}"
        sorted_trivial_premises = sorted(
            trivial_premises,
            key=lambda line: dep_idx[
                Statement.from_tokens(
                    translate_sentence(mp, line.statement.to_str().split(" ")),
                    dep_graph,
                ).to_str()
            ],
        )
        for line in sorted_trivial_premises:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            trivial_items.append(f"{statement_str} [{dep_idx[statement_str]}]")
        for line in trivial_aux:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            if statement_str not in dep_idx:
                dep_idx[statement_str] = f"{len(dep_idx):03d}"
        sorted_trivial_aux = sorted(
            trivial_aux,
            key=lambda line: dep_idx[
                Statement.from_tokens(
                    translate_sentence(mp, line.statement.to_str().split(" ")),
                    dep_graph,
                ).to_str()
            ],
        )
        for line in sorted_trivial_aux:
            statement_str = Statement.from_tokens(
                translate_sentence(mp, line.statement.to_str().split(" ")),
                dep_graph,
            ).to_str()
            trivial_items.append(f"{statement_str} [{dep_idx[statement_str]}]")
        if len(trivial_items) > 0:
            trivial = "<trivial> " + " ; ".join(trivial_items) + " ; </trivial> "
        else:
            trivial = ""
        return trivial

    @staticmethod
    def _generate_proof_section(mp, dep_idx, proof_steps, dep_graph: DependencyGraph):
        """Generate proof section"""
        proof = "<proof> "
        proof_steps_formatted = []
        for k, line in enumerate(proof_steps):
            if NUMERICAL_CHECK not in line.reason and IN_PREMISES not in line:
                proof_steps_formatted.append(
                    ProblemWorker._rediger_new_format(line, mp, dep_idx, dep_graph)
                )
        proof += " ; ".join(proof_steps_formatted) + " ; </proof>"
        return proof

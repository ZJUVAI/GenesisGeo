from __future__ import annotations

import os
import re
import string
import time
from collections import defaultdict
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import ray

from newclid.DDAR.build import DDAR
from newclid.algebraic_reasoning.algebraic_manipulator import AlgebraicManipulator
from newclid.configs import load_solver_config
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.formulations.clause import Clause, translate_sentence
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.problem import ProblemJGEX
from newclid.ddar_build_input import build_ddar_input
from newclid.numerical.geometries import PointNum
from newclid.predicates.collinearity import Coll
from newclid.predicates.congruence import Cong
from newclid.predicates.cyclic import Cyclic
from newclid.predicates.equal_angles import EqAngle
from newclid.predicates.equal_ratios import EqRatio
from newclid.predicates.midpoint import MidPoint
from newclid.predicates.parallelism import Para
from newclid.predicates.perpendicularity import Perp
from newclid.proof import ProofState
from newclid.statement import Statement

if TYPE_CHECKING:
    from newclid.formulations.rule import Rule


DEFAULT_DDAR_CONFIG = load_solver_config()

_SIMPLE_PREDICATES = {
    "perp": Perp.to_constructive,
    "para": Para.to_constructive,
    "cong": Cong.to_constructive,
    "midp": MidPoint.to_constructive,
    "coll": Coll.to_constructive,
    "cyclic": Cyclic.to_constructive,
    "eqratio": EqRatio.to_constructive,
}


def classify_build_exception(exc: Exception) -> str:
    message = str(exc)
    for fragment, label in [
        ("already used", "build_reused_point_error"),
        ("InvalidIntersectError", "build_numerical_error"),
        ("InvalidReduceError", "build_reduce_error"),
        ("PointTooCloseError", "build_point_too_close"),
        ("PointTooFarError", "build_point_too_far"),
        ("ConstructionError", "build_requirement_error"),
    ]:
        if fragment in message:
            return label
    return "build_definition_error"


def get_new_point_name(problem: ProblemJGEX) -> str:
    num_points = sum(len(clause.points) for clause in problem.constructions)
    letter_part = string.ascii_lowercase[num_points % 26]
    number_part = num_points // 26
    return f"{letter_part}{number_part - 1}" if number_part else letter_part


def try_dsl_to_constructions(content: str) -> str | None:
    try:
        points, premises = content.split(";")[0].split(" : ")
    except ValueError:
        return None
    point_names = points.strip().split()
    if len(point_names) != 1:
        return None
    point = point_names[0]

    premise_segments = [
        s.strip() for s in re.split(r"\s*\[\d+\]", premises) if s.strip()
    ]
    if len(premise_segments) > 2:
        return None
    if not premise_segments:
        return f"{point} = free {point}"

    constructions: list[str] = []
    for premise in premise_segments:
        parts = premise.split()
        if not parts or not parts[0].isalpha():
            return None
        try:
            result = translate_dsl_to_construction(point, parts[0], parts[1:])
        except (ValueError, TypeError):
            return None
        if result is None:
            return None
        constructions.append(result)
    construction = f"{point} = {', '.join(constructions)}"
    try:
        Clause.parse_line(construction)
    except ValueError:
        return None
    return construction


def try_full_aux_dsl_to_constructions(content: str) -> str | None:
    construction_parts: list[str] = []
    for segment in content.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        segment = re.sub(r"^x00\s+", "", segment)
        construction = try_dsl_to_constructions(segment)
        if construction is None:
            return None
        construction_parts.append(construction)
    if not construction_parts:
        return None
    constructions = "; ".join(construction_parts)
    try:
        Clause.parse_line(constructions)
    except ValueError:
        return None
    return constructions


def translate_dsl_to_construction(
    point: str, predicate: str, args: list[str]
) -> str | None:
    if point not in args:
        return None
    handler = _SIMPLE_PREDICATES.get(predicate)
    if handler is not None:
        return handler(point, tuple(args))
    if predicate != "eqangle":
        return f"{predicate} {' '.join(args)}"

    def _arrange(a: str, b: str, c: str, d: str) -> tuple[str, str, str] | None:
        if a == c:
            return (b, a, d)
        if a == d:
            return (b, a, c)
        if b == c:
            return (a, b, d)
        if b == d:
            return (a, b, c)
        return None

    a, b, c, d, e, f, g, h = args
    if len(set(args)) == 8:
        _map = {
            h: f"on_aline0 {h} {a} {b} {c} {d} {e} {f} {g}",
            g: f"on_aline0 {g} {a} {b} {c} {d} {e} {f} {h}",
            f: f"on_aline0 {f} {c} {d} {a} {b} {g} {h} {e}",
            e: f"on_aline0 {e} {c} {d} {a} {b} {g} {h} {f}",
            d: f"on_aline0 {d} {e} {f} {g} {h} {a} {b} {c}",
            c: f"on_aline0 {c} {e} {f} {g} {h} {a} {b} {d}",
            b: f"on_aline0 {b} {g} {h} {e} {f} {c} {d} {a}",
            a: f"on_aline0 {a} {g} {h} {e} {f} {c} {d} {b}",
        }
        if point in _map:
            return _map[point]

    if len({a, b, c, d}) == 4 and len({a, b, e, f}) == 3:
        a, b, c, d, e, f, g, h = a, b, e, f, c, d, g, h
    return EqAngle.to_constructive(point, _arrange(a, b, c, d) + _arrange(e, f, g, h))


def problem_to_dsl(problem: ProblemJGEX, defs: dict[str, DefinitionJGEX]) -> str:
    dep_idx: dict[Statement, str] = {}
    dep_graph = DependencyGraph(AlgebraicManipulator())
    grouped_statements: dict[str, list[Statement]] = defaultdict(list)

    for construction in problem.constructions:
        for constr_sentence in construction.sentences:
            cdef = defs[constr_sentence[0]]
            if len(constr_sentence) == len(cdef.declare):
                mapping = dict(zip(cdef.declare[1:], constr_sentence[1:]))
            else:
                points = tuple(point.split("@")[0] for point in construction.points)
                mapping = dict(zip(cdef.declare[1:], points + constr_sentence[1:]))
            for points, basics in cdef.basics:
                resolved_points = tuple(mapping[name] for name in points)
                if not basics:
                    grouped_statements[" ".join(resolved_points)] = []
                for basic in basics:
                    stmt = Statement.from_tokens(
                        translate_sentence(mapping, basic), dep_graph
                    )
                    grouped_statements[" ".join(resolved_points)].append(stmt)

    problem_parts: list[str] = []
    for key, statements in grouped_statements.items():
        rendered = key + " : "
        for stmt in statements:
            if stmt not in dep_idx:
                dep_idx[stmt] = f"{len(dep_idx):03d}"
            rendered += stmt.to_str() + f" [{dep_idx[stmt]}] "
        problem_parts.append(rendered.strip())

    goals = [Statement.from_tokens(goal, dep_graph).to_str() for goal in problem.goals]
    return (
        "<problem> "
        + " ; ".join(problem_parts)
        + " ? "
        + " ; ".join(goals)
        + " </problem>"
    )


def _extract_args(stmt) -> list[str]:
    return [str(pt) if isinstance(pt, Fraction) else pt.name for pt in stmt.args]


def extract_points(proof: ProofState) -> list[tuple[str, Any, Any]]:
    return [
        (name, pt.num.x, pt.num.y)
        for name, pt in proof.symbols_graph.name2node.items()
        if isinstance(pt.num, PointNum)
    ]


def extract_premises(proof: ProofState) -> list[tuple[str, list[str]]]:
    return [
        (stmt.predicate.NAME, _extract_args(stmt))
        for stmt in proof.dep_graph.hyper_graph
    ]


def extract_goals(proof: ProofState) -> list[tuple[str, list[str]]]:
    return [(stmt.predicate.NAME, _extract_args(stmt)) for stmt in proof.goals]


def run_ddar_c(proof: ProofState, ddar_config: dict[str, bool] | None = None) -> bool:
    config = DEFAULT_DDAR_CONFIG if ddar_config is None else ddar_config
    solved, _ = DDAR.run_ddar(
        "",
        extract_points(proof),
        extract_premises(proof),
        extract_goals(proof),
        500,
        config,
    )
    return solved


class BeamQueue:
    """Keep only the top-k nodes according to their scores."""

    def __init__(self, max_size: int = 512):
        self.queue: list[tuple[float, tuple[int, ...], int, Any]] = []
        self.max_size = max_size
        self.counter = 0

    @staticmethod
    def _sort_key(entry: tuple) -> tuple:
        score, stable_key, insertion_order, _ = entry
        return (-score, stable_key, insertion_order)

    def _sorted_entries(self):
        return sorted(self.queue, key=self._sort_key)

    def add(self, node: object, val: float, *, stable_key: tuple[int, ...]) -> None:
        self.queue.append((val, stable_key, self.counter, node))
        self.counter += 1
        if len(self.queue) > self.max_size:
            self.queue = self._sorted_entries()[: self.max_size]

    def __iter__(self):
        for val, _, _, node in self._sorted_entries():
            yield val, node

    def iter_entries(self):
        yield from self._sorted_entries()

    def __len__(self) -> int:
        return len(self.queue)

    def map_nodes(self, func) -> None:
        self.queue = [
            (val, sk, order, func(node)) for val, sk, order, node in self.queue
        ]


def build_problem_proof(problem, defs, *, max_attempts: int = 100) -> ProofState:
    return ProofState.build_problemJGEX(
        problemJGEX=problem,
        defsJGEX=defs,
        rng=np.random.default_rng(998244353),
        max_attempts=max_attempts,
        problem_path=None,
    )


def run_ddar_task(
    problem,
    defs,
    rules: list["Rule"] | None = None,
    *,
    return_proof: bool = False,
    ddar_config: dict[str, bool] | None = None,
):
    # Timings here measure work inside one remote task and can sum to more than
    # end-to-end wall time when many tasks run in parallel.
    eval_start = time.time()
    ddar_worker_id = f"{ray.util.get_node_ip_address()}:{os.getpid()}"
    build_start = eval_start

    try:
        build_start = time.time()
        points, premises, goals = build_ddar_input(
            problem,
            defs,
            np.random.default_rng(998244353),
            max_attempts=100,
            only_useful_points=False,
        )
        build_end = time.time()
        ddar_build_work_time_s = build_end - build_start
    except Exception as exc:
        now = time.time()
        result = {
            "status": "invalid",
            "elapsed_time": now - eval_start,
            "ddar_worker_id": ddar_worker_id,
            "ddar_started_at_unix_s": eval_start,
            "ddar_finished_at_unix_s": now,
            "ddar_build_started_at_unix_s": build_start,
            "ddar_build_finished_at_unix_s": now,
            "ddar_engine_started_at_unix_s": None,
            "ddar_engine_finished_at_unix_s": None,
            "ddar_build_work_time_s": now - build_start,
            "ddar_engine_work_time_s": 0.0,
            "error_type": classify_build_exception(exc),
            "error_message": str(exc),
        }
        if return_proof:
            result["proof"] = None
        return result

    try:
        ddar_start = time.time()
        config = DEFAULT_DDAR_CONFIG if ddar_config is None else ddar_config
        solved, _ = DDAR.run_ddar("", points, premises, goals, 500, config)
        ddar_end = time.time()
    except Exception as exc:
        now = time.time()
        result = {
            "status": "invalid",
            "elapsed_time": now - eval_start,
            "ddar_worker_id": ddar_worker_id,
            "ddar_started_at_unix_s": eval_start,
            "ddar_finished_at_unix_s": now,
            "ddar_build_started_at_unix_s": build_start,
            "ddar_build_finished_at_unix_s": build_end,
            "ddar_engine_started_at_unix_s": ddar_start,
            "ddar_engine_finished_at_unix_s": now,
            "ddar_build_work_time_s": ddar_build_work_time_s,
            "ddar_engine_work_time_s": now - ddar_start,
            "error_type": "engine_error",
            "error_message": str(exc),
        }
        if return_proof:
            result["proof"] = None
        return result

    now = time.time()
    result = {
        "status": "solved" if solved else "unsolved",
        "elapsed_time": now - eval_start,
        "ddar_worker_id": ddar_worker_id,
        "ddar_started_at_unix_s": eval_start,
        "ddar_finished_at_unix_s": now,
        "ddar_build_started_at_unix_s": build_start,
        "ddar_build_finished_at_unix_s": build_end,
        "ddar_engine_started_at_unix_s": ddar_start,
        "ddar_engine_finished_at_unix_s": ddar_end,
        "ddar_build_work_time_s": ddar_build_work_time_s,
        "ddar_engine_work_time_s": ddar_end - ddar_start,
    }
    if return_proof:
        result["proof"] = build_problem_proof(problem, defs)
    return result


@ray.remote(num_cpus=1)
def run_ddar_remote(
    problem,
    defs,
    rules: list["Rule"] | None = None,
    *,
    return_proof: bool = False,
    ddar_config: dict[str, bool] | None = None,
):
    return run_ddar_task(
        problem, defs, rules, return_proof=return_proof, ddar_config=ddar_config
    )

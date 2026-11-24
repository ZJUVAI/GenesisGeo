from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from typing import TYPE_CHECKING, Any

import ray

from newclid.agent.agents_interface import DeductiveAgent
from newclid.evaluation.search_runtime import (
    BeamQueue,
    run_ddar_c,
    run_ddar_remote,
    try_dsl_to_constructions,
    try_full_aux_dsl_to_constructions,
)
from newclid.formulations.problem import ProblemJGEX
from newclid.proof import ProofState

if TYPE_CHECKING:
    from newclid.formulations.rule import Rule


HTTP_WORKERS = max(1, int(os.environ.get("EVAL_HTTP_WORKERS", "16")))
RESPONSE_PREFIX = "<aux> x00"


def _rank(aux_dsl_scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(aux_dsl_scores.items(), key=lambda item: item[1], reverse=True)


class BaseAgent(DeductiveAgent, ABC):
    def __init__(
        self,
        *,
        decoding_size: int,
        beam_size: int,
        search_depth: int,
        search_version: str = "v1",
        ddar_config: dict[str, bool] | None = None,
        trace_writer=None,
    ) -> None:
        if search_version not in {"v1", "v2", "hybrid"}:
            raise ValueError(f"Unsupported search_version: {search_version}")
        self.problemJGEX: ProblemJGEX | None = None
        self.decoding_size = decoding_size
        self.beam_size = beam_size
        self.search_depth = search_depth
        self.search_version = search_version
        self.ddar_config = ddar_config
        self.trace_writer = trace_writer
        self._defs_ref: Any | None = None
        self._rules_ref: Any | None = None
        self._step = 0
        self._ddar_calls = 0
        self._ddar_wall = 0.0
        self._llm_calls = 0
        self._llm_wall = 0.0

    def step(self, proof: ProofState, rules: list["Rule"]) -> bool:
        del proof, rules
        return True

    def base_ddar_proof(self, proof: ProofState) -> ProofState:
        return proof

    def prepare_search(self, proof: ProofState) -> None:
        del proof

    @abstractmethod
    def build_request(
        self,
        *,
        mode: str,
        depth: int,
        request_id: str,
        problem: ProblemJGEX,
        aux_prefix: str,
        proof: ProofState,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def request_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_request_remote_kwargs(
        self,
        *,
        mode: str,
        depth: int,
        request_id: str,
        problem: ProblemJGEX,
        aux_prefix: str,
        proof: ProofState,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_request_from_remote_kwargs(self, kwargs: dict[str, Any]) -> Any:
        raise NotImplementedError

    def run(
        self, proof: ProofState, rules: list["Rule"], timeout: int = 3600
    ) -> dict[str, Any]:
        t0 = time.time()
        deadline = t0 + timeout
        self._step = 0
        self._ddar_calls = 0
        self._ddar_wall = 0.0
        self._llm_calls = 0
        self._llm_wall = 0.0
        self._max_pending = self._max_pending_ddar()

        for goal in proof.goals:
            if not goal.check_numerical():
                return self._infos(t0, False, f"{goal.pretty()} fails numerical check")

        base_proof = self.base_ddar_proof(proof)
        base_start = time.perf_counter()
        self._ddar_calls += 1
        if run_ddar_c(base_proof, self.ddar_config):
            self._ddar_wall += time.perf_counter() - base_start
            return self._infos(t0, True)
        self._ddar_wall += time.perf_counter() - base_start

        if self.problemJGEX is None:
            return self._infos(t0, False, "Missing problemJGEX.")

        self.prepare_search(base_proof)
        self._defs_ref = ray.put(base_proof.defs)
        self._rules_ref = ray.put(rules)

        modes = (
            ("v1", "v2") if self.search_version == "hybrid" else (self.search_version,)
        )
        error = "Tried but failed."
        try:
            for mode in modes:
                self._trace("search_mode", mode=mode)
                if time.time() >= deadline:
                    return self._infos(t0, False, "Timeout")
                solved, error = self._search(mode, base_proof, deadline)
                if solved:
                    return self._infos(t0, True)
            if time.time() >= deadline:
                return self._infos(t0, False, "Timeout")
            return self._infos(t0, False, error)
        except Exception as exc:
            return self._infos(t0, False, f"{type(exc).__name__}: {exc}")

    def _search(
        self, mode: str, proof: ProofState, deadline: float
    ) -> tuple[bool, str]:
        beam = BeamQueue(max_size=self.beam_size)
        beam.add(node=((), self.problemJGEX, ""), val=0.0, stable_key=())

        for depth in range(self.search_depth):
            self._step = depth + 1
            frontier = list(beam)
            self._trace(
                "depth_start", mode=mode, depth=depth, frontier_size=len(frontier)
            )
            if not frontier:
                self._trace("depth_end", mode=mode, depth=depth, next_frontier_size=0)
                break
            if time.time() >= deadline:
                return False, "Timeout"

            last_depth = depth == self.search_depth - 1
            next_beam = BeamQueue(max_size=self.beam_size)
            requests_list, context = self._build_requests(mode, depth, frontier, proof)
            if not requests_list:
                beam = next_beam
                self._trace("depth_end", mode=mode, depth=depth, next_frontier_size=0)
                continue

            pending: list[Any] = []
            meta: dict[Any, dict[str, Any]] = {}
            solved = False
            timed_out = False
            lm_start = time.time()
            last_lm_done = lm_start
            request_iter = iter(requests_list)

            executor = ThreadPoolExecutor(max_workers=HTTP_WORKERS)
            futures = {
                executor.submit(
                    self.request_completions,
                    {**request, "_deadline_unix_s": deadline},
                ): (request, time.time())
                for request in islice(request_iter, HTTP_WORKERS)
            }
            try:
                while futures and not solved:
                    remaining_s = deadline - time.time()
                    if remaining_s <= 0:
                        timed_out = True
                        break
                    done, _ = wait(
                        futures,
                        timeout=min(1.0, remaining_s),
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        continue

                    for future in done:
                        request, request_started = futures.pop(future)
                        self._llm_calls += 1
                        result: dict[str, Any] | None = None
                        try:
                            result = future.result()
                        except Exception as exc:
                            last_lm_done = time.time()
                            self._trace(
                                "lm_error",
                                mode=mode,
                                depth=depth,
                                request_id=request.get("request_id"),
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                                request_elapsed_s=round(
                                    last_lm_done - request_started, 6
                                ),
                            )
                        else:
                            last_lm_done = float(
                                result.get("completed_at_unix_s", time.time())
                            )
                            self._trace(
                                "lm_result",
                                mode=mode,
                                depth=depth,
                                request_id=result.get("request_id"),
                                candidate_count=len(result.get("aux_dsl_scores", {})),
                                request_elapsed_s=round(
                                    last_lm_done - request_started, 6
                                ),
                            )

                        if time.time() < deadline and not solved:
                            try:
                                next_request = next(request_iter)
                            except StopIteration:
                                pass
                            else:
                                futures[
                                    executor.submit(
                                        self.request_completions,
                                        {**next_request, "_deadline_unix_s": deadline},
                                    )
                                ] = (next_request, time.time())

                        if result is not None:
                            solved = self._submit(
                                result,
                                context,
                                pending,
                                meta,
                                next_beam,
                                last_depth,
                                deadline,
                                mode=mode,
                                depth=depth,
                            ) or self._collect(
                                pending,
                                meta,
                                next_beam,
                                last_depth,
                                deadline,
                                mode=mode,
                                depth=depth,
                                block=False,
                            )
                            if solved:
                                break
            finally:
                executor.shutdown(wait=not (solved or timed_out), cancel_futures=True)

            self._llm_wall += max(last_lm_done - lm_start, 0.0)
            if timed_out:
                self._cancel(pending)
                self._trace(
                    "depth_end",
                    mode=mode,
                    depth=depth,
                    next_frontier_size=len(next_beam),
                    solved=False,
                )
                return False, "Timeout"
            if not solved:
                solved = self._collect(
                    pending,
                    meta,
                    next_beam,
                    last_depth,
                    deadline,
                    mode=mode,
                    depth=depth,
                    block=True,
                )
            if solved:
                self._cancel(pending)
                self._trace(
                    "depth_end",
                    mode=mode,
                    depth=depth,
                    next_frontier_size=len(next_beam),
                    solved=True,
                )
                return True, ""

            beam = next_beam
            self._trace(
                "depth_end",
                mode=mode,
                depth=depth,
                next_frontier_size=len(beam),
                solved=False,
            )

        return False, "Tried but failed."

    def _build_requests(
        self,
        mode: str,
        depth: int,
        frontier: list[tuple[float, tuple[tuple[int, ...], ProblemJGEX, str]]],
        proof: ProofState,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray must be initialized before evaluation request build."
            )

        entries: list[dict[str, Any]] = []
        refs = []
        for prev_score, (path_key, problem, aux_prefix) in frontier:
            suffix = "root" if not path_key else "-".join(map(str, path_key))
            request_id = f"d{depth}_p{suffix}"
            entry = {
                "prev_score": prev_score,
                "path_key": path_key,
                "problem": problem,
                "request_id": request_id,
            }
            entries.append(entry)
            refs.append(
                self.build_request_from_remote_kwargs(
                    self.build_request_remote_kwargs(
                        mode=mode,
                        depth=depth,
                        request_id=request_id,
                        problem=problem,
                        aux_prefix=aux_prefix,
                        proof=proof,
                    )
                )
            )

        requests_list: list[dict[str, Any]] = []
        context: dict[str, dict[str, Any]] = {}
        for entry, ref in zip(entries, refs):
            request_id = entry["request_id"]
            try:
                request = ray.get(ref)
            except Exception as exc:
                self._trace(
                    "request_build_error",
                    mode=mode,
                    depth=depth,
                    request_id=request_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                continue
            requests_list.append(request)
            context[request_id] = {
                "prev_score": entry["prev_score"],
                "path_key": entry["path_key"],
                "problem": entry["problem"],
                "request": request,
            }
            self._trace(
                "lm_request",
                mode=mode,
                depth=depth,
                request_id=request_id,
                response_prefix=request.get("response_prefix"),
                new_point_name=request.get("new_point_name"),
            )
        return requests_list, context

    def _submit(
        self,
        result: dict[str, Any],
        context: dict[str, dict[str, Any]],
        pending: list[Any],
        meta: dict[Any, dict[str, Any]],
        next_beam: BeamQueue,
        last_depth: bool,
        deadline: float,
        *,
        mode: str,
        depth: int,
    ) -> bool:
        ctx = context[result["request_id"]]
        response_prefix = str(ctx["request"]["response_prefix"])
        aux_dsl_thinks = result.get("aux_dsl_thinks", {})

        for rank, (aux_dsl, score) in enumerate(
            _rank(result.get("aux_dsl_scores", {}))
        ):
            if not aux_dsl.startswith(response_prefix):
                continue
            aux_body = aux_dsl[len(response_prefix) :].strip()
            if bool(getattr(self, "think", False)):
                aux_construction = try_full_aux_dsl_to_constructions(aux_body)
            else:
                aux_construction = try_dsl_to_constructions(aux_body)
            model_think = aux_dsl_thinks.get(aux_dsl)
            self._trace(
                "candidate_parse",
                mode=mode,
                depth=depth,
                request_id=result["request_id"],
                candidate_rank=rank,
                aux_dsl=aux_dsl,
                parsed=aux_construction is not None,
                model_think=model_think,
            )
            if aux_construction is None:
                continue

            try:
                appended_problem = ctx["problem"].with_more_construction(
                    aux_construction
                )
                new_problem = appended_problem.renamed()
                new_constructions = new_problem.constructions[
                    len(ctx["problem"].constructions) :
                ]
                renamed_construction = "; ".join(str(c) for c in new_constructions)
            except Exception as exc:
                self._trace(
                    "candidate_build",
                    mode=mode,
                    depth=depth,
                    request_id=result["request_id"],
                    candidate_rank=rank,
                    construction_text=aux_construction,
                    built=False,
                    error_message=str(exc),
                )
                continue

            self._trace(
                "candidate_build",
                mode=mode,
                depth=depth,
                request_id=result["request_id"],
                candidate_rank=rank,
                construction_text=renamed_construction,
                raw_construction_text=aux_construction,
                built=True,
            )

            while len(pending) >= self._max_pending:
                if self._collect(
                    pending,
                    meta,
                    next_beam,
                    last_depth,
                    deadline,
                    mode=mode,
                    depth=depth,
                    block=True,
                ):
                    return True

            future = run_ddar_remote.options(max_retries=0).remote(
                new_problem,
                self._defs_ref,
                self._rules_ref,
                ddar_config=self.ddar_config,
            )
            self._ddar_calls += 1
            pending.append(future)
            meta[future] = {
                "prev_score": ctx["prev_score"],
                "path_key": ctx["path_key"],
                "rank": rank,
                "score": score,
                "problem": new_problem,
                "child_aux_prefix": aux_dsl[len("<aux>") :],
                "request_id": result["request_id"],
                "construction_text": renamed_construction,
                "raw_construction_text": aux_construction,
            }
            self._trace(
                "ddar_submit",
                mode=mode,
                depth=depth,
                request_id=result["request_id"],
                candidate_rank=rank,
                construction_text=renamed_construction,
                raw_construction_text=aux_construction,
            )
        return False

    def _collect(
        self,
        pending: list[Any],
        meta: dict[Any, dict[str, Any]],
        next_beam: BeamQueue,
        last_depth: bool,
        deadline: float,
        *,
        mode: str,
        depth: int,
        block: bool,
    ) -> bool:
        while pending:
            if block and time.time() >= deadline:
                self._cancel(pending)
                return False

            wait_start = time.perf_counter()
            done, remaining = ray.wait(
                pending, num_returns=1, timeout=1.0 if block else 0.0
            )
            self._ddar_wall += time.perf_counter() - wait_start
            if not done:
                if block:
                    continue
                return False

            pending[:] = remaining
            result = ray.get(done[0])
            info = meta.pop(done[0])
            self._trace(
                "ddar_result",
                mode=mode,
                depth=depth,
                request_id=info["request_id"],
                candidate_rank=info["rank"],
                status=result.get("status"),
                elapsed_time=result.get("elapsed_time"),
                construction_text=info.get("construction_text"),
                raw_construction_text=info.get("raw_construction_text"),
                error_type=result.get("error_type"),
                error_message=result.get("error_message"),
            )

            if result.get("status") == "solved":
                self._cancel(pending)
                return True

            if result.get("status") == "unsolved" and not last_depth:
                path_key = info["path_key"] + (info["rank"],)
                next_beam.add(
                    node=(path_key, info["problem"], info["child_aux_prefix"]),
                    val=float(info["prev_score"]) + float(info["score"]),
                    stable_key=path_key,
                )

            if not block:
                return False
        return False

    def _cancel(self, pending: list[Any]) -> None:
        for future in pending:
            try:
                ray.cancel(future, force=False)
            except Exception:
                pass
        pending.clear()

    def _max_pending_ddar(self) -> int:
        try:
            return max(1, 2 * int(ray.cluster_resources().get("CPU", 1)))
        except Exception:
            return 1

    def _trace(self, event: str, **payload: Any) -> None:
        if self.trace_writer is not None:
            self.trace_writer.log(event, **payload)

    def _infos(
        self, t0: float, success: bool, error: str | None = None
    ) -> dict[str, Any]:
        infos: dict[str, Any] = {
            "runtime": time.time() - t0,
            "success": success,
            "steps": self._step,
            "llm_calls": self._llm_calls,
            "ddar_calls": self._ddar_calls,
            "llm_real_time_s": round(self._llm_wall, 3),
            "ddar_real_time_s": round(self._ddar_wall, 3),
        }
        if error:
            infos["error"] = error
        return infos

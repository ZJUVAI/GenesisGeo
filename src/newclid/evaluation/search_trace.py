from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "item"


def get_git_commit(repo_root: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root or "."),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_attempt_key(
    request_id: str | None, candidate_rank: int | None, node_id: int | None
) -> str:
    if request_id is not None and candidate_rank is not None:
        return f"{request_id}:{candidate_rank}"
    if node_id is not None:
        return f"node:{node_id}"
    return "unknown"


class AttemptWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        self._fp.write("\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class AttemptAggregator:
    def __init__(self, writer: AttemptWriter) -> None:
        self.writer = writer
        self.pending: dict[str, dict[str, Any]] = {}
        self.node_to_key: dict[int, str] = {}

    def _resolve_key(self, record: dict[str, Any]) -> str:
        key = record.get("attempt_key") or build_attempt_key(
            record.get("request_id"),
            record.get("candidate_rank"),
            record.get("node_id"),
        )
        return key

    def _common(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            k: record.get(k)
            for k in (
                "run_id",
                "route",
                "agent",
                "problem_name",
                "problem_index",
                "ts_utc",
                "elapsed_s",
            )
        }

    def _base_attempt(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._common(record),
            "attempt_type": "base_ddar",
            "attempt_key": record.get("attempt_key")
            or build_attempt_key(None, None, record.get("node_id")),
            "attempt_id": record.get("node_id"),
            "node_id": record.get("node_id"),
            "parent_node_id": record.get("parent_node_id"),
            "depth": record.get("depth"),
            "request_id": None,
            "candidate_rank": None,
            "beam_score_before": None,
            "beam_score_after": None,
            "status": None,
            "decision": None,
            "ddar_status": None,
            "ddar_elapsed_time": None,
            "ddar_build_work_time_s": None,
            "ddar_engine_work_time_s": None,
            "raw_aux_text": None,
            "construction_text": None,
            "model_think": None,
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        }

    def _candidate_attempt(self, record: dict[str, Any]) -> dict[str, Any]:
        request_id = record.get("request_id")
        candidate_rank = record.get("candidate_rank")
        node_id = record.get("node_id")
        return {
            **self._common(record),
            "attempt_type": "candidate",
            "attempt_key": record.get("attempt_key")
            or build_attempt_key(request_id, candidate_rank, node_id),
            "attempt_id": node_id,
            "node_id": node_id,
            "parent_node_id": record.get("parent_node_id"),
            "depth": record.get("depth"),
            "request_id": request_id,
            "candidate_rank": candidate_rank,
            "beam_score_before": record.get("beam_score_before"),
            "beam_score_after": record.get("beam_score_after"),
            "status": record.get("decision"),
            "decision": record.get("decision"),
            "ddar_status": None,
            "ddar_elapsed_time": None,
            "ddar_build_work_time_s": None,
            "ddar_engine_work_time_s": None,
            "raw_aux_text": record.get("raw_aux_text"),
            "construction_text": record.get("construction_text"),
            "model_think": record.get("model_think"),
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        }

    def _register_node(self, record: dict[str, Any], attempt_key: str) -> None:
        node_id = record.get("node_id")
        if node_id is not None:
            self.node_to_key[node_id] = attempt_key

    def process(self, record: dict[str, Any]) -> None:
        event = record["event"]

        if event == "base_ddar":
            key = self._resolve_key(record)
            attempt = self._base_attempt(record)
            attempt["attempt_key"] = key
            self.pending[key] = attempt
            self._register_node(record, key)
            return

        if event == "candidate_transition":
            key = self._resolve_key(record)
            attempt = self.pending.pop(key, None)
            if attempt is None:
                attempt = self._candidate_attempt(record)
            else:
                attempt.update(
                    {
                        **self._candidate_attempt(record),
                        "ddar_status": attempt.get("ddar_status"),
                        "ddar_elapsed_time": attempt.get("ddar_elapsed_time"),
                        "raw_aux_text": record.get(
                            "raw_aux_text", attempt.get("raw_aux_text")
                        ),
                        "construction_text": record.get(
                            "construction_text", attempt.get("construction_text")
                        ),
                        "model_think": record.get(
                            "model_think", attempt.get("model_think")
                        ),
                        "error_type": attempt.get("error_type"),
                        "error_message": attempt.get("error_message"),
                    }
                )
            self._register_node(record, attempt["attempt_key"])
            if record.get("decision") == "ddar_submitted":
                self.pending[attempt["attempt_key"]] = attempt
            else:
                self.writer.write(attempt)
            return

        if event == "ddar_result":
            key = record.get("attempt_key")
            if key is None and record.get("node_id") is not None:
                key = self.node_to_key.get(record["node_id"])
            if key is None:
                key = build_attempt_key(None, None, record.get("node_id"))
            attempt = self.pending.pop(key, self._base_attempt(record))
            attempt["attempt_key"] = key
            for field in (
                "ddar_status",
                "ddar_elapsed_time",
                "ddar_build_work_time_s",
                "ddar_engine_work_time_s",
                "error_type",
                "error_message",
            ):
                src = {"ddar_status": "status"}.get(field, field)
                attempt[field] = record.get(src)
            attempt["status"] = record.get("status")
            status = record.get("status")
            if attempt.get("attempt_type") == "base_ddar":
                attempt["decision"] = status
                self.writer.write(attempt)
                return
            if status in ("solved", "invalid"):
                attempt["decision"] = status
                self.writer.write(attempt)
            else:
                attempt["decision"] = "unsolved"
                self.pending[key] = attempt

    def close(self) -> None:
        for key in sorted(self.pending):
            self.writer.write(self.pending[key])
        self.pending.clear()
        self.writer.close()


class TraceWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        route: str,
        agent: str,
        problem_name: str,
        problem_index: int,
        start_time: float,
        attempts_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.route = route
        self.agent = agent
        self.problem_name = problem_name
        self.problem_index = problem_index
        self.start_time = start_time
        self._fp = open(self.path, "a", encoding="utf-8")
        self._attempt_aggregator = (
            AttemptAggregator(AttemptWriter(attempts_path))
            if attempts_path is not None
            else None
        )

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "run_id": self.run_id,
            "route": self.route,
            "agent": self.agent,
            "problem_name": self.problem_name,
            "problem_index": self.problem_index,
            "ts_utc": utc_now_iso(),
            "elapsed_s": round(time.time() - self.start_time, 6),
            **payload,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        self._fp.write("\n")
        self._fp.flush()
        if self._attempt_aggregator is not None:
            self._attempt_aggregator.process(record)

    def close(self) -> None:
        if self._attempt_aggregator is not None:
            self._attempt_aggregator.close()
        self._fp.close()


class TraceRun:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        route: str,
        agent: str,
        dataset_path: str | Path,
        model_path: str | list[str],
        params: dict[str, Any],
        run_name: str | None = None,
        run_timestamp: str | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        dataset_slug = sanitize_filename(Path(dataset_path).stem)
        model_repr = (
            "__".join(str(x) for x in model_path)
            if isinstance(model_path, list)
            else str(model_path)
        )
        model_slug = sanitize_filename(Path(model_repr).name or model_repr)
        if run_name is None:
            run_name = f"{sanitize_filename(route)}_{dataset_slug}_{model_slug}"
        timestamp = run_timestamp or timestamp_slug()
        self.run_id = f"{sanitize_filename(run_name)}_{timestamp}"
        self.run_dir = Path(root_dir) / self.run_id
        self.problem_dir = self.run_dir / "problems"
        self.attempts_dir = self.run_dir / "attempts"
        self.problem_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": self.run_id,
            "route": route,
            "agent": agent,
            "dataset_path": str(dataset_path),
            "model_path": model_path,
            "git_commit": get_git_commit(repo_root),
            "argv": sys.argv,
            "trace_outputs": ["raw_events"],
            **params,
        }
        with open(self.run_dir / "run_meta.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2, sort_keys=True)
            fp.write("\n")

    def create_problem_writer(
        self,
        *,
        problem_index: int,
        problem_name: str,
        route: str,
        agent: str,
        start_time: float,
    ) -> TraceWriter:
        filename = f"{problem_index:04d}_{sanitize_filename(problem_name)}.jsonl"
        return TraceWriter(
            self.problem_dir / filename,
            run_id=self.run_id,
            route=route,
            agent=agent,
            problem_name=problem_name,
            problem_index=problem_index,
            start_time=start_time,
            attempts_path=None,
        )

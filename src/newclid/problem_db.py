from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from newclid.formulations.problem import ProblemJGEX


# The on-disk layout is intentionally small:
# - one directory per base problem
# - one index.json for fast strict-key membership checks
# - one jsonl file per result category for append/merge-friendly records
RESULT_CATEGORIES = ("solved", "unsolved", "invalid")
RESULT_FILES = {category: f"{category}.jsonl" for category in RESULT_CATEGORIES}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dataset_name_from_path(problems_path: str | Path) -> str:
    return Path(problems_path).stem


def normalize_problem(problem: ProblemJGEX) -> str:
    return str(problem)


def normalize_aux(base_problem: ProblemJGEX, augmented_problem: ProblemJGEX) -> str:
    # Only hash the newly added constructions, not the whole augmented problem.
    aux_constructions = augmented_problem.constructions[
        len(base_problem.constructions) :
    ]
    return "; ".join(str(clause) for clause in aux_constructions)


def build_problem_key(problem: ProblemJGEX) -> str:
    return sha256_text(normalize_problem(problem))


def slugify_problem_name(problem_name: str, problem_key: str) -> str:
    cleaned = problem_name.lower().replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r'[\/\\:\*\?"<>\|]+', "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or f"problem_{problem_key[:8]}"


def get_git_commit(repo_root: str | Path | None = None) -> str:
    cwd = Path(repo_root or ".")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def classify_build_exception(exc: Exception) -> str:
    message = str(exc)
    if "InvalidIntersectError" in message:
        return "build_numerical_error"
    if "InvalidReduceError" in message:
        return "build_reduce_error"
    if "PointTooCloseError" in message:
        return "build_point_too_close"
    if "PointTooFarError" in message:
        return "build_point_too_far"
    if "ConstructionError" in message:
        return "build_requirement_error"
    if "ValueError" in message:
        return "build_definition_error"
    return "build_definition_error"


def summarize_problem_db_runtime(
    runtime: ProblemDBRuntime | None,
) -> dict[str, Any] | None:
    if runtime is None:
        return None
    return {
        "cache_hits": dict(runtime.cache_hits),
        "new_records": {
            category: len(runtime.pending_records[category])
            for category in RESULT_CATEGORIES
        },
    }


def _default_index() -> dict[str, dict[str, bool]]:
    return {category: {} for category in RESULT_CATEGORIES}


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            strict_key = record.get("strict_key")
            if strict_key:
                records[strict_key] = record
    return records


def _write_jsonl(path: Path, records: dict[str, dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for strict_key in sorted(records):
            f.write(json.dumps(records[strict_key], ensure_ascii=False, sort_keys=True))
            f.write("\n")


@dataclass(frozen=True)
class ProblemDBLookup:
    # normalized_aux is stored for writing records; strict_key is the lookup key.
    normalized_aux: str | None = None
    strict_key: str | None = None
    hit_category: str | None = None

    @property
    def is_enabled(self) -> bool:
        return self.strict_key is not None


@dataclass
class ProblemDBRuntime:
    # Runtime only handles lookup and buffering of records produced in the current run.
    # It does not touch the filesystem beyond loading the existing index.
    db_root: str | Path
    problems_path: str | Path
    base_problem: ProblemJGEX
    existing_index: dict[str, dict[str, bool]] = field(default_factory=_default_index)
    pending_records: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: {category: {} for category in RESULT_CATEGORIES}
    )
    cache_hits: dict[str, int] = field(
        default_factory=lambda: {category: 0 for category in RESULT_CATEGORIES}
    )

    def __post_init__(self) -> None:
        self.db_root = str(self.db_root)
        self.problems_path = str(self.problems_path)
        self.dataset_name = dataset_name_from_path(self.problems_path)
        self.normalized_problem = normalize_problem(self.base_problem)
        self.problem_key = build_problem_key(self.base_problem)
        self.problem_dirname = self._resolve_problem_dirname()
        # Merge any caller-provided in-memory index with the persisted index on disk.
        loaded_index = self._load_existing_index()
        for category in RESULT_CATEGORIES:
            loaded_index[category].update(self.existing_index.get(category, {}))
        self.existing_index = loaded_index

    def problem_meta(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "problem_name": self.base_problem.name,
            "problem_dirname": self.problem_dirname,
            "problem_file": self.problems_path,
            "normalized_problem": self.normalized_problem,
            "problem_key": self.problem_key,
        }

    def lookup_problem(self, augmented_problem: ProblemJGEX) -> ProblemDBLookup:
        # strict_key is derived from normalized_aux, so the same aux text maps to one cache key.
        normalized_aux, strict_key = self.make_aux_state(augmented_problem)
        return ProblemDBLookup(
            normalized_aux=normalized_aux,
            strict_key=strict_key,
            hit_category=self.lookup(strict_key),
        )

    def make_aux_state(self, augmented_problem: ProblemJGEX) -> tuple[str, str]:
        normalized_aux = normalize_aux(self.base_problem, augmented_problem)
        return normalized_aux, sha256_text(normalized_aux)

    def lookup(self, strict_key: str) -> str | None:
        # Check buffered records first so the current process can reuse results immediately.
        for category in RESULT_CATEGORIES:
            if strict_key in self.pending_records[category]:
                self.cache_hits[category] += 1
                return category
            if self.existing_index.get(category, {}).get(strict_key):
                self.cache_hits[category] += 1
                return category
        return None

    def record_ddar_result(
        self, lookup: ProblemDBLookup, ddar_result: dict[str, Any]
    ) -> None:
        if not lookup.is_enabled:
            return

        # Store the minimal payload needed for future cache hits and lightweight analysis.
        category = ddar_result["status"]
        record: dict[str, Any] = {
            "strict_key": lookup.strict_key,
            "normalized_aux": lookup.normalized_aux,
        }
        if category == "invalid":
            record["error_type"] = ddar_result.get(
                "error_type", "build_definition_error"
            )
            if ddar_result.get("error_message"):
                record["error_message"] = ddar_result["error_message"]
        else:
            record["elapsed_time"] = ddar_result["elapsed_time"]
        self.pending_records[category][lookup.strict_key] = record

    def export_payload(self) -> dict[str, Any]:
        return {
            "meta": self.problem_meta(),
            "records": {
                category: list(self.pending_records[category].values())
                for category in RESULT_CATEGORIES
            },
            "cache_hits": dict(self.cache_hits),
        }

    def _dataset_dir(self) -> Path:
        return Path(self.db_root) / self.dataset_name

    def _resolve_problem_dirname(self) -> str:
        dataset_dir = self._dataset_dir()
        preferred = slugify_problem_name(self.base_problem.name, self.problem_key)
        preferred_dir = dataset_dir / preferred
        if preferred_dir.exists():
            meta = _load_json(preferred_dir / "meta.json", {})
            if meta.get("problem_key") == self.problem_key:
                return preferred
            return f"{preferred}_{self.problem_key[:8]}"
        fallback = dataset_dir / f"{preferred}_{self.problem_key[:8]}"
        if fallback.exists():
            meta = _load_json(fallback / "meta.json", {})
            if meta.get("problem_key") == self.problem_key:
                return fallback.name
        return preferred

    def _load_existing_index(self) -> dict[str, dict[str, bool]]:
        index_path = self._dataset_dir() / self.problem_dirname / "index.json"
        index = _load_json(index_path, _default_index())
        for category in RESULT_CATEGORIES:
            index.setdefault(category, {})
        return index


class ProblemDBWriter:
    # Writer is the only component that mutates the on-disk database.
    def __init__(self, db_root: str | Path, repo_root: str | Path | None = None):
        self.db_root = Path(db_root)
        self.commit = get_git_commit(repo_root)

    def write_payload(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        problem_dir = self._ensure_problem_dir(payload["meta"])
        self._write_records(problem_dir, payload.get("records", {}))

    def _ensure_problem_dir(self, meta: dict[str, Any]) -> Path:
        problem_dir = self.db_root / meta["dataset_name"] / meta["problem_dirname"]
        problem_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta(problem_dir, meta)
        return problem_dir

    def _write_meta(self, problem_dir: Path, meta: dict[str, Any]) -> None:
        meta_path = problem_dir / "meta.json"
        existing = _load_json(meta_path, {})
        now = utc_now_iso()
        _write_json(
            meta_path,
            {
                **meta,
                "created_commit": existing.get("created_commit", self.commit),
                "updated_commit": self.commit,
                "created_at": existing.get("created_at", now),
                "updated_at": now,
            },
        )
        index_path = problem_dir / "index.json"
        index = _load_json(index_path, _default_index())
        # Ensure all categories are always present even if some files are still empty.
        for category in RESULT_CATEGORIES:
            index.setdefault(category, {})
        _write_json(index_path, index)

    def _write_records(
        self, problem_dir: Path, records_by_category: dict[str, list[dict[str, Any]]]
    ) -> None:
        index_path = problem_dir / "index.json"
        index = _load_json(index_path, _default_index())
        loaded_records = {
            category: _load_jsonl_map(problem_dir / RESULT_FILES[category])
            for category in RESULT_CATEGORIES
        }

        for category in RESULT_CATEGORIES:
            for record in records_by_category.get(category, []):
                strict_key = record["strict_key"]
                # In the current evaluation flow, existing strict_keys are normally not re-run
                # through DDAR, so category transitions should be rare. We still enforce this
                # invariant defensively to keep the on-disk database consistent if old data,
                # manual edits, or future pipeline changes introduce conflicting states.
                self._remove_conflicting_records(
                    strict_key, category, index, loaded_records
                )
                loaded_records[category][strict_key] = self._merge_record(
                    loaded_records[category].get(strict_key),
                    record,
                )
                index.setdefault(category, {})[strict_key] = True

        for category in RESULT_CATEGORIES:
            _write_jsonl(problem_dir / RESULT_FILES[category], loaded_records[category])
        _write_json(index_path, index)

    def _remove_conflicting_records(
        self,
        strict_key: str,
        category: str,
        index: dict[str, dict[str, bool]],
        loaded_records: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        for other in RESULT_CATEGORIES:
            if other == category:
                continue
            index.setdefault(other, {}).pop(strict_key, None)
            loaded_records[other].pop(strict_key, None)

    def _merge_record(
        self,
        existing: dict[str, Any] | None,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        # Current callers usually write each strict_key once per category, so this merge path
        # mainly acts as a defensive "last write wins for known fields" safeguard.
        if existing is None:
            return dict(record)
        merged = dict(existing)
        merged["normalized_aux"] = record["normalized_aux"]
        if "elapsed_time" in record:
            merged["elapsed_time"] = record["elapsed_time"]
        if "error_type" in record:
            merged["error_type"] = record["error_type"]
        if "error_message" in record:
            merged["error_message"] = record["error_message"]
        return merged

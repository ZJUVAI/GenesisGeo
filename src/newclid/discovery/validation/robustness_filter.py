"""
Robustness pre-filter for Part 1.

Strips coordinates from fl_problem, rebuilds with a fresh random seed, and
numerically verifies the goal.  Records whose goal fails the numerical check
are discarded — their original proof was likely a coordinate- dependent
coincidence rather than a robust geometric truth.

Groups records by seed (same seed = same construction, different goals) so
each unique construction is only rebuilt once.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from tqdm import tqdm


def _strip_coords(fl_problem: str) -> str:
    """Remove @coord annotations from an fl_problem string."""
    return re.sub(r"@-?[0-9.]+(?:_-?[0-9.]+)?", "", fl_problem)


def _parse_goals(fl_problem: str) -> list[str]:
    """Extract semicolon-separated goal predicates from fl_problem."""
    if "?" not in fl_problem:
        return []
    goal_text = fl_problem.split("?", 1)[1].strip()
    return [g.strip() for g in goal_text.split(";") if g.strip()]


def filter_records(
    records: list[dict[str, Any]],
    rebuild_seed: int = 999983,
    max_attempts: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter records by numerical robustness.

    Parameters
    ----------
    records : list[dict]
        Raw JSONL records with ``fl_problem`` and ``seed`` fields.
    rebuild_seed : int
        Seed used for the rebuild (must differ from any original seed).
    max_attempts : int
        Max build attempts for the GeometricSolverBuilder.

    Returns
    -------
    (kept, stats) : (list[dict], dict)
        ``kept`` contains records that passed; ``stats`` has
        ``total``, ``kept``, ``build_fail``, ``goal_fail``.
    """
    # Group by seed: all records with the same seed share the same
    # construction, so we only need to rebuild once per seed group.
    groups: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["seed"]].append(rec)

    # Lazy imports — avoid dragging in heavy deps at module level
    from newclid.api import GeometricSolverBuilder
    from newclid.dependencies.symbols import Point
    from newclid.discovery.validation.counterexample_search import evaluate_predicate

    kept: list[dict] = []
    stats = {"total": len(records), "kept": 0, "build_fail": 0, "goal_fail": 0}

    for seed, group in tqdm(groups.items(), desc="[robustness]", unit="seed"):
        # Use the first record's fl_problem as the construction template
        fp_raw = group[0].get("fl_problem", "")
        constr = _strip_coords(fp_raw).split("?")[0].strip()
        if not constr:
            stats["build_fail"] += len(group)
            continue

        # Rebuild once for this seed group
        try:
            builder = GeometricSolverBuilder(seed=rebuild_seed).load_problem_from_txt(
                constr
            )
            solver = builder.build(max_attempts=max_attempts)
        except Exception:
            stats["build_fail"] += len(group)
            continue

        # Collect point coordinates
        pts: dict[str, Any] = {}
        for node in solver.proof.symbols_graph.nodes_of_type(Point):
            pts[node.name] = node.num

        # Check each record's goal(s)
        for rec in group:
            fp_rec = _strip_coords(rec.get("fl_problem", ""))
            goals = _parse_goals(fp_rec)
            if not goals:
                stats["goal_fail"] += 1
                continue

            all_ok = True
            for goal_str in goals:
                tokens = goal_str.split()
                if not tokens:
                    continue
                ok, _viol = evaluate_predicate(tokens[0], tokens[1:], pts)
                if not ok:
                    all_ok = False
                    break

            if all_ok:
                kept.append(rec)
                stats["kept"] += 1
            else:
                stats["goal_fail"] += 1

    return kept, stats


def filter_file(
    input_path: str,
    output_path: str,
    rebuild_seed: int = 999983,
    max_attempts: int = 100,
    n_workers: int = 1,
    limit: int | None = None,
) -> dict[str, int]:
    """Filter a JSONL file by numerical robustness.

    Reads ``input_path``, writes surviving records to ``output_path``.
    When ``n_workers > 1``, uses Ray for parallelism across seed groups.

    Returns
    -------
    stats : dict
        ``total``, ``kept``, ``build_fail``, ``goal_fail``.
    """
    # Read all records
    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    if n_workers <= 1:
        kept, stats = filter_records(records, rebuild_seed, max_attempts)
    else:
        # Ray parallel: split by seed groups across workers
        import ray
        from newclid.discovery.reduction.parallel import ensure_ray

        ensure_ray(n_workers)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=n_workers)

        # Group and split
        groups: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            groups[rec.get("seed", 0)].append(rec)

        seed_list = list(groups.items())
        chunk_size = max(1, len(seed_list) // n_workers)
        chunks = [
            seed_list[i : i + chunk_size] for i in range(0, len(seed_list), chunk_size)
        ]

        @ray.remote(num_cpus=1)
        def _filter_chunk(chunk, _rebuild_seed, _max_attempts):
            chunk_records = []
            for _seed, group in chunk:
                chunk_records.extend(group)
            return filter_records(chunk_records, _rebuild_seed, _max_attempts)

        futures = [_filter_chunk.remote(c, rebuild_seed, max_attempts) for c in chunks]
        results = ray.get(futures)

        kept = []
        stats = {"total": len(records), "kept": 0, "build_fail": 0, "goal_fail": 0}
        for k, s in results:
            kept.extend(k)
            for key in ("kept", "build_fail", "goal_fail"):
                stats[key] += s.get(key, 0)

        ray.shutdown()

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats

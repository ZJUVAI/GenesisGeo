"""Stream, deduplicate, and certify rules with bounded worker queues."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from itertools import groupby, islice
from pathlib import Path
import json
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

from newclid.configs import load_solver_config
from newclid.process_workers import Workers, configure_runtime

from .certification import certify
from .compression import Rule, compress_rule_bank, split
from .extraction import extract
from .rules import split_rule_text


def publish_rules(
    output,
    certified,
    bounded_coordinates,
    *,
    algebraic_compression=True,
    statistics=None,
):
    counts = Counter(rule.name for rule in certified)
    compressed = compress_rule_bank(
        [rule for rule in certified if counts[rule.name] == 1],
        allow_aliased_points=load_solver_config()["allow_duplicate_points"],
        bounded_coordinates=bounded_coordinates,
        algebraic_compression=algebraic_compression,
        statistics=statistics,
    )
    by_id = {rule.name: rule for rule in compressed}
    exported = 0
    with TemporaryDirectory(prefix=".export-", dir=output) as scratch:
        pipe_path = Path(scratch) / "rules_discovered.txt"
        with pipe_path.open("w") as pipe:
            for original in certified:
                rule = (
                    original if counts[original.name] > 1 else by_id.get(original.name)
                )
                if rule is not None:
                    pipe.write(rule.pipe() + "\n")
                    exported += 1
        pipe_path.replace(output / "rules_discovered.txt")
    return exported


def extract_candidates(source, output, limit, stats, timing):
    with (
        TemporaryDirectory(prefix=".index-", dir=output) as scratch,
        closing(sqlite3.connect(f"{scratch}/dedup.sqlite")) as database,
        source.open() as stream,
    ):
        database.execute("CREATE TABLE rules (text TEXT PRIMARY KEY) WITHOUT ROWID")
        for line, raw in enumerate(islice(stream, limit), 1):
            started = perf_counter()
            stats["proofs"] += 1
            fresh = []
            try:
                for row in extract(json.loads(raw), line):
                    stats["occurrences"] += 1
                    if database.execute(
                        "INSERT OR IGNORE INTO rules VALUES (?)", (row["rule_text"],)
                    ).rowcount:
                        stats["candidates"] += 1
                        fresh.append(row)
            except (ValueError, IndexError, KeyError) as error:
                raise ValueError(f"Invalid proof on line {line}: {error}") from error
            if line % 4096 == 0:
                database.commit()
            timing["extraction_and_dedup"] += perf_counter() - started
            yield from fresh


def read_candidates(source, limit, stats):
    with source.open() as stream:
        for line in islice(stream, limit):
            stats["candidates"] += 1
            yield json.loads(line)


def _batches(rows):
    def seed_key(row):
        return row["seed"] if row.get("seed") is not None else row["rule_id"]

    for _, siblings in groupby(rows, key=seed_key):
        while window := list(islice(siblings, 48)):
            window.sort(key=lambda row: row["rule_text"].split(" => ", 1)[0])
            for offset in range(0, len(window), 6):
                yield window[offset : offset + 6]


def _schedule(rows, workers, rule_timeout, singular_timeout):
    children = Workers(limit=32)
    pending = _batches(rows)

    def run_batch(batch):
        return [
            children.call(certify, (row, singular_timeout), rule_timeout)
            for row in batch
        ]

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            active = {
                executor.submit(run_batch, batch): batch
                for batch in islice(pending, workers)
            }
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    batch = active.pop(future)
                    results = future.result()
                    following = next(pending, None)
                    if following is not None:
                        active[executor.submit(run_batch, following)] = following
                    for row, worker in zip(batch, results, strict=True):
                        if worker["status"] == "error":
                            raise RuntimeError(
                                f"Rule {row['rule_id']}: {worker['error']}"
                            )
                        yield worker.get("output", worker)
    finally:
        children.close()
        pending.close()


def run(
    source,
    output,
    *,
    workers=16,
    limit=None,
    rule_timeout=6,
    singular_timeout=1,
    candidate_input=False,
    bounded_coordinates=False,
    algebraic_compression=True,
):
    configure_runtime(
        "newclid.discovery_experimental.certification", "newclid.api", "flint"
    )
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    stats, timing, outcomes, worker_timing, backends = (Counter() for _ in range(5))
    rows = (
        read_candidates(source, limit, stats)
        if candidate_input
        else extract_candidates(
            source,
            output,
            limit,
            stats,
            timing,
        )
    )
    certified = []
    with closing(_schedule(rows, workers, rule_timeout, singular_timeout)) as jobs:
        for result in jobs:
            status = result["status"]
            outcomes[status] += 1
            worker_timing.update(result.get("timings", {}))
            if status == "certified":
                row = result["rule"]
                backends[result["backend"]] += 1
                premises, conclusion = split_rule_text(row["rule_text"])
                certified.append(
                    Rule(
                        row["rule_id"],
                        tuple(premises),
                        split(conclusion),
                        split(row["guards"]),
                    )
                )
    compression_start = perf_counter()
    compression = {}
    exported = publish_rules(
        output,
        certified,
        bounded_coordinates,
        algebraic_compression=algebraic_compression,
        statistics=compression,
    )
    return {
        "input": str(source.resolve()),
        "workers": workers,
        "rule_timeout": rule_timeout,
        "singular_timeout": singular_timeout,
        "source": dict(stats),
        "extraction_seconds": timing["extraction_and_dedup"],
        "worker_phase_seconds": dict(worker_timing),
        "certified": outcomes["certified"],
        "exported": exported,
        "compression_seconds": perf_counter() - compression_start,
        "compression": compression,
        "outcomes": dict(outcomes),
        "certified_backends": dict(backends),
        "seconds": perf_counter() - started,
    }

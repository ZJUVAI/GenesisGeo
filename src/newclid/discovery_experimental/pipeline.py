"""Stream, deduplicate, and certify rules with bounded worker queues."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from itertools import groupby, islice
import json
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

from .certification import certify
from .extraction import extract
from .rules import to_pipe_format
from newclid.process_workers import Workers, configure_runtime


def emit(stream, row):
    stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def extract_candidates(source, output, limit, stats, timing):
    with (
        TemporaryDirectory(prefix=".index-", dir=output) as scratch,
        closing(sqlite3.connect(f"{scratch}/dedup.sqlite")) as database,
        (output / "candidates.jsonl").open("w") as candidates,
        (output / "origins.jsonl").open("w") as origins,
        source.open() as stream,
    ):
        database.execute("CREATE TABLE rules (text TEXT PRIMARY KEY) WITHOUT ROWID")
        for line, raw in enumerate(islice(stream, limit), 1):
            started = perf_counter()
            stats["proofs"] += 1
            try:
                extracted = list(extract(json.loads(raw), line))
            except (ValueError, IndexError, KeyError) as error:
                raise ValueError(f"Invalid proof on line {line}: {error}") from error
            fresh = []
            for row in extracted:
                stats["occurrences"] += 1
                emit(
                    origins,
                    {
                        key: row[key]
                        for key in ("rule_id", "seed", "line", "proof_step")
                    },
                )
                if database.execute(
                    "INSERT OR IGNORE INTO rules VALUES (?)", (row["rule_text"],)
                ).rowcount:
                    stats["candidates"] += 1
                    emit(candidates, row)
                    fresh.append(row)
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
                        yield row, worker.get("output", worker)
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
    bank_path = output / ".rules.jsonl.pending"
    pipe_path = output / ".rules.pipe.txt.pending"
    with (
        bank_path.open("w") as bank,
        pipe_path.open("w") as pipe,
        (output / "failed.jsonl").open("w") as failures,
        closing(_schedule(rows, workers, rule_timeout, singular_timeout)) as jobs,
    ):
        for row, result in jobs:
            status = result["status"]
            outcomes[status] += 1
            worker_timing.update(result.get("timings", {}))
            if status == "certified":
                rule = result["rule"]
                backends[result["backend"]] += 1
                emit(bank, rule)
                pipe.write(
                    to_pipe_format(rule["rule_id"], rule["rule_text"], rule["guards"])
                    + "\n"
                )
            else:
                failure = {"rule_id": row["rule_id"], "status": status}
                if "reason" in result:
                    failure["reason"] = result["reason"]
                emit(failures, failure)
            if sum(outcomes.values()) % 512 == 0:
                print(f"certify: {sum(outcomes.values())} processed", flush=True)
    bank_path.replace(output / "rules.jsonl")
    pipe_path.replace(output / "rules.pipe.txt")
    report = {
        "input": str(source.resolve()),
        "workers": workers,
        "rule_timeout": rule_timeout,
        "singular_timeout": singular_timeout,
        "source": dict(stats),
        "extraction_seconds": timing["extraction_and_dedup"],
        "worker_phase_seconds": dict(worker_timing),
        "certified": outcomes["certified"],
        "outcomes": dict(outcomes),
        "certified_backends": dict(backends),
        "seconds": perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

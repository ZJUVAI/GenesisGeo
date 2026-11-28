"""Evaluate frozen file-rule packages on explicitly supplied problem lists."""

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import threading
import os
import signal
import time
from pathlib import Path

from newclid.api import CSolver, DDAR, GeometricSolverBuilder
from newclid.formulations.problem import ProblemJGEX
from newclid.formulations.rule import Rule


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve(job):
    (
        name,
        statement,
        seed,
        variant,
        additions,
        solve_seconds,
        wall_seconds,
        build_attempts,
        guarded,
    ) = job
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError("process wall limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(wall_seconds)
    try:
        solver = (
            GeometricSolverBuilder(seed=seed)
            .load_rules_from_file(Path("src/newclid/default_configs/rules.txt"))
            .load_problem_from_txt(statement)
            .build(max_attempts=build_attempts)
        )
        if guarded:
            from newclid.evaluation.symbolic_rule_bank import (
                COMPILE_PID,
                RULE_BANK,
                RULE_NAMES,
            )

            custom_rules, rule_names = [], frozenset()
            if variant == "candidate":
                if RULE_BANK is None or COMPILE_PID == os.getpid():
                    raise RuntimeError("Symbolic rule bank was not preloaded")
                custom_rules, rule_names = RULE_BANK, RULE_NAMES
            native_solver = CSolver(solver=solver, seed=seed)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.alarm(solve_seconds)
            solved = native_solver.run(custom_rules=custom_rules, goal_proofs_only=True)
            signal.alarm(0)
        else:
            extra_rules = Rule.parse_text(additions)
            rule_names = {rule.descrption for rule in extra_rules}
            solver.rules.extend(extra_rules)
            solved = solver.run(timeout=solve_seconds)
        info = solver.run_infos
        status = (
            "solved"
            if solved
            else "unsolved"
            if info.get("runtime", solve_seconds) < solve_seconds
            else "timeout"
        )
        reasons = set()
        pending = list(solver.goals) if solved else []
        visited = set()
        graph = solver.proof.dep_graph.hyper_graph
        while pending:
            statement = pending.pop()
            if statement in visited:
                continue
            visited.add(statement)
            dependency = graph.get(statement)
            if dependency:
                reasons.add(dependency.reason)
                pending.extend(dependency.why)
        return {
            "problem": name,
            "seed": seed,
            "variant": variant,
            "status": status,
            "seconds": time.monotonic() - started,
            "error": "",
            "proof_rules": ",".join(sorted(reasons & rule_names)),
        }
    except TimeoutError as exc:
        return {
            "problem": name,
            "seed": seed,
            "variant": variant,
            "status": "wall_timeout",
            "seconds": time.monotonic() - started,
            "error": str(exc),
            "proof_rules": "",
        }
    except Exception as exc:
        return {
            "problem": name,
            "seed": seed,
            "variant": variant,
            "status": "error",
            "seconds": time.monotonic() - started,
            "error": repr(exc),
            "proof_rules": "",
        }
    finally:
        signal.alarm(0)


PROCESS_LOCK = threading.Lock()


def deliver(writer, job):
    try:
        writer.send(solve(job))
    finally:
        writer.close()


def bounded(job):
    name, _, seed, variant, _, _, wall_seconds, *_ = job
    started = time.monotonic()
    context = multiprocessing.get_context("forkserver")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=deliver, args=(writer, job))
    try:
        with PROCESS_LOCK:
            child.start()
        writer.close()
        remaining = max(0, wall_seconds - (time.monotonic() - started))
        if reader.poll(remaining):
            try:
                return reader.recv()
            except EOFError:
                child.join()
                timed_out = child.exitcode == -signal.SIGALRM
                status = "timeout" if timed_out else "error"
                error = (
                    "native solve time limit"
                    if timed_out
                    else "worker exited without a result"
                )
        else:
            status = "wall_timeout"
            error = "process wall limit"
        return {
            "problem": name,
            "seed": seed,
            "variant": variant,
            "status": status,
            "seconds": time.monotonic() - started,
            "error": error,
            "proof_rules": "",
        }
    finally:
        reader.close()
        writer.close()
        with PROCESS_LOCK:
            if child.pid is not None:
                if child.exitcode is None:
                    child.kill()
                child.join()
                child.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", action="append", nargs=2, metavar=("PATH", "COUNT"), required=True
    )
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    parser.add_argument("--solve-seconds", type=int, default=60)
    parser.add_argument("--wall-seconds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--build-attempts", type=int, default=64)
    args = parser.parse_args()
    if (
        min(args.solve_seconds, args.wall_seconds, args.workers, args.build_attempts)
        <= 0
        or args.wall_seconds < args.solve_seconds
    ):
        parser.error("positive limits required and wall limit must cover solve limit")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("repeat seeds must be distinct")
    manifest = json.loads(args.manifest.read_text())
    for filename, expected in manifest["sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("Frozen input changed: " + filename)
    frozen_hashes = {
        Path(path).resolve(): value for path, value in manifest["sha256"].items()
    }
    additions_digest = frozen_hashes.get(args.additions.resolve())
    if additions_digest is None:
        raise ValueError("Additions are absent from frozen manifest")
    k0 = Path("src/newclid/default_configs/rules.txt")
    if k0.resolve() not in frozen_hashes:
        raise ValueError("K0 is absent from frozen manifest")
    additions = args.additions.read_text()
    rule_lines = [line.strip() for line in additions.splitlines() if line.strip()]
    guarded = args.additions.name.endswith(".pipe.txt") or any(
        "|" in line for line in rule_lines
    )
    if guarded:
        DDAR.compile_custom_theorems(rule_lines)
        additions = ""
    suites = []
    for filename, count in args.suite:
        path = Path(filename)
        lines = [line for line in path.read_text().splitlines() if line]
        expected = int(count)
        if len(lines) != 2 * expected:
            raise ValueError("Unexpected line count: " + str(path))
        problems = ProblemJGEX.parse_txt_file(path)
        if len(problems) != expected:
            raise ValueError("Unexpected unique problem count: " + str(path))
        suites.append((path, problems))
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "seeds": args.seeds,
        "solve_seconds": args.solve_seconds,
        "wall_seconds": args.wall_seconds,
        "workers": args.workers,
        "build_max_attempts": args.build_attempts,
        "evaluator_sha256": digest(Path(__file__)),
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "manifest": manifest,
        "suites": {
            str(path): {"count": len(problems), "sha256": digest(path)}
            for path, problems in suites
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    preload = ["newclid.api"]
    if guarded:
        os.environ["NEWCLID_SYMBOLIC_RULE_BANK"] = str(args.additions.resolve())
        os.environ["NEWCLID_SYMBOLIC_RULE_BANK_SHA256"] = additions_digest
        preload.append("newclid.evaluation.symbolic_rule_bank")
    multiprocessing.set_forkserver_preload(preload)
    summary = {}
    for repeat, seed in enumerate(args.seeds, 1):
        out = args.output / ("repeat_" + str(repeat))
        out.mkdir()
        repeat_summary = {}
        for path, problems in suites:
            tasks = [
                (
                    name,
                    str(problem),
                    seed,
                    variant,
                    rules,
                    args.solve_seconds,
                    args.wall_seconds,
                    args.build_attempts,
                    guarded,
                )
                for name, problem in problems.items()
                for variant, rules in [("baseline", ""), ("candidate", additions)]
            ]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as pool:
                records = list(pool.map(bounded, tasks))
            if len(records) != 2 * len(problems):
                raise RuntimeError("Incomplete result set: " + str(path))
            with (out / (path.stem + ".csv")).open("x", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "problem",
                        "seed",
                        "variant",
                        "status",
                        "seconds",
                        "error",
                        "proof_rules",
                    ],
                )
                writer.writeheader()
                writer.writerows(records)
            variants = {}
            for variant in ("baseline", "candidate"):
                counts = collections.Counter(
                    r["status"] for r in records if r["variant"] == variant
                )
                variants[variant] = dict(counts)
            pairs = list(zip(records[::2], records[1::2]))
            if any(
                a["problem"] != b["problem"]
                or a["variant"] != "baseline"
                or b["variant"] != "candidate"
                for a, b in pairs
            ):
                raise RuntimeError("Paired records are misaligned: " + str(path))
            gains = collections.Counter(
                a["status"]
                for a, b in pairs
                if a["status"] != "solved" and b["status"] == "solved"
            )
            losses = collections.Counter(
                b["status"]
                for a, b in pairs
                if a["status"] == "solved" and b["status"] != "solved"
            )
            repeat_summary[path.stem] = {
                "total": len(problems),
                "variants": variants,
                "gained": sum(gains.values()),
                "gained_by_baseline_status": dict(gains),
                "lost": sum(losses.values()),
                "lost_by_candidate_status": dict(losses),
            }
        (out / "summary.json").write_text(json.dumps(repeat_summary, indent=2) + "\n")
        summary["repeat_" + str(repeat)] = {"seed": seed, "suites": repeat_summary}
        print(
            json.dumps({"repeat": repeat, "seed": seed, "suites": repeat_summary}),
            flush=True,
        )
    aggregate = {}
    for path, _ in suites:
        per_repeat = [
            summary["repeat_" + str(i)]["suites"][path.stem]
            for i in range(1, len(args.seeds) + 1)
        ]
        aggregate[path.stem] = {}
        for variant in ("baseline", "candidate"):
            values = [item["variants"][variant].get("solved", 0) for item in per_repeat]
            aggregate[path.stem][variant] = {
                "scores": values,
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
    (args.output / "summary.json").write_text(
        json.dumps({"repeats": summary, "aggregate": aggregate}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--symbolic":
    symbolic_script = Path(__file__).with_name("evaluation_symbolic.py")
    os.execv(sys.executable, [sys.executable, str(symbolic_script), *sys.argv[2:]])

import ray
from rich.live import Live
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from newclid.agent.vllm import (
    Qwen3Agent,
    Qwen3VLAgent,
    Qwen3VLTextAgent,
    discover_served_model,
)
from newclid.api import GeometricSolverBuilder
from newclid.configs import load_solver_config
from newclid.evaluation.search_trace import TraceRun, get_git_commit, timestamp_slug


LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def validate_think_search_version(*, think: bool, search_version: str) -> None:
    if think and search_version != "v1":
        raise ValueError(
            "--think true only supports --search_version v1; "
            f"got {search_version!r}."
        )


def slugify(value: str, default: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip().rstrip("/"))
    return cleaned.strip("._") or default


def served_model_slugs(served_model_name: str) -> tuple[str, str]:
    path = Path(served_model_name.rstrip("/"))
    checkpoint_slug = slugify(path.name or served_model_name)
    model_part = path.parent.name or path.name or served_model_name
    return slugify(model_part), checkpoint_slug


def build_eval_output_stem(
    *,
    agent_type: str,
    search_version: str,
    problems_path: Path,
    served_model_name: str,
    decoding_size: int,
    beam_size: int,
    search_depth: int,
    timestamp: str,
    commit_short: str,
) -> str:
    search_slug = search_version[1:] if search_version.startswith("v") else search_version
    model_slug, checkpoint_slug = served_model_slugs(served_model_name)
    return (
        f"eval_vllm_{agent_type}_{model_slug}_{checkpoint_slug}_{problems_path.stem}"
        f"_sv{search_slug}_d{decoding_size}_b{beam_size}_s{search_depth}_{timestamp}_{commit_short}"
    )


@dataclass(frozen=True)
class EvalConfig:
    problems_path: Path
    vllm_base_url: str
    agent_type: str
    served_model_name: str
    decoding_size: int
    beam_size: int
    search_depth: int
    search_version: str
    think: bool
    timeout: int
    render_root: Path
    ddar_config: dict[str, bool]



@dataclass(frozen=True)
class SolveOutcome:
    solved: bool
    elapsed_s: float
    run_infos: dict[str, object]


def create_agent(config: EvalConfig, *, trace_writer=None):
    common = {
        "base_url": config.vllm_base_url,
        "served_model_name": config.served_model_name,
        "decoding_size": config.decoding_size,
        "beam_size": config.beam_size,
        "search_depth": config.search_depth,
        "search_version": config.search_version,
        "ddar_config": config.ddar_config,
        "trace_writer": trace_writer,
    }
    if config.agent_type == "qwen3_text":
        return Qwen3Agent(**common, think=config.think)
    if config.agent_type == "qwen3_vl_text":
        return Qwen3VLTextAgent(**common)
    if config.agent_type == "qwen3_vl":
        return Qwen3VLAgent(**common, render_root=config.render_root)
    raise ValueError(f"Unsupported agent type: {config.agent_type}")


def solve_one_problem(
    *, problem_name: str, config: EvalConfig, trace_writer=None
) -> SolveOutcome:
    t0 = time.perf_counter()
    builder = GeometricSolverBuilder().load_problem_from_file(
        config.problems_path, problem_name, rename=True
    )
    agent = create_agent(config, trace_writer=trace_writer)
    solver = builder.with_deductive_agent(agent).build()
    solved = solver.run(timeout=config.timeout)
    return SolveOutcome(
        solved=solved,
        elapsed_s=time.perf_counter() - t0,
        run_infos=solver.run_infos,
    )


# (name, status, elapsed, llm_calls, ddar_calls, llm_wall, ddar_wall)
TaskRow = tuple[str, str, float, int, int, float, float]


def render_table(tasks: list[TaskRow], start_time: float) -> Table:
    solved = sum(status == "Success" for _, status, *_ in tasks)
    done = sum(status != "Pending" for _, status, *_ in tasks)
    table = Table()
    table.add_column(
        f"Problem Names ({solved} Solved / {done} Processed / {len(tasks)} Total)",
        justify="left",
        no_wrap=True,
    )
    table.add_column("Status", justify="center")
    table.add_column("Calls (LM/DDAR)", justify="right")
    table.add_column(f"Time (LM/DDAR) ({time.time() - start_time:.2f}s)", justify="right")
    priority = {"Failed": 0, "Pending": 1, "Success": 2}
    for name, status, elapsed, llm_calls, ddar_calls, llm_wall, ddar_wall in sorted(
        tasks, key=lambda item: priority.get(item[1], 99)
    ):
        if status == "Pending":
            table.add_row(name, status, "-", "-")
        else:
            table.add_row(
                name,
                status,
                f"{llm_calls} / {ddar_calls}",
                f"{elapsed:.2f}s ({llm_wall:.2f}/{ddar_wall:.2f})",
            )
    return table


def _outcome_to_row(problem_name: str, outcome: SolveOutcome) -> TaskRow:
    info = outcome.run_infos
    return (
        problem_name,
        "Success" if outcome.solved else "Failed",
        outcome.elapsed_s,
        int(info.get("llm_calls", 0)),
        int(info.get("ddar_calls", 0)),
        float(info.get("llm_real_time_s", 0.0)),
        float(info.get("ddar_real_time_s", 0.0)),
    )


def solve_problems_vllm(
    *,
    filepath: Path,
    vllm_base_url: str,
    agent_type: str,
    decoding_size: int,
    beam_size: int,
    search_depth: int,
    search_version: str,
    think: bool,
    ray_num_cpus: int | None,
    timeout: int,
    log_dir: str | None,
    enable_trace: bool,
    using_exp: bool = True,
):
    validate_think_search_version(think=think, search_version=search_version)

    main_t0 = time.perf_counter()
    if not filepath.exists():
        raise FileNotFoundError(f"Problems file not found: {filepath}")

    problem_names = [
        line.strip()
        for line in filepath.read_text(encoding="utf-8").splitlines()[::2]
        if line.strip()
    ]
    served_model_name, server_models = discover_served_model(vllm_base_url)
    ddar_config = load_solver_config(using_exp=using_exp)

    if not ray.is_initialized():
        ray.init(num_cpus=ray_num_cpus, include_dashboard=False, ignore_reinit_error=True)

    output_dir = Path(log_dir) if log_dir else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    render_root = output_dir / "_rendered"
    render_root.mkdir(parents=True, exist_ok=True)

    run_timestamp = timestamp_slug()
    commit_short = get_git_commit(REPO_ROOT)[:7]
    output_stem = build_eval_output_stem(
        agent_type=agent_type,
        search_version=search_version,
        problems_path=filepath,
        served_model_name=served_model_name,
        decoding_size=decoding_size,
        beam_size=beam_size,
        search_depth=search_depth,
        timestamp=run_timestamp,
        commit_short=commit_short,
    )
    model_slug, checkpoint_slug = served_model_slugs(served_model_name)

    trace_run = None
    if enable_trace:
        trace_run = TraceRun(
            output_dir,
            route="evaluation_vllm",
            agent=agent_type,
            dataset_path=filepath,
            model_path=served_model_name,
            run_name=output_stem,
            run_timestamp=run_timestamp,
            params={
                "vllm_base_url": vllm_base_url,
                "served_model_name": served_model_name,
                "server_models": server_models,
                "decoding_size": decoding_size,
                "beam_size": beam_size,
                "search_depth": search_depth,
                "timeout": timeout,
                "search_version": search_version,
                "think": think,
                "using_exp": using_exp,
                "ray_num_cpus": ray_num_cpus,
            },
            repo_root=REPO_ROOT,
        )

    config = EvalConfig(
        problems_path=filepath,
        vllm_base_url=vllm_base_url,
        agent_type=agent_type,
        served_model_name=served_model_name,
        decoding_size=decoding_size,
        beam_size=beam_size,
        search_depth=search_depth,
        search_version=search_version,
        think=think,
        timeout=timeout,
        render_root=render_root,
        ddar_config=ddar_config,
    )

    print(
        f"Total problems: {len(problem_names)}\n"
        f"agent={agent_type}\n"
        f"served_model_name={served_model_name}\n"
        f"model_slug={model_slug}\n"
        f"checkpoint_slug={checkpoint_slug}\n"
        f"vllm_base_url={vllm_base_url}\n"
        f"search_version={search_version}\n"
        f"think={think}\n"
        f"ray_n_cpus={ray.available_resources().get('CPU')}\n"
        f"using_exp={using_exp}",
        flush=True,
    )
    if trace_run is not None:
        print(f"Trace run directory: {trace_run.run_dir}", flush=True)

    tasks: list[TaskRow] = [(name, "Pending", 0.0, 0, 0, 0.0, 0.0) for name in problem_names]
    start_time = time.time()

    try:
        with Live(refresh_per_second=1) as live:
            for idx, problem_name in enumerate(problem_names):
                print(f"[start] problem={problem_name}", flush=True)
                problem_t0 = time.time()

                problem_render_root = render_root / slugify(problem_name, "problem")
                problem_render_root.mkdir(parents=True, exist_ok=True)
                problem_config = EvalConfig(**{**config.__dict__, "render_root": problem_render_root})

                trace_writer = None
                if trace_run is not None:
                    trace_writer = trace_run.create_problem_writer(
                        problem_index=idx,
                        problem_name=problem_name,
                        route="evaluation_vllm",
                        agent=agent_type,
                        start_time=problem_t0,
                    )
                    trace_writer.log(
                        "problem_start",
                        dataset_path=str(filepath),
                        model_path=served_model_name,
                    )

                try:
                    outcome = solve_one_problem(
                        problem_name=problem_name,
                        config=problem_config,
                        trace_writer=trace_writer,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    print(
                        f"Warning: solver crashed on '{problem_name}': {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    outcome = SolveOutcome(
                        solved=False,
                        elapsed_s=time.time() - problem_t0,
                        run_infos={"error": f"{type(exc).__name__}: {exc}"},
                    )

                tasks[idx] = _outcome_to_row(problem_name, outcome)
                _, status, elapsed, llm_calls, ddar_calls, llm_wall, ddar_wall = tasks[idx]

                if trace_writer is not None:
                    trace_writer.log(
                        "problem_end",
                        success=outcome.solved,
                        runtime=outcome.run_infos.get("runtime"),
                        final_error=outcome.run_infos.get("error"),
                    )
                    trace_writer.close()

                print(
                    f"[stats] problem={problem_name}"
                    f" solved={outcome.solved}"
                    f" elapsed={elapsed:.2f}s"
                    f" llm_calls={llm_calls}"
                    f" ddar_calls={ddar_calls}"
                    f" llm_wall={llm_wall:.2f}s"
                    f" ddar_wall={ddar_wall:.2f}s",
                    flush=True,
                )
                if "error" in outcome.run_infos:
                    print(
                        f"[error] problem={problem_name} reason={outcome.run_infos['error']}",
                        flush=True,
                    )
                live.update(render_table(tasks, start_time))
            live.update(render_table(tasks, start_time))

        csv_path = output_dir / f"{output_stem}.csv"
        solved_count = sum(1 for _, status, *_ in tasks if status == "Success")
        total_time = sum(elapsed for _, _, elapsed, *_ in tasks)
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [f"Dataset: {filepath.stem}, Model: {served_model_name}, Checkpoint: {checkpoint_slug}, "
                 f"Solved: {solved_count}/{len(tasks)}, Total Time: {total_time:.2f}s"]
            )
            writer.writerow(["Problem Name", "Solved", "LM Calls", "DDAR Calls",
                             "LM Time(s)", "DDAR Time(s)", "Total Time(s)"])
            for name, status, elapsed, llm_calls, ddar_calls, llm_wall, ddar_wall in tasks:
                writer.writerow([name, "√" if status == "Success" else "x",
                                 llm_calls, ddar_calls,
                                 f"{llm_wall:.2f}", f"{ddar_wall:.2f}", f"{elapsed:.2f}"])

        print(f"Results saved to {csv_path}", flush=True)
        print(f"[session] total_wall_time_s={time.perf_counter() - main_t0:.2f}", flush=True)
        return {
            "csv_path": csv_path,
            "solved_count": solved_count,
            "total_problems": len(tasks),
            "total_time": total_time,
            "served_model_name": served_model_name,
            "model_slug": model_slug,
            "checkpoint_slug": checkpoint_slug,
            "trace_dir": trace_run.run_dir if trace_run is not None else None,
        }
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOGLEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(description="vLLM-only evaluation.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["qwen3_text", "qwen3_vl", "qwen3_vl_text"],
    )
    parser.add_argument("--problems_path", type=str, required=True)
    parser.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--decoding_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--search_depth", type=int, default=4)
    parser.add_argument("--search_version", type=str, default="hybrid", choices=["v1", "v2", "hybrid"])
    parser.add_argument(
        "--think",
        type=parse_bool,
        default=False,
        help=(
            "For text eval, true starts with <think> and extracts the generated <aux>; "
            "false inserts an empty <think> block before <aux>."
        ),
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--ray_num_cpus", type=int, default=None)
    parser.add_argument("--using_exp", type=lambda v: v.strip().lower() not in {"false", "0", "no", "n", "off"}, default=True)
    parser.add_argument("--enable_trace", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    try:
        validate_think_search_version(
            think=args.think,
            search_version=args.search_version,
        )
    except ValueError as exc:
        parser.error(str(exc))

    solve_problems_vllm(
        filepath=Path(args.problems_path),
        vllm_base_url=args.vllm_base_url,
        agent_type=args.agent,
        decoding_size=args.decoding_size,
        beam_size=args.beam_size,
        search_depth=args.search_depth,
        search_version=args.search_version,
        think=args.think,
        ray_num_cpus=args.ray_num_cpus,
        timeout=args.timeout,
        log_dir=args.log_dir,
        enable_trace=args.enable_trace,
        using_exp=args.using_exp,
    )


if __name__ == "__main__":
    main()

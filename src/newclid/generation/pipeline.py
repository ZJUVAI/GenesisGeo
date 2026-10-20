import logging
import math
import os
import argparse
import json
import time
import sys
from datetime import timedelta, datetime
import ray
import re
from millify import millify
import uuid

from newclid.configs import default_defs_path
from newclid.formulations.definition import DefinitionJGEX
from newclid.generation.statistics import Statistics, get_first_predicate
from newclid.generation.filter import GoalFilter
from newclid.generation.worker import ProblemWorker
from newclid.generation.writer import Writer
from newclid.discovery.utils.rule_parser import to_pipe_format

# logging.basicConfig(level=logging.DEBUG, force=True)


class ProblemPipeline:
    def __init__(
        self,
        n_clauses=5,
        n_threads=1,
        output_dir="dataset",
        n_samples=100,
        timeout=3600,
        max_level=500,
        img=0,
        aux_only=0,
        clear=False,
        add_auxiliary=True,
        max_auxiliary_points=2,
        prune=True,
        remove_coords=False,
        construction_config=None,
        seed_cache=False,
        base_seed=42,
        using_log=True,
        using_exp=False,
        direct_png=True,
        img_pixels=512,
        custom_rules_path=None,
        triple_symmetry=True,
    ):
        if img_pixels <= 0:
            raise ValueError("img_pixels must be a positive integer")

        self.n_clauses = n_clauses
        self.n_samples = n_samples
        self.n_threads = n_threads
        self.timeout = timeout
        self.max_level = max_level
        self.max_auxiliary_points = max_auxiliary_points
        self.output_dir = output_dir
        self.file_prefix = (
            f"geometry_clauses{self.n_clauses}_samples{millify(self.n_samples)}"
        )
        self.hashed_problems = set()
        self.filter = GoalFilter()
        self.defs = DefinitionJGEX.to_dict(
            DefinitionJGEX.parse_txt_file(default_defs_path())
        )
        self.img = img
        self.aux_only = aux_only
        self.clear = clear
        self.add_auxiliary = add_auxiliary
        self.prune = prune
        self.remove_coords = remove_coords
        self.construction_config = construction_config
        self.using_log = using_log
        self.using_exp = using_exp
        self.direct_png = direct_png
        self.img_pixels = img_pixels
        self.custom_rules_path = custom_rules_path
        self.triple_symmetry = triple_symmetry
        self.custom_rules = load_custom_rules(custom_rules_path)
        if self.custom_rules:
            logging.info(
                "Loaded %d custom rule(s) from %s",
                len(self.custom_rules),
                self.custom_rules_path,
            )

        self.use_seed_cache = seed_cache
        self.base_seed = base_seed
        self.seed_cache = {}
        self.cache_file = os.path.join(self.output_dir, "seed_cache.jsonl")
        if self.use_seed_cache and os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        key = (entry["seed"], entry["n_clauses"])
                        self.seed_cache[key] = entry
            logging.info("Loaded %d seeds from cache", len(self.seed_cache))

        # Serialize defs for Ray remote tasks
        defs_data = {k: v._asdict() for k, v in self.defs.items()}

        # Generate unique session ID
        session_id = uuid.uuid4().hex[:16]
        logging.info(f"Session ID: {session_id}")
        logging.info(
            "Construction config: %s",
            "external" if self.construction_config is not None else "default",
        )

        # Initialize writer
        dated_dir = os.path.join(self.output_dir, datetime.now().strftime("%Y%m%d"))
        self.writer = Writer(
            output_dir=dated_dir,
            file_prefix=self.file_prefix,
            img_mode=self.img,
            defs_data=defs_data,
            session_id=session_id,
            direct_png=self.direct_png,
            img_pixels=self.img_pixels,
        )

        # Initialize statistics reporter
        self.summary_reporter = Statistics(
            report_path=os.path.join(dated_dir, self.file_prefix + "_report.json")
        )

    def problem_hash_filter(self, data: list, key: str) -> list[str]:
        """Check if the input has already been written to the output file."""
        filtered_data = []
        for d in data:
            key_hash = hash(d[key])
            if key_hash not in self.hashed_problems:
                self.hashed_problems.add(key_hash)
                filtered_data.append(d)
        return filtered_data

    def _resume_from_existing(self):
        """Resume from existing JSONL file: update base_seed and hashed_problems."""
        filepath = os.path.join(self.writer.output_dir, self.file_prefix + ".jsonl")
        if not os.path.exists(filepath):
            return

        max_seed = -1
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "llm_input_renamed" in entry:
                    self.hashed_problems.add(hash(entry["llm_input_renamed"]))
                if "seed" in entry:
                    max_seed = max(max_seed, entry["seed"])
                count += 1

        self.writer.written_count = count
        if max_seed >= 0:
            self.base_seed = max_seed + 1
        logging.info(
            f"Resumed from existing data: {count} records, "
            f"new base_seed={self.base_seed}"
        )

    def generate(self):
        if self.clear:
            self.writer.clear()
        else:
            self._resume_from_existing()

        def task_generator():
            for i in range(10**9):
                seed = self.base_seed + i
                fl_statement = None

                if self.use_seed_cache:
                    cache_key = (seed, self.n_clauses)
                    if cache_key in self.seed_cache:
                        entry = self.seed_cache[cache_key]
                        if not entry.get("has_real_aux", False):
                            continue
                        fl_statement = entry.get("fl_statement")

                yield (
                    i,
                    seed,
                    self.n_clauses,
                    self.max_level,
                    self.using_log,
                    self.using_exp,
                    self.img,
                    self.aux_only,
                    self.add_auxiliary,
                    self.max_auxiliary_points,
                    self.prune,
                    self.remove_coords,
                    self.construction_config,
                    fl_statement,
                    self.custom_rules,
                    self.triple_symmetry,
                )

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=self.n_threads)
        task_iterator = task_generator()
        max_pending = int(self.n_threads * 1.5)
        pending_tasks = {}

        HEARTBEAT_INTERVAL = 30  # seconds; print progress even with 0 new samples

        start_time = time.time()
        last_logged_written = self.writer.written_count
        last_logged_time = start_time
        last_heartbeat_time = start_time
        smoothed_speed = None
        tasks_since_log = 0
        failed_tasks_since_log = 0
        # 被 DEADLINE 截断、且已经产出过 aux 样本的任务数（见 worker.py）。
        aux_truncated_since_log = 0
        aux_truncated_total = 0
        # 反向重构统计（见 worker.py::_try_reverse_reconstruction）：候选数、
        # stage-a 直接可证被丢弃数、stage-b 加了辅助点仍证不出被丢弃数、
        # 最终保留数——全部累计到当前进程存活期间。
        recon_candidates_total = 0
        recon_solved_without_aux_total = 0
        recon_unsolved_with_aux_total = 0
        recon_kept_total = 0

        while self.writer.written_count < self.n_samples:
            done, _ = ray.wait(list(pending_tasks.keys()), num_returns=1, timeout=10)

            if done:
                task_id = done[0]
                task_success = True
                try:
                    data, summary = ray.get(task_id)

                    if "error" in summary:
                        task_success = False
                except Exception as e:
                    logging.error(f"Task failed: {e}")
                    task_success = False

                _, seed = pending_tasks.pop(task_id)
                tasks_since_log += 1
                if not task_success:
                    failed_tasks_since_log += 1
                else:
                    if summary.get("aux_truncated", False):
                        aux_truncated_since_log += 1
                        aux_truncated_total += 1
                    recon_candidates_total += summary.get("recon_candidates", 0)
                    recon_solved_without_aux_total += summary.get(
                        "recon_solved_without_aux", 0
                    )
                    recon_unsolved_with_aux_total += summary.get(
                        "recon_unsolved_with_aux", 0
                    )
                    recon_kept_total += summary.get("recon_kept", 0)

                if task_success:
                    if self.use_seed_cache:
                        has_real_aux = (
                            summary.get("has_real_aux", False) if summary else False
                        )
                        cache_entry = {
                            "seed": seed,
                            "n_clauses": self.n_clauses,
                            "has_real_aux": has_real_aux,
                            "fl_statement": summary.get("fl_statement")
                            if has_real_aux
                            else None,
                        }
                        cache_key = (seed, self.n_clauses)
                        if cache_key not in self.seed_cache:
                            self.seed_cache[cache_key] = cache_entry
                            os.makedirs(self.output_dir, exist_ok=True)
                            with open(self.cache_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(cache_entry) + "\n")

                    data = self.problem_hash_filter(data, "llm_input_renamed")
                    if data:
                        summary["n_samples"] = len(data)
                        summary["n_filtered_samples"] = (
                            summary["n_samples_raw"] - summary["n_samples"]
                        )
                        summary["goals"] = [
                            re.search(r"\?\s*(\w+)", d["fl_problem"]).group(1)
                            for d in data
                        ]
                        summary["first_predicate"] = [
                            get_first_predicate(d["fl_problem"]) for d in data
                        ]
                        summary["n_premises"] = [d["n_premises"] for d in data]
                        summary["n_proof_steps"] = [d["n_proof_steps"] for d in data]
                        self.writer.write_data(data)
                        self.summary_reporter.add(summary)

                        # Log progress when new data is written to file
                        if self.writer.written_count > last_logged_written:
                            now = time.time()
                            elapsed_time = now - start_time
                            instant_elapsed = max(now - last_logged_time, 1e-6)
                            instant_written = (
                                self.writer.written_count - last_logged_written
                            )
                            instant_speed = instant_written / instant_elapsed
                            alpha = 1 - math.exp(-instant_elapsed / 500)
                            smoothed_speed = (
                                instant_speed
                                if smoothed_speed is None
                                else alpha * instant_speed
                                + (1 - alpha) * smoothed_speed
                            )
                            remaining = max(
                                0, self.n_samples - self.writer.written_count
                            )
                            eta_seconds = remaining / max(smoothed_speed, 1e-6)
                            pending_draw = len(self.writer.pending_draw_tasks)
                            pending_write = len(self.writer.pending_write_data)
                            logging.info(
                                f"{millify(self.writer.written_count)}/{millify(self.n_samples)} "
                                f"(+{self.writer.written_count - last_logged_written:3d}, pending: {pending_draw}+{pending_write}) "
                                f"in {elapsed_time:5.0f}s | "
                                f"Total: {summary['total_time']:3.0f}s = "
                                f"Gen: {summary['generation_time']:1.0f} + "
                                f"DDAR: {summary['runtime']:2.0f} + "
                                f"Proc: {summary['process_goal_runtime']:3.0f} | "
                                f"Speed: {smoothed_speed:3.0f} samp/s | "
                                f"ETA: {timedelta(seconds=int(eta_seconds))} | "
                                f"AuxTrunc: {aux_truncated_total} | "
                                f"Recon: cand={recon_candidates_total} "
                                f"noAux={recon_solved_without_aux_total} "
                                f"aux_fail={recon_unsolved_with_aux_total} "
                                f"kept={recon_kept_total}"
                            )
                            last_logged_written = self.writer.written_count
                            last_logged_time = now
                            last_heartbeat_time = now
                            tasks_since_log = 0
                            failed_tasks_since_log = 0
                            aux_truncated_since_log = 0
            # Heartbeat: print progress periodically even when no new samples
            # were written, so a stalled run is distinguishable from a slow one.
            now = time.time()
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                pending_draw = len(self.writer.pending_draw_tasks)
                pending_write = len(self.writer.pending_write_data)
                logging.info(
                    f"{millify(self.writer.written_count)}/{millify(self.n_samples)} "
                    f"(+0, {tasks_since_log} tasks done in last {now - last_heartbeat_time:.0f}s, "
                    f"{failed_tasks_since_log} failed, "
                    f"{aux_truncated_since_log} aux-truncated "
                    f"({aux_truncated_total} total), "
                    f"pending: {len(pending_tasks)} running "
                    f"+ {pending_draw}+{pending_write} draw/write) "
                    f"in {now - start_time:5.0f}s | "
                    f"Recon: cand={recon_candidates_total} "
                    f"noAux={recon_solved_without_aux_total} "
                    f"aux_fail={recon_unsolved_with_aux_total} "
                    f"kept={recon_kept_total}"
                )
                last_heartbeat_time = now
                tasks_since_log = 0
                failed_tasks_since_log = 0
                aux_truncated_since_log = 0

            for task, (s_time, _) in list(pending_tasks.items()):
                if time.time() - s_time > self.timeout:
                    print(f"⚠️ Task {task} timeout. Canceling")
                    ray.cancel(task, force=True)
                    del pending_tasks[task]

            while len(pending_tasks) < max_pending:
                task_args = next(task_iterator)
                seed = task_args[1]
                pending_tasks[
                    ProblemWorker.ray_process_single_problem.remote(task_args)
                ] = (
                    time.time(),
                    seed,
                )

        # Cancel any remaining problem generation tasks
        for task in pending_tasks.keys():
            ray.cancel(task, force=True)

        # Wait for all pending draw tasks to complete before shutdown
        self.writer.write_data([], force=True)

        ray.shutdown()

        final_elapsed_time = time.time() - start_time
        self.summary_reporter.total_elapsed_time = final_elapsed_time
        self.summary_reporter.total_samples_generated = self.writer.written_count
        logging.info(
            f"Generate {self.writer.written_count} samples in {final_elapsed_time:.2f}s."
        )
        self.summary_reporter.output_report()


def load_construction_config(config_path: str | None) -> dict | None:
    """Load optional construction config JSON from disk."""
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Construction config JSON must contain a top-level object.")
    return config


def load_custom_rules(rules_path: str | None) -> list[str] | None:
    """Load extra DDAR rules and convert them to CSolver's pipe format.

    Accepts a `.txt` file with one `premises => conclusion` rule per line
    (same grammar as `default_configs/rules.txt`, e.g. the output of the
    discovery/reduction pipeline's `reduced_rules.txt`), or a `.jsonl` file
    where each line is a JSON object with a `rule_text` field (e.g.
    `reduced_rules.jsonl`).
    """
    if not rules_path:
        return None

    rule_texts: list[str] = []
    guards_list: list[str] = []
    is_jsonl = rules_path.endswith(".jsonl") or rules_path.endswith(".json")
    with open(rules_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                rec = json.loads(line)
                rule_texts.append(rec["rule_text"])
                guards_list.append(rec.get("guards", ""))
            else:
                rule_texts.append(line)
                guards_list.append("")

    return [
        to_pipe_format(f"custom_{i:05d}", rule_texts[i],
                       guards=guards_list[i] if i < len(guards_list) else "")
        for i in range(len(rule_texts))
    ]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def write_cli_args(params_path: str, args: argparse.Namespace):
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "command": " ".join(sys.argv),
                "argv": sys.argv,
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Create problem fl - nl dataset")
    # General parameters
    parser.add_argument(
        "--log_level",
        required=False,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    parser.add_argument(
        "--n_clauses",
        required=False,
        type=int,
        default=15,
        help="Number of clauses in generated problems",
    )
    parser.add_argument(
        "--n_samples",
        required=False,
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--n_threads",
        required=False,
        type=int,
        default=10,
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--timeout",
        required=False,
        type=int,
        default=3600,
        help="Timeout for individual tasks (seconds)",
    )
    parser.add_argument(
        "--max_level",
        required=False,
        type=int,
        default=500,
        help="Maximum DDAR search depth",
    )
    parser.add_argument(
        "--using_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CSolver logarithmic equations (default: enabled)",
    )
    parser.add_argument(
        "--using_exp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CSolver exponential equation form (default: disabled)",
    )
    parser.add_argument(
        "--construction_config",
        required=False,
        default=None,
        help="Optional JSON file defining construction sets and sampler steps",
    )
    parser.add_argument(
        "--custom_rules_path",
        required=False,
        default=None,
        help="Optional extra DDAR rules file (.txt with 'premises => conclusion' "
        "lines, or .jsonl with a 'rule_text' field per line) to add to the "
        "solver's rule set, e.g. discovery/reduction output reduced_rules.txt",
    )
    parser.add_argument(
        "--seed_cache",
        required=False,
        action="store_true",
        default=False,
        help="Use seed cache to skip seeds without real auxiliary points",
    )
    parser.add_argument(
        "--base_seed",
        required=False,
        type=int,
        default=42,
        help="Base seed for problem generation",
    )
    # Auxiliary point parameters
    parser.add_argument(
        "--add_auxiliary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add auxiliary points (default: enabled)",
    )
    parser.add_argument(
        "--max_auxiliary_points",
        required=False,
        type=int,
        default=2,
        help="Maximum number of auxiliary points per problem",
    )
    parser.add_argument(
        "--aux_only",
        required=False,
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Auxiliary data filter: "
        "0=all data, "
        "1=include data without aux with 0.1 probability, "
        "2=only data with aux",
    )
    # Output parameters
    parser.add_argument(
        "--dir", required=False, default="./datasets", help="Output directory"
    )
    parser.add_argument(
        "--img",
        required=False,
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Image generation mode: "
        "0=no images, "
        "1=with annotations only, "
        "2=without annotations only, "
        "3=both",
    )
    parser.add_argument(
        "--direct_png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save PNG directly instead of using svg -> png conversion (default: enabled)",
    )
    parser.add_argument(
        "--img_pixels",
        required=False,
        type=positive_int,
        default=512,
        help="Output image width in pixels (default: 512)",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prune clauses to preserve only deepest clause chain (default: enabled)",
    )
    parser.add_argument(
        "--remove_coords",
        action="store_true",
        default=False,
        help="Remove coordinate information from final clause output",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        default=False,
        help="Clear old dataset files before generation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    construction_config = load_construction_config(args.construction_config)
    generator = ProblemPipeline(
        n_clauses=args.n_clauses,
        n_threads=args.n_threads,
        output_dir=args.dir,
        n_samples=args.n_samples,
        timeout=args.timeout,
        max_level=args.max_level,
        img=args.img,
        aux_only=args.aux_only,
        clear=args.clear,
        add_auxiliary=args.add_auxiliary,
        max_auxiliary_points=args.max_auxiliary_points,
        prune=args.prune,
        remove_coords=args.remove_coords,
        construction_config=construction_config,
        seed_cache=args.seed_cache,
        base_seed=args.base_seed,
        using_log=args.using_log,
        using_exp=args.using_exp,
        direct_png=args.direct_png,
        img_pixels=args.img_pixels,
        custom_rules_path=args.custom_rules_path,
    )
    write_cli_args(
        os.path.join(generator.writer.output_dir, generator.file_prefix + "_params.json"),
        args,
    )
    generator.generate()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Single-threaded (non-Ray) version of pipeline.py.

Processes geometry problems one at a time in a simple loop,
writing results directly to JSONL without any parallel framework.
"""

import logging
import math
import os
import argparse
import json
import re
import time
from datetime import timedelta, datetime
from millify import millify
import uuid

from newclid.configs import default_defs_path
from newclid.formulations.definition import DefinitionJGEX
from newclid.generation.statistics import Statistics, get_first_predicate
from newclid.generation.filter import GoalFilter
from newclid.generation.worker import ProblemWorker

# logging.basicConfig(level=logging.DEBUG, force=True)


class LinearProblemPipeline:
    """Single-threaded version of ProblemPipeline — no Ray dependency."""

    def __init__(
        self,
        n_clauses=5,
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
        triple_symmetry=True,
    ):
        if img_pixels <= 0:
            raise ValueError("img_pixels must be a positive integer")

        self.n_clauses = n_clauses
        self.n_samples = n_samples
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
        self.triple_symmetry = triple_symmetry

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

        session_id = uuid.uuid4().hex[:16]
        logging.info(f"Session ID: {session_id}")
        logging.info(
            "Construction config: %s",
            "external" if self.construction_config is not None else "default",
        )

        # Output paths
        dated_dir = os.path.join(self.output_dir, datetime.now().strftime("%Y%m%d"))
        self.output_dated_dir = dated_dir
        os.makedirs(dated_dir, exist_ok=True)
        self.jsonl_path = os.path.join(dated_dir, self.file_prefix + ".jsonl")
        self.report_path = os.path.join(dated_dir, self.file_prefix + "_report.json")

        self.written_count = 0

        # Initialize statistics reporter
        self.summary_reporter = Statistics(report_path=self.report_path)

    def problem_hash_filter(self, data: list, key: str) -> list:
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
        if not os.path.exists(self.jsonl_path):
            return

        max_seed = -1
        count = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
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

        self.written_count = count
        if max_seed >= 0:
            self.base_seed = max_seed + 1
        logging.info(
            f"Resumed from existing data: {count} records, "
            f"new base_seed={self.base_seed}"
        )

    def clear_output(self):
        """Remove existing output files."""
        import shutil

        if os.path.exists(self.jsonl_path):
            os.remove(self.jsonl_path)
        imgs_dir = os.path.join(self.output_dated_dir, "imgs")
        imgs_png_dir = os.path.join(self.output_dated_dir, "imgs_png")
        if os.path.exists(imgs_dir):
            shutil.rmtree(imgs_dir)
        if os.path.exists(imgs_png_dir):
            shutil.rmtree(imgs_png_dir)

    def _write_jsonl(self, data: list):
        """Append data items to JSONL file directly."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for item in data:
                item.pop("_timings", None)
                item.pop("draw_data", None)
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        self.written_count += len(data)

    def generate(self):
        if self.clear:
            self.clear_output()
        else:
            self._resume_from_existing()

        start_time = time.time()
        last_logged_written = self.written_count
        last_logged_time = start_time
        smoothed_speed = None

        problem_index = 0
        while self.written_count < self.n_samples:
            seed = self.base_seed + problem_index
            fl_statement = None

            # Check seed cache
            if self.use_seed_cache:
                cache_key = (seed, self.n_clauses)
                if cache_key in self.seed_cache:
                    entry = self.seed_cache[cache_key]
                    if not entry.get("has_real_aux", False):
                        problem_index += 1
                        continue
                    fl_statement = entry.get("fl_statement")

            task_args = (
                problem_index,
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
                None,  # custom_rules (unsupported by this pipeline)
                self.triple_symmetry,
            )

            # Process single problem (call static method directly, no Ray)
            task_success = False
            try:
                data, summary = ProblemWorker._process_single_problem(task_args)
                if "error" not in summary:
                    task_success = True
            except Exception as e:
                logging.error(f"Task failed (seed={seed}): {e}")
                import traceback
                traceback.print_exc()

            if task_success:
                # Update seed cache
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

                # Dedup
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

                    self._write_jsonl(data)
                    self.summary_reporter.add(summary)

                    # Log progress
                    if self.written_count > last_logged_written:
                        now = time.time()
                        elapsed_time = now - start_time
                        instant_elapsed = max(now - last_logged_time, 1e-6)
                        instant_written = (
                            self.written_count - last_logged_written
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
                            0, self.n_samples - self.written_count
                        )
                        eta_seconds = remaining / max(smoothed_speed, 1e-6)
                        logging.info(
                            f"{millify(self.written_count)}/{millify(self.n_samples)} "
                            f"(+{self.written_count - last_logged_written:3d}) "
                            f"in {elapsed_time:5.0f}s | "
                            f"Total: {summary['total_time']:3.0f}s = "
                            f"Gen: {summary['generation_time']:1.0f} + "
                            f"DDAR: {summary['runtime']:2.0f} + "
                            f"Proc: {summary['process_goal_runtime']:3.0f} | "
                            f"Speed: {smoothed_speed:3.0f} samp/s | "
                            f"ETA: {timedelta(seconds=int(eta_seconds))}"
                        )
                        last_logged_written = self.written_count
                        last_logged_time = now

            problem_index += 1

        final_elapsed_time = time.time() - start_time
        self.summary_reporter.total_elapsed_time = final_elapsed_time
        self.summary_reporter.total_samples_generated = self.written_count
        logging.info(
            f"Generate {self.written_count} samples in {final_elapsed_time:.2f}s."
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Create problem fl - nl dataset (linear)")
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
        default=100,
        help="Number of samples to generate",
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
        help="Save PNG directly (default: enabled)",
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
        default=True,
        help="Clear old dataset files before generation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    construction_config = load_construction_config(args.construction_config)
    generator = LinearProblemPipeline(
        n_clauses=args.n_clauses,
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
    )
    generator.generate()


if __name__ == "__main__":
    main()

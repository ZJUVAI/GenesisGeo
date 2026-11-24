from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.evaluation as evaluation
from scripts.upload_eval_to_swanlab import parse_eval_csv


class _FakeLive:
    def __init__(self, *args, **kwargs):
        self.last_render = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, renderable):
        self.last_render = renderable


class EvaluationOutputTests(unittest.TestCase):
    def test_parse_bool_accepts_explicit_true_false(self):
        self.assertTrue(evaluation.parse_bool("true"))
        self.assertTrue(evaluation.parse_bool("1"))
        self.assertFalse(evaluation.parse_bool("false"))
        self.assertFalse(evaluation.parse_bool("0"))
        with self.assertRaises(Exception):
            evaluation.parse_bool("maybe")

    def test_think_true_only_allows_v1_search(self):
        evaluation.validate_think_search_version(think=True, search_version="v1")
        evaluation.validate_think_search_version(think=False, search_version="hybrid")

        with self.assertRaisesRegex(ValueError, "only supports --search_version v1"):
            evaluation.validate_think_search_version(
                think=True,
                search_version="hybrid",
            )

    def test_output_stem_contains_model_checkpoint_dataset_and_commit_slugs(self):
        stem = evaluation.build_eval_output_stem(
            agent_type="qwen3_text",
            search_version="hybrid",
            problems_path=Path("/tmp/demo.txt"),
            served_model_name="/models/Qwen3/checkpoint-42",
            decoding_size=8,
            beam_size=16,
            search_depth=3,
            timestamp="20260615T000000Z",
            commit_short="abcdef0",
        )

        self.assertEqual(
            stem,
            "eval_vllm_qwen3_text_Qwen3_checkpoint-42_demo"
            "_svhybrid_d8_b16_s3_20260615T000000Z_abcdef0",
        )

    def test_eval_csv_uses_legacy_seven_column_rows_with_model_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            problems_path = Path(tmpdir) / "dataset.txt"
            problems_path.write_text("p1\nunused\n", encoding="utf-8")

            with patch.object(evaluation, "Live", _FakeLive):
                with patch.object(
                    evaluation,
                    "discover_served_model",
                    return_value=("/models/Qwen3/checkpoint-42", ["model"]),
                ):
                    with patch.object(
                        evaluation,
                        "load_solver_config",
                        return_value={"using_exp": True},
                    ):
                        with patch.object(
                            evaluation.ray, "is_initialized", side_effect=[False, True]
                        ):
                            with patch.object(evaluation.ray, "init"):
                                with patch.object(
                                    evaluation.ray,
                                    "cluster_resources",
                                    return_value={"CPU": 1},
                                ):
                                    with patch.object(
                                        evaluation.ray,
                                        "available_resources",
                                        return_value={"CPU": 1},
                                    ):
                                        with patch.object(evaluation.ray, "shutdown"):
                                            with patch.object(
                                                evaluation,
                                                "timestamp_slug",
                                                return_value="20260615T000000Z",
                                            ):
                                                with patch.object(
                                                    evaluation,
                                                    "solve_one_problem",
                                                    return_value=evaluation.SolveOutcome(
                                                        solved=True,
                                                        elapsed_s=1.25,
                                                        run_infos={
                                                            "llm_calls": 2,
                                                            "ddar_calls": 3,
                                                            "llm_real_time_s": 0.4,
                                                            "ddar_real_time_s": 0.5,
                                                        },
                                                    ),
                                                ):
                                                    result = evaluation.solve_problems_vllm(
                                                        filepath=problems_path,
                                                        vllm_base_url="http://localhost:8000",
                                                        agent_type="qwen3_text",
                                                        decoding_size=8,
                                                        beam_size=16,
                                                        search_depth=3,
                                                        search_version="hybrid",
                                                        think=False,
                                                        ray_num_cpus=1,
                                                        timeout=30,
                                                        log_dir=tmpdir,
                                                        enable_trace=False,
                                                        using_exp=True,
                                                    )

            csv_path = Path(result["csv_path"])
            with csv_path.open(newline="", encoding="utf-8") as fp:
                rows = list(csv.reader(fp))

        self.assertIn("Model: /models/Qwen3/checkpoint-42", rows[0][0])
        self.assertIn("Checkpoint: checkpoint-42", rows[0][0])
        self.assertEqual(
            rows[1],
            [
                "Problem Name",
                "Solved",
                "LM Calls",
                "DDAR Calls",
                "LM Time(s)",
                "DDAR Time(s)",
                "Total Time(s)",
            ],
        )
        self.assertEqual(rows[2], ["p1", "√", "2", "3", "0.40", "0.50", "1.25"])
        self.assertIn("Qwen3_checkpoint-42", csv_path.name)

    def test_upload_parser_accepts_model_checkpoint_summary_and_dynamic_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "eval.csv"
            csv_path.write_text(
                "Dataset: demo, Model: /m/Qwen3/ckpt, Checkpoint: ckpt, Solved: 1/2, Total Time: 3.50s\n"
                "Problem Name,Solved,LM Calls,DDAR Calls,LM Time(s),DDAR Time(s),Total Time(s)\n"
                "p1,√,1,2,0.10,0.20,1.00\n",
                encoding="utf-8",
            )

            parsed = parse_eval_csv(csv_path)

        self.assertEqual(parsed[0], "demo")
        self.assertEqual(parsed[1], "/m/Qwen3/ckpt")
        self.assertEqual(parsed[2], "ckpt")
        self.assertEqual(parsed[3:6], (1, 2, 3.5))
        self.assertEqual(parsed[6][0], "Problem Name")
        self.assertEqual(len(parsed[6]), 7)


if __name__ == "__main__":
    unittest.main()

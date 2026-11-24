import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from newclid.evaluation.search_runtime import (
    try_dsl_to_constructions,
    try_full_aux_dsl_to_constructions,
)
from newclid.training.aux_dsl import (
    extract_aux_body,
    extract_first_tagged_aux_block,
)
from newclid.training.grpo_rewards import (
    AuxEvaluationResult,
    AuxReward,
    AuxRewardEvaluator,
)

SAMPLE_FL_PROBLEM = (
    "a b c = risos a b c; d = eqdistance d a b c, angle_bisector d b a c; "
    "e f g h = centroid e f g h a d c ? perp b e c f"
)
SAMPLE_COMPLETION = (
    "<aux> x00 i : cong a i b c [012] eqangle a b a i a i a c [013] ; </aux>"
)


class TestGRPOTrainingCore(unittest.TestCase):
    def test_reward_batch_uses_configured_workers_and_preserves_order(self):
        reward = AuxReward(cpu_workers=2)
        completions = [
            "<aux> x00 i : coll a b i [012] ; </aux>",
            "<aux> x00 i : coll a c i [012] ; </aux>",
        ]
        try:
            results = reward.evaluate_batch(
                completions,
                fl_problem=[SAMPLE_FL_PROBLEM, SAMPLE_FL_PROBLEM],
            )
        finally:
            reward.close()

        self.assertIn("coll a b i", results[0].normalized_aux)
        self.assertIn("coll a c i", results[1].normalized_aux)
        self.assertTrue(all(result.build_ok for result in results))

    def test_reward_cpu_workers_can_be_set_by_environment(self):
        with mock.patch.dict("os.environ", {"NEWCLID_GRPO_CPU_WORKERS": "3"}):
            reward = AuxReward()

        self.assertEqual(reward.cpu_workers, 3)
        reward.close()

    def test_reward_cpu_workers_default_to_serial(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            reward = AuxReward()

        self.assertEqual(reward.cpu_workers, 1)
        self.assertIsNone(reward._executor)

    def test_aux_helpers_extract_tagged_block(self):
        self.assertEqual(
            extract_aux_body(SAMPLE_COMPLETION),
            "x00 i : cong a i b c [012] eqangle a b a i a i a c [013] ;",
        )
        self.assertEqual(
            extract_first_tagged_aux_block(SAMPLE_COMPLETION),
            SAMPLE_COMPLETION,
        )
        self.assertIsNone(extract_first_tagged_aux_block("<proof> no aux </proof>"))

    def test_single_aux_parser_ignores_later_segments(self):
        self.assertEqual(
            try_dsl_to_constructions("e : coll a b e [002] ; f : perp e f a b [003] ;"),
            "e = on_line e a b",
        )

    def test_full_aux_parser_parses_all_segments_together(self):
        self.assertEqual(
            try_full_aux_dsl_to_constructions(
                "x00 e : coll a b e [002] ; x00 f : perp e f a b [003] ;"
            ),
            "e = on_line e a b; f = on_tline f e a b",
        )

    def test_reward_evaluator_adds_multi_aux_constructions_once(self):
        evaluator = AuxRewardEvaluator(build_max_attempts=100)
        with (
            mock.patch.object(evaluator, "_run_ddar", return_value=True),
            mock.patch(
                "newclid.training.grpo_rewards.ProblemJGEX.from_text"
            ) as mocked_from_text,
            mock.patch("newclid.training.grpo_rewards.ProofState.build_problemJGEX"),
        ):
            problem = mock.MagicMock()
            problem.with_more_construction.return_value = "new_problem"
            mocked_from_text.return_value = problem

            result = evaluator.evaluate(
                "<aux> x00 e : coll a b e [002] ; x00 f : perp e f a b [003] ; </aux>",
                SAMPLE_FL_PROBLEM,
            )

        self.assertTrue(result.format_ok)
        self.assertTrue(result.build_ok)
        self.assertEqual(result.ddar_status, "solved")
        problem.with_more_construction.assert_called_once_with(
            "e = on_line e a b; f = on_tline f e a b"
        )

    def test_prepare_grpo_aux_dataset_converts_tagged_rows(self):
        script_path = Path("scripts/grpo/prepare_grpo_aux_dataset.py")
        spec = importlib.util.spec_from_file_location(
            "prepare_grpo_aux_dataset", script_path
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            rows = [
                {
                    "fl_problem": SAMPLE_FL_PROBLEM,
                    "llm_input_renamed": "<problem> prompt </problem>",
                    "llm_output_renamed": SAMPLE_COMPLETION,
                },
                {
                    "fl_problem": SAMPLE_FL_PROBLEM,
                    "llm_input_renamed": "<problem> prompt </problem>",
                    "llm_output_renamed": "<proof> no aux here </proof>",
                },
            ]
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            kept, skipped = module.convert_dataset(input_path, output_path)
            written = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual((kept, skipped), (1, 1))
        self.assertEqual(
            written,
            [
                {
                    "query": "<problem> prompt </problem>",
                    "fl_problem": SAMPLE_FL_PROBLEM,
                    "response": SAMPLE_COMPLETION,
                }
            ],
        )

    def test_plugin_registers_aux_reward_without_swift_installed(self):
        swift_module = types.ModuleType("swift")
        rewards_module = types.ModuleType("swift.rewards")

        class DummyORM:
            pass

        rewards_module.ORM = DummyORM
        rewards_module.orms = {}
        swift_module.rewards = rewards_module

        with mock.patch.dict(
            "sys.modules",
            {"swift": swift_module, "swift.rewards": rewards_module},
        ):
            script_path = Path("scripts/grpo/plugin.py")
            spec = importlib.util.spec_from_file_location("grpo_plugin", script_path)
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(spec.loader)
            spec.loader.exec_module(module)

        self.assertIn("aux_reward", rewards_module.orms)

    def test_plugin_logs_each_rollout_breakdown_to_jsonl_and_wandb(self):
        swift_module = types.ModuleType("swift")
        rewards_module = types.ModuleType("swift.rewards")
        wandb_module = types.ModuleType("wandb")

        class DummyORM:
            pass

        rewards_module.ORM = DummyORM
        rewards_module.orms = {}
        swift_module.rewards = rewards_module
        wandb_module.run = object()
        wandb_module.log = mock.Mock()

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.dict(
                "sys.modules",
                {
                    "swift": swift_module,
                    "swift.rewards": rewards_module,
                    "wandb": wandb_module,
                },
            ),
        ):
            script_path = Path("scripts/grpo/plugin.py")
            spec = importlib.util.spec_from_file_location(
                "grpo_plugin_breakdown_test", script_path
            )
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(spec.loader)
            spec.loader.exec_module(module)

            breakdown_path = Path(tmp_dir) / "reward_breakdown.jsonl"
            reward = module.AuxReward(
                cpu_workers=1,
                reward_log_interval=1,
                reward_breakdown_path=str(breakdown_path),
            )
            results = [
                AuxEvaluationResult("aux-1", True, True, "solved", None, 1.0),
                AuxEvaluationResult("aux-2", True, True, "unsolved", None, 0.25),
                AuxEvaluationResult(
                    "aux-3", True, False, "build_invalid", "build", -0.25
                ),
                AuxEvaluationResult(
                    None, False, False, "format_invalid", "format", -0.5
                ),
            ]
            kwargs = {"trainer_state": types.SimpleNamespace(global_step=7)}
            reward._record_reward_rollout(results, kwargs)
            reward._record_reward_rollout(results, kwargs)
            reward.close()

            records = [
                json.loads(line) for line in breakdown_path.read_text().splitlines()
            ]

        self.assertEqual(
            [record["type"] for record in records],
            ["header", "rollout", "rollout"],
        )
        self.assertEqual(records[1]["step"], 7)
        self.assertEqual(records[1]["rollout_call"], 1)
        self.assertEqual(records[2]["rollout_call"], 2)
        self.assertEqual(records[1]["step_rollout_call"], 1)
        self.assertEqual(records[2]["step_rollout_call"], 2)
        self.assertEqual(records[1]["samples"], 4)
        self.assertEqual(records[2]["samples"], 8)
        self.assertEqual(
            records[1]["status_ratios"],
            {
                "solved": 0.25,
                "valid_unsolved": 0.25,
                "build_invalid": 0.25,
                "format_invalid": 0.25,
                "engine_error": 0.0,
            },
        )
        self.assertEqual(wandb_module.log.call_count, 2)
        metrics = wandb_module.log.call_args.args[0]
        self.assertEqual(metrics["reward_breakdown/samples"], 8)
        self.assertEqual(metrics["reward_breakdown/solved_ratio"], 0.25)
        self.assertEqual(metrics["reward_breakdown/format_invalid_ratio"], 0.25)
        self.assertEqual(wandb_module.log.call_args.kwargs, {"commit": False})


if __name__ == "__main__":
    unittest.main()

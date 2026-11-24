from __future__ import annotations

import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from newclid.agent.base import BaseAgent
from newclid.agent.vllm import (
    Qwen3Agent,
    Qwen3VLAgent,
    Qwen3VLTextAgent,
    _extract_aux_dsl,
    _parse_scored_choices,
)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [99]


def _agent_kwargs():
    return {
        "base_url": "http://localhost:8000",
        "served_model_name": "Qwen/Qwen3",
        "decoding_size": 2,
        "beam_size": 4,
        "search_depth": 1,
    }


class VLLMHelperTests(unittest.TestCase):
    def test_parse_scored_choices_rebuilds_aux_dsl(self):
        aux_dsl_scores, aux_dsl_thinks = _parse_scored_choices(
            choices=[
                {
                    "message": {"content": "free a</aux>"},
                    "token_ids": [17, 99],
                    "logprobs": {"content": [{"logprob": -0.2}, {"logprob": -1.0}]},
                }
            ],
            request={
                "response_prefix": "<aux> x00",
                "new_point_name": "a",
            },
            stop_token_id=99,
        )

        self.assertEqual(aux_dsl_scores, {"<aux> x00 a : free a": -0.2})
        self.assertEqual(aux_dsl_thinks, {"<aux> x00 a : free a": ""})

    def test_parse_scored_choices_can_extract_generated_aux_block(self):
        choices = [
            {
                "message": {
                    "content": " useful reasoning</think>\n\n<aux> x00 z : free z</aux>"
                },
                "token_ids": [17, 18, 99],
                "logprobs": {
                    "content": [
                        {"logprob": -0.4},
                        {"logprob": -0.2},
                        {"logprob": -1.0},
                    ]
                },
            },
            {
                "message": {"content": " no aux here"},
                "token_ids": [19],
                "logprobs": {"content": [{"logprob": -0.1}]},
            },
        ]
        request = {"response_prefix": "<aux> x00"}
        aux_dsl_scores, aux_dsl_thinks = _parse_scored_choices(
            choices=choices,
            request=request,
            stop_token_id=99,
            extract_aux_from_output=True,
        )

        self.assertEqual(list(aux_dsl_scores), ["<aux> x00 z : free z"])
        self.assertAlmostEqual(aux_dsl_scores["<aux> x00 z : free z"], -0.3)
        self.assertEqual(aux_dsl_thinks["<aux> x00 z : free z"], "useful reasoning")
        self.assertEqual(_extract_aux_dsl("x <aux> y </aux> z"), "<aux> y")


class Qwen3AgentTests(unittest.TestCase):
    def test_request_completions_uses_chat_endpoint_not_completions(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs())

        class _Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"content": "free a"},
                            "token_ids": [17],
                            "logprobs": {"content": [{"logprob": -0.2}]},
                        },
                        {
                            "message": {"content": "free b"},
                            "token_ids": [18],
                            "logprobs": {"content": [{"logprob": -0.3}]},
                        },
                    ]
                }

        with patch.object(agent.session, "post", return_value=_Response()) as mock_post:
            result = agent.request_completions(
                {
                    "request_id": "r0",
                    "messages": [],
                    "response_prefix": "<aux> x00",
                    "new_point_name": "a",
                }
            )

        self.assertIn("/v1/chat/completions", mock_post.call_args.args[0])
        self.assertNotIn("/v1/completions", mock_post.call_args.args[0])
        self.assertEqual(result["request_id"], "r0")
        self.assertEqual(len(result["aux_dsl_scores"]), 2)

    def test_request_completions_builds_thinking_generation_payload(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs(), think=True)

        class _Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": " reason</think>\n\n<aux> x00 z : free z"
                            },
                            "token_ids": [17],
                            "logprobs": {"content": [{"logprob": -0.2}]},
                        },
                        {
                            "message": {
                                "content": " other</think>\n\n<aux> x00 y : free y"
                            },
                            "token_ids": [18],
                            "logprobs": {"content": [{"logprob": -0.3}]},
                        },
                    ]
                }

        with patch.object(agent.session, "post", return_value=_Response()) as mock_post:
            result = agent.request_completions(
                {
                    "request_id": "r0",
                    "messages": [],
                    "response_prefix": "<aux> x00",
                }
            )

        payload = mock_post.call_args.kwargs["json"]
        self.assertFalse(payload["continue_final_message"])
        self.assertTrue(payload["add_generation_prompt"])
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("stop", payload)
        self.assertNotIn("stop_token_ids", payload)
        self.assertEqual(result["aux_dsl_scores"]["<aux> x00 z : free z"], -0.2)
        self.assertEqual(result["aux_dsl_thinks"]["<aux> x00 z : free z"], "reason")

    def test_request_completions_builds_non_thinking_continuation_payload(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs(), think=False)

        class _Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"content": "free a"},
                            "token_ids": [17],
                            "logprobs": {"content": [{"logprob": -0.2}]},
                        },
                        {
                            "message": {"content": "free b"},
                            "token_ids": [18],
                            "logprobs": {"content": [{"logprob": -0.3}]},
                        },
                    ]
                }

        with patch.object(agent.session, "post", return_value=_Response()) as mock_post:
            agent.request_completions(
                {
                    "request_id": "r0",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<think>\n\n</think>\n\n<aux> x00 a :",
                        }
                    ],
                    "response_prefix": "<aux> x00",
                    "new_point_name": "a",
                }
            )

        payload = mock_post.call_args.kwargs["json"]
        self.assertTrue(payload["continue_final_message"])
        self.assertFalse(payload["add_generation_prompt"])
        self.assertEqual(payload["max_tokens"], 100)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual(payload["stop"], [" ;"])
        self.assertEqual(payload["stop_token_ids"], [99])

    def test_hybrid_search_falls_back_to_v2_after_v1_failure_with_cumulative_counts(
        self,
    ):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs(), search_version="hybrid")
        agent.problemJGEX = object()
        calls: list[str] = []

        def fake_search(self, mode, proof, deadline):
            del proof, deadline
            calls.append(mode)
            self._llm_calls += 1
            self._ddar_calls += 2
            return (mode == "v2"), "Tried but failed."

        with patch("newclid.agent.base.run_ddar_c", return_value=False):
            with patch.object(agent, "prepare_search"):
                with patch.object(BaseAgent, "_search", new=fake_search):
                    with patch.object(BaseAgent, "_trace"):
                        with patch("newclid.agent.base.ray.put", return_value="ref"):
                            result = agent.run(
                                proof=types.SimpleNamespace(
                                    goals=[],
                                    defs={},
                                ),
                                rules=[],
                                timeout=30,
                            )

        self.assertEqual(calls, ["v1", "v2"])
        self.assertTrue(result["success"])
        self.assertEqual(result["llm_calls"], 2)
        self.assertEqual(result["ddar_calls"], 5)

    def test_search_exception_returns_cumulative_counts(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs())
        agent.problemJGEX = object()

        def fake_search(self, mode, proof, deadline):
            del mode, proof, deadline
            self._llm_calls += 3
            self._ddar_calls += 7
            raise RuntimeError("render failed")

        with patch("newclid.agent.base.run_ddar_c", return_value=False):
            with patch.object(agent, "prepare_search"):
                with patch.object(BaseAgent, "_search", new=fake_search):
                    with patch.object(BaseAgent, "_trace"):
                        with patch("newclid.agent.base.ray.put", return_value="ref"):
                            result = agent.run(
                                proof=types.SimpleNamespace(goals=[], defs={}),
                                rules=[],
                                timeout=30,
                            )

        self.assertFalse(result["success"])
        self.assertEqual(result["llm_calls"], 3)
        self.assertEqual(result["ddar_calls"], 8)
        self.assertEqual(result["error"], "RuntimeError: render failed")

    def test_search_ignores_single_lm_request_error(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs())
        agent.problemJGEX = object()
        trace_events: list[tuple[str, dict]] = []

        def fake_request_completions(request):
            if request["request_id"] == "bad":
                raise TimeoutError("request timed out")
            return {
                "request_id": request["request_id"],
                "aux_dsl_scores": {},
                "aux_dsl_thinks": {},
                "completed_at_unix_s": 10.0,
            }

        with patch.object(
            BaseAgent,
            "_build_requests",
            return_value=(
                [{"request_id": "bad"}, {"request_id": "good"}],
                {"bad": {"request": {}}, "good": {"request": {}}},
            ),
        ):
            with patch.object(
                agent, "request_completions", side_effect=fake_request_completions
            ):
                with patch.object(agent, "_submit", return_value=False):
                    with patch.object(
                        agent,
                        "_trace",
                        side_effect=lambda event, **payload: trace_events.append(
                            (event, payload)
                        ),
                    ):
                        solved, error = agent._search(
                            "v1",
                            proof=types.SimpleNamespace(),
                            deadline=time.time() + 30,
                        )

        self.assertFalse(solved)
        self.assertEqual(error, "Tried but failed.")
        self.assertEqual(agent._llm_calls, 2)
        self.assertTrue(
            any(
                event == "lm_error"
                and payload["request_id"] == "bad"
                and payload["error_type"] == "TimeoutError"
                for event, payload in trace_events
            )
        )

    def test_prepare_request_v2_reuses_root_problem_with_separator(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs(), search_version="v2")
        agent.problemJGEX = object()
        agent._root_problem_dsl = "<problem> root </problem>"
        with patch("newclid.agent.vllm.get_new_point_name", return_value="b"):
            request = agent.build_request(
                mode="v2",
                depth=0,
                request_id="d0_proot",
                problem=object(),
                aux_prefix=" x00 a : free a",
                proof=types.SimpleNamespace(defs={}),
            )

        self.assertEqual(request["query"], "<problem> root </problem>")
        self.assertEqual(request["response_prefix"], "<aux> x00 a : free a ; x00")

    def test_text_think_true_omits_assistant_prefix(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs(), think=True)
        agent.problemJGEX = object()
        with patch(
            "newclid.agent.vllm.problem_to_dsl",
            return_value="<problem> text </problem>",
        ) as mock_dsl:
            with patch("newclid.agent.vllm.get_new_point_name", return_value="a"):
                request = agent.build_request(
                    mode="v2",
                    depth=0,
                    request_id="d0_proot",
                    problem=object(),
                    aux_prefix="",
                    proof=types.SimpleNamespace(defs={}),
                )

        mock_dsl.assert_called_once()
        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "<problem> text </problem>"},
            ],
        )
        self.assertEqual(request["response_prefix"], "<aux> x00")
        self.assertEqual(request["new_point_name"], "")
        self.assertTrue(agent.server_info()["think"])

    def test_text_think_false_prefills_empty_think_and_aux_prefix(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3Agent(**_agent_kwargs(), think=False)
        agent.problemJGEX = object()
        agent._root_problem_dsl = "<problem> root </problem>"
        with patch("newclid.agent.vllm.get_new_point_name", return_value="a"):
            request = agent.build_request(
                mode="v2",
                depth=0,
                request_id="d0_proot",
                problem=object(),
                aux_prefix="",
                proof=types.SimpleNamespace(defs={}),
            )

        self.assertEqual(
            request["messages"][2]["content"],
            "<think>\n\n</think>\n\n<aux> x00 a :",
        )


class Qwen3VLAgentTests(unittest.TestCase):
    def test_build_requests_skips_frontier_entries_that_fail_to_render_remotely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()
            ):
                agent = Qwen3VLAgent(
                    **_agent_kwargs(),
                    search_version="v1",
                    render_root=Path(tmpdir),
                )
            traces = []
            with patch("newclid.agent.base.ray.is_initialized", return_value=True):
                with patch(
                    "newclid.agent.base.ray.get",
                    side_effect=[
                        RuntimeError("bad geometry"),
                        {
                            "request_id": "d1_p1",
                            "response_prefix": "<aux> x00",
                            "new_point_name": "b",
                        },
                    ],
                ):
                    with patch.object(
                        agent,
                        "_trace",
                        side_effect=lambda event, **kw: traces.append((event, kw)),
                    ):
                        with patch.object(
                            agent,
                            "build_request_remote_kwargs",
                            side_effect=[
                                {"request_id": "d1_p0"},
                                {"request_id": "d1_p1"},
                            ],
                        ):
                            with patch.object(
                                agent,
                                "build_request_from_remote_kwargs",
                                side_effect=["bad-ref", "good-ref"],
                            ):
                                requests, context = agent._build_requests(
                                    mode="v1",
                                    depth=1,
                                    frontier=[
                                        (0.0, ((0,), object(), "")),
                                        (1.0, ((1,), object(), " x00 a : free a")),
                                    ],
                                    proof=types.SimpleNamespace(defs={}),
                                )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["request_id"], "d1_p1")
        self.assertEqual(list(context), ["d1_p1"])
        self.assertEqual(traces[0][0], "request_build_error")
        self.assertEqual(traces[0][1]["error_message"], "bad geometry")
        self.assertEqual(traces[1][0], "lm_request")

    def test_vl_text_build_request_uses_text_only_messages_with_vl_prefix(self):
        with patch("newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()):
            agent = Qwen3VLTextAgent(**_agent_kwargs(), search_version="v1")
        agent.problemJGEX = object()
        with patch(
            "newclid.agent.vllm.problem_to_dsl",
            return_value="<problem> text only </problem>",
        ):
            with patch("newclid.agent.vllm.get_new_point_name", return_value="c"):
                request = agent.build_request(
                    mode="v1",
                    depth=0,
                    request_id="d0_proot",
                    problem=object(),
                    aux_prefix="",
                    proof=types.SimpleNamespace(defs={}),
                )

        self.assertEqual(request["request_id"], "d0_proot")
        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "<problem> text only </problem>"},
                {"role": "assistant", "content": "<aux> x00 c :"},
            ],
        )
        self.assertNotIn("image_data_url", request)
        self.assertEqual(request["response_prefix"], "<aux> x00")

    def test_prepare_request_rebuilds_proof_and_embeds_png_as_data_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            render_root = Path(tmpdir)
            with patch(
                "newclid.agent.vllm._load_tokenizer", return_value=_FakeTokenizer()
            ):
                agent = Qwen3VLAgent(
                    **_agent_kwargs(),
                    search_version="v1",
                    render_root=render_root,
                )
            agent.problemJGEX = object()
            rebuilt_proof = types.SimpleNamespace(defs={}, rng=object())
            problem = object()
            with patch(
                "newclid.agent.vllm.problem_to_dsl",
                return_value="<problem> visual </problem>",
            ):
                with patch("newclid.agent.vllm.get_new_point_name", return_value="c"):
                    with patch(
                        "newclid.agent.vllm.build_problem_proof",
                        return_value=rebuilt_proof,
                    ) as mock_build_proof:
                        with patch(
                            "newclid.agent.vllm.draw_clause_figure",
                            return_value=object(),
                        ):
                            with patch(
                                "newclid.agent.vllm.save_figure_as_png"
                            ) as mock_save:

                                def _write_png(fig, png_path, img_pixels, direct_png):
                                    del fig, img_pixels, direct_png
                                    Path(png_path).write_bytes(b"\x89PNG\r\n\x1a\nfake")

                                mock_save.side_effect = _write_png
                                request = agent.build_request(
                                    mode="v1",
                                    depth=0,
                                    request_id="d0_proot",
                                    problem=problem,
                                    aux_prefix="",
                                    proof=types.SimpleNamespace(defs={}),
                                )

        mock_build_proof.assert_called_once_with(problem, {})
        self.assertTrue(request["image_data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(request["messages"][2]["content"], "<aux> x00 c :")
        self.assertEqual(request["messages"][1]["content"][0]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()

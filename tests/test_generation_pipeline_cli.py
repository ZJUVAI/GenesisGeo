import sys
from types import SimpleNamespace

import pytest

from newclid.algebraic_reasoning.algebraic_manipulator import AlgebraicManipulator
from newclid.dependencies.dependency import Dependency
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.formulations.clause import Clause
from newclid.generation import pipeline as generation_pipeline
from newclid.generation.worker import ProblemWorker
from newclid.statement import Statement


@pytest.mark.parametrize(
    (
        "argv",
        "expected_using_log",
        "expected_using_exp",
        "expected_direct_png",
        "expected_img_pixels",
    ),
    [
        (
            ["pipeline.py", "--n_samples", "1", "--dir", "./tmp-datasets"],
            True,
            False,
            True,
            512,
        ),
        (
            [
                "pipeline.py",
                "--n_samples",
                "1",
                "--no-using_log",
                "--using_exp",
                "--no-direct_png",
                "--img_pixels",
                "768",
            ],
            False,
            True,
            False,
            768,
        ),
    ],
)
def test_pipeline_cli_image_render_args(
    monkeypatch,
    argv,
    expected_using_log,
    expected_using_exp,
    expected_direct_png,
    expected_img_pixels,
):
    captured = {}

    class DummyPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.writer = SimpleNamespace(output_dir="./tmp-datasets")
            self.file_prefix = "dummy"

        def generate(self):
            captured["generated"] = True

    monkeypatch.setattr(generation_pipeline, "ProblemPipeline", DummyPipeline)
    monkeypatch.setattr(
        generation_pipeline, "load_construction_config", lambda path: None
    )
    monkeypatch.setattr(generation_pipeline, "write_cli_args", lambda path, args: None)
    monkeypatch.setattr(sys, "argv", argv)

    generation_pipeline.main()

    assert captured["using_log"] is expected_using_log
    assert captured["using_exp"] is expected_using_exp
    assert captured["direct_png"] is expected_direct_png
    assert captured["img_pixels"] == expected_img_pixels
    assert captured["generated"] is True


def test_problem_worker_passes_csolver_equation_flags(monkeypatch):
    captured = {}

    class DummyCSolver:
        def __init__(self, problem, seed=None, solver=None, **kwargs):
            captured["problem"] = problem
            captured["seed"] = seed
            captured["solver"] = solver
            captured["kwargs"] = kwargs

        def run(self, max_level=500, custom_rules=None):
            captured["max_level"] = max_level
            return True

    dep_graph = DependencyGraph(AlgebraicManipulator())
    dummy_solver = SimpleNamespace(
        run_infos={"runtime": 0.0},
        proof=SimpleNamespace(
            dep_graph=dep_graph, symbols_graph=dep_graph.symbols_graph
        ),
    )
    dummy_builder = SimpleNamespace(problemJGEX=SimpleNamespace(constructions=[]))

    monkeypatch.setattr("newclid.generation.worker.CSolver", DummyCSolver)
    monkeypatch.setattr(
        ProblemWorker,
        "_build_solver",
        staticmethod(
            lambda fl_statement, max_attempts=1: (dummy_solver, dummy_builder)
        ),
    )
    monkeypatch.setattr(
        ProblemWorker,
        "_generate_possible_goals",
        staticmethod(lambda solver: ([], 0.0)),
    )
    monkeypatch.setattr(
        ProblemWorker,
        "_get_all_premise",
        staticmethod(lambda clauses, proof: ({}, {})),
    )

    data, summary = ProblemWorker._process_single_problem(
        (
            0,
            123,
            5,
            77,
            False,
            True,
            0,
            0,
            True,
            2,
            True,
            False,
            None,
            "a b c = triangle a b c ? perp a b b c",
        )
    )

    assert data == []
    assert summary["fl_statement"] == "a b c = triangle a b c ? perp a b b c"
    assert captured["max_level"] == 77
    assert captured["kwargs"]["using_log"] is False
    assert captured["kwargs"]["using_exp"] is True


def test_minimal_aux_search_passes_csolver_equation_flags(monkeypatch):
    captured_kwargs = []

    class DummyGoal:
        def to_str(self):
            return "goal"

        def check(self, explain=True):
            return False

    class DummyCSolver:
        def __init__(self, problem, seed=None, solver=None, **kwargs):
            captured_kwargs.append(kwargs)

        def run(
            self,
            max_level=500,
            *,
            custom_rules=None,
            goal_checks_only=False,
            goal_proofs_only=False,
        ):
            return False

    class DummyGeometricSolver:
        def __init__(self, proof_state, rules, agent):
            self.goals = [DummyGoal()]

    monkeypatch.setattr("newclid.generation.worker.CSolver", DummyCSolver)
    monkeypatch.setattr(
        "newclid.generation.worker.GeometricSolver", DummyGeometricSolver
    )
    monkeypatch.setattr(
        "newclid.generation.worker.ProofState.build_predicates",
        staticmethod(lambda **kwargs: SimpleNamespace()),
    )

    _, timings = ProblemWorker._find_minimal_aux_clauses_new(
        solver_builder=SimpleNamespace(defs={}, rules=[], seed=123),
        goals_str=["goal"],
        expanded_premises=[],
        expanded_aux_groups=[[SimpleNamespace()], [SimpleNamespace()]],
        aux_only=0,
        rng=SimpleNamespace(random=lambda: 1.0),
        using_log=False,
        using_exp=True,
    )

    assert len(captured_kwargs) == 4
    assert all(kwargs["using_log"] is False for kwargs in captured_kwargs)
    assert all(kwargs["using_exp"] is True for kwargs in captured_kwargs)
    assert set(timings) == {
        "build_predicates_time",
        "build_solver_time",
        "run_solver_time",
    }


def test_problem_worker_point_mapping_uses_predicate_order():
    mapping = ProblemWorker._create_point_mapping(
        ["d", "a", "c"],
        ["b"],
    )

    assert mapping == {
        "d": "a",
        "a": "b",
        "c": "c",
        "b": "d",
    }


def test_problem_worker_proof_reuses_canonicalized_premise_ids():
    source_dep_graph = DependencyGraph(AlgebraicManipulator())
    output_dep_graph = DependencyGraph(AlgebraicManipulator())
    premise = Statement.from_tokens(("coll", "x", "y", "z"), source_dep_graph)
    conclusion = Statement.from_tokens(("coll", "x", "y", "w"), source_dep_graph)
    assert premise is not None
    assert conclusion is not None

    mapping = {"x": "c", "y": "b", "z": "a", "w": "d"}
    clause = Clause(points=("x", "y", "z"), sentences=())
    dep_idx = {}

    problem = ProblemWorker._generate_problem_predicates_section(
        mapping,
        dep_idx,
        {clause: [(("x", "y", "z"), (premise,))]},
        [clause],
        [],
        output_dep_graph,
    )
    proof = ProblemWorker._generate_proof_section(
        mapping,
        dep_idx,
        [Dependency(conclusion, "r00", (premise,))],
        output_dep_graph,
    )

    assert "coll a b c [000]" in problem
    assert "coll b c d [001] r00 [000]" in proof
    assert "[002]" not in proof

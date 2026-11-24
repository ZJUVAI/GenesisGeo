from types import SimpleNamespace

from newclid.algebraic_reasoning.algebraic_manipulator import AlgebraicManipulator
from newclid.agent.base import BaseAgent
from newclid.configs import default_defs_path
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.evaluation.search_runtime import (
    problem_to_dsl,
    try_full_aux_dsl_to_constructions,
)
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.problem import ProblemJGEX
from newclid.generation.worker import ProblemWorker
from newclid.statement import Statement


def _load_defs():
    return DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))


def _identity_mapping(problem: ProblemJGEX):
    names: set[str] = set()
    for construction in problem.constructions:
        names.update(point.split("@")[0] for point in construction.points)
        for sentence in construction.sentences:
            names.update(sentence[1:])
    for goal in problem.goals:
        names.update(goal[1:])
    return {name: name for name in names}


def _generation_problem_dsl(problem: ProblemJGEX) -> str:
    source_dep_graph = DependencyGraph(AlgebraicManipulator())
    output_dep_graph = DependencyGraph(AlgebraicManipulator())
    clause2basics, _ = ProblemWorker._get_all_premise(
        list(problem.constructions),
        SimpleNamespace(dep_graph=source_dep_graph),
    )
    goals = [Statement.from_tokens(goal, source_dep_graph) for goal in problem.goals]
    assert all(goal is not None for goal in goals)
    return ProblemWorker._generate_problem_predicates_section(
        _identity_mapping(problem),
        {},
        clause2basics,
        list(problem.constructions),
        goals,
        output_dep_graph,
    )


def test_problem_dsl_uses_basic_order():
    problem = ProblemJGEX.from_text("a b c = triangle12 a b c ? cong a b a c")
    defs = _load_defs()

    dsl = problem_to_dsl(problem, defs)

    assert (
        dsl
        == "<problem> a : ; b : ; c : rconst a b a c 1/2 [000] ? cong a b a c </problem>"
    )

    problem = ProblemJGEX.from_text(
        "x y z i a b c = incenter2 x y z i a b c ? cong x y x y"
    )
    dsl = problem_to_dsl(problem, defs)

    assert (
        dsl
        == "<problem> i : eqangle a b a i a i a c [000] eqangle a c c i c i b c [001] ; "
        "x : coll b c x [002] perp b c i x [003] ; "
        "y : coll a c y [004] perp a c i y [005] ; "
        "z : coll a b z [006] perp a b i z [007] ? cong x y x y </problem>"
    )


def test_full_aux_dsl_parses_all_semicolon_separated_aux_points():
    assert (
        try_full_aux_dsl_to_constructions(
            "e : coll a b e [000] ; x00 f : coll a c f [001] ;"
        )
        == "e = on_line e a b; f = on_line f a c"
    )


def test_search_submit_renames_whole_problem_before_ddar(monkeypatch):
    class Agent(BaseAgent):
        def build_request(self, **kwargs):
            return {}

        def build_request_remote_kwargs(self, **kwargs):
            return {}

        def build_request_from_remote_kwargs(self, kwargs):
            return {}

        def request_completions(self, request):
            return {}

    class FakeRemote:
        def options(self, **kwargs):
            return self

        def remote(self, problem, *args, **kwargs):
            captured["problem"] = problem
            return "future"

    captured = {}
    monkeypatch.setattr("newclid.agent.base.run_ddar_remote", FakeRemote())
    defs = _load_defs()
    root = ProblemJGEX.from_text("a b = segment a b ? coll a b a")
    agent = Agent(decoding_size=1, beam_size=4, search_depth=2)
    agent.problemJGEX = root
    agent._defs_ref = "defs"
    agent._rules_ref = "rules"
    agent._max_pending = 10
    pending = []
    meta = {}

    solved = agent._submit(
        {
            "request_id": "d0_proot",
            "aux_dsl_scores": {"<aux> x00 z : coll a z b [000]": 0.0},
        },
        {
            "d0_proot": {
                "prev_score": 0.0,
                "path_key": (),
                "problem": root,
                "request": {"response_prefix": "<aux> x00"},
                "defs": defs,
            }
        },
        pending,
        meta,
        next_beam=[],
        last_depth=False,
        deadline=9999999999.0,
        mode="v1",
        depth=0,
    )

    assert solved is False
    assert str(captured["problem"]).startswith("a b = segment a b; c = on_line c a b")
    assert meta["future"]["child_aux_prefix"] == " x00 z : coll a z b [000]"
    assert meta["future"]["construction_text"] == "c = on_line c a b"
    assert meta["future"]["raw_construction_text"] == "z = on_line z a b"


def test_generation_construction_round_trips_to_eval_predicates():
    source_dep_graph = DependencyGraph(AlgebraicManipulator())
    output_dep_graph = DependencyGraph(AlgebraicManipulator())
    source_problem = ProblemJGEX.from_text(
        "a = free a; b = free b; x = on_line x a b ? coll a b x"
    )
    clause2basics, _ = ProblemWorker._get_all_premise(
        list(source_problem.constructions),
        SimpleNamespace(dep_graph=source_dep_graph),
    )
    goals = [Statement.from_tokens(("coll", "a", "b", "x"), source_dep_graph)]
    assert goals[0] is not None

    mapping = {"a": "c", "b": "b", "x": "a"}
    point_coords = {
        "a": (0.0, 0.0),
        "b": (1.0, 0.0),
        "c": (2.0, 0.0),
    }
    fl_problem = ProblemWorker._generate_problem_clauses_section(
        mapping,
        list(source_problem.constructions),
        goals,
        point_coords,
        output_dep_graph,
    )
    generation_dsl = ProblemWorker._generate_problem_predicates_section(
        mapping,
        {},
        clause2basics,
        list(source_problem.constructions),
        goals,
        output_dep_graph,
    )

    eval_dsl = problem_to_dsl(ProblemJGEX.from_text(fl_problem), _load_defs())

    assert eval_dsl == generation_dsl


def test_benchmark_problem_eval_and_generation_translation_match():
    problem = ProblemJGEX.from_text(
        "translated_imo_2019_p6\n"
        "a b c = triangle a b c; "
        "d e f i = incenter2 d e f i a b c; "
        "r = on_tline r d e f, on_circle r i d; "
        "p = on_line p r a, on_circle p i d; "
        "o1 = circle o1 p c e; "
        "o2 = circle o2 p b f; "
        "q = on_circle q o1 p, on_circle q o2 p; "
        "t = on_line t p q, on_line t i d ? perp a t a i"
    )

    assert _generation_problem_dsl(problem) == problem_to_dsl(problem, _load_defs())

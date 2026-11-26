"""Core regressions for numerical domains, equality transfer and symbol closure."""

import itertools
import math

import pytest

from newclid.api import DDAR
from newclid.algebraic_reasoning.tables import Table
from newclid.dependencies.symbols import Circle, Line, Point
from newclid.dependencies.symbols_graph import SymbolsGraph
from newclid.numerical.geometries import PointNum
from tests.native_predicate_probe import query


def test_congruence_rejects_native_close_vertices_and_preserves_reflection():
    translated = [
        ("A", 10000.0, 0.0),
        ("B", 10001.0, 0.0),
        ("C", 10000.0, 1.0),
        ("P", 0.0, 0.0),
        ("Q", 1.0, 0.0),
        ("R", 0.0, 1.0),
    ]
    direct = [
        ("A", 0.0, 0.0),
        ("B", 2.0, 0.0),
        ("C", 0.0, 1.0),
        ("P", 3.0, 2.0),
        ("Q", 5.0, 2.0),
        ("R", 3.0, 3.0),
    ]
    reflected = [("A", 0.0, 0.0), ("B", 2.0, 0.0), ("C", 0.0, 1.0), ("D", 0.0, -1.0)]
    assert query(
        [
            dict(points=translated, statements=[("contri", list("ABCPQR"))]),
            dict(
                points=direct,
                statements=[("contri", list("ABCPQR")), ("contrir", list("ABCPQR"))],
            ),
            dict(
                points=reflected,
                statements=[("contrir", list("ABCABD")), ("contri", list("ABCABD"))],
            ),
        ]
    ) == [[False], [True, False], [True, False]]


def test_angle_spelling_periodicity_and_invalid_input():
    points = [
        ("a", 0.0, 0.0),
        ("b", 1.0, 0.0),
        ("c", 0.0, 0.0),
        ("d", math.sqrt(3) / 2, 0.5),
    ]
    spellings = ["30o", "1/6", "1pi/6", "210o", "-150o"]
    assert query(
        [
            dict(
                points=points,
                statements=[("aconst", ["a", "b", "c", "d", s]) for s in spellings],
            )
        ]
    ) == [[True] * 5]
    parallel = [("a", 0.0, 0.0), ("b", 1.0, 0.0), ("c", 2.0, 0.0)]
    values = ["2147483647", "-2147483647", "0oops", "0/1junk", "0/0"]
    assert query(
        [
            dict(
                points=parallel,
                statements=[("aconst", ["a", "b", "a", "c", s]) for s in values],
            )
        ]
    ) == [[True, True, False, False, False]]


def test_angle_preserves_relative_tolerance_and_rejects_large_error():
    cases = []
    for angle, constant in [
        (math.pi / 4 + 1e-4, "45o"),
        (math.pi - 1e-4, "180o"),
        (math.pi / 4 + 0.01, "45o"),
    ]:
        points = [
            ("a", 0.0, 0.0),
            ("b", 1.0, 0.0),
            ("c", 0.0, 0.0),
            ("d", math.cos(angle), math.sin(angle)),
        ]
        cases.append(
            dict(points=points, statements=[("aconst", ["a", "b", "c", "d", constant])])
        )
    assert query(cases) == [[True], [True], [False]]


def test_equality_transfer_is_independent_of_premise_order():
    points = [
        ("p", 0.0, 0.0),
        ("q", 0.0, 0.0),
        ("r", 0.0, 0.0),
        ("a", 1.0, 0.0),
        ("b", 1.0, 1.0),
        ("c", 1.0, 2.0),
    ]
    premises = [
        ("perp", ["p", "a", "b", "c"]),
        ("eqpoint", ["p", "q"]),
        ("eqpoint", ["q", "r"]),
    ]
    config = {f"r{i:02d}": False for i in range(1000)}
    config.update(
        using_ar=False,
        using_log=False,
        using_exp=False,
        verbose=False,
        allow_duplicate_points=False,
        export_goal_proofs_only=True,
    )
    for order in itertools.permutations(premises):
        solved, graph = DDAR.run_ddar(
            "transfer", points, list(order), [("perp", ["r", "a", "b", "c"])], 3, config
        )
        assert solved and any(step[2] == "Transfer" for step in graph)
        # Every dependency must precede its conclusion in the exported proof.
        seen = set()
        for conclusion, dependencies, _ in graph:
            assert all(tuple(p) in seen for p in dependencies)
            seen.add(tuple(conclusion))
    solved, _ = DDAR.run_ddar(
        "coincidence", points, premises[:1], [("perp", ["r", "a", "b", "c"])], 3, config
    )
    assert not solved


def make_points(coordinates):
    graph = SymbolsGraph()
    points = {}
    for name, x, y in coordinates:
        p = graph.new_node(Point, name)
        p.num = PointNum(x, y)
        points[name] = p
    return graph, points


def test_line_merge_rescans_overlaps_without_merging_one_shared_point():
    graph, p = make_points([(n, float(i), 0.0) for i, n in enumerate("adgbx")])
    orphan = graph.line_thru_pair(p["b"], p["g"], Table())
    separate = graph.line_thru_pair(p["b"], p["x"], Table())
    Line.make_coll([p[n] for n in "adg"], None)
    representative, merged = Line.make_coll([p[n] for n in "abd"], None)
    assert orphan in merged and orphan.rep() is representative
    assert separate not in merged and separate.rep() is separate
    assert {p[n] for n in "abdg"} <= representative.points


def test_failed_numeric_line_does_not_register_partial_symbol():
    graph, p = make_points([("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    with pytest.raises(ValueError):
        graph.line_thru_pair(p["a"], p["b"], Table())
    assert not graph.nodes_of_type(Line) and "ab" not in graph.name2node


def test_circle_merge_rescans_overlaps_and_preserves_dependencies():
    from newclid.dependencies.dependency import Dependency
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement

    deps = DependencyGraph(None)
    coordinates = [
        ("a", 1.0, 0.0),
        ("b", 0.0, 1.0),
        ("c", -1.0, 0.0),
        ("d", 0.0, -1.0),
        ("e", 0.6, 0.8),
        ("f", -0.6, 0.8),
    ]
    for name, x, y in coordinates:
        p = deps.symbols_graph.new_node(Point, name)
        p.num = PointNum(x, y)
    premises = []
    for names in ("bdef", "abce", "abcd"):
        fact = Statement.from_tokens(("cyclic", *names), deps)
        Dependency.mk(fact, "Premise", ()).add()
        premises.append(fact)
    goal = Statement.from_tokens(("cyclic", *"adef"), deps)
    assert goal.check() and len(deps.symbols_graph.nodes_of_type(Circle)) == 1
    assert set(deps.hyper_graph[goal].why) == set(premises)

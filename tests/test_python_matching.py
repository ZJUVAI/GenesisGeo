"""Core Python/native matching regressions: binding, constants and goal anchors."""

import itertools
import math

import pytest

from newclid import matchinC
from newclid.api import GeometricSolverBuilder
from newclid.formulations.rule import Rule
from newclid.statement import Statement
from newclid.dependencies.dependency import Dependency


def test_incremental_bindings_preserve_repeated_variables_and_mapping_order():
    points = [(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    expected = []
    for b, a, m in itertools.product(range(4), repeat=3):
        if points[m] == (
            (points[a][0] + points[b][0]) / 2,
            (points[a][1] + points[b][1]) / 2,
        ):
            expected.append(dict(m=m, a=a, b=b))
    premise = ["midp", "m", "a", "b"]
    assert matchinC.mapping_normal_theorem([premise, premise], points) == expected
    assert matchinC.mapping_normal_theorem([], points) == [{}]


def test_scalar_constants_and_parallel_endpoint_reversal():
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)]
    identity = dict(a=0, b=1, c=2, d=3)
    for value in ("90o", "1/2", "1pi/2"):
        assert identity in matchinC.mapping_normal_theorem(
            [["aconst", "a", "b", "c", "d", value]], points
        )
    ratio_points = [(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 1.0)]
    for value in ("2", "2/1"):
        assert identity in matchinC.mapping_normal_theorem(
            [["rconst", "a", "b", "c", "d", value]], ratio_points
        )
    parallel = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for suffix in (["c", "d"], ["d", "c"]):
        assert identity in matchinC.mapping_normal_theorem(
            [["para", "a", "b", *suffix]], parallel
        )

    chain = []
    for angle in (math.pi / 6, math.pi / 6 + 0.0003, math.pi / 6 + 0.0006):
        chain.extend([(0.0, 0.0), (math.cos(angle), math.sin(angle))])
    assert dict(a=0, b=1, c=2, d=3, e=4, f=5) in matchinC.mapping_normal_theorem(
        [["para", *"abcdef"]], chain
    )


def test_malformed_atoms_and_unbound_equation_variables_raise():
    for atoms in ([[]], [["aconst"]], [["aconst", "a", "b", "c", "d", "1/0"]]):
        with pytest.raises(ValueError):
            matchinC.mapping_normal_theorem(atoms, [(0.0, 0.0)])
    with pytest.raises(ValueError):
        matchinC.mapping_eq_theorem(
            [["eqangle", *"abababab"], ["midp", "m", "a", "b"]],
            [(0.0, 0.0)],
            [(0,) * 8],
        )

    assert dict(a=0, b=1, c=0, d=1) in matchinC.mapping_normal_theorem(
        [["aconst", "a", "b", "c", "d", "-9223372036854775808/-1"]],
        [(0.0, 0.0), (1.0, 0.0)],
    )


def test_equation_premise_binds_points_outside_its_anchor():
    solver = (
        GeometricSolverBuilder(seed=1)
        .load_problem_from_txt(
            "a@-1_0 b@1_0 c@0_1 = triangle a b c; m@0_0 = midpoint m a b ? cong a m m b"
        )
        .build()
    )
    rule = Rule.from_string(
        "eqangle a b a c a b a c, midp m a b => cong a m m b", "anchored midpoint"
    )
    matches = list(solver.proof.matcher.match_theorem(rule))
    (goal,) = solver.goals
    assert any(dep.statement == goal for dep in matches)
    assert all(len(dep.why) == 2 for dep in matches)


def test_requested_ratio_anchor_enables_a_proof_without_assuming_the_goal():
    problem = (
        "b d f = triangle b d f; c = on_pline0 c d b f; "
        "g = foot g b d f; j = foot j c b f ? eqratio b f b g d f c j"
    )
    rule = (
        "two triangle areas\npara b f c d, perp b g d f, coll d f g, "
        "perp b f c j, coll b f j, ncoll b d f => eqratio b f b g d f c j\n"
    )
    base = GeometricSolverBuilder(seed=1004665).load_problem_from_txt(problem).build()
    assert not base.run(timeout=2)
    solver = (
        GeometricSolverBuilder(seed=1004665)
        .load_rules_from_txt(rule)
        .load_problem_from_txt(problem)
        .build()
    )
    assert solver.run(timeout=2)
    (goal,) = solver.goals
    assert solver.proof.dep_graph.hyper_graph[goal].reason == "two triangle areas"
    matcher = solver.proof.matcher
    assert matcher.ratio_goal_anchors
    matcher.update()
    assert not matcher.ratio_goal_anchors and not matcher.cache


def test_equation_matching_with_literal_constants_uses_complete_bindings():
    for text, target in [
        ("eqangle a b a c a b a c, aconst a b a c 90o => perp a b a c", "perp a b a c"),
        ("aconst a b a c 90o => eqangle a b a c a c a b", "eqangle a b a c a c a b"),
    ]:
        solver = (
            GeometricSolverBuilder(seed=1)
            .load_problem_from_txt("a@0_0 b@1_0 c@0_1 = triangle a b c ? " + target)
            .build()
        )
        graph = solver.proof.dep_graph
        fact = Statement.from_tokens(tuple("aconst a b a c 90o".split()), graph)
        Dependency.mk(fact, "Premise", ()).add()
        rule = Rule.from_string(text, "constant angle")
        (goal,) = solver.goals
        assert any(
            dep.statement == goal for dep in solver.proof.matcher.match_theorem(rule)
        )

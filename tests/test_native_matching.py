"""Core native matching regressions with and without shared indexes."""

import re

from newclid.api import DDAR
from newclid.configs import load_solver_config


def solve(rules, premises, optimized, points=None):
    points = points or [
        ("a", 0.0, 0.0),
        ("b", -1.0, 0.0),
        ("c", 1.0, 0.0),
        ("d", 0.0, 1.0),
    ]
    config = {f"r{i:02d}": False for i in range(1000)}
    config.update(
        using_ar=False,
        targeted_ar=False,
        using_log=False,
        using_exp=False,
        verbose=False,
        allow_duplicate_points=False,
    )
    config.update(
        {
            name: optimized
            for name in (
                "custom_projection_index",
                "custom_shared_prefix",
                "custom_early_guards",
                "custom_bitset_plan",
                "custom_numeric_memo",
            )
        }
    )
    return DDAR.run_ddar_with_custom_theorems(
        "matching", points, premises, [("cong", ["a", "b", "a", "c"])], rules, 3, config
    )


def test_shared_indexes_preserve_logical_premises_and_guards():
    facts = [("midp", ["a", "b", "c"]), ("perp", ["a", "b", "a", "d"])]
    for naming in ("upper", "long"):
        for guard, expected in [("ncoll a b d", True), ("ncoll a b c", False)]:
            rules = [
                f"r{i}|midp a b c,perp a b a d|cong a b a c|{guard}" for i in range(3)
            ]

            def rename(m):
                return m[0].upper() if naming == "upper" else "point_" + m[0]

            rules = [re.sub(r"\b[abcd]\b", rename, r) for r in rules]
            for known in (False, True):
                results = [
                    solve(rules, facts if known else [], flag) for flag in (False, True)
                ]
                assert results[0] == results[1]
                assert results[0][0] == (known and expected)


def test_shared_prefix_does_not_reuse_another_rules_guard_acceptance():
    facts = [("midp", ["a", "b", "c"]), ("perp", ["a", "b", "a", "d"])]
    rules = [
        "valid|midp a b c,perp a b a d|perp a b a d|ncoll a b d",
        "invalid|midp a b c,perp a b a d|cong a b a c|ncoll a b c",
    ]
    assert not solve(rules, facts, True)[0]


def test_custom_rule_guards_still_reject_aliased_points():
    points = [("a", 0.0, 0.0), ("b", 1.0, 0.0), ("c", 0.0, 1.0)]
    premises = [("perp", list("abac"))]
    goals = [("cong", list("abac"))]
    config = load_solver_config(
        using_ar=False, targeted_ar=False, using_log=False, using_exp=False
    )
    config.update({f"r{i:02d}": False for i in range(1000)})

    def solves(rule, settings):
        return DDAR.run_ddar_with_custom_theorems(
            "aliased_points", points, premises, goals, [rule], 3, settings
        )[0]

    four_variables = "four|perp A B C D|cong A B C D"
    assert solves(four_variables, config)
    assert not solves(four_variables, {**config, "allow_duplicate_points": False})
    distinct_segments = "|diff A C,diff A D,diff B C,diff B D"
    assert not solves(four_variables + distinct_segments, config)


def test_wide_rule_uses_the_non_bitset_fallback():
    body = ",".join(f"midp v{i} v{i + 1} v{i + 2}" for i in range(0, 66, 3))
    rule = f"wide|{body}|cong v0 v1 v0 v2"
    points = [("a", 0.0, 0.0), ("b", 1.0, 0.0), ("c", 0.0, 1.0)]
    assert solve([rule], [], False, points) == solve([rule], [], True, points)


def test_wrong_predicate_width_does_not_index_past_a_relation_row():
    for body in ("coll A B C D", "coll " + " ".join(f"v{i}" for i in range(34))):
        for enabled in (False, True):
            assert not solve(
                [body.replace("coll ", "bad|coll ", 1) + "|cong A B A C"], [], enabled
            )[0]


def test_ratio_index_contains_the_native_tolerance_window():
    points = [("a", 0.0, 0.0), ("b", 1.0, 0.0), ("c", 0.0, 1.0001)]
    premise = ("eqratio", ["a", "b", "a", "b", "a", "c", "a", "b"])
    goal = ("eqangle", ["b", "a", "b", "c", "c", "b", "c", "a"])
    rule = "isosceles|eqratio A B A B A C A B|eqangle B A B C C B C A"
    config = {f"r{i:02d}": False for i in range(1000)}
    config.update(
        using_ar=False,
        targeted_ar=False,
        using_log=False,
        using_exp=False,
        allow_duplicate_points=False,
        verbose=False,
    )
    for cached in (False, True):
        assert DDAR.run_ddar_with_custom_theorems(
            "ratio",
            points,
            [premise],
            [goal],
            [rule],
            3,
            {**config, "use_predicate_cache": cached},
        )[0]

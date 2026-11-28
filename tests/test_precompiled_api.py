"""Core compiled-bank/API compatibility and per-problem state isolation."""

import pytest

from newclid.api import CSolver, DDAR


def run(bank, *, known=True, changed=False, **overrides):
    points = [
        ("a", 0.0, 0.0),
        ("b", -1.0, 0.0),
        ("c", 1.0, 0.0),
        ("d", 1.0 if changed else 0.0, 1.0),
    ]
    facts = [("midp", ["a", "b", "c"]), ("ncoll", ["a", "b", "d"])] if known else []
    config = {f"r{i:02d}": False for i in range(1000)}
    config.update(
        using_ar=False,
        targeted_ar=False,
        using_log=False,
        using_exp=False,
        immutable_dependency_tokens=True,
        allow_duplicate_points=False,
    )
    config.update(overrides)
    runner = (
        DDAR.run_ddar_with_compiled_theorems
        if isinstance(bank, DDAR.CompiledRuleBank)
        else DDAR.run_ddar_with_custom_theorems
    )
    return runner(
        "bank", points, facts, [("perp", ["a", "b", "a", "d"])], bank, 3, config
    )


def test_text_and_compiled_banks_have_identical_proofs_and_guard_behavior():
    for guard in ("ncoll A B D", "ncoll A B C"):
        rules = [f"checked|midp A B C,ncoll A B D|perp A B A D|{guard}"]
        bank = DDAR.compile_custom_theorems(rules)
        assert len(bank) == 1 and run(bank) == run(rules)
        assert run(bank)[0] == (guard == "ncoll A B D")


def test_reusing_bank_does_not_share_numeric_or_logical_state():
    rules = ["checked|midp A B C,ncoll A B D|perp A B A D|ncoll A B D"]
    bank = DDAR.CompiledRuleBank(rules)
    rules.clear()
    assert run(bank)[0]
    assert not run(bank, known=False)[0]
    assert not run(bank, changed=True)[0]
    assert run(bank)[0]
    with pytest.raises(AttributeError):
        bank.rules = []


@pytest.mark.parametrize(
    "variables", [("p0", "p1", "p2", "p3"), ("p0", "p01", "A", "p26")]
)
def test_static_plan_preserves_rule_proofs(variables):
    a, b, c, d = variables
    rules = [
        f"valid|midp {a} {b} {c},ncoll {a} {b} {d}|perp {a} {b} {a} {d}|diff {a} {b}",
        f"blocked|midp {a} {b} {c},ncoll {a} {b} {d}|perp {a} {b} {a} {d}|ncoll {a} {b} {c}",
    ]
    bank = DDAR.compile_custom_theorems(rules)
    expected = run(rules)
    assert expected[0]
    for static_plan in (False, True):
        assert (
            run(bank, custom_static_plan=static_plan, custom_validate_plan=True)
            == expected
        )


def test_csolver_accepts_both_formats_and_empty_banks():
    problem = "a b c = triangle a b c; d = midpoint d a b ? cong a d b d"
    rules = ["midpoint|midp D A B|cong A D B D"]
    for light in (False, True):
        for bank in (
            rules,
            DDAR.compile_custom_theorems(rules),
            DDAR.compile_custom_theorems([]),
        ):
            assert CSolver(problem, seed=123, light=light, build_max_attempts=10).run(
                custom_rules=bank
            )
    for budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="build_max_attempts"):
            CSolver(problem, build_max_attempts=budget)


def test_compiler_rejects_silently_truncated_rules():
    for text in (
        "missing separators",
        "r||cong A B A C",
        "r|midp A B C|cong A B A C|ncoll",
        "r|midp A B C|cong A B A C|ncoll A B D|extra",
        "r|midp A B C,|cong A B A C",
    ):
        with pytest.raises(ValueError):
            DDAR.compile_custom_theorems([text])
    assert len(DDAR.compile_custom_theorems([])) == 0

import json

from newclid.discovery_experimental.certification import certify
from newclid.discovery_experimental.compression import Rule, compress_rule_bank
from newclid.discovery_experimental.compression.algebra import reduce_premises
from newclid.discovery_experimental.compression.coverage import (
    merge_conclusions,
    remove_renamed_covered_rules,
)
from newclid.discovery_experimental.compression.guards import trim_guards
from newclid.discovery_experimental.compression.premises import pack_premises
from newclid.discovery_experimental.pipeline import run


def test_stock_runtime_can_use_only_supported_guards(monkeypatch):
    from newclid.api import DDAR
    from newclid.discovery_experimental import runtime

    monkeypatch.setattr(runtime, "supports_extended_guards", lambda: False)
    monkeypatch.delattr(DDAR, "check_numerical_statements", raising=False)
    row = {
        "rule_id": "midpoint",
        "rule_text": "midp m a b => cong a m b m",
        "points": [
            {"name": "a", "x": 0, "y": 0},
            {"name": "b", "x": 2, "y": 0},
            {"name": "m", "x": 1, "y": 0},
        ],
    }
    result = certify((row, 1))
    assert result["status"] == "certified"
    assert result["backend"] == "flint"
    assert all(g.startswith("nperp ") for g in result["rule"]["guards"].split(", "))


def test_discovery_exports_only_deduplicated_ddar_rules(tmp_path):
    problem = {
        "seed": 5,
        "fl_problem": "a@0_0 = free a; b@2_0 = free b; m@1_0 = midpoint m a b ? cong a m b m",
        "llm_input_renamed": "<problem> m : midp m a b [001] ? cong a m b m </problem>",
        "llm_output_renamed": (
            "<aux> x00 u : coll a b u [002] ; </aux>"
            "<proof> cong a m b m [003] rule [001] [002] ; </proof>"
        ),
    }
    dataset = tmp_path / "sources.jsonl"
    dataset.write_text(
        json.dumps(problem) + "\n" + json.dumps({**problem, "seed": 6}) + "\n"
    )
    output = tmp_path / "result"
    report = run(dataset, output, workers=1, rule_timeout=30)
    assert report["source"] == {"proofs": 2, "occurrences": 2, "candidates": 1}
    assert report["certified"] == 1
    assert report["exported"] == 1
    assert "reduce_algebraic_premises" in report["compression"]
    assert {path.name for path in output.iterdir()} == {"rules_discovered.txt"}
    assert (output / "rules_discovered.txt").read_text().count("\n") == 1


def test_rule_compression_preserves_guards_and_requires_aliasing_for_contraction():
    premises = ("cong p0 p1 p0 p2", "cong p0 p1 p0 p3")
    conclusions = ("cong p1 p0 p2 p0",)
    narrow = Rule("narrow", premises, conclusions, ("diff p2 p3", "diff p2 p3"))
    broad = Rule("broad", premises, conclusions, ())

    [contracted] = compress_rule_bank([broad], allow_aliased_points=True)
    assert contracted.premises == premises[:1]
    assert contracted.conclusions == conclusions

    [distinct] = compress_rule_bank([broad], allow_aliased_points=False)
    assert distinct == broad

    [guarded] = compress_rule_bank([narrow], allow_aliased_points=True)
    assert guarded.premises == premises
    assert guarded.guards == ("diff p2 p3",)

    assert compress_rule_bank([narrow, broad], allow_aliased_points=True) == [
        contracted
    ]


def test_rename_coverage_requires_the_corresponding_guard():
    short = Rule("short", ("perp p0 p1 p0 p2",), ("ncoll p0 p1 p2",), ("diff p1 p2",))
    long = Rule(
        "long",
        ("perp p4 p5 p4 p6", "cong p4 p5 p4 p6"),
        ("ncoll p4 p5 p6",),
        ("diff p5 p6",),
    )
    assert remove_renamed_covered_rules([long, short]) == [short]
    unguarded = long._replace(guards=())
    assert remove_renamed_covered_rules([unguarded, short]) == [unguarded, short]


def test_merge_keeps_each_conclusion_and_preserves_distinct_guards():
    first = Rule("a", ("midp p0 p1 p2",), ("coll p0 p1 p2",), ())
    second = first._replace(name="b", conclusions=("cong p0 p1 p0 p2",))
    guarded = second._replace(name="guarded", guards=("diff p1 p2",))
    assert merge_conclusions([first, second, guarded]) == [
        first._replace(conclusions=first.conclusions + second.conclusions),
        guarded,
    ]


def test_pack_premises_preserves_the_existing_circle_domains():
    rule = Rule(
        "circle",
        ("cong p0 p1 p0 p2", "cong p0 p2 p0 p3"),
        ("cyclic p1 p2 p3 p4",),
        ("ncoll p1 p2 p3", "ncoll p0 p1 p2", "ncoll p0 p2 p3"),
    )
    assert pack_premises(rule) == rule._replace(premises=("circle p0 p1 p2 p3",))
    weak = rule._replace(guards=rule.guards[:1])
    assert pack_premises(weak) == weak
    midpoint = Rule(
        "midpoint", ("coll p0 p1 p2", "cong p0 p1 p0 p2"), ("coll p0 p1 p2",), ()
    )
    assert pack_premises(midpoint).premises == ("midp p0 p1 p2",)


def test_cyclic_removes_only_the_noncollinearity_check_it_performs():
    rule = Rule(
        "cyclic",
        ("cyclic p0 p1 p2 p3",),
        ("cong p0 p1 p2 p3",),
        ("ncoll p0 p1 p2", "ncoll p0 p1 p3"),
    )
    assert trim_guards(rule, bounded_coordinates=False).guards == ("ncoll p0 p1 p3",)


def test_algebraic_compression_uses_retained_length_relations():
    rule = Rule(
        "length_chain",
        ("cong p0 p1 p2 p3", "cong p2 p3 p4 p5", "cong p0 p1 p4 p5"),
        ("perp p0 p1 p4 p5",),
        ("diff p1 p4",),
    )
    assert reduce_premises(rule) == rule._replace(premises=rule.premises[:2])
    assert compress_rule_bank([rule], allow_aliased_points=False) == [
        rule._replace(premises=rule.premises[:2])
    ]
    assert compress_rule_bank(
        [rule], allow_aliased_points=False, algebraic_compression=False
    ) == [rule]


def test_rename_coverage_does_not_treat_constants_as_points():
    rule = Rule("half", ("coll p0 p1 p2",), ("rconst p0 p1 p1 p2 1/2",), ())
    third = rule._replace(name="third", conclusions=("rconst p0 p1 p1 p2 1/3",))
    assert remove_renamed_covered_rules([rule, third]) == [rule, third]

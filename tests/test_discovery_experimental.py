import json

from newclid.discovery_experimental.certification import certify
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


def test_stream_preserves_origins_after_deduplication(tmp_path):
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
    assert (output / "rules.pipe.txt").read_text().count("\n") == 1
    origins = [
        json.loads(line) for line in (output / "origins.jsonl").read_text().splitlines()
    ]
    assert [row["seed"] for row in origins] == [5, 6]

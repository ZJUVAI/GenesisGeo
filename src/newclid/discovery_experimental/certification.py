"""Select existing-point guards and qualify a rule for export."""

from collections import defaultdict
from itertools import combinations
from time import perf_counter

from .algebra.orientation_conditions import clock_candidates, side_candidates
from .algebra.search import prove
from .rules import canonical, rule_points, split_rule_text
from .runtime import (
    check_runtime,
    check_statements,
    prepare,
    stock_guard,
    supports_extended_guards,
)


def _candidate_pool(rule_text):
    p, g = rule_text.split(" => ")
    facts = [f.strip() for f in p.split(",")]
    goals = [f.strip() for f in g.split(",")]
    names = sorted({n for f in facts + goals for n in f.split()[1:]})
    pool = ["diff " + " ".join(pair) for pair in combinations(names, 2)]
    forbidden = {frozenset(f.split()[1:]) for f in goals if f.split()[0] == "ncoll"}
    pool.extend(
        "ncoll " + " ".join(t)
        for t in combinations(names, 3)
        if frozenset(t) not in forbidden
    )
    edges = set()
    for fact in facts + goals:
        kind, *args = fact.split()
        if kind in {"coll", "ncoll", "cyclic", "midp"}:
            edges.update(tuple(sorted(pair)) for pair in combinations(args, 2))
        elif kind in {"contri", "contrir"}:
            edges.update(
                tuple(sorted(pair))
                for triangle in (args[:3], args[3:])
                for pair in combinations(triangle, 2)
            )
        else:
            edges.update(tuple(sorted(pair)) for pair in zip(args[::2], args[1::2]))
    for left, right in combinations(sorted(e for e in edges if len(set(e)) == 2), 2):
        for kind in ("npara", "nperp"):
            if kind == "npara" and set(left) & set(right):
                continue
            pool.append(kind + " " + " ".join(left + right))
    return pool


def candidate_guards(row):
    rule = row["rule_text"]
    premises, goal = split_rule_text(rule)
    excluded = {canonical(f) for f in [*filter(None, premises), goal]}
    allowed = rule_points(rule, "")
    goal_names = set(goal.split()[1:])
    best = []
    basic = _candidate_pool(rule)
    orientation = clock_candidates(row, rule, max_pairs=24)
    for proposed in (basic, orientation, side_candidates(row, rule, max_pairs=16)):
        proposed = list(
            dict.fromkeys(
                f
                for f in proposed
                if set(f.split()[1:]) <= allowed and canonical(f) not in excluded
            )
        )
        proposed = [
            guard
            for guard, valid in zip(
                proposed, check_statements(row, proposed), strict=True
            )
            if valid
        ]
        families = defaultdict(list)
        for guard in proposed:
            families[guard.split()[0]].append(guard)
        limits = {
            "ncoll": 24,
            "npara": 20,
            "diff": 12,
            "nperp": 12,
            "sameclock": 16,
            "sameside": 16,
        }
        for family, options in families.items():
            ranked = sorted(
                options,
                key=lambda guard: (-len(set(guard.split()[1:]) & goal_names), guard),
            )
            best.extend(ranked[: limits[family]])
    best.extend(
        g
        for g in row.get("source_guards", ())
        if canonical(g) not in excluded
        and set(g.split()[1:]) <= allowed
        and check_statements(row, [g])[0]
    )
    if not supports_extended_guards():
        best = [stock_guard(guard) for guard in best]
    return list(dict.fromkeys(best))


def certify(task):
    row, singular_timeout = task
    text = row["rule_text"]
    timings = {}
    started = perf_counter()
    try:
        prepared = prepare(row)
        premises, goal = split_rule_text(prepared["runtime_rule_text"])
        required = list(prepared["guards"])
        for fact in [*premises, goal]:
            kind, *points = fact.split()
            if kind == "cyclic":
                required.extend("ncoll " + " ".join(t) for t in combinations(points, 3))
        required = list(dict.fromkeys(required))
        runtime = prepared["runtime_guards"]
        facts = [*premises, *required, *runtime, goal]
        failed = [
            fact
            for fact, holds in zip(facts, check_statements(row, facts), strict=True)
            if not holds
        ]
        timings["source"] = perf_counter() - started
        if failed:
            return {"status": "rejected_source", "reason": failed, "timings": timings}
        status, selected, backend = prove(
            prepared["runtime_rule_text"],
            required,
            lambda: candidate_guards(row),
            seconds=singular_timeout,
            timings=timings,
        )
    except (ValueError, KeyError, NotImplementedError) as error:
        return {"status": "unsupported", "reason": str(error), "timings": timings}
    if status != "certified":
        return {"status": status, "backend": backend, "timings": timings}

    start = perf_counter()
    conditions = []
    seen = {canonical(p) for p in premises}
    for guard in [*required, *runtime, *selected]:
        # Only syntactic duplicates and explicit premises are removed here.
        identity = canonical(guard)
        if identity not in seen:
            conditions.append(guard)
            seen.add(identity)
    exported = {
        **row,
        "rule_text": prepared["runtime_rule_text"],
        "guards": ", ".join(conditions),
    }
    if canonical(goal) in {canonical(g) for g in conditions}:
        return {"status": "circular_guard", "timings": timings}
    if rule_points(exported["rule_text"], exported["guards"]) != rule_points(text):
        return {"status": "changed_points", "timings": timings}
    checked = check_runtime(exported)
    timings["runtime"] = perf_counter() - start
    if not checked["passed"]:
        return {
            "status": "rejected_runtime",
            "backend": backend,
            "timings": timings,
            **({"reason": checked["error"]} if "error" in checked else {}),
        }
    return {
        "status": "certified",
        "backend": backend,
        "timings": timings,
        "rule": exported,
    }

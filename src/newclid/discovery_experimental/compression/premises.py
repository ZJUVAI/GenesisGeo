"""Rewrite rule premises using the original bound points."""

import re

from ..rules import canonical, parse_predicate, split_rule_text
from .guards import distinct_pairs
from .model import Rule


DISCOVERY_POINT = re.compile(r"p\d+\Z")


def point_names(literals: tuple[str, ...]) -> set[str]:
    return {
        token
        for literal in literals
        for token in parse_predicate(literal)[1]
        if DISCOVERY_POINT.fullmatch(token)
    }


def contract(rule: Rule) -> Rule:
    protected = point_names(rule.conclusions)
    while True:
        points = point_names(rule.premises)
        for source in sorted(points - protected):
            affected_premises = {
                s for s in rule.premises if source in parse_predicate(s)[1]
            }
            affected_guards = {
                s for s in rule.guards if source in parse_predicate(s)[1]
            }
            premises = tuple(s for s in rule.premises if s not in affected_premises)
            guards = tuple(s for s in rule.guards if s not in affected_guards)
            premise_keys = {canonical(s) for s in premises}
            guard_keys = {canonical(s) for s in guards}
            for target in sorted(points - {source}):
                if not all(
                    project(s, source, target) in premise_keys
                    for s in affected_premises
                ):
                    continue
                if not all(
                    project(s, source, target) in guard_keys for s in affected_guards
                ):
                    continue
                rule = rule._replace(premises=premises, guards=guards)
                break
            else:
                continue
            break
        else:
            return rule


def project(literal: str, source: str, target: str):
    kind, points = parse_predicate(literal)
    return canonical(" ".join([kind, *(target if p == source else p for p in points)]))


def simplify_collinear_parallel_premises(rule_text):
    premises, goal = split_rule_text(rule_text)
    lines = [
        set(args) for kind, args in map(parse_predicate, premises) if kind == "coll"
    ]
    shared = set()
    merged = []
    while lines:
        line = lines.pop()
        for index, other in enumerate(lines):
            overlap = sorted(line & other)
            if len(overlap) >= 2:
                shared.add(tuple(overlap[:2]))
                lines[index] |= line
                lines.extend(merged)
                merged.clear()
                break
        else:
            merged.append(line)

    kept = []
    guards = set()
    for premise in premises:
        kind, args = parse_predicate(premise)
        if kind not in {"para", "eqangle"} or not any(
            set(args) <= line for line in merged
        ):
            kept.append(premise)
            continue
        segments = set(zip(args[::2], args[1::2]))
        if any(a == b for a, b in segments):
            raise ValueError("degenerate premise segment")
        guards.update("diff " + " ".join(sorted(pair)) for pair in segments | shared)
    return {
        "rule_text": ", ".join(kept) + " => " + goal,
        "extra_guards": sorted(guards),
    }


def pack_premises(rule: Rule) -> Rule:
    """Use existing DDAR predicates for equivalent conjunctions.

    A circle checks the two consecutive radius equalities. Its noncollinearity
    and nonzero radii must already hold independently of those equalities.
    """
    active = dict.fromkeys(rule.premises)
    by_key = {
        canonical(fact): fact
        for fact in rule.premises
        if parse_predicate(fact)[0] in {"coll", "cong"}
    }
    packed = []
    for fact in rule.premises:
        kind, args = parse_predicate(fact)
        if kind != "coll" or len(set(args)) != 3:
            continue
        collinear = fact
        if collinear not in active:
            continue
        for middle in args:
            a, b = sorted(set(args) - {middle})
            lengths = by_key.get(canonical(f"cong {middle} {a} {middle} {b}"))
            if lengths in active:
                del active[collinear], active[lengths]
                packed.append(f"midp {middle} {a} {b}")
                break

    for guard in rule.guards:
        kind, args = parse_predicate(guard)
        if kind != "ncoll" or len(set(args)) != 3:
            continue
        for center in sorted(point_names(tuple(active)) - set(args)):
            for b in args:
                a, c = sorted(set(args) - {b})
                left = by_key.get(canonical(f"cong {center} {a} {center} {b}"))
                right = by_key.get(canonical(f"cong {center} {b} {center} {c}"))
                if left not in active or right not in active:
                    continue
                remaining = tuple(fact for fact in active if fact not in {left, right})
                distinct = distinct_pairs(rule._replace(premises=remaining))
                if not all(
                    tuple(sorted((center, point))) in distinct for point in args
                ):
                    continue
                del active[left], active[right]
                packed.append(f"circle {center} {a} {b} {c}")
                break
    return rule._replace(premises=tuple(dict.fromkeys((*active, *packed))))

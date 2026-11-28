"""Run compression passes on certified DDAR rules."""

from time import perf_counter

from .algebra import reduce_premises
from .coverage import (
    merge_conclusions,
    remove_covered_rules,
    remove_renamed_covered_rules,
)
from .guards import trim_guards
from .model import Rule, split
from .premises import contract, pack_premises, simplify_collinear_parallel_premises

__all__ = [
    "Rule",
    "compress_rule_bank",
    "simplify_collinear_parallel_premises",
    "split",
]


def compress_rule_bank(
    source: list[Rule],
    *,
    allow_aliased_points: bool,
    bounded_coordinates: bool = False,
    algebraic_compression: bool = True,
    statistics: dict | None = None,
) -> list[Rule]:
    def apply(name, transform):
        nonlocal source
        started = perf_counter()
        source = transform(source)
        if statistics is not None:
            statistics[name] = {
                "rules": len(source),
                "premises": sum(len(rule.premises) for rule in source),
                "guards": sum(len(rule.guards) for rule in source),
                "conclusions": sum(len(rule.conclusions) for rule in source),
                "seconds": perf_counter() - started,
            }

    if statistics is not None:
        apply("input", lambda rules: rules)
    if allow_aliased_points:
        apply("contract_points", lambda rules: [contract(rule) for rule in rules])
    apply(
        "trim_guards",
        lambda rules: [trim_guards(rule, bounded_coordinates) for rule in rules],
    )
    apply("remove_covered_rules", remove_covered_rules)
    apply("remove_renamed_covered_rules", remove_renamed_covered_rules)
    apply("pack_premises", lambda rules: [pack_premises(rule) for rule in rules])
    if algebraic_compression:
        apply(
            "reduce_algebraic_premises",
            lambda rules: [reduce_premises(rule) for rule in rules],
        )
    apply("merge_conclusions", merge_conclusions)
    return source

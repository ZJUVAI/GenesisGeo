"""Remove covered rules and share identical antecedents."""

from collections import Counter, defaultdict
from functools import lru_cache
import re

from ..rules import canonical, orbit, parse_predicate
from .model import Rule


def remove_covered_rules(rules: list[Rule]) -> list[Rule]:
    groups = defaultdict(list)
    for rule in rules:
        groups[frozenset(map(canonical, rule.conclusions))].append(rule)

    covered = set()
    for group in groups.values():
        retained = []
        for rule in sorted(group, key=lambda r: len(r.premises) + len(r.guards)):
            premises = frozenset(map(canonical, rule.premises))
            guards = frozenset(map(canonical, rule.guards))
            if any(p <= premises and g <= guards for p, g in retained):
                covered.add(rule.name)
            else:
                retained.append((premises, guards))
    return [rule for rule in rules if rule.name not in covered]


# These symmetries preserve one predicate's equation. Circle radius comparisons
# and numerical guards keep their argument order.
_SYMMETRIC = {"coll", "cong", "para", "perp", "eqangle", "eqratio", "cyclic", "midp"}
_POINT = re.compile(r"p[0-9]+\Z")


def _argument_pattern(args):
    names = {}
    return tuple(names.setdefault(arg, len(names)) for arg in args)


@lru_cache(maxsize=65536)
def _shape(literal, symmetric):
    kind, args = parse_predicate(literal)
    variants = orbit(literal) if symmetric and kind in _SYMMETRIC else (args,)
    return kind, min(map(_argument_pattern, variants))


def _literals(rule):
    return [
        (field, _shape(literal, field != 2), literal)
        for field, items in enumerate((rule.conclusions, rule.premises, rule.guards))
        for literal in items
    ]


def _match(source, target_options, remaining=2048):
    tasks = []
    for field, key, literal in source:
        variants = target_options.get((field, key))
        if not variants:
            return False
        tasks.append((field, parse_predicate(literal)[1], variants))
    tasks.sort(key=lambda task: (task[0] != 0, len(task[2])))

    def search(index, bound, used):
        nonlocal remaining
        if index == len(tasks):
            return True
        _, args, candidates = tasks[index]
        for candidate in candidates:
            if remaining == 0:
                return False
            remaining -= 1
            extended, taken = bound.copy(), used.copy()
            for a, b in zip(args, candidate, strict=True):
                if a in extended:
                    if extended[a] != b:
                        break
                elif b in taken:
                    break
                else:
                    extended[a] = b
                    taken.add(b)
            else:
                if search(index + 1, extended, taken):
                    return True
        return False

    return search(0, {}, set())


def _options(literals):
    options = defaultdict(set)
    for field, key, literal in literals:
        kind, args = parse_predicate(literal)
        options[field, key].update(
            orbit(literal) if field != 2 and kind in _SYMMETRIC else (args,)
        )
    return {key: tuple(sorted(values)) for key, values in options.items()}


def remove_renamed_covered_rules(rules: list[Rule]) -> list[Rule]:
    """Drop an implication only when an injective renaming proves coverage."""
    groups = defaultdict(list)
    covered = set()
    ordered = sorted(rules, key=lambda r: len(r.premises) + len(r.guards))
    for rule in ordered:
        if len(rule.conclusions) != 1:
            continue
        if any(
            not _POINT.fullmatch(arg)
            for literal in (*rule.premises, *rule.conclusions, *rule.guards)
            for arg in parse_predicate(literal)[1]
        ):
            continue
        literals = _literals(rule)
        counts = Counter((field, key) for field, key, _ in literals)
        retained = groups[_shape(rule.conclusions[0], True)]
        options = None
        # Coverage is optional. Limiting search merely leaves more rules intact.
        for fingerprint, source in retained[:128]:
            if fingerprint <= counts:
                if options is None:
                    options = _options(literals)
                if _match(source, options):
                    covered.add(rule.name)
                    break
        else:
            retained.append((counts, literals))
    return [rule for rule in rules if rule.name not in covered]


def merge_conclusions(rules: list[Rule]) -> list[Rule]:
    """Match exact antecedents once while retaining all their conclusions."""
    groups = {}
    merged = {}
    for rule in rules:
        key = frozenset(rule.premises), frozenset(rule.guards)
        if key in groups:
            name = groups[key]
            first = merged[name]
            conclusions = tuple(dict.fromkeys((*first.conclusions, *rule.conclusions)))
            merged[name] = first._replace(conclusions=conclusions)
        else:
            groups[key] = rule.name
            merged[rule.name] = rule
    return list(merged.values())

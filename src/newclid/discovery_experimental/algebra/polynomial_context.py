"""Drop line identities implied by collinear premises and explicit domains."""

from ..rules import parse_predicate, split_rule_text


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

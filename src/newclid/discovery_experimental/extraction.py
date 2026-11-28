"""Read generated DDAR proofs and extract their auxiliary-free boundaries."""

import math
import re
from hashlib import sha256

from .rules import canonical, is_point, orbit

_FACT = re.compile(r"([^\[\];:]+?)\s*\[(\d+)\]")
_ID = re.compile(r"\[(\d+)\]")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_COORDINATE = re.compile(rf"\b([A-Za-z][A-Za-z0-9_]*)@({_NUMBER})_({_NUMBER})")
SUPPORTED = frozenset(
    {
        "coll",
        "cyclic",
        "para",
        "perp",
        "cong",
        "eqangle",
        "eqratio",
        "midp",
        "circle",
        "contri",
        "contrir",
        "simtri",
        "simtrir",
        "diff",
        "ncoll",
        "npara",
        "nperp",
        "sameclock",
    }
)


def section(text, tag):
    return text.partition(f"<{tag}>")[2].partition(f"</{tag}>")[0]


def point_names(fact):
    return {token for token in fact.split()[1:] if is_point(token)}


def facts(text):
    return {
        identifier: " ".join(body.split()) for body, identifier in _FACT.findall(text)
    }


def normalize_fact(fact):
    kind, *args = fact.split()
    if (
        kind == "para"
        and len(set(args)) == 3
        and len(args) == 4
        and all(a != b for a, b in zip(args[::2], args[1::2]))
    ):
        fact = "coll " + " ".join(sorted(set(args)))
    kind, args = canonical(fact)
    return " ".join((kind, *args))


def rewrite_angle(fact, coordinates):
    if not fact.startswith("eqangle ") or len(fact.split()) != 9:
        return fact
    for a, b, c, d, e, f, g, h in orbit(fact):
        if a == e and b == d and c == g and f == h and len({a, b, c, f}) == 4:
            return f"cyclic {a} {b} {c} {f}"
        if a == g and b == c == e and d == f == h and len({a, b, d}) == 3:
            if {a, b, d} <= coordinates.keys() and math.isclose(
                math.dist(coordinates[a], coordinates[b]),
                math.dist(coordinates[a], coordinates[d]),
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                return f"cong {a} {b} {a} {d}"
    return fact


def normalize(premises, goal, coordinates):
    if any(f.split()[0] not in SUPPORTED for f in (*premises, goal)):
        return None
    premises = sorted(
        {normalize_fact(rewrite_angle(fact, coordinates)) for fact in premises}
    )
    goal = normalize_fact(rewrite_angle(goal, coordinates))
    if goal in premises:
        return None
    names = dict.fromkeys(
        token for f in premises for token in f.split()[1:] if is_point(token)
    )
    if not point_names(goal) <= names.keys() or not names.keys() <= coordinates.keys():
        return None
    rename = {name: f"p{i}" for i, name in enumerate(names)}

    def renamed(fact):
        kind, *args = fact.split()
        return normalize_fact(" ".join((kind, *(rename.get(arg, arg) for arg in args))))

    text = ", ".join(sorted(map(renamed, premises))) + " => " + renamed(goal)
    points = [
        {"name": name, "x": coordinates[old][0], "y": coordinates[old][1]}
        for old, name in rename.items()
    ]
    return text, points, rename


def extract(record, line):
    problem = section(record["llm_input_renamed"], "problem").partition("?")[0]
    output = record["llm_output_renamed"]
    aux = section(output, "aux")
    coordinates = {
        name: (float(x), float(y))
        for name, x, y in _COORDINATE.findall(record["fl_problem"])
    }
    auxiliaries = {
        token
        for part in aux.split(";")
        if ":" in part
        for token in part.split(":", 1)[0].split()[1:]
    } - coordinates.keys()
    if not auxiliaries:
        return
    numerical = facts(section(output, "numerical_check"))
    trivial = facts(section(output, "trivial"))
    known = facts(problem)
    known.update(facts(aux))
    known.update(numerical)
    known.update(trivial)
    dependencies = {}
    for step in section(output, "proof").split(";"):
        match = _ID.search(step)
        if match is None:
            continue
        root = match.group(1)
        known[root] = " ".join(step[: match.start()].split())
        dependencies[root] = _ID.findall(step[match.end() :])

    internal = {key for key, fact in known.items() if point_names(fact) & auxiliaries}
    for root, inputs in dependencies.items():
        if root in internal or not inputs:
            continue
        frontier, checks, visited = set(), set(), set()
        pending = list(inputs)
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in internal:
                pending.extend(dependencies.get(node, ()))
            elif node in numerical:
                checks.add(node)
            elif node in known and node not in trivial:
                frontier.add(node)
        if not (internal & visited) or not frontier:
            continue
        if not visited <= known.keys():
            continue
        normalized = normalize(
            [known[node] for node in sorted(frontier)], known[root], coordinates
        )
        if normalized is None:
            continue
        text, points, rename = normalized
        hints = []
        for node in sorted(checks):
            kind, *args = known[node].split()
            if kind in {
                "diff",
                "ncoll",
                "npara",
                "nperp",
                "sameclock",
                "sameside",
            } and all(arg in rename for arg in args):
                hints.append(" ".join((kind, *(rename[arg] for arg in args))))
        yield {
            "rule_id": sha256(text.encode()).hexdigest()[:24],
            "rule_text": text,
            "points": points,
            "seed": record.get("seed"),
            "line": line,
            "proof_step": root,
            "source_guards": hints,
        }

"""Pure operations on rule text, predicate symmetries, and source point names."""

import re
from functools import lru_cache
from itertools import permutations

_PREPARSE_GENERATORS: tuple[tuple[int, ...], ...] = (
    (1, 0, 2, 3, 4, 5, 6, 7),  # seg1 内部交换
    (0, 1, 3, 2, 4, 5, 6, 7),  # seg2 内部
    (0, 1, 2, 3, 5, 4, 6, 7),  # seg3 内部
    (0, 1, 2, 3, 4, 5, 7, 6),  # seg4 内部
    (2, 3, 0, 1, 6, 7, 4, 5),  # seg1↔seg2 且 seg3↔seg4（两角同时反向）
    (4, 5, 6, 7, 0, 1, 2, 3),  # (seg1,seg2)↔(seg3,seg4)（两角整体交换）
    (0, 1, 4, 5, 2, 3, 6, 7),  # seg2↔seg3（转置）
    (6, 7, 2, 3, 4, 5, 0, 1),  # seg1↔seg4
)


def _build_group8(gens: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """由生成元求对称群的全部置换（闭包，模块加载时算一次）。"""
    identity = tuple(range(8))
    group: set[tuple[int, ...]] = {identity}
    frontier = [identity]
    while frontier:
        cur = frontier.pop()
        for g in gens:
            nxt = tuple(cur[g[i]] for i in range(8))
            if nxt not in group:
                group.add(nxt)
                frontier.append(nxt)
    return tuple(group)


_PREPARSE_GROUP: tuple[tuple[int, ...], ...] = _build_group8(_PREPARSE_GENERATORS)


@lru_cache(maxsize=4096)
def is_point(token):
    return re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token) is not None


@lru_cache(maxsize=65536)
def parse(text):
    predicate, *args = text.split()
    return predicate, tuple(args)


@lru_cache(maxsize=65536)
def orbit(text):
    predicate, args = parse(text)
    if predicate in {"coll", "ncoll", "cyclic", "diff"}:
        values = permutations(args)
    elif predicate in {"cong", "perp", "para", "nperp", "npara"}:
        first, second = args[:2], args[2:]
        values = (
            left + right
            for a, b in ((first, second), (second, first))
            for left in (a, a[::-1])
            for right in (b, b[::-1])
        )
    elif predicate in {"eqangle", "eqratio"}:
        values = (tuple(args[i] for i in p) for p in _PREPARSE_GROUP)
    elif predicate == "midp":
        values = [args, (args[0], args[2], args[1])]
    elif predicate == "circle":
        values = (args[:1] + p for p in permutations(args[1:]))
    elif predicate == "sameclock":

        def rotations(triangle):
            return triangle, triangle[1:] + triangle[:1], triangle[2:] + triangle[:2]

        ordered = [a + b for a in rotations(args[:3]) for b in rotations(args[3:])]
        values = ordered + [v[3:] + v[:3] for v in ordered]
    elif predicate in {"contri", "contrir", "simtri", "simtrir"}:
        ordered = [
            tuple(args[i] for i in p) + tuple(args[i + 3] for i in p)
            for p in permutations(range(3))
        ]
        values = ordered + [v[3:] + v[:3] for v in ordered]
    else:
        values = [args]
    return tuple(sorted(set(values)))


@lru_cache(maxsize=65536)
def canonical(text):
    return parse(text)[0], min(orbit(text))


def split_rule_text(text):
    premises, conclusion = text.split(" => ", 1)
    return [
        item.strip() for item in premises.split(",") if item.strip()
    ], conclusion.strip()


def parse_predicate(text):
    return parse(text.strip())


def rule_points(text, guards=""):
    premises, conclusion = split_rule_text(text)
    literals = [*premises, conclusion, *filter(None, map(str.strip, guards.split(",")))]
    return frozenset(
        name for item in literals for name in item.split()[1:] if is_point(name)
    )


def to_pipe_format(identifier, text, guards=""):
    premises, conclusion = split_rule_text(text)
    fields = [identifier, ",".join(premises), conclusion]
    if guards:
        fields.append(guards)
    return "|".join(fields)

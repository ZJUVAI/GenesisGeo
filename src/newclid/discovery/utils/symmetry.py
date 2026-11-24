from __future__ import annotations

from typing import Callable
from newclid.discovery.data_models import PredicateInstance


Args = tuple[str, ...]

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def compare_args(arg1: str, arg2: str) -> int:
    letter1, digits1 = arg1[0], arg1[1:]
    letter2, digits2 = arg2[0], arg2[1:]
    n1 = int(digits1) if digits1 != "" else -1
    n2 = int(digits2) if digits2 != "" else -1
    if n1 != n2:
        return -1 if n1 < n2 else 1
    if letter1 != letter2:
        return -1 if letter1 < letter2 else 1
    return 0


# ---------------------------------------------------------------------------
# 谓词 → 对称性类型 查表
# ---------------------------------------------------------------------------

_UNORDERED: frozenset[str] = frozenset({"coll", "ncoll", "cyclic"})
_SWAP_PAIRS: frozenset[str] = frozenset(
    {"cong", "para", "npara", "perp", "nperp", "eqpoint"}
)
_PREPARSE: frozenset[str] = frozenset({"eqangle", "eqratio"})
_VERTEX_MAP: frozenset[str] = frozenset({"contri", "simtri", "contrir", "simtrir"})
_HEAD_FIXED: frozenset[str] = frozenset({"midp", "constline", "circle"})
_SPECIAL: frozenset[str] = frozenset({"sameside"})
_NONE: frozenset[str] = frozenset({"aconst", "rconst"})

# ---------------------------------------------------------------------------
# 各对称性类型的规范化实现（待补全）
# ---------------------------------------------------------------------------


def _normalize_unordered(args: Args) -> Args:
    """所有参数按字典序排序。"""
    from functools import cmp_to_key

    args = sorted(args, key=cmp_to_key(compare_args))
    return args


def _normalize_swap_pairs(args: Args) -> Args:
    """参数按 2 个一组分组，组内排序，组间排序。"""
    from functools import cmp_to_key

    pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    normalized_pairs = [
        tuple(sorted(pair, key=cmp_to_key(compare_args))) for pair in pairs
    ]
    normalized_pairs.sort(
        key=cmp_to_key(
            lambda p1, p2: compare_args(p1[0], p2[0]) or compare_args(p1[1], p2[1])
        )
    )
    return tuple(arg for pair in normalized_pairs for arg in pair)


# eqangle / eqratio 对称群生成元（作用于 8 个参数位置；4 条线段 seg1..seg4）。
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


def _normalize_preparse(args: Args) -> Args:
    """8 参数 eqangle / eqratio 的规范化：在对称群下取字典序最小的等价写法。

    结构：eqangle a b c d e f g h 表示 ∠(AB,CD) = ∠(EF,GH)；eqratio 同构。
    合法对称（作用于 4 条线段 seg1..seg4，见 _PREPARSE_GENERATORS）：
      - 段内可换（线无方向）：a↔b, c↔d, e↔f, g↔h
      - 两角整体交换：(seg1,seg2) ↔ (seg3,seg4)
      - 两角同时反向：seg1↔seg2 且 seg3↔seg4
      - 转置（有向角/比例恒等式）：seg2↔seg3；seg1↔seg4

    注意：单独交换 seg1↔seg2（而不同时换 seg3↔seg4）**不是**合法对称
    （有向角会变号），故不能对两个半组各自独立排序——必须在真正的对称群
    轨道里取最小。
    """
    from functools import cmp_to_key

    def _tuple_cmp(t1: Args, t2: Args) -> int:
        for x, y in zip(t1, t2):
            c = compare_args(x, y)
            if c != 0:
                return c
        return 0

    orbit = {tuple(args[p[i]] for i in range(8)) for p in _PREPARSE_GROUP}
    return min(orbit, key=cmp_to_key(_tuple_cmp))


def _normalize_vertex_map(args: Args) -> Args:
    """6 参数按 3+3 分组，三种顶点循环对应取字典序最小。适用：contri, simtri。

    simtri/contri a b c d e f 表示 △ABC ∼/≅ △DEF。
    对称性：顶点对应循环等价，即
      (a,b,c,d,e,f) ~ (b,c,a,e,f,d) ~ (c,a,b,f,d,e)
    取三种循环中字典序最小的。
    """
    a, b, c, d, e, f = args
    candidates = [
        (a, b, c, d, e, f),
        (b, c, a, e, f, d),
        (c, a, b, f, d, e),
        (d, e, f, a, b, c),
        (e, f, d, b, c, a),
        (f, d, e, c, a, b),
        (a, c, b, d, f, e),
        (c, b, a, f, e, d),
        (b, a, c, e, d, f),
        (d, f, e, a, c, b),
        (f, e, d, c, b, a),
        (e, d, f, b, a, c),
    ]

    def cmp_candidate(t1: tuple, t2: tuple) -> int:
        for x, y in zip(t1, t2):
            c = compare_args(x, y)
            if c != 0:
                return c
        return 0

    best = candidates[0]
    for cand in candidates[1:]:
        if cmp_candidate(cand, best) < 0:
            best = cand
    return best


def _normalize_head_fixed(args: Args) -> Args:
    """首参固定，其余参数按字典序排序。"""
    from functools import cmp_to_key

    head = args[0]
    rest = tuple(sorted(args[1:], key=cmp_to_key(compare_args)))
    return (head,) + rest


def _normalize_sameside(args: Args) -> Args:
    """sameside a b c x y z：a 在 b→c 的同侧如 x 在 y→z 的同侧。

    对称性：
      - (b, c) 内部可换
      - (y, z) 内部可换
      - 两个三元组 (a,b,c) ↔ (x,y,z) 可换
    """
    from functools import cmp_to_key

    a, b, c, x, y, z = args
    # 内部排序
    bc = tuple(sorted((b, c), key=cmp_to_key(compare_args)))
    yz = tuple(sorted((y, z), key=cmp_to_key(compare_args)))
    cand1 = (a,) + bc + (x,) + yz
    cand2 = (x,) + yz + (a,) + bc

    def cmp_tuple(t1: tuple, t2: tuple) -> int:
        for u, v in zip(t1, t2):
            r = compare_args(u, v)
            if r != 0:
                return r
        return 0

    return cand1 if cmp_tuple(cand1, cand2) <= 0 else cand2


_SPECIAL_HANDLERS: dict[str, Callable[[Args], Args]] = {
    "sameside": _normalize_sameside,
}


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


def normalize_predicate(predicate: PredicateInstance) -> Args:
    """按谓词 `name` 的对称性类型规范化参数 `args`，返回规范化后的元组。"""
    type = predicate.predicate
    args = predicate.args

    if type in _UNORDERED:
        return _normalize_unordered(args)
    if type in _SWAP_PAIRS:
        return _normalize_swap_pairs(args)
    if type in _PREPARSE:
        return _normalize_preparse(args)
    if type in _VERTEX_MAP:
        return _normalize_vertex_map(args)
    if type in _HEAD_FIXED:
        return _normalize_head_fixed(args)
    if type in _SPECIAL:
        return _SPECIAL_HANDLERS[type](args)
    if type in _NONE:
        return args

    raise ValueError(f"Unknown predicate for symmetry normalization: {type!r}")


if __name__ == "__main__":
    # 仅测试主接口 normalize_predicate
    test_cases: list[tuple[str, tuple[str, ...]]] = [
        # unordered
        ("coll", ("a", "c", "b")),
        ("cyclic", ("d", "b", "a", "c")),
        # swap-pairs
        ("cong", ("d", "a", "c", "b")),
        ("para", ("b", "a", "d", "c")),
        ("perp", ("c", "d", "a", "b")),
        ("eqpoint", ("b", "a")),
        # preparse
        ("eqangle", ("b", "a", "d", "c", "f", "e", "h", "g")),
        ("eqratio", ("h", "g", "c", "d", "f", "e", "b", "a")),
        # vertex-map
        ("simtri", ("c", "a", "b", "f", "d", "e")),
        ("contri", ("f", "e", "d", "b", "c", "a")),
        # head-fixed
        ("midp", ("m", "b", "a")),
        ("constline", ("a", "q", "p")),
        ("circle", ("o", "c", "a", "b")),
        # special
        ("sameside", ("a", "c", "b", "x", "z", "y")),
        ("sameside", ("x", "z", "y", "a", "c", "b")),
        # none
        ("aconst", ("a", "b", "c", "d", "x")),
        ("rconst", ("d", "c", "a", "b", "r")),
        # 带数字后缀的参数测试
        ("coll", ("a1", "a", "b0")),
        ("cong", ("b", "a1", "a", "b0")),
    ]

    print(f"{'Predicate':<10} {'Input':<40} -> {'Output'}")
    print("-" * 90)
    for name, args in test_cases:
        pred = PredicateInstance(predicate=name, args=args)
        try:
            out = normalize_predicate(pred)
            print(f"{name:<10} {str(args):<40} -> {out}")
        except Exception as exc:
            print(f"{name:<10} {str(args):<40} -> ERROR: {exc}")

"""eqpoint 等价点合并（Part 1 命题后处理）。

``eqpoint a b`` 表示点 a 与点 b 数值重合——通常是随机采点采样时的偶然退化，
而不是有意义的几何条件。若直接把 eqpoint 保留为前提，生成的规则里会出现
"因为两个点碰巧是同一个点"这种偶然事实（例如 conclusion 直接就是某条
premise 换了个点名），价值不大甚至是平凡的。

本模块对每条命题的 eqpoint 前提做 Union-Find 合并：
1. 把每一对 eqpoint(a, b) 的两个点名合并为同一等价类，代表元取字典序最小者。
2. 用代表元重写其余前提与结论中的点名。
3. 去掉 eqpoint 前提本身。
4. 重写后若某条前提退化（同一预测的点参数出现重复）则丢弃该前提。
5. 重写后若结论退化，或结论与某条前提完全相同（规则变得平凡），整条命题丢弃。
"""

from __future__ import annotations

from newclid.discovery.data_models import PredicateInstance, PropositionRecord

Pred = PredicateInstance


def _extract_eqpoint_pairs(premises: tuple[Pred, ...]) -> list[tuple[str, str]]:
    return [
        (p.args[0], p.args[1])
        for p in premises
        if p.predicate == "eqpoint" and len(p.args) == 2
    ]


class _UnionFind:
    """字符串键的并查集，代表元固定取字典序最小者。"""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra <= rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb


def _build_representative_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    uf = _UnionFind()
    for a, b in pairs:
        uf.union(a, b)
    points = {p for pair in pairs for p in pair}
    return {p: uf.find(p) for p in points}


def _rewrite(pred: Pred, rep: dict[str, str]) -> Pred:
    return PredicateInstance(
        predicate=pred.predicate,
        args=tuple(rep.get(a, a) for a in pred.args),
    )


# 退化判断需要按谓词的参数结构分组检查——不同"线段"/"三角形"之间共享端点是
# 合法的几何情况（如 perp a e e g 表示两条线段在 e 点垂直相交），只有同一
# 线段/同一三角形内部出现重复点才算退化。分组沿用 utils/symmetry.py 的谓词分类。
_SEGMENT_PAIR_PREDICATES = frozenset(
    {"cong", "para", "npara", "perp", "nperp", "eqpoint"}
)
_SEGMENT_QUAD_PREDICATES = frozenset({"eqangle", "eqratio", "aconst", "rconst"})
# 否定形式(ncoll/nperp/npara)与正向形式共享同一套"参数须两两不同"约束——
# ncoll a b c 断言 A,B,C 不共线，若 b==c 之类同名参数出现，命题本身已无意义
# (退化前就该丢弃)，之前漏掉 ncoll 导致 "coll C E E" 这类命题被放行。
_ALL_DISTINCT_PREDICATES = frozenset(
    {"coll", "ncoll", "cyclic", "midp", "constline", "circle"}
)
_TRIANGLE_PAIR_PREDICATES = frozenset({"contri", "simtri", "contrir", "simtrir"})


def _has_dup(args: tuple[str, ...]) -> bool:
    return len(set(args)) < len(args)


def _segments(point_args: tuple[str, ...]) -> list[frozenset[str]]:
    """按 2 个一组切出线段集合（用集合比较，忽略线段内部顺序）。"""
    return [frozenset(point_args[i : i + 2]) for i in range(0, len(point_args) - 1, 2)]


def _is_degenerate(pred: Pred) -> bool:
    """按谓词的几何结构判断重写后的参数是否退化。

    两类退化：
    - 同一线段/三角形内部点重合（如 cong a b a d 里线段 (a,b) 变成了 (a,a)）。
    - 两条本应独立的线段被重写为完全相同的线段，谓词退化为自比较的恒真句
      （如 eqpoint 合并后 para a f a g 变成 para a f a f）。
    """
    name, args = pred.predicate, pred.args
    if name in _SEGMENT_PAIR_PREDICATES:
        # 4 参数 = 2 条线段 (0,1) (2,3)，线段内两点不可相同，两条线段也不可完全相同
        segs = _segments(args)
        if any(_has_dup(args[i : i + 2]) for i in range(0, len(args) - 1, 2)):
            return True
        return len(set(segs)) < len(segs)
    if name in _SEGMENT_QUAD_PREDICATES:
        # 8/5 参数中前面部分是 4 条线段 (0,1)(2,3)(4,5)(6,7)，末尾常量参数忽略
        point_args = tuple(a for a in args if a and not a[0].isdigit())
        if any(
            _has_dup(point_args[i : i + 2]) for i in range(0, len(point_args) - 1, 2)
        ):
            return True
        segs = _segments(point_args)
        # 参照 DDAR EqAngle::trivial() / EqRatio::trivial()：
        # 谓词形式为 (seg0,seg1) = (seg2,seg3)（左角/比 = 右角/比），恒真只有两种情况：
        # 1) 两侧线段对完全相同 (seg0,seg1) == (seg2,seg3)：断言"角/比等于自身"；
        # 2) 两侧各自内部线段相同 seg0==seg1 且 seg2==seg3：两侧同时退化为 0/1，恒为 0=0 或 1=1。
        # 仅一侧内部线段相同、另一侧不同，不是恒真句，而是在断言另一侧也必须退化为 0/1，
        # 这是真实的几何约束（如 eqangle c j c j e j f j 断言 EJ∥FJ），不能丢弃。
        if len(segs) == 4:
            return (segs[0] == segs[2] and segs[1] == segs[3]) or (
                segs[0] == segs[1] and segs[2] == segs[3]
            )
        return False
    if name in _TRIANGLE_PAIR_PREDICATES:
        # 6 参数 = 2 个三角形 (0,1,2) (3,4,5)，三角形内三点须两两不同
        return _has_dup(args[0:3]) or _has_dup(args[3:6])
    if name in _ALL_DISTINCT_PREDICATES:
        return _has_dup(args)
    if name == "sameside":
        # sameside a b c x y z：(a,b,c) 与 (x,y,z) 各自内部须两两不同
        return _has_dup(args[0:3]) or _has_dup(args[3:6])
    return _has_dup(args)


def merge_eqpoints(prop: PropositionRecord) -> PropositionRecord | None:
    """合并一条命题中的所有 eqpoint 等价点。

    Returns
    -------
    PropositionRecord | None
        无 eqpoint 前提时原样返回；合并后规则退化为平凡时返回 None。
    """
    pairs = _extract_eqpoint_pairs(prop.premises)
    if not pairs:
        return prop

    rep = _build_representative_map(pairs)

    rewritten_premises: list[Pred] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for p in prop.premises:
        if p.predicate == "eqpoint":
            continue
        rp = _rewrite(p, rep)
        if _is_degenerate(rp):
            continue
        key = (rp.predicate, rp.args)
        if key in seen:
            continue
        seen.add(key)
        rewritten_premises.append(rp)

    conclusion = _rewrite(prop.conclusion, rep)
    if _is_degenerate(conclusion):
        return None
    if (conclusion.predicate, conclusion.args) in seen:
        return None  # 结论与某条前提完全相同，规则平凡

    used_points = {a for p in [*rewritten_premises, conclusion] for a in p.args}
    points = tuple(pt for pt in prop.points if pt.name in used_points)

    return PropositionRecord(
        proposition_id=prop.proposition_id,
        seed=prop.seed,
        index_in_seed=prop.index_in_seed,
        premises=tuple(rewritten_premises),
        conclusion=conclusion,
        points=points,
    )


def merge_eqpoints_batch(
    props: list[PropositionRecord],
) -> tuple[list[PropositionRecord], int, int]:
    """对一批命题做 eqpoint 合并。

    Returns
    -------
    (kept, merged_count, dropped_count)
        kept: 处理后的命题列表（未涉及 eqpoint 的原样保留）。
        merged_count: 实际发生了 eqpoint 合并且保留下来的命题数。
        dropped_count: 合并后判定为平凡而丢弃的命题数。
    """
    kept: list[PropositionRecord] = []
    merged_count = 0
    dropped_count = 0
    for prop in props:
        has_eqpoint = any(p.predicate == "eqpoint" for p in prop.premises)
        result = merge_eqpoints(prop)
        if result is None:
            dropped_count += 1
            continue
        if has_eqpoint:
            merged_count += 1
        kept.append(result)
    return kept, merged_count, dropped_count

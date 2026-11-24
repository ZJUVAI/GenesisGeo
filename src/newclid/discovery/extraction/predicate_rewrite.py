"""谓词转化（规范化 Step 1）。

把语义等价、但更易处理的谓词就地替换（1:1，不新增规则）：
- ``para w x y z``（两线段共点）           → ``coll`` 三个不同点
- ``eqangle a c b c a d b d``（模对称）    → ``cyclic a b c d``
- ``eqangle a b b c b c a c``（模对称）    → ``cong a b a c``（需坐标校验 |ab|≈|ac|）

eqangle 对称群较大（含有向角转置恒等式），故采用**对称轨道匹配**：枚举该 eqangle
在对称群下的全部等价写法，与模板精确匹配，命中才转化，零误判。
"""

from __future__ import annotations

import math

Args = tuple[str, ...]
Pred = tuple[str, Args]
Coord = dict[str, tuple[float, float]]


# ---------------------------------------------------------------------------
# eqangle 对称轨道
# ---------------------------------------------------------------------------
# 参数分 4 段（角的 4 条线）：seg1=(0,1) seg2=(2,3) seg3=(4,5) seg4=(6,7)。
_EQANGLE_GENERATORS: tuple[tuple[int, ...], ...] = (
    (1, 0, 2, 3, 4, 5, 6, 7),  # seg1 内部交换
    (0, 1, 3, 2, 4, 5, 6, 7),  # seg2 内部
    (0, 1, 2, 3, 5, 4, 6, 7),  # seg3 内部
    (0, 1, 2, 3, 4, 5, 7, 6),  # seg4 内部
    (2, 3, 0, 1, 6, 7, 4, 5),  # (ab,cd)->(cd,ab),(ef,gh)->(gh,ef)
    (4, 5, 6, 7, 0, 1, 2, 3),  # 两个角整体交换 seg12 <-> seg34
    (0, 1, 4, 5, 2, 3, 6, 7),  # 转置: seg2 <-> seg3 (cd<->ef)
    (6, 7, 2, 3, 4, 5, 0, 1),  # seg1 <-> seg4 (ab<->gh)
)


def _apply(perm: tuple[int, ...], args: Args) -> Args:
    return tuple(args[perm[i]] for i in range(len(perm)))


def _build_group(
    generators: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """由生成元求出对称群的全部置换（闭包，只算一次）。"""
    identity = tuple(range(8))
    group: set[tuple[int, ...]] = {identity}
    frontier = [identity]
    while frontier:
        cur = frontier.pop()
        for g in generators:
            nxt = tuple(cur[g[i]] for i in range(8))
            if nxt not in group:
                group.add(nxt)
                frontier.append(nxt)
    return tuple(group)


# 预计算 eqangle 对称群（模块加载时算一次），避免每次调用重复 BFS。
_EQANGLE_GROUP: tuple[tuple[int, ...], ...] = _build_group(_EQANGLE_GENERATORS)


def _eqangle_orbit(args: Args) -> set[Args]:
    """eqangle 参数在对称群下的全部等价写法。"""
    return {tuple(args[p[i]] for i in range(8)) for p in _EQANGLE_GROUP}


# ---------------------------------------------------------------------------
# 各转化规则
# ---------------------------------------------------------------------------


def _rewrite_para(args: Args) -> Pred | None:
    """para w x y z，若两线段共点则转 coll（三个不同点）。"""
    if len(args) != 4:
        return None
    w, x, y, z = args
    if not ({w, x} & {y, z}):
        return None
    union: list[str] = []
    for p in (w, x, y, z):
        if p not in union:
            union.append(p)
    if len(union) != 3:  # 退化（线段相同等）不转
        return None
    return ("coll", tuple(union))


def _match_cyclic(orbit: set[Args]) -> Pred | None:
    """匹配 eqangle a c b c a d b d → cyclic a b c d。"""
    for t in orbit:
        if t[0] == t[4] and t[1] == t[3] and t[2] == t[6] and t[5] == t[7]:
            a, c, b, d = t[0], t[1], t[2], t[5]
            if len({a, b, c, d}) == 4:
                return ("cyclic", (a, b, c, d))
    return None


def _match_cong_shape(orbit: set[Args]) -> tuple[str, str, str] | None:
    """匹配 eqangle a b b c b c a c 的结构，返回 (a, b, c)（未做数值校验）。"""
    for t in orbit:
        if t[0] == t[6] and t[1] == t[2] == t[4] and t[3] == t[5] == t[7]:
            a, b, c = t[0], t[1], t[3]
            if len({a, b, c}) == 3:
                return (a, b, c)
    return None


def _dist(coord: Coord, p: str, q: str) -> float | None:
    if p not in coord or q not in coord:
        return None
    (x1, y1), (x2, y2) = coord[p], coord[q]
    return math.hypot(x1 - x2, y1 - y2)


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


def rewrite_predicate(name: str, args: Args, coord: Coord) -> Pred:
    """对单个谓词尝试转化；未命中则原样返回 (name, args)。"""
    if name == "para":
        r = _rewrite_para(args)
        if r is not None:
            return r
    elif name == "eqangle" and len(args) == 8:
        orbit = _eqangle_orbit(args)
        cyc = _match_cyclic(orbit)
        if cyc is not None:
            return cyc
        shape = _match_cong_shape(orbit)
        if shape is not None:
            a, b, c = shape
            dab, dac = _dist(coord, a, b), _dist(coord, a, c)
            if (
                dab is not None
                and dac is not None
                and math.isclose(dab, dac, rel_tol=1e-6, abs_tol=1e-9)
            ):
                return ("cong", (a, b, a, c))
    return (name, tuple(args))

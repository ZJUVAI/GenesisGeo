from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# 规则文本拆分
# ---------------------------------------------------------------------------


def split_rule_text(rule_text: str) -> tuple[list[str], str]:
    """将 rule_text 拆分为 (premise_strings, conclusion_string)。

    Parameters
    ----------
    rule_text : str
        如 "cong a b c d, perp a b b c => para a c d b"

    Returns
    -------
    (premises, conclusion)
        premises: ["cong a b c d", "perp a b b c"]
        conclusion: "para a c d b"
    """
    lhs, rhs = rule_text.split("=>")
    premises = [p.strip() for p in lhs.split(",")]
    conclusion = rhs.strip()
    return premises, conclusion


def parse_predicate(clause_str: str) -> tuple[str, tuple[str, ...]]:
    """解析单个谓词字符串为 (name, args)。

    Parameters
    ----------
    clause_str : str
        如 "cong a b c d"

    Returns
    -------
    (name, args)
        ("cong", ("a", "b", "c", "d"))
    """
    parts = clause_str.strip().split()
    return parts[0], tuple(parts[1:])


def parse_rule_text(
    rule_text: str,
) -> tuple[list[tuple[str, tuple[str, ...]]], tuple[str, tuple[str, ...]]]:
    """完整解析规则文本为结构化数据。

    Returns
    -------
    (premise_predicates, conclusion_predicate)
        premise_predicates: [("cong", ("a","b","c","d")), ...]
        conclusion_predicate: ("para", ("a","c","d","b"))
    """
    premises, conclusion = split_rule_text(rule_text)
    premise_predicates = [parse_predicate(p) for p in premises]
    conclusion_predicate = parse_predicate(conclusion)
    return premise_predicates, conclusion_predicate


# ---------------------------------------------------------------------------
# 点名收集与重命名
# ---------------------------------------------------------------------------


def collect_points_in_order(
    premises: list[tuple[str, tuple[str, ...]]],
    conclusion: tuple[str, tuple[str, ...]],
) -> list[str]:
    """按首次出现顺序收集所有点名（先 premises 后 conclusion）。"""
    seen: dict[str, None] = {}
    for _, args in [*premises, conclusion]:
        for p in args:
            if p and not p[0].isdigit() and p not in seen:
                seen[p] = None
    return list(seen)


def build_rename_map(points_in_order: list[str]) -> dict[str, str]:
    """生成 original → a, b, c, ... 的重命名映射。"""

    def to_name(i: int) -> str:
        s, n = "", i + 1
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(ord("a") + r) + s
        return s

    return {p: to_name(i) for i, p in enumerate(points_in_order)}


def rename_rule_text(rule_text: str, rename_map: dict[str, str]) -> str:
    """将 rule_text 中的点名按 rename_map 替换。"""
    premises, conclusion = parse_rule_text(rule_text)

    def fmt(pred: tuple[str, tuple[str, ...]]) -> str:
        name, args = pred
        return " ".join([name, *(rename_map.get(a, a) for a in args)])

    return ", ".join(fmt(p) for p in premises) + " => " + fmt(conclusion)


def rename_by_first_appearance(rule_text: str) -> str:
    """解析 rule_text，按首次出现顺序重命名所有点为 a, b, c, ...，返回重命名后的 rule_text。"""
    premises, conclusion = parse_rule_text(rule_text)
    points = collect_points_in_order(premises, conclusion)
    return rename_rule_text(rule_text, build_rename_map(points))


# ---------------------------------------------------------------------------
# 规则文本重建
# ---------------------------------------------------------------------------


def build_rule_text(
    premises: Sequence[tuple[str, tuple[str, ...]]],
    conclusion: tuple[str, tuple[str, ...]],
) -> str:
    """从结构化数据重建规则文本字符串。

    Returns
    -------
    str
        如 "cong a b c d, perp a b b c => para a c d b"
    """

    def fmt(pred: tuple[str, tuple[str, ...]]) -> str:
        name, args = pred
        return " ".join([name, *args])

    lhs = ", ".join(fmt(p) for p in premises)
    rhs = fmt(conclusion)
    return f"{lhs} => {rhs}" if lhs else f"=> {rhs}"


# ---------------------------------------------------------------------------
# CSolver pipe 格式转换
# ---------------------------------------------------------------------------


def to_pipe_format(rule_id: str, rule_text: str, guards: str = "") -> str:
    """将规则转为 CSolver 接受的 pipe 自定义规则格式。

    Parameters
    ----------
    rule_id : str
        规则 ID，如 "r00000042_0"
    rule_text : str
        规范化规则文本，如 "cong a b c d => para e f g h"
    guards : str
        可选的守卫谓词（逗号分隔），只做数值检查不参与推理

    Returns
    -------
    str
        CSolver pipe 格式字符串。格式: id|premises|conclusions[|guards]
    """
    if "=>" not in rule_text:
        return f"{rule_id}||"
    premise_part, conclusion_part = rule_text.split("=>", 1)
    premises = ",".join(p.strip() for p in premise_part.split(",") if p.strip())
    conclusions = ",".join(c.strip() for c in conclusion_part.split(",") if c.strip())
    pipe = f"{rule_id}|{premises}|{conclusions}"
    if guards.strip():
        pipe += f"|{guards.strip()}"
    return pipe


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def count_premises(rule_text: str) -> int:
    """计算规则中 premises 的数量。"""
    raise NotImplementedError


def extract_conclusion_predicate(rule_text: str) -> str:
    """提取结论谓词名称。

    Parameters
    ----------
    rule_text : str
        如 "cong a b c d => para e f g h"

    Returns
    -------
    str
        "para"
    """
    raise NotImplementedError


def has_missing_points(rule_text: str) -> bool:
    """检查 conclusion 中是否存在未在 premises 中出现的点。"""
    raise NotImplementedError


if __name__ == "__main__":
    tests = [
        "cong a b c d, perp a b b c => para a c d b",
        "coll a b c => para a b a c",
        "eqangle a b c d e f g h => simtri a b c d e f",
    ]
    for rule in tests:
        premises, conclusion = parse_rule_text(rule)
        print(f"Rule: {rule}")
        print(f"  Premises: {premises}")
        print(f"  Conclusion: {conclusion}")
        print()

    print("=== rename_by_first_appearance ===")
    rename_cases = [
        (
            "cong x y z w, perp x y y z => para x z w y",
            "cong a b c d, perp a b b c => para a c d b",
        ),
        ("coll p q r => para p q p r", "coll a b c => para a b a c"),
        (
            "eqangle b d b e c d c e => para b c d e",
            "eqangle a b a c d b d c => para a d b c",
        ),
        (
            "cong a b c d, perp a b b c => para a c d b",
            "cong a b c d, perp a b b c => para a c d b",
        ),
    ]
    for src, expected in rename_cases:
        got = rename_by_first_appearance(src)
        ok = "OK" if got == expected else "FAIL"
        print(f"  [{ok}] {src}")
        print(f"         -> {got}")
        if got != expected:
            print(f"     want: {expected}")

    print()
    print("=== collect_points_in_order ===")
    p, c = parse_rule_text("cong x y z w, perp x y y z => para x z w y")
    print(f"  points: {collect_points_in_order(p, c)}")

    print("=== build_rename_map (28 points -> aa, ab) ===")
    pts = [f"p{i}" for i in range(28)]
    m = build_rename_map(pts)
    print(f"  p0={m['p0']}, p25={m['p25']}, p26={m['p26']}, p27={m['p27']}")

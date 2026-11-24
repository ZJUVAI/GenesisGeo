"""将区分谓词加入规则文本。

读取 ndg_discovery.discover_all 的输出，为每条规则选择最佳区分谓词，
生成带 NDG 守卫的规则。

选择策略：
  1. 只保留 true_at_good=True 的，或能被否定的（perp→nperp, obtuse→acute等）
  2. 按优先级排序：ncoll/npara/diff > perp/nperp > acute/obtuse > sameclock
  3. true_at_good=False 的谓词取其否定形式
  4. 取优先级最高的 1 个加入规则

供 reduction.orchestrator._run_ndg_stage 直接 import 调用（apply_ndg），不提供独立 CLI。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

# ============================================================================
# Priority and negation
# ============================================================================

# 优先级：数字越小越优先
PRIORITY = {
    # Strong NDG candidates (non-degeneracy, orientation)
    "ncoll": 0,
    "npara": 0,
    "nperp": 0,
    "diff": 0,
    "obtuse": 2,
    "acute": 2,
    "sameclock": 3,
    # Weak / suspicious: positive predicates that are often construction
    # artifacts.  Only used as a last resort when no better NDG survives.
    "perp": 5,
}

# Positive geometric predicates that are almost never valid NDG guards.
# (Their negations — ncoll, npara, nperp — ARE valid and handled above.)
SUBSTANTIVE_PREDICATES = {
    "para",
    "cong",
    "cyclic",
    "coll",
    "eqangle",
    "eqratio",
    "simtri",
    "simtrir",
    "contri",
    "contrir",
    "midp",
}

# 否定映射：true_at_good=False 时，如果能取反就用映射
NEGATION = {
    "perp": "nperp",
    "nperp": "perp",
    "obtuse": "acute",
    "acute": "obtuse",
    "ncoll": None,  # 否定是 coll，太泛，不取反
    "npara": None,
    "sameclock": None,
    "diff": None,
}


def _parse_predicate(pred_str: str) -> tuple[str, list[str]]:
    """'perp A D B C' → ('perp', ['A','D','B','C'])"""
    parts = pred_str.strip().split()
    return parts[0], parts[1:]


def _format_predicate(name: str, args: list[str]) -> str:
    return f"{name} {' '.join(args)}"


def _conclusion_predicate_set(rule_text: str) -> set[tuple[str, tuple[str, ...]]]:
    """规则结论的 (name, args) 集合，含所有对称等价写法。

    用于 select_best_predicate 排除"guard 恰好就是结论本身"这种退化情形——
    候选谓词搜索(generate_all_candidates)本就会包含结论谓词自身(它在好/坏
    样本上天然 100% 区分), select_best_predicate 若不排除它, 唯一优先级更
    高的候选恰好都缺席时, 会选出"这条规则成立的前提是结论成立"这种空话。
    """
    from newclid.discovery.reverse_construction import predicate_variants
    from newclid.discovery.utils.rule_parser import split_rule_text

    _prem_strs, concl_str = split_rule_text(rule_text)
    out: set[tuple[str, tuple[str, ...]]] = set()
    for c in concl_str.split(","):
        c = c.strip()
        if not c:
            continue
        name, args = _parse_predicate(c)
        for variant in predicate_variants(name, args):
            out.add((name, tuple(variant)))
    return out


def extra_ncoll_guards_for_disjoint_para(rule_text: str) -> list[str]:
    """为前提中"无公共点"的每条 para A B C D 补一条 ncoll A B C 守卫（纯文本操作）。

    背景：para A B C D, para A C B D => cong A B C D 这类规则，若 A,B,C 恰好
    共线，两条 para 前提在共线退化下仍然成立（共线时任意"线段"都相互平行），
    但等长关系不再被保证（例如 A=(0,0),B=(1,0),C=(2,0),D=(5,0)：|AB|=1≠3=|CD|，
    见本次调试实测）——需要 ncoll 排除这个退化。

    只在 A,B,C,D 两两不同（para A B A C 这种共享点的写法已由
    predicate_rewrite 转成 coll，不会走到这里）、且前提里还没有涉及 A,B,C
    这三点的 coll/ncoll 时才补。这是一条不依赖数值反例搜索的保守追加：
    宁可多一条从未真正需要过的 guard，不放过共线这一退化分支。
    """
    from newclid.discovery.utils.rule_parser import parse_predicate, split_rule_text

    prem_strs, _concl_str = split_rule_text(rule_text)
    existing_coll_triples: set[frozenset] = set()
    para_clauses: list[tuple[str, ...]] = []
    for p in prem_strs:
        p = p.strip()
        if not p:
            continue
        name, args = parse_predicate(p)
        if name in ("coll", "ncoll") and len(args) == 3:
            existing_coll_triples.add(frozenset(args))
        elif name == "para" and len(args) == 4 and len(set(args)) == 4:
            para_clauses.append(tuple(args))

    guards: list[str] = []
    seen: set[frozenset] = set()
    for a, b, c, _d in para_clauses:
        triple = frozenset((a, b, c))
        if triple in existing_coll_triples or triple in seen:
            continue
        seen.add(triple)
        guards.append(f"ncoll {' '.join(sorted(triple))}")
    return guards


def select_best_predicate(
    surviving: list[dict],
    conclusion_predicates: set[tuple[str, tuple[str, ...]]] | None = None,
) -> dict | None:
    """从存活谓词列表中选择最佳的一个。

    conclusion_predicates: 规则结论的 (name, args) 集合（按 predicate_variants
        全部对称写法展开）。候选谓词若与结论的某个写法完全相同 —— 或其否定
        与之相同 —— 会被排除：guard 的作用是限定前提的适用范围，不能是
        "结论本身"或"结论的否定"，否则等于把规则改写成"若结论成立则结论成立"
        的空话，或者反过来直接否定了整条规则。仅在候选优先级列表(PRIORITY
        中 0/2/3 三档)恰好全部被"结论/其否定"占满、别无选择时才会体现出
        差异 —— 正常情况下更高优先级的其它候选会先被选中，这只是补一道
        兜底防线。

    返回 {predicate_str, true_at_good, negated, priority} 或 None。
    """
    if not surviving:
        return None

    conclusion_predicates = conclusion_predicates or set()

    # 处理每个谓词：如果是 true_at_good=False，尝试取反
    candidates = []
    for s in surviving:
        pred_str = s["predicate"]
        true_at_good = s["true_at_good"]
        name, args = _parse_predicate(pred_str)
        if (name, tuple(args)) in conclusion_predicates:
            continue  # guard 不能就是结论本身

        priority = PRIORITY.get(name, 5)

        if true_at_good:
            candidates.append(
                {
                    "predicate": pred_str,
                    "true_at_good": True,
                    "negated": False,
                    "priority": priority,
                    "name": name,
                    "args": args,
                }
            )
        else:
            # true_at_good=False → 在好配置下不成立，需要取反
            neg_name = NEGATION.get(name)
            if (
                neg_name is not None
                and (neg_name, tuple(args)) not in conclusion_predicates
            ):
                # 取反后的谓词同样要排除"恰好是结论本身"的情形——例如结论是
                # perp A C A E 时，候选 nperp A C A E 本身不在结论集合里，
                # 取反后却变成 perp A C A E，必须在这里再检查一次。
                candidates.append(
                    {
                        "predicate": _format_predicate(neg_name, args),
                        "true_at_good": True,  # 取反后，好配置下应该成立
                        "negated": True,
                        "priority": PRIORITY.get(neg_name, 2),
                        "name": neg_name,
                        "args": args,
                        "original_predicate": pred_str,
                    }
                )
            # else: 无法取反, 或取反后即结论本身 → 丢弃

    # Filter out substantive predicates (perp, para, cong, etc.) that are
    # almost always construction artifacts rather than genuine NDG conditions.
    candidates = [c for c in candidates if c["name"] not in SUBSTANTIVE_PREDICATES]

    if not candidates:
        return None

    # 去重：相同 predicate 文本只保留优先级最高的
    seen = {}
    for c in candidates:
        key = c["predicate"]
        if key not in seen or c["priority"] < seen[key]["priority"]:
            seen[key] = c

    # 按优先级排序，取最优
    best = min(seen.values(), key=lambda c: (c["priority"], c["predicate"]))
    return best


def apply_ndg(
    dist_file: str,
    rules_file: str,
    output_file: str,
    degeneracy_threshold: float = 0.001,
) -> tuple[dict[str, Any], list[dict]]:
    """主逻辑：读取区分谓词 + 规则，输出带 NDG 的规则。

    Returns (stats, dropped)：dropped 是逐条 {rule_id, rule_text, reason, ...}
    记录，供调用方（如 orchestrator 的 audit log）追溯每条规则被丢弃的原因，
    而不只是 stats 里的聚合计数。

    degeneracy_threshold: status="pass"（反例搜索未找到反例，视为恒成立）的规则，
        如果其 min_degeneracy（reverse_construction 记录的、构造新点时两条定义直线
        夹角归一化 sin 值的最小值，见 reverse_construction._line_line_degeneracy）
        低于此阈值，说明该规则的求解在采样到的某个分支上几乎退化为无穷解——
        "反例搜索找不到反例"很可能只是因为这个退化区域是测度为零的边界，随机
        重建难以采样到，而不是规则真的恒成立（典型案例：custom_00045，两条直线
        在特定几何条件下重合，新点从"唯一确定"退化为"一条自由直线"）。
        这类规则改为丢弃（reason="degenerate_construction"），不再无条件放行。
        设为 0 或负数可关闭此检查（保留原始 pass-through 行为）。
    """
    # 加载区分谓词结果
    print(f"[ndg_apply] Loading distinguishing results: {dist_file}")
    dist_recs = []
    with open(dist_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dist_recs.append(json.loads(line))

    # 加载规则
    print(f"[ndg_apply] Loading rules: {rules_file}")
    rules = []
    with open(rules_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rules.append(json.loads(line))

    # 匹配：优先用 rule_id，否则按行号
    dist_map = {}
    for d in dist_recs:
        rid = d.get("rule_id")
        if rid:
            dist_map[rid] = d

    # 处理每条规则：只保留两类，其余丢弃
    #   1. status=success 且能选出合法候选谓词 → 保留 (加 NDG)
    #   2. status=pass (所有 seed 都没找到 bad) → 保留原规则不变 (视为恒成立)
    #   其余 (insufficient_pairs / insufficient_good_sets / success 但无合法候选) → 丢弃
    kept = []
    dropped = []
    stats = defaultdict(int)

    for i, rule in enumerate(rules):
        rule_id = rule["rule_id"]
        rule_text = rule["rule_text"]
        # 优先 rule_id 匹配，否则行号匹配
        dist_rec = dist_map.get(rule_id)
        if dist_rec is None and i < len(dist_recs):
            dist_rec = dist_recs[i]
            drid = dist_rec.get("rule_id")
            if drid and drid != rule_id:
                dist_rec = None

        if dist_rec is None:
            stats["dropped_no_dist_data"] += 1
            dropped.append(
                {"rule_id": rule_id, "rule_text": rule_text, "reason": "no_dist_data"}
            )
            continue

        status = dist_rec.get("status")

        if status == "pass":
            min_deg = dist_rec.get("min_degeneracy")
            if (
                degeneracy_threshold > 0
                and min_deg is not None
                and min_deg < degeneracy_threshold
            ):
                # 反例搜索未找到反例, 但构造新点时曾撞到近乎平行(退化)的两条
                # 定义直线 —— 这片退化区域是测度为零的边界, 随机反例搜索几乎
                # 采不到, 不能当作"恒成立"的证据. 丢弃而不是放行.
                stats["dropped_degenerate_construction"] += 1
                dropped.append(
                    {
                        "rule_id": rule_id,
                        "rule_text": rule_text,
                        "reason": "degenerate_construction",
                        "min_degeneracy": min_deg,
                    }
                )
                continue
            # 所有 seed 都没找到 bad → 视为恒成立，原样保留
            stats["kept_pass"] += 1
            extra_guards = extra_ncoll_guards_for_disjoint_para(rule_text)
            kept.append(
                {
                    "rule_id": rule_id,
                    "seed": rule.get("seed"),
                    "index_in_seed": rule.get("index_in_seed", 0),
                    "rule_text": rule_text,
                    "points": rule.get("points", []),
                    "guards": ", ".join(extra_guards) if extra_guards else "",
                    "ndg_added": None,
                    "status": "pass",
                    "min_degeneracy": min_deg,
                }
            )
            continue

        if status != "success":
            # insufficient_pairs / insufficient_good_sets 等 → 丢弃
            stats[f"dropped_{status}"] += 1
            dropped.append(
                {"rule_id": rule_id, "rule_text": rule_text, "reason": status}
            )
            continue

        # status == "success": 尝试选出合法候选谓词
        surviving = dist_rec.get("surviving_predicates", [])
        conclusion_predicates = _conclusion_predicate_set(rule_text)
        best = (
            select_best_predicate(surviving, conclusion_predicates)
            if surviving
            else None
        )
        if best is None:
            stats["dropped_no_valid_candidate"] += 1
            dropped.append(
                {
                    "rule_id": rule_id,
                    "rule_text": rule_text,
                    "reason": "no_valid_candidate",
                }
            )
            continue

        # NDG 谓词作为守卫，不混入 rule_text
        # rule_text 保留原样；guards 段在 pipe 转换时附加为第四段
        extra_guards = extra_ncoll_guards_for_disjoint_para(rule_text)
        guard_str = ", ".join([best["predicate"], *extra_guards])

        stats["kept_ndg_added"] += 1
        kept.append(
            {
                "rule_id": rule_id,
                "seed": rule.get("seed"),
                "index_in_seed": rule.get("index_in_seed", 0),
                "rule_text": rule_text,  # 原规则文本，不修改
                "points": rule.get("points", []),
                "guards": guard_str,
                "original_rule_text": rule_text,
                "ndg_added": best["predicate"],
                "ndg_priority": best["priority"],
                "ndg_negated": best["negated"],
                "true_at_good": best["true_at_good"],
                "n_surviving_total": len(surviving),
                "n_pairs": dist_rec.get("n_pairs", 0),
                "status": "success",
            }
        )

    # 保存：只写入 kept（success+NDG 和 pass 两类）
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 同时写一份纯规则文本 (.txt)，一行一条 rule_text，方便人工浏览。
    # 有 guards 时追加 " | guard1, guard2, ..." 方便查看。
    txt_path = os.path.splitext(output_file)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for rec in kept:
            line = rec["rule_text"]
            g = rec.get("guards", "")
            if g:
                line += " | " + g
            f.write(line + "\n")

    # 统计
    print(f"\n{'=' * 50}")
    print(
        f"[ndg_apply] Kept {len(kept)}/{len(rules)} rules "
        f"(NDG added: {stats['kept_ndg_added']}, pass-through: {stats['kept_pass']})"
    )
    print(f"[ndg_apply] Dropped {len(dropped)}/{len(rules)} rules:")
    for k, v in stats.items():
        if k.startswith("dropped_"):
            print(f"  {k[8:]}: {v}")
    print(f"[ndg_apply] Saved to: {output_file}")

    # 展示几个示例
    print("\n[ndg_apply] Examples (NDG added):")
    shown = 0
    for rec in kept:
        if rec.get("ndg_added"):
            print(f"  {rec['rule_id']}: + {rec['ndg_added']}")
            print(f"    {rec['original_rule_text'][:80]}...")
            print(f"    → {rec['rule_text'][:80]}...")
            shown += 1
        if shown >= 5:
            break

    return dict(stats), dropped

"""Part 2 规约编排。

规约策略（作用于 Part 1 的 normalized_rules.jsonl）：
  0. （可选）点数预过滤 + 前提数预过滤：丢弃涉及点数 >= max_points_drop 或
     前提数 >= max_premises_drop 的规则。
  0.5.（可选）NDG 发现 + 应用：必须在任何规约（1/2/3）之前做。规约阶段的
     subsumption 判定会把 sources 当作无条件成立的定理喂给 CSolver；若
     sources 本身是"看似恒成立、实际需要 guard 才成立"的假规则，用它去
     淘汰别的规则这个决定就可能是错的，且规约丢弃不可逆——被误伐的规则
     不会再有机会被复查。提前做完 NDG，让此后所有规约阶段使用的 sources
     都已经过反例搜索验证、带上正确的 guard。              [ndg_discovery/ndg_apply]
  1. 按 seed 分组，组内贪心规约（Ray 并行，每组一任务）。       [seed_reducer]
  2. 全部存活规则分治规约（打乱分块并行 → 汇总 → 迭代到稳定）。  [divide_conquer_reducer]
  3. 按前提精确分组，组内规约一次（补上分治阶段打乱分块可能漏掉的
     "同前提互推"冗余；此后不再对存活规则的前提/结论做任何丢弃）。
                                                                [seed_reducer]
     同时落盘此刻存活规则对应的 rule_text -> seed 出现次数统计子集
     （反查最终规则的原始泛化程度用，合并之前的"完整规则"快照）。
  4. 合并前提完全一致的存活规则 p→g1, p→g2 ⇒ p→g1,g2。NDG 已在阶段 0.5
     提前做完，此后规则前提（含 guards）不再变化，可放在规约流程最后。
                                                                [seed_reducer.merge]

规约判定原子操作（subsumption_tester）：
  CSolver(points=R坐标, premises=R前提, goals=[R结论]).run(custom_rules=其它规则) 是否 solved。
  custom_rules 可多条 → 支持"多条规则合力推出第三条"；custom_rules 会带上
  各 source 自己的 guards（若有），CSolver 在应用该 source 作为定理前先
  数值检查 guards 是否满足。

并行：Ray，bounded in-flight（ray.wait 完成即补），避免 worker 闲置。
"""

from __future__ import annotations

import json
import os
from typing import Any

from newclid.discovery.reduction import divide_conquer_reducer, seed_reducer
from newclid.discovery.reduction.subsumption_tester import RuleItem, load_rules


def _write(rules: list[RuleItem], output_path: str) -> None:
    txt_path = os.path.splitext(output_path)[0] + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rules:
            f.write(
                json.dumps(
                    {
                        "rule_id": r.rule_id,
                        "seed": r.seed,
                        "index_in_seed": r.index_in_seed,
                        "rule_text": r.rule_text,
                        "points": [{"name": n, "x": x, "y": y} for n, x, y in r.points],
                        "guards": r.guards,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in rules:
            if r.guards:
                f.write(f"{r.rule_text} | {r.guards}\n")
            else:
                f.write(r.rule_text + "\n")


def _run_ndg_stage(
    rules: list[RuleItem],
    cfg: dict[str, Any],
    output_dir: str,
    out_dir: str,
) -> tuple[list[RuleItem], dict[str, dict], dict[str, Any]]:
    """阶段 0.5：NDG 发现 + 应用。委托 ndg_discovery.discover_all + ndg_apply.apply_ndg。

    config: part2_reduction.ndg.{n_seeds, n_ce_trials, n_workers, degeneracy_threshold,
                                  rule_timeout_seconds, normalized_rules_path,
                                  occurrences_path, source_dataset_path}
    normalized_rules_path/occurrences_path/source_dataset_path 为 null 时自动取
    Part 1 的默认输出路径（{output_dir}/part1/normalized_rules.jsonl 等）与
    part1_extract.input（原始数据集）——discover_all 靠它们把规则映射回原始
    fl_problem 以重建多组坐标做反例搜索，与 NDG 现在跑在哪个阶段无关。

    Returns (kept_rules, audit, stats)：audit 记录每条被丢弃规则的 {stage, reason}；
    stats 是 ndg_discovery/ndg_apply 的聚合统计（键加 ndg_ 前缀）。
    """
    import json as _json

    from newclid.discovery.ndg_apply import apply_ndg
    from newclid.discovery.ndg_discovery import discover_all, save_discover_results

    ndg_cfg = cfg.get("part2_reduction", {}).get("ndg", {})
    n_seeds = ndg_cfg.get("n_seeds", 10)
    n_ce_trials = ndg_cfg.get("n_ce_trials", 3)
    n_workers = ndg_cfg.get("n_workers", 8)
    rule_timeout_seconds = ndg_cfg.get("rule_timeout_seconds", 120.0)
    degeneracy_threshold = ndg_cfg.get("degeneracy_threshold", 0.001)
    min_good_ratio = ndg_cfg.get("min_good_ratio", 0.0)

    part1_dir = os.path.join(output_dir, "part1")
    normalized_rules_path = ndg_cfg.get("normalized_rules_path") or os.path.join(
        part1_dir, "normalized_rules.jsonl"
    )
    occurrences_path = ndg_cfg.get("occurrences_path") or os.path.join(
        part1_dir, "rule_seed_occurrences_all.json"
    )
    source_dataset_path = ndg_cfg.get("source_dataset_path") or cfg.get(
        "part1_extract", {}
    ).get("input")
    if not source_dataset_path:
        raise ValueError(
            "part2_reduction.ndg.enabled=True 需要 source_dataset_path（原始数据集），"
            "请显式指定，或确保 part1_extract.input 已配置"
        )

    # discover_all 本身只用 rule_id/rule_text（靠 RuleTracer 通过
    # normalized_rules_path/occurrences_path 反查坐标与原始 fl_problem 做反例
    # 搜索）。但 apply_ndg 随后会把这份文件里的 seed/index_in_seed/points
    # 原样搬进它的输出记录（见 ndg_apply.apply_ndg 的 kept.append），若这里
    # 不带上这些字段，NDG 应用后的坐标/seed 信息会全部丢失为空。
    rules_file = os.path.join(out_dir, "ndg_input.jsonl")
    with open(rules_file, "w", encoding="utf-8") as f:
        for r in rules:
            f.write(
                _json.dumps(
                    {
                        "rule_id": r.rule_id,
                        "seed": r.seed,
                        "index_in_seed": r.index_in_seed,
                        "rule_text": r.rule_text,
                        "points": [{"name": n, "x": x, "y": y} for n, x, y in r.points],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("[part2][ndg] Step 1: discovering distinguishing predicates...")
    results = discover_all(
        rules_file=rules_file,
        normalized_rules_path=normalized_rules_path,
        occurrences_path=occurrences_path,
        source_dataset_path=source_dataset_path,
        n_seeds=n_seeds,
        n_ce_trials=n_ce_trials,
        n_workers=n_workers,
        rule_timeout_seconds=rule_timeout_seconds,
        min_good_ratio=min_good_ratio,
    )
    dist_file = save_discover_results(results, out_dir)

    print("[part2][ndg] Step 2: applying NDG predicates...")
    ndg_output_file = os.path.join(out_dir, "ndg_output.jsonl")
    apply_stats, apply_dropped = apply_ndg(
        dist_file,
        rules_file,
        ndg_output_file,
        degeneracy_threshold=degeneracy_threshold,
    )

    audit: dict[str, dict] = {}
    for rec in apply_dropped:
        audit[rec["rule_id"]] = {
            "rule_text": rec["rule_text"],
            "status": "dropped",
            "stage": "ndg",
            "reason": rec.get("reason"),
        }

    # ndg_apply 的输出记录含 rule_id/seed/index_in_seed/rule_text/points/guards
    # （rule_text 本身不改写，guard 单独存于 guards 字段）——RuleItem.from_record
    # 重新解析 rule_text 即可得到 premises/goal/premise_count，直接复用。
    kept: list[RuleItem] = []
    with open(ndg_output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            kept.append(RuleItem.from_record(_json.loads(line)))

    stats = {f"ndg_{k}": v for k, v in apply_stats.items()}
    stats["ndg_input"] = len(rules)
    stats["ndg_kept"] = len(kept)
    print(f"[part2][ndg] {len(rules)} -> {len(kept)} 条")
    return kept, audit, stats


def run_part2(
    cfg: dict[str, Any],
    output_dir: str,
    input_path: str,
) -> tuple[str, dict[str, Any]]:
    """Part 2 规约完整流程。

    config: part2_reduction.{max_points_drop, max_premises_drop, n_workers, use_ray,
                             chunk_size, config_path}
    """
    part2_cfg = cfg.get("part2_reduction", {})
    max_points_drop = part2_cfg.get("max_points_drop", 8)
    max_premises_drop = part2_cfg.get("max_premises_drop")
    n_workers = part2_cfg.get("n_workers") or cfg.get("global", {}).get("n_workers", 30)
    use_ray = part2_cfg.get("use_ray", True)
    chunk_size = part2_cfg.get("chunk_size", 200)
    config_path = part2_cfg.get("csolver_config_path")

    out_dir = os.path.join(output_dir, "part2")
    os.makedirs(out_dir, exist_ok=True)
    output_path = part2_cfg.get("output") or os.path.join(
        out_dir, "reduced_rules.jsonl"
    )

    # 加载
    rules = load_rules(input_path)
    stats: dict[str, Any] = {"input": len(rules)}

    # 额外的、始终可用作 sources 但不参与本轮规约的规则集合（如上一轮迭代已
    # 验证过的规则库）。不计入 rules/audit/output，只在每次 subsumption 判定
    # 时附加进 sources 里，帮助本轮规则被更早判定为冗余，同时自身不会被丢弃
    # 或合并、也不需要经过本轮的 NDG。
    extra_rules_path = part2_cfg.get("extra_rules_path")
    extra_sources: list[RuleItem] | None = None
    if extra_rules_path:
        extra_sources = load_rules(extra_rules_path)
        stats["extra_sources"] = len(extra_sources)
        print(
            f"[part2] 加载额外 sources 规则(不参与规约): {len(extra_sources)} 条 <- {extra_rules_path}"
        )

    # 全量 audit：rule_id -> {rule_text, status, stage, ...}，最终落盘为 rule_audit_log.jsonl。
    # 记录范围是 Part 2 的输入（即 Part 1 产出的全部规则），逐阶段追加"dropped/merged"记录，
    # 未被任何阶段记录的规则视为最终"kept"。
    audit: dict[str, dict] = {}

    # 小规模测试：只取前 N 条
    limit = part2_cfg.get("limit")
    if limit is not None:
        rules = rules[:limit]
        print(f"[part2] limit={limit}, 只处理前 {len(rules)} 条")

    # 记录进入 Part 2 规约流程的完整规则集（rule_id -> rule_text），供最终
    # audit log 反查"未被任何阶段记录 = 存活到底"的规则文本。
    initial_rule_texts = {r.rule_id: r.rule_text for r in rules}

    # 阶段 0：点数预过滤 + 前提数预过滤 + perp 前提数过滤
    max_perp_premises = part2_cfg.get("max_perp_premises")
    if (
        max_points_drop is not None
        or max_premises_drop is not None
        or max_perp_premises is not None
    ):
        before = rules
        if max_points_drop is not None:
            rules = [r for r in rules if len(r.points) < max_points_drop]
        if max_premises_drop is not None:
            rules = [r for r in rules if r.premise_count < max_premises_drop]
        if max_perp_premises is not None:
            rules = [
                r
                for r in rules
                if sum(1 for name, _args in r.premises if name == "perp")
                <= max_perp_premises
            ]
        dropped_ids = {r.rule_id for r in before} - {r.rule_id for r in rules}
        by_id = {r.rule_id: r for r in before}
        for rid in dropped_ids:
            audit[rid] = {
                "rule_text": by_id[rid].rule_text,
                "status": "dropped",
                "stage": "premise_num_filter",
                "reason": f"points>={max_points_drop} or premises>={max_premises_drop} "
                f"or perp_premises>{max_perp_premises}",
            }
        stats["after_premise_filter"] = len(rules)
        print(
            f"[part2] 点数<{max_points_drop}"
            f"{' 且 前提数<' + str(max_premises_drop) if max_premises_drop is not None else ''}"
            f"{' 且 perp 前提数<=' + str(max_perp_premises) if max_perp_premises is not None else ''}"
            f" 过滤后: {len(rules)} 条"
        )
        _write(rules, os.path.join(out_dir, "premise_num_filtered.jsonl"))

    # 阶段 0.5：NDG 发现 + 应用（必须在任何规约之前做，否则规约阶段用来
    # 判定 target 是否冗余的 sources 可能是"看似恒成立、实际需要 guard 才
    # 成立"的假规则——CSolver 把 sources 当无条件真定理喂进去，用假规则
    # 淘汰别的规则这个决定本身就可能是错的，且规约丢弃不可逆）。
    ndg_cfg = part2_cfg.get("ndg", {})
    if ndg_cfg.get("enabled", False):
        rules, ndg_stage_audit, ndg_stats = _run_ndg_stage(
            rules,
            cfg,
            output_dir,
            out_dir,
        )
        audit.update(ndg_stage_audit)
        stats.update(ndg_stats)
        stats["after_ndg"] = len(rules)
        _write(rules, os.path.join(out_dir, "ndg_applied.jsonl"))

    # 阶段 1：seed 分组规约
    rules, stage_audit = seed_reducer.reduce_by_seed(
        rules,
        n_workers=n_workers,
        use_ray=use_ray,
        config_path=config_path,
        extra_sources=extra_sources,
    )
    audit.update(stage_audit)
    stats["after_seed_reduce"] = len(rules)
    _write(rules, os.path.join(out_dir, "seed_reduced.jsonl"))

    # 阶段 3：分治规约
    rules, stage_audit = divide_conquer_reducer.reduce(
        rules,
        chunk_size=chunk_size,
        n_workers=n_workers,
        use_ray=use_ray,
        config_path=config_path,
        extra_sources=extra_sources,
    )
    audit.update(stage_audit)
    stats["after_divide_conquer"] = len(rules)
    _write(rules, os.path.join(out_dir, "divide_reduced.jsonl"))

    # 阶段 3.5：合并前先按前提精确分组，组内规约一次（补上分治阶段打乱分块
    # 可能漏掉的"同前提互推"冗余），保证 merge_same_premise 合并时组内不再
    # 发生规约层面的丢弃 —— 此后不会再对存活规则的前提/结论做任何丢弃或改写。
    rules, stage_audit = seed_reducer.reduce_by_premise_group(
        rules,
        n_workers=n_workers,
        use_ray=use_ray,
        config_path=config_path,
        extra_sources=extra_sources,
    )
    audit.update(stage_audit)
    stats["after_premise_group_reduce"] = len(rules)
    _write(rules, os.path.join(out_dir, "premise_group_reduced.jsonl"))

    # 落盘"规约结束、合并之前"存活规则的 rule_text -> seed 出现次数子集：
    # 从 Part 1 落盘的全量统计（rule_seed_occurrences_all.json）里，按当前存活
    # 规则的 rule_text 筛出对应条目。
    _write_survived_occurrences(rules, input_path, out_dir)

    # 阶段 4：合并前提完全一致的规则（p→g1, p→g2 ⇒ p→g1,g2）。放在所有规约
    # 阶段之后——NDG 已在阶段 0.5 提前做完，此后规则的前提集合（含 guards）
    # 不会再变化，合并不会把"日后才加 guard 而变得前提不同"的规则错误归并。
    rules, merge_audit = seed_reducer.merge_same_premise(rules)
    audit.update(merge_audit)
    stats["after_merge"] = len(rules)

    _write(rules, output_path)
    stats["kept"] = len(rules)
    stats["output"] = output_path

    audit_path = _write_audit_log(rules, audit, initial_rule_texts, out_dir)
    stats["audit_log"] = audit_path

    print(
        f"[part2] 规约完成: {stats['input']} -> {len(rules)} 条, 结果 -> {output_path}"
    )
    return output_path, stats


def _write_audit_log(
    final_rules: list[RuleItem],
    audit: dict[str, dict],
    initial_rule_texts: dict[str, str],
    out_dir: str,
) -> str:
    """把每条入参规则的最终归宿写成 JSONL：kept / dropped（哪个阶段、被哪些
    sources 判定冗余）/ merged（并入了哪条规则）。

    覆盖范围是 initial_rule_texts（即进入 Part 2 前的全部规则）；凡是未出现在
    audit 里、又不是最终存活规则代表的（因为 merge 时把多条结论合到了代表规则
    上，代表规则本身的 rule_text 已被改写，不等于其原始 rule_text），都视为
    "kept"——即未被任何阶段丢弃或吞并，原样存活到最后。
    """
    # 存活规则用它们在最终输出里的 rule_text（若被 merge_same_premise 合并过
    # 结论，文本已改写，须与 reduced_rules.jsonl 保持一致，而不是入参时的原文）。
    survived_texts = {r.rule_id: r.rule_text for r in final_rules}
    records: list[dict] = []
    for rule_id, rule_text in initial_rule_texts.items():
        if rule_id in audit:
            rec = {"rule_id": rule_id, "rule_text": rule_text, **audit[rule_id]}
        elif rule_id in survived_texts:
            rec = {
                "rule_id": rule_id,
                "rule_text": survived_texts[rule_id],
                "status": "kept",
            }
        else:
            # 理论上不应发生：既不在 audit（未被记录丢弃/合并），又不在最终存活集合。
            rec = {"rule_id": rule_id, "rule_text": rule_text, "status": "unknown"}
        records.append(rec)

    audit_path = os.path.join(out_dir, "rule_audit_log.jsonl")
    with open(audit_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_kept = sum(1 for r in records if r["status"] == "kept")
    n_dropped = sum(1 for r in records if r["status"] == "dropped")
    n_merged = sum(1 for r in records if r["status"] == "merged")
    n_unknown = sum(1 for r in records if r["status"] == "unknown")
    print(
        f"[part2] audit log 已保存 -> {audit_path} "
        f"(kept={n_kept} dropped={n_dropped} merged={n_merged} unknown={n_unknown})"
    )
    return audit_path


def _write_survived_occurrences(
    rules: list[RuleItem], input_path: str, out_dir: str
) -> None:
    """从 Part 1 落盘的全量 rule_text->seed 出现次数统计里，筛出当前存活规则
    （规约结束、合并之前）对应的子集，另存一份，便于反查最终规则的原始泛化程度。

    全量统计文件默认与 Part 1 输出（normalized_rules.jsonl）同目录，名为
    rule_seed_occurrences_all.json；找不到则跳过（不阻塞规约流程）。
    """
    all_path = os.path.join(
        os.path.dirname(input_path), "rule_seed_occurrences_all.json"
    )
    if not os.path.exists(all_path):
        print(
            f"[part2] 未找到全量 rule_text->seed 统计文件 {all_path}, 跳过存活子集落盘"
        )
        return

    with open(all_path, "r", encoding="utf-8") as f:
        all_occurrences: dict[str, dict[str, int]] = json.load(f)

    survived_texts = {r.rule_text for r in rules}
    survived_occurrences = {
        text: all_occurrences[text]
        for text in survived_texts
        if text in all_occurrences
    }

    survived_path = os.path.join(out_dir, "rule_seed_occurrences_survived.json")
    with open(survived_path, "w", encoding="utf-8") as f:
        json.dump(survived_occurrences, f, ensure_ascii=False, indent=2)
    print(
        f"[part2] 存活规则(合并前) rule_text->seed 出现次数统计已保存 -> {survived_path} "
        f"({len(survived_occurrences)}/{len(survived_texts)} 条匹配到全量统计)"
    )

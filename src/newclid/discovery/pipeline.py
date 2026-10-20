"""Pipeline 主入口与编排（对应伪代码 §3）。

串行执行 Part 1（规则提取）和 Part 2（规则规约），维护 last_output 自动链接。

Part 2（reduction.orchestrator.run_part2）内部含 NDG 发现+应用阶段
（part2_reduction.ndg，紧接在点数/前提数预过滤之后、seed_reduce 之前执行）——
NDG 必须在任何规约判定之前完成，否则规约阶段用作 sources 的规则可能是"看似
恒成立、实际需要 guard 才成立"的假规则，用它淘汰别的规则这个决定本身就可能
是错的，且不可逆。曾经独立的 Part 3（NDG）已并入 Part 2，不再是单独阶段。

当前进度：Part 1 仅实现「构图」子步（读取数据 + 构建完整证明图）。
后续子步（剪枝、命题提取、规则文本化、去重、落盘）与 Part 2 暂以注释占位，
随实现推进逐步接入；测试脚本 / config 入口保持不变。
"""

from __future__ import annotations

import os
from typing import Any

from tqdm import tqdm

from newclid.discovery.config import load_config, resolve_output
from newclid.discovery.data_models import PipelineSummary
from newclid.discovery.extraction.graph_manager import build_proof_graphs


def run_pipeline(config_path: str) -> None:
    """Pipeline 顶层入口。

    流程：
    1. load_config(config_path) → cfg。
    2. 创建 output_dir，初始化 PipelineSummary。
    3. Part 1: 若 enabled → run_part1(cfg, output_dir, input_path)，记录 stats。
    4. Part 2: 若 enabled → run_part2(cfg, output_dir, input_path)，
       input_path 自动继承 Part 1 输出（若未显式指定）。
    5. summary.save() → output_dir/pipeline_summary.json。
    """
    cfg = load_config(config_path)
    output_dir = cfg["global"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    summary = PipelineSummary(output_dir=output_dir)

    last_output: str | None = None

    # ---------------- Part 1: 规则提取 ----------------
    part1_cfg = cfg["part1_extract"]
    if part1_cfg.get("enabled"):
        input_path = part1_cfg.get("input")
        if not input_path:
            raise ValueError("Part 1 enabled 但未指定 input")
        output_path, stats = run_part1(cfg, output_dir, input_path)
        summary.record_part("part1_extract", True, stats)
        last_output = output_path
    else:
        summary.record_part("part1_extract", False, {})
        # part1 关闭时，沿用 part1 本会产出的默认输出路径，
        # 让 part2.input=null 也能直接读取上一次 part1 的结果。
        last_output = resolve_output(
            part1_cfg.get("output"), output_dir, "part1",
            "normalized_rules.jsonl" if part1_cfg.get("normalize", True) else "propositions.jsonl",
        )

    # ---------------- Part 2: 规则规约（含 NDG 发现，见 part2_reduction.ndg） ----------------
    part2_cfg = cfg["part2_reduction"]
    if part2_cfg.get("enabled"):
        input_path = part2_cfg.get("input") or last_output
        if not input_path:
            raise ValueError("Part 2 enabled 但无可用 input")
        output_path, stats = run_part2(cfg, output_dir, input_path)
        summary.record_part("part2_reduction", True, stats)
        last_output = output_path
    else:
        summary.record_part("part2_reduction", False, {})

    # ---------------- 汇总 ----------------
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    import json

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[pipeline] summary written to {summary_path}")


def run_part1(
    cfg: dict[str, Any],
    output_dir: str,
    input_path: str,
) -> tuple[str, dict[str, Any]]:
    """执行 Part 1: 规则提取。

    当前仅实现第一子步：读取数据 + 构建完整证明图，并打印统计。
    完整的 Steps 1-7（剪枝 / 命题提取 / 规范化 / 去重 / 落盘）随实现推进接入。

    Returns
    -------
    (output_path, stats)
    """
    part1_cfg = cfg["part1_extract"]
    limit = part1_cfg.get("limit")          # 取前 N 条
    sample = part1_cfg.get("sample")        # 随机抽 N 条（优先于 limit）
    random_seed = part1_cfg.get("random_seed")

    # Step 0 (optional): robustness pre-filter.  Strip coordinates from each
    # fl_problem, rebuild with a fresh seed, and numerically verify the goal.
    # Discards records whose conclusion is a coordinate-dependent coincidence.
    robust_cfg = part1_cfg.get("robustness_filter", {})
    if robust_cfg.get("enabled", False):
        from newclid.discovery.validation.robustness_filter import filter_file

        robust_input = input_path
        robust_output = os.path.join(output_dir, "robustness_filtered.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        r_stats = filter_file(
            robust_input, robust_output,
            rebuild_seed=robust_cfg.get("rebuild_seed", 999983),
            max_attempts=robust_cfg.get("max_attempts", 100),
            n_workers=robust_cfg.get("n_workers", 1),
            limit=limit if not sample else None,
        )
        print(f"[part1] 鲁棒性过滤: {r_stats['total']} -> {r_stats['kept']} "
              f"(build失败={r_stats['build_fail']}, goal不成立={r_stats['goal_fail']})")
        input_path = robust_output  # Use filtered data for subsequent steps

    # 跳过读数据/构图/拆解/命题提取，直接从已落盘的 propositions.jsonl 做规范化去重。
    # 用于命题提取已完成、只想重跑/调参规范化去重这一步的场景。
    normalize_only_input = part1_cfg.get("normalize_only_input")
    if normalize_only_input:
        return _run_normalize_only(cfg, output_dir, normalize_only_input)

    prop_path = resolve_output(
        None, output_dir, "part1", "propositions.jsonl",
    )
    output_path = resolve_output(
        part1_cfg.get("output"), output_dir, "part1", "normalized_rules.jsonl",
    )
    decompose = part1_cfg.get("decompose", False)
    merge_eqpoints_flag = part1_cfg.get("merge_eqpoints", True)

    # 构图 + 单步折叠 + 拆解 + 命题提取 + eqpoint 合并：
    # 数据量大时（config: part1_extract.extract_n_workers > 1）走 Ray 并行版，
    # 按 chunk 把整条处理链路下放到 worker；否则走串行版，逐子步单独统计打印。
    extract_n_workers = part1_cfg.get("extract_n_workers") or 0
    if extract_n_workers and extract_n_workers > 1:
        from newclid.discovery.extraction.parallel_extract import extract_propositions_parallel
        from newclid.discovery.extraction.proposition_extractor import (
            proposition_to_output,
            proposition_to_text,
        )
        from newclid.discovery.utils.jsonl_io import write_jsonl

        print(f"[part1] 并行构图+命题提取({extract_n_workers} workers): {input_path}")
        rule_skip_predicates = part1_cfg.get("rule_skip_predicates") or []
        propositions, failures, pstats = extract_propositions_parallel(
            input_path,
            limit=limit, sample=sample, random_seed=random_seed,
            simplify=part1_cfg.get("simplify", True),
            decompose=decompose,
            merge_eqpoints_flag=merge_eqpoints_flag,
            n_workers=extract_n_workers,
            chunk_size=part1_cfg.get("extract_chunk_size", 2000),
            rule_skip_predicates=rule_skip_predicates,
        )
        stats: dict[str, Any] = {
            "built": pstats["built"],
            "failures": pstats["failures"],
            "collapsed_steps": pstats["collapsed_steps"],
            "subgraphs": pstats["subgraphs"] if decompose else None,
            "skipped_by_predicate": pstats["skipped_by_predicate"],
            "skipped_by_eqpoint": pstats["skipped_by_eqpoint"],
            "eqpoint_merged": pstats["eqpoint_merged"],
            "eqpoint_dropped_trivial": pstats["eqpoint_dropped_trivial"],
            "propositions": len(propositions),
        }
        print(f"[part1] built={pstats['built']} failures={pstats['failures']} "
              f"collapsed_steps={pstats['collapsed_steps']} "
              f"subgraphs={pstats['subgraphs']} "
              f"skipped_by_predicate={pstats['skipped_by_predicate']} "
              f"skipped_by_eqpoint={pstats['skipped_by_eqpoint']} "
              f"propositions={len(propositions)}")
        print(f"[part1] eqpoint 合并: {pstats['eqpoint_merged']} 条命题合并等价点, "
              f"{pstats['eqpoint_dropped_trivial']} 条合并后平凡被丢弃")
        for f in failures[:5]:
            print("  FAIL", f)
        if propositions:
            p = propositions[0]
            print(f"[part1] sample proposition {p.proposition_id}: {proposition_to_text(p)}")

        os.makedirs(os.path.dirname(prop_path), exist_ok=True)
        write_jsonl([proposition_to_output(p) for p in propositions], prop_path)
        print(f"[part1] 命题已保存 -> {prop_path}")
    else:
        print(f"[part1] 读取数据并构图: {input_path}")
        graphs, failures = build_proof_graphs(
            input_path, limit=limit, sample=sample, random_seed=random_seed,
        )

        # 单前提推导折叠（config: part1_extract.simplify，默认开启）
        collapsed_total = 0
        if part1_cfg.get("simplify", True):
            for g in tqdm(graphs, desc="[part1] 单步折叠", unit="图"):
                collapsed_total += g.simplify_single_step()
            print(f"[part1] 单步折叠: 共移除 {collapsed_total} 步")

        # 按辅助点拆解证明图（config: part1_extract.decompose）
        # 拆解后，后续绘制 / 统计以子图为准。
        subgraphs: list = []
        if decompose:
            from newclid.discovery.extraction.decomposer import decompose_by_aux
            for g in tqdm(graphs, desc="[part1] 拆解辅助点子图", unit="图"):
                subgraphs.extend(decompose_by_aux(g))
            print(f"[part1] 拆解: {len(graphs)} 张原图 -> {len(subgraphs)} 张含辅助点子图")

        # 需要绘制 / 统计的目标图集
        target_graphs = subgraphs if decompose else graphs

        # 整图丢弃，两条规则（见 graph_skip_reason）：rule_skip_predicates 命中的
        # 谓词只要在图中任一 fact 出现就丢整图；裸 eqpoint（两个参数都不是辅助点）
        # 恒定丢弃 —— 它有资格成为命题前提，会泄漏进提取出的规则。含辅助点的
        # eqpoint 是证明内部的正常中间步骤，保留。
        rule_skip_predicates = part1_cfg.get("rule_skip_predicates") or []
        from newclid.discovery.extraction.graph_manager import graph_skip_reason
        skip_set = set(rule_skip_predicates)
        skipped_by_predicate = 0
        skipped_by_eqpoint = 0
        kept_graphs = []
        for g in target_graphs:
            reason = graph_skip_reason(g, skip_set)
            if reason == "predicate":
                skipped_by_predicate += 1
            elif reason == "eqpoint":
                skipped_by_eqpoint += 1
            else:
                kept_graphs.append(g)
        target_graphs = kept_graphs
        print(f"[part1] 整图过滤: 丢弃 {skipped_by_predicate} 张(命中 {sorted(skip_set)}) "
              f"+ {skipped_by_eqpoint} 张(裸 eqpoint), 剩余 {len(target_graphs)} 张")

        n = len(target_graphs)
        n_facts = sum(len(g.facts) for g in target_graphs)
        n_rules = sum(len(g.rules) for g in target_graphs)
        stats = {
            "built": len(graphs),
            "failures": len(failures),
            "collapsed_steps": collapsed_total,
            "subgraphs": len(subgraphs) if decompose else None,
            "skipped_by_predicate": skipped_by_predicate,
            "skipped_by_eqpoint": skipped_by_eqpoint,
            "avg_facts_per_graph": round(n_facts / n, 2) if n else 0,
            "avg_rules_per_graph": round(n_rules / n, 2) if n else 0,
        }
        print(f"[part1] built={len(graphs)} failures={len(failures)} "
              f"targets={n} avg_facts={stats['avg_facts_per_graph']} "
              f"avg_rules={stats['avg_rules_per_graph']}")
        for f in failures[:5]:
            print("  FAIL", f)

        # 抽样打印一条，便于人工核对结构
        if target_graphs:
            g = target_graphs[0]
            print(f"[part1] sample {g.problem_id}: {g.summary()}")

        # 可选：可视化证明图（config: part1_extract.draw_dir / draw_limit）
        # 拆解开启时绘制的是子图；同时把对应原始数据导出到同名 .json，便于溯源。
        draw_dir = part1_cfg.get("draw_dir")
        if draw_dir and target_graphs:
            import json as _json

            from newclid.discovery.extraction.visualizer import draw_proof_graph
            draw_limit = part1_cfg.get("draw_limit") or len(target_graphs)
            os.makedirs(draw_dir, exist_ok=True)
            drawn = 0
            for g in target_graphs[:draw_limit]:
                stem = f"proof_{g.problem_id.replace(':', '_').replace('#', '__')}"
                draw_proof_graph(g, os.path.join(draw_dir, stem + ".png"))
                # 原始数据 sidecar
                sidecar = {
                    "problem_id": g.problem_id,
                    "seed": g.seed,
                    "index_in_seed": g.index_in_seed,
                    **(g.raw_record or {}),
                }
                with open(os.path.join(draw_dir, stem + ".json"), "w", encoding="utf-8") as sf:
                    _json.dump(sidecar, sf, ensure_ascii=False, indent=2)
                drawn += 1
            stats["drawn"] = drawn
            print(f"[part1] 已绘制 {drawn} 张图(含原始数据 .json) -> {draw_dir}")

        # 命题提取：无辅助点前提 → 结论（config: part1_extract.extract_propositions）
        propositions: list = []
        if part1_cfg.get("extract_propositions", True):
            from newclid.discovery.extraction.proposition_extractor import (
                extract_propositions,
                proposition_to_output,
                proposition_to_text,
            )
            from newclid.discovery.utils.jsonl_io import write_jsonl

            propositions = extract_propositions(target_graphs)
            stats["propositions"] = len(propositions)
            print(f"[part1] 命题提取: {len(propositions)} 条")
            if propositions:
                p = propositions[0]
                print(f"[part1] sample proposition {p.proposition_id}: {proposition_to_text(p)}")

            # eqpoint 等价点合并（config: part1_extract.merge_eqpoints，默认开启）
            # eqpoint a b 表示两点数值重合，多为采点退化而非有意义的几何条件，
            # 用 Union-Find 把等价点合并为代表元、去掉 eqpoint 前提本身。
            if merge_eqpoints_flag:
                from newclid.discovery.extraction.eqpoint_merge import merge_eqpoints_batch

                propositions, merged, dropped = merge_eqpoints_batch(propositions)
                stats["eqpoint_merged"] = merged
                stats["eqpoint_dropped_trivial"] = dropped
                stats["propositions"] = len(propositions)
                print(f"[part1] eqpoint 合并: {merged} 条命题合并等价点, "
                      f"{dropped} 条合并后平凡被丢弃, 剩余 {len(propositions)} 条")

            # 落盘原始命题（规则文本 + 点坐标）
            os.makedirs(os.path.dirname(prop_path), exist_ok=True)
            write_jsonl([proposition_to_output(p) for p in propositions], prop_path)
            print(f"[part1] 命题已保存 -> {prop_path}")

    # 规范化 + 去重（config: part1_extract.normalize）
    if part1_cfg.get("normalize", True) and propositions:
        rules, occurrences = _normalize_dedup(part1_cfg, propositions)
        stats["normalized_rules"] = len(rules)
        print(f"[part1] 规范化去重: {len(propositions)} 命题 -> {len(rules)} 条规则")

        # Bridge-point elimination after normalization (configurable).
        # Normalization converts para→coll and renames points, which can
        # expose bridge points invisible in the raw propositions.  Running
        # here (instead of before normalization) catches all of them.
        bridge_cfg = part1_cfg.get("bridge_elimination", {})
        if bridge_cfg.get("enabled", False) and rules:
            from dataclasses import replace
            from newclid.discovery.reduction.bridge_elimination import bridge_point_eliminate

            n_bridge = 0
            for i, r in enumerate(rules):
                new_text = bridge_point_eliminate(r.rule_text)
                if new_text and new_text != r.rule_text:
                    rules[i] = replace(r, rule_text=new_text)
                    n_bridge += 1
            if n_bridge:
                stats["normalized_rules"] = len(rules)
                print(f"[part1] 桥接点消除: {n_bridge} 条被简化")

        if rules:
            print(f"[part1] sample rule {rules[0].rule_id}: {rules[0].rule_text}")
        from newclid.discovery.extraction.normalizer import rule_to_output
        from newclid.discovery.utils.jsonl_io import write_jsonl

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_jsonl([rule_to_output(r) for r in rules], output_path)
        stats["output"] = output_path
        print(f"[part1] 规则已保存 -> {output_path}")

        occurrences_path = os.path.join(os.path.dirname(output_path), "rule_seed_occurrences_all.json")
        _write_occurrences(occurrences, occurrences_path)
        stats["seed_occurrences_all"] = occurrences_path
        print(f"[part1] 去重前 rule_text->seed 出现次数统计已保存 -> {occurrences_path}")
    else:
        output_path = prop_path
        stats["output"] = prop_path

    return output_path, stats


def _write_occurrences(occurrences: dict[str, dict[str, int]], path: str) -> None:
    """把 rule_text -> {seed: count} 的溯源统计落盘为 JSON。"""
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(occurrences, f, ensure_ascii=False, indent=2)


def _normalize_dedup(
    part1_cfg: dict[str, Any], propositions: list,
) -> tuple[list, dict[str, dict[str, int]]]:
    """按配置在串行 / Ray 并行的 normalize_and_dedup 之间选择。

    config: part1_extract.normalize_n_workers（null/<=1 = 串行）。
    """
    from newclid.discovery.extraction.normalizer import (
        normalize_and_dedup,
        normalize_and_dedup_parallel,
    )

    n_workers = part1_cfg.get("normalize_n_workers") or 0
    if n_workers and n_workers > 1:
        chunk_size = part1_cfg.get("normalize_chunk_size", 5000)
        return normalize_and_dedup_parallel(
            propositions, n_workers=n_workers, chunk_size=chunk_size,
        )
    return normalize_and_dedup(propositions)


def _run_normalize_only(
    cfg: dict[str, Any], output_dir: str, propositions_path: str,
) -> tuple[str, dict[str, Any]]:
    """只跑规范化去重：从已落盘的 propositions.jsonl 读入，跳过读数据/构图等步骤。"""
    from newclid.discovery.extraction.normalizer import rule_to_output
    from newclid.discovery.extraction.proposition_extractor import load_propositions
    from newclid.discovery.utils.jsonl_io import write_jsonl

    part1_cfg = cfg["part1_extract"]
    print(f"[part1] 从已提取命题直接规范化去重: {propositions_path}")
    propositions = load_propositions(propositions_path)
    stats: dict[str, Any] = {"propositions": len(propositions)}

    rules, occurrences = _normalize_dedup(part1_cfg, propositions)
    stats["normalized_rules"] = len(rules)
    print(f"[part1] 规范化去重: {len(propositions)} 命题 -> {len(rules)} 条规则")
    if rules:
        print(f"[part1] sample rule {rules[0].rule_id}: {rules[0].rule_text}")

    output_path = resolve_output(
        part1_cfg.get("output"), output_dir, "part1", "normalized_rules.jsonl",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl([rule_to_output(r) for r in rules], output_path)
    stats["output"] = output_path
    print(f"[part1] 规则已保存 -> {output_path}")

    occurrences_path = os.path.join(os.path.dirname(output_path), "rule_seed_occurrences_all.json")
    _write_occurrences(occurrences, occurrences_path)
    stats["seed_occurrences_all"] = occurrences_path
    print(f"[part1] 去重前 rule_text->seed 出现次数统计已保存 -> {occurrences_path}")
    return output_path, stats


def run_part2(
    cfg: dict[str, Any],
    output_dir: str,
    input_path: str,
) -> tuple[str, dict[str, Any]]:
    """执行 Part 2: 规则规约（第一步：按前提数量过滤）。委托 reduction.orchestrator。"""
    from newclid.discovery.reduction.orchestrator import run_part2 as _run_part2
    return _run_part2(cfg, output_dir, input_path)



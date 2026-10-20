"""Part 1 构图 + 命题提取的 Ray 并行版本。

串行版（pipeline.run_part1）逐条构建 SingleProofGraph、再统一做单步折叠 /
拆解 / 命题提取 / eqpoint 合并；数据量大（百万条）时这几步都是纯 CPU 且逐条
独立，天然可按 chunk 并行。

并行策略：把原始记录切成若干 chunk，每个 chunk 在一个 Ray worker 内串行跑完
"构图 -> 单步折叠 -> 拆解 -> 命题提取 -> eqpoint 合并"整条链路，只把最终的
PropositionRecord 列表和统计量传回主进程（避免跨进程传输体量大的
SingleProofGraph 对象）。
"""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from newclid.discovery.data_models import PropositionRecord


def _process_chunk(
    records: list[tuple[int, Any, int, dict[str, Any]]],
    simplify: bool,
    decompose: bool,
    merge_eqpoints_flag: bool,
    rule_skip_predicates: list[str] | None = None,
) -> tuple[list[PropositionRecord], list[dict[str, Any]], dict[str, int]]:
    """单个 chunk 内串行跑完构图 -> 折叠 -> 拆解 -> 谓词过滤 -> 命题提取 -> eqpoint 合并。"""
    from newclid.discovery.extraction.decomposer import decompose_by_aux
    from newclid.discovery.extraction.eqpoint_merge import merge_eqpoints
    from newclid.discovery.extraction.graph_manager import (
        SingleProofGraph,
        graph_skip_reason,
    )
    from newclid.discovery.extraction.proposition_extractor import extract_proposition

    skip_set = set(rule_skip_predicates or [])

    propositions: list[PropositionRecord] = []
    failures: list[dict[str, Any]] = []
    built = 0
    collapsed_total = 0
    subgraphs_count = 0
    skipped_by_predicate = 0
    skipped_by_eqpoint = 0
    merged_count = 0
    dropped_count = 0

    for line_no, seed, index_in_seed, record in records:
        try:
            graph = SingleProofGraph.build_from_result_record(
                record, seed=seed, index_in_seed=index_in_seed,
            )
        except Exception as exc:  # 单条失败不影响整批
            failures.append(
                {"seed": seed, "index_in_seed": index_in_seed, "line": line_no, "error": str(exc)}
            )
            continue
        built += 1

        if simplify:
            collapsed_total += graph.simplify_single_step()

        target_graphs = [graph]
        if decompose:
            subs = decompose_by_aux(graph)
            target_graphs = subs
            subgraphs_count += len(subs)

        # 整图丢弃：rule_skip_predicates 命中 → "predicate"；裸 eqpoint → "eqpoint"
        # （后者恒定生效 —— 见 graph_skip_reason）。两条规则对每张图分别计数，
        # 便于在 pipeline_summary 里分辨 eqpoint 这条不变量实际拦下了多少。
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

        for g in target_graphs:
            prop = extract_proposition(g)
            if prop is None:
                continue
            if merge_eqpoints_flag:
                has_eqpoint = any(p.predicate == "eqpoint" for p in prop.premises)
                merged_prop = merge_eqpoints(prop)
                if merged_prop is None:
                    dropped_count += 1
                    continue
                if has_eqpoint:
                    merged_count += 1
                prop = merged_prop
            propositions.append(prop)

    stats = {
        "built": built,
        "failures": len(failures),
        "collapsed_steps": collapsed_total,
        "subgraphs": subgraphs_count,
        "skipped_by_predicate": skipped_by_predicate,
        "skipped_by_eqpoint": skipped_by_eqpoint,
        "eqpoint_merged": merged_count,
        "eqpoint_dropped_trivial": dropped_count,
    }
    return propositions, failures, stats


def extract_propositions_parallel(
    input_path: str,
    *,
    limit: int | None = None,
    sample: int | None = None,
    random_seed: int | None = None,
    simplify: bool = True,
    decompose: bool = False,
    merge_eqpoints_flag: bool = True,
    n_workers: int = 30,
    chunk_size: int = 2000,
    rule_skip_predicates: list[str] | None = None,
) -> tuple[list[PropositionRecord], list[dict[str, Any]], dict[str, Any]]:
    """构图 + 单步折叠 + 拆解 + 谓词过滤 + 命题提取 + eqpoint 合并的 Ray 并行版。

    与串行版（build_proof_graphs + 逐步处理）等价，只是把每条记录的整条处理
    链路下放到 Ray worker，主进程只做 chunk 切分与统计汇总。

    Returns
    -------
    (propositions, failures, stats)
        propositions: 合并后的 PropositionRecord 列表。
        failures: 构图失败记录。
        stats: 汇总统计（built / failures / collapsed_steps / subgraphs /
        skipped_by_predicate / skipped_by_eqpoint / eqpoint_merged /
        eqpoint_dropped_trivial）。
    """
    import ray

    from newclid.discovery.extraction.graph_manager import select_records
    from newclid.discovery.reduction.parallel import ensure_ray, run_bounded

    chosen = select_records(input_path, limit=limit, sample=sample, random_seed=random_seed)
    chunks = [chosen[i:i + chunk_size] for i in range(0, len(chosen), chunk_size)]

    if not chunks:
        return [], [], {
            "built": 0, "failures": 0, "collapsed_steps": 0,
            "subgraphs": 0, "skipped_by_predicate": 0, "skipped_by_eqpoint": 0,
            "eqpoint_merged": 0, "eqpoint_dropped_trivial": 0,
        }

    ensure_ray(n_workers)
    remote = ray.remote(_process_chunk)
    args = [(c, simplify, decompose, merge_eqpoints_flag, rule_skip_predicates) for c in chunks]

    propositions: list[PropositionRecord] = []
    failures: list[dict[str, Any]] = []
    agg: dict[str, int] = {
        "built": 0, "failures": 0, "collapsed_steps": 0,
        "subgraphs": 0, "skipped_by_predicate": 0, "skipped_by_eqpoint": 0,
        "eqpoint_merged": 0, "eqpoint_dropped_trivial": 0,
    }
    for props, fails, stats in tqdm(
        run_bounded(remote, args, inflight=n_workers),
        total=len(chunks), desc="[part1] 构图+命题提取(并行)", unit="块",
    ):
        propositions.extend(props)
        failures.extend(fails)
        for k in agg:
            agg[k] += stats.get(k, 0)

    return propositions, failures, agg

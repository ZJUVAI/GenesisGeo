#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分叉子图挖掘一键运行脚本。
- 目的：`python run_gspan_branched_demo.py` 即按脚本内推荐超参运行，不会卡住。
- 特点：集中管理超参 CONFIG；默认采用小数据、强剪枝与限流（扩展步数上限 + 标签全局支持剪枝）。
- 注意：为避免包的重依赖，这里直接按路径加载 proof_graph.py。
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
import importlib.util
import json
import time
from datetime import datetime
import threading
import queue


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "../src")
PROOF_GRAPH_PY = os.path.join(SRC_DIR, "newclid", "data_discovery", "proof_graph.py")
# 默认使用更小的 lil 数据集以保证快速稳定运行
DEFAULT_JSON = os.path.join(SRC_DIR, "newclid", "data_discovery", "r07_expanded_problems_results.json")

# 集中管理的默认超参（稳妥不挂的一组）
CONFIG = {
    # 数据集：小集，速度稳
    "json": DEFAULT_JSON,
    # 支持度（字符串，内部解析）
    "min_support": "2",
    # 结构阈值（建议提升规则节点数以获得更有意义的图形）
    "min_rule_nodes": 3,
    "min_edges": 3,
    # 分叉特有阈值：至少一个 rule 入度>=2 可更聚焦（0 表示不强制）
    "min_rule_indeg2_count": 1,
    # 搜索上限
    "max_nodes": 10,
    "max_edges": 16,
    # 代表性嵌入数
    "sample_embeddings": 2,
    # 安全阈值：扩展步数上限与日志频率（防爆炸）
    "debug_limit_expansions": 100000,
    "debug_log_every": 10000,
    # 额外安全：全局按标签题目覆盖度进行剪枝（建议开启）
    # 按标签题目覆盖度剪枝（F:*, R:* 标签均适用）
    "prune_low_support_labels": True,
    # 按规则 code 的题目覆盖度剪枝（R:code）
    "prune_by_rule": True,
    # 是否允许接入生产者规则（向上接 1 层）
    "attach_producer": True,
    "max_producer_depth": 1,
    # 跳过 unknown 前提（减少噪声与节点数）
    "skip_unknown": True,
    # 变量闭包兜底检查：结论变量 ⊆ 前提变量（默认关闭）
    "enable_var_closure_check": True,
    # 可选：总时长预算（秒）。None 表示不限制
    "time_budget_seconds": None,
    # 输出
    "top_k": 5,
    # 安静模式
    "quiet": True,
    # 新增：流式写入 & 挖掘并行（默认关闭，维持旧行为）
    "stream_write": 1,      # 1: 主线程消费即写，0: 聚合后一次写
    "parallel_mining": 1,   # 1: 按种子并行挖掘（子线程仅搜索，主线程转换/去重/过滤/写入）
    "mining_workers": 10,    # 挖掘工作线程数（parallel_mining=1 时生效）
    # schema 过滤：是否直接丢弃包含 unknown() 的结果
    "drop_unknown_in_schema": 1,
}


def _load_proof_graph_module():
    spec = importlib.util.spec_from_file_location("proof_graph", PROOF_GRAPH_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from {PROOF_GRAPH_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_min_support(s: str):
    s = s.strip()
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return int(s)


def main(argv=None):
    # 允许覆盖 CONFIG，但不需要输入任何参数也能跑
    parser = argparse.ArgumentParser(description="Run branched subgraph mining (one-click)")
    parser.add_argument("--json", default=CONFIG["json"], help=f"Input results json (default: {CONFIG['json']})")
    # 输出目录：默认到 data_discovery/data
    default_out_dir = os.path.join(SRC_DIR, "newclid", "data_discovery", "data")
    parser.add_argument("--out-dir", default=default_out_dir, help="Directory to save input copy and mining results")
    parser.add_argument("--mode", choices=["fact_rule", "rule_only"], default="fact_rule", help="Mining on full graph (fact_rule) or rules-only graph (rule_only)")
    parser.add_argument("--min-support", default=CONFIG["min_support"], type=str)
    parser.add_argument("--min-rule-nodes", default=CONFIG["min_rule_nodes"], type=int)
    parser.add_argument("--min-edges", default=CONFIG["min_edges"], type=int)
    parser.add_argument("--min-rule-indeg2-count", default=CONFIG["min_rule_indeg2_count"], type=int)
    parser.add_argument("--max-nodes", default=CONFIG["max_nodes"], type=int)
    parser.add_argument("--max-edges", default=CONFIG["max_edges"], type=int)
    parser.add_argument("--sample-embeddings", default=CONFIG["sample_embeddings"], type=int)
    parser.add_argument("--debug-limit-expansions", default=CONFIG["debug_limit_expansions"], type=int)
    parser.add_argument("--debug-log-every", default=CONFIG["debug_log_every"], type=int)
    parser.add_argument("--time-budget-seconds", default=CONFIG["time_budget_seconds"], type=float)
    parser.add_argument("--prune-low-support-labels", default=int(CONFIG["prune_low_support_labels"]), type=int,
                        help="1 to enable, 0 to disable (label-level global support pruning)")
    parser.add_argument("--prune-by-rule", default=int(CONFIG["prune_by_rule"]), type=int,
                        help="1 to enable, 0 to disable (rule-level global support pruning)")
    parser.add_argument("--attach-producer", default=int(CONFIG["attach_producer"]), type=int,
                        help="1 to enable, 0 to disable (attach producer rule up to max_producer_depth)")
    parser.add_argument("--max-producer-depth", default=CONFIG["max_producer_depth"], type=int)
    parser.add_argument("--skip-unknown", default=int(CONFIG["skip_unknown"]), type=int,
                        help="1 to skip F:unknown premises during expansion, 0 to include")
    parser.add_argument("--enable-var-closure-check", default=int(CONFIG["enable_var_closure_check"]), type=int,
                        help="1 to enforce conclusion vars subset of premise vars at finalize, 0 to skip")
    parser.add_argument("--top-k", default=CONFIG["top_k"], type=int)
    parser.add_argument("--quiet", action="store_true" if not CONFIG["quiet"] else "store_false")
    # 新增：流式写入 & 挖掘并行
    parser.add_argument("--stream-write", default=CONFIG["stream_write"], type=int,
                        help="1 to stream write (consume-and-write in main thread), 0 to aggregate then write")
    parser.add_argument("--parallel-mining", default=CONFIG["parallel_mining"], type=int,
                        help="1 to enable seed-level parallel mining; 0 to use legacy single-thread mining")
    parser.add_argument("--mining-workers", default=CONFIG["mining_workers"], type=int,
                        help="Number of worker threads for seed-level mining when --parallel-mining=1")
    # schema 过滤
    parser.add_argument("--drop-unknown-in-schema", default=CONFIG["drop_unknown_in_schema"], type=int,
                        help="1 to drop any pattern whose schema contains unknown()")
    args = parser.parse_args(argv)

    mod = _load_proof_graph_module()
    ProofGraph = getattr(mod, "ProofGraph")
    GSpanMiner = getattr(mod, "GSpanMiner")

    min_support = parse_min_support(args.min_support)

    # 计时：解析与构图
    t0 = time.time()
    pg = ProofGraph(verbose=not CONFIG["quiet"] if not hasattr(args, "quiet") else args.quiet is False,
                    log_level=logging.INFO)
    pg.from_results_json(args.json)
    t1 = time.time()

    miner = GSpanMiner(
        pg,
        min_support=min_support,
        min_rule_nodes=args.min_rule_nodes,
        min_edges=args.min_edges,
        max_nodes=args.max_nodes,
        sample_embeddings=args.sample_embeddings,
    )
    # 计时：挖掘
    patterns = []
    t2 = None  # 将在后续收集阶段结束后设置
    if int(args.parallel_mining) == 0:
        if args.mode == "fact_rule":
            patterns = miner.run_branched(
                min_rule_indeg2_count=args.min_rule_indeg2_count,
                max_edges=args.max_edges,
                debug_limit_expansions=args.debug_limit_expansions,
                debug_log_every=args.debug_log_every,
                time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                prune_low_support_labels=bool(args.prune_low_support_labels),
                prune_by_rule=bool(args.prune_by_rule),
                attach_producer=bool(args.attach_producer),
                max_producer_depth=int(args.max_producer_depth),
                skip_unknown=bool(args.skip_unknown),
                enable_var_closure_check=bool(args.enable_var_closure_check),
            )
        else:
            patterns = miner.run_rules_only(
                max_edges=args.max_edges,
                debug_limit_expansions=args.debug_limit_expansions,
                debug_log_every=args.debug_log_every,
                time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                prune_low_support_labels=bool(args.prune_low_support_labels),
                prune_by_rule=bool(args.prune_by_rule),
                enable_var_closure_check=bool(args.enable_var_closure_check),
            )
        t2 = time.time()
        print(f"total patterns: {len(patterns)}")

    # ------- 工具：schema 解析/过滤/规范化 去重 -------
    import re as _re
    _ATOM_RE = _re.compile(r"\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*")

    def parse_schema(schema: str):
        """返回 (prem_atoms, concl_atom)，atom 为 (pred, [args...])。"""
        if not schema:
            return [], None
        if "=>" not in schema:
            return [], None
        left, right = schema.split("=>", 1)
        left = left.strip()
        right = right.strip()
        prem_atoms = []
        if left:
            for part in left.split("∧"):
                m = _ATOM_RE.match(part.strip())
                if not m:
                    continue
                pred = m.group(1)
                args = [a.strip() for a in m.group(2).split(",") if a.strip()]
                prem_atoms.append((pred, args))
        m = _ATOM_RE.match(right)
        concl = None
        if m:
            pred = m.group(1)
            args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            concl = (pred, args)
        return prem_atoms, concl

    def has_unknown(schema: str) -> bool:
        return "unknown(" in schema or schema.strip().startswith("unknown()")

    def concl_args_subset_of_prem(schema: str) -> bool:
        prem, concl = parse_schema(schema)
        if not concl:
            return False
        prem_args = set()
        for _, args in prem:
            prem_args.update(args)
        return set(concl[1]).issubset(prem_args)

    def canonical_key(schema: str) -> str:
        """对 schema 做变量规范化 + 前提无序排序，得到稳定键。"""
        prem, concl = parse_schema(schema)
        # 变量首次出现顺序 -> X1,X2,...（跨 premise+conclusion）
        order = []
        def vnorm(arg: str) -> str:
            if arg not in order:
                order.append(arg)
            return f"X{order.index(arg)+1}"
        prem_norm = []
        for pred, args in prem:
            prem_norm.append((pred, tuple(vnorm(a) for a in args), len(args)))
        prem_norm.sort(key=lambda x: (x[0], x[2], x[1]))
        concl_norm = None
        if concl:
            concl_norm = (concl[0], tuple(vnorm(a) for a in concl[1]), len(concl[1]))
        return json.dumps({
            "prem": [(p, list(a), k) for (p,a,k) in prem_norm],
            "concl": None if concl_norm is None else [concl_norm[0], list(concl_norm[1]), concl_norm[2]],
        }, ensure_ascii=False, sort_keys=True)

    # 打印 Top-K（仅在非并行挖掘路径下，先简单转换，不过滤），便于快速查看
    if int(args.parallel_mining) == 0:
        K = max(0, int(args.top_k))
        for i, p in enumerate(patterns[:K]):
            expr, vmap = (miner.pattern_to_schema_branched(p) if args.mode=="fact_rule" else miner.pattern_to_schema_rules_only(p))
            labels = p.get("labels", [])
            print(f"[{i}] support={p.get('support')} pids={p.get('pids')} nodes={len(p.get('nodes', []))} edges={len(p.get('edges', []))}")
            print(f"     labels: {labels}")
            print(f"     edges: {p.get('edges')}")
            print(f"     schema: {expr}")
            if vmap:
                print(f"     var_map: {vmap}")

    # 结果保存到 data_discovery/data
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Debug: 收集含 unknown 的 schema 的题目并落盘（rule_only 模式常见）
    def _write_unknown_problems_txt():
        try:
            unknown_details = []
            unknown_pids = set()
            for i, p in enumerate(patterns):
                if args.mode == "fact_rule":
                    expr, _ = miner.pattern_to_schema_branched(p)
                else:
                    expr, _ = miner.pattern_to_schema_rules_only(p)
                if not isinstance(expr, str):
                    continue
                if "unknown" in expr:
                    pids = p.get("pids", [])
                    for pid in pids:
                        unknown_pids.add(pid)
                    unknown_details.append({
                        "rank": i,
                        "support": p.get("support"),
                        "pids": pids,
                        "schema": expr,
                    })

            if not unknown_details:
                return None

            out_txt = os.path.join(out_dir, "unknown_problem.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(f"created_at: {datetime.now().isoformat()}\n")
                f.write(f"mode: {args.mode}\n")
                f.write(f"total_patterns: {len(patterns)}\n")
                f.write(f"unknown_patterns: {len(unknown_details)}\n")
                f.write(f"unique_problem_ids: {len(unknown_pids)}\n")
                f.write("unknown_problem_ids:\n")
                for pid in sorted(unknown_pids):
                    f.write(f"- {pid}\n")
                f.write("\npatterns_with_unknown:\n")
                for item in unknown_details:
                    f.write(f"[rank {item['rank']}] support={item['support']} pids={item['pids']}\n")
                    f.write(f"schema: {item['schema']}\n\n")
            print(f"unknown problems written to: {out_txt}")
            return out_txt
        except Exception as e:
            print(f"failed to write unknown_problem.txt: {e}")
            return None

    # 仅在 rule_only 下默认写出，便于排查数据来源问题
    if args.mode == "rule_only":
        _write_unknown_problems_txt()

    # 按你的要求：不在结果中包含 input.json 信息

    # 汇总元信息
    def count_pg_types():
        fact = sum(1 for nd in pg.nodes.values() if nd.get("type") == "fact")
        rule = sum(1 for nd in pg.nodes.values() if nd.get("type") == "rule")
        return fact, rule

    fact_cnt, rule_cnt = count_pg_types()
    merged_nodes = len(miner.G.nodes)
    merged_edges = sum(len(vs) for vs in miner.G.out_edges.values())
    problems = len(pg.fact_id_map)

    # ---------- 结果收集（按新方案 B）：可选并行挖掘 + 主线程去重/过滤/写入 ----------
    tmp_ndjson = os.path.join(out_dir, f"tmp_stream_{args.mode}.ndjson")
    do_stream = int(args.stream_write) == 1

    # 统一的主线程收集器
    def collector_init():
        return {
            "sig_seen": set(),         # 结构签名去重（labels+edges）
            "schema_seen": set(),      # canonical schema 去重
            "patterns_all": [],
            "dropped_unknown": 0,
            "dropped_var_subset": 0,
            "dedup_sig": 0,
            "dedup_schema": 0,
            "written": 0,
        }

    coll = collector_init()
    wf = None
    if do_stream:
        wf = open(tmp_ndjson, "w", encoding="utf-8")

    def process_one_pattern(p: dict):
        # 第一层：结构签名去重
        sig = (tuple(p.get("labels", [])), tuple(sorted(p.get("edges", []))))
        if sig in coll["sig_seen"]:
            coll["dedup_sig"] += 1
            return
        coll["sig_seen"].add(sig)
        # schema 转换（主线程）
        expr, _ = (miner.pattern_to_schema_branched(p) if args.mode=="fact_rule" else miner.pattern_to_schema_rules_only(p))
        # 过滤
        if args.drop_unknown_in_schema and has_unknown(expr):
            coll["dropped_unknown"] += 1
            return
        if bool(args.enable_var_closure_check) and not concl_args_subset_of_prem(expr):
            coll["dropped_var_subset"] += 1
            return
        # 第二层：schema 去重
        key = canonical_key(expr) if expr else None
        if key and key in coll["schema_seen"]:
            coll["dedup_schema"] += 1
            return
        if key:
            coll["schema_seen"].add(key)
        item = {
            "support": p.get("support"),
            "pids_count": len(p.get("pids", [])),
            "nodes": len(p.get("nodes", [])),
            "edges": len(p.get("edges", [])),
            "labels": p.get("labels", []),
            "schema": expr,
        }
        if do_stream and wf is not None:
            wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            coll["written"] += 1
        else:
            coll["patterns_all"].append(item)

    if int(args.parallel_mining) == 1:
        print(f"[mining-parallel] enabled with workers={int(args.mining_workers)}", flush=True)

        # 构建种子
        if args.mode == "fact_rule":
            seeds = miner.build_branched_seeds(prune_by_rule=bool(args.prune_by_rule), skip_unknown=bool(args.skip_unknown))
        else:
            seeds = miner.build_rules_only_seeds(prune_by_rule=bool(args.prune_by_rule))
        print(f"[mining-parallel] seeds={len(seeds)}", flush=True)

        # 输出队列（子线程 emit 原始模式，主线程消费）
        qout: "queue.Queue[dict]" = queue.Queue(maxsize=1000)

        def emit_fn(pobj: dict):
            # 子线程将原始模式放入队列，由主线程处理
            try:
                qout.put(pobj, timeout=1.0)
            except queue.Full:
                pass

        # worker：扩展一个或多个种子
        def worker_seed(job_q: "queue.Queue[dict]"):
            while True:
                try:
                    seed = job_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    if args.mode == "fact_rule":
                        miner.expand_branched_from_seed(
                            seed,
                            min_rule_indeg2_count=args.min_rule_indeg2_count,
                            max_edges=args.max_edges,
                            debug_limit_expansions=args.debug_limit_expansions,
                            debug_log_every=args.debug_log_every,
                            time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                            prune_low_support_labels=bool(args.prune_low_support_labels),
                            prune_by_rule=bool(args.prune_by_rule),
                            attach_producer=bool(args.attach_producer),
                            max_producer_depth=int(args.max_producer_depth),
                            skip_unknown=bool(args.skip_unknown),
                            enable_var_closure_check=bool(args.enable_var_closure_check),
                            emit=emit_fn,
                        )
                    else:
                        miner.expand_rules_only_from_seed(
                            seed,
                            max_edges=args.max_edges,
                            debug_limit_expansions=args.debug_limit_expansions,
                            debug_log_every=args.debug_log_every,
                            time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                            prune_low_support_labels=bool(args.prune_low_support_labels),
                            prune_by_rule=bool(args.prune_by_rule),
                            enable_var_closure_check=bool(args.enable_var_closure_check),
                            emit=emit_fn,
                        )
                finally:
                    job_q.task_done()

        # 准备种子任务
        jobs: "queue.Queue[dict]" = queue.Queue()
        for s in seeds:
            jobs.put(s)

        # 启动 worker
        n_workers = max(1, int(args.mining_workers))
        threads = [threading.Thread(target=worker_seed, args=(jobs,), daemon=True) for _ in range(n_workers)]
        for t in threads:
            t.start()

        # 主线程消费输出（边出边处理/可选写入）
        while True:
            try:
                obj = qout.get(timeout=0.2)
                process_one_pattern(obj)
                qout.task_done()
            except queue.Empty:
                # 检查 worker 是否全部结束、队列是否为空
                alive = any(t.is_alive() for t in threads)
                if not alive and qout.empty() and jobs.empty():
                    break

        # 收尾
        jobs.join()
        qout.join()
        t2 = time.time()
    else:
        # 旧路径：矿工一次性返回所有模式，主线程统一处理
        for p in patterns:
            process_one_pattern(p)
        if t2 is None:
            t2 = time.time()

    # 收集完成：若启用流式写入，读取 NDJSON 汇总到内存结构
    patterns_all: list[dict] = coll["patterns_all"]
    if do_stream and wf is not None:
        wf.close()
        with open(tmp_ndjson, "r", encoding="utf-8") as rf:
            for line in rf:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                patterns_all.append(obj)
        try:
            os.remove(tmp_ndjson)
        except Exception:
            pass

    # Top-K 摘要（按出现顺序）
    topn = max(0, int(args.top_k))
    patterns_summary: list[dict] = patterns_all[:topn]

    result_obj = {
        "created_at": datetime.now().isoformat(),
        "problems": problems,
        "proof_graph": {
            "nodes_total": len(pg.nodes),
            "edges_total": len(pg.edges),
            "fact_nodes": fact_cnt,
            "rule_nodes": rule_cnt,
            "stats": pg.stats,
        },
        "merged_graph": {
            "nodes_total": merged_nodes,
            "edges_total": merged_edges,
        },
        "params": {
            "mode": args.mode,
            "min_support": args.min_support,
            "min_rule_nodes": args.min_rule_nodes,
            "min_edges": args.min_edges,
            "min_rule_indeg2_count": args.min_rule_indeg2_count,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
            "sample_embeddings": args.sample_embeddings,
            "debug_limit_expansions": args.debug_limit_expansions,
            "debug_log_every": args.debug_log_every,
            "time_budget_seconds": args.time_budget_seconds,
            "prune_low_support_labels": bool(args.prune_low_support_labels),
            "prune_by_rule": bool(args.prune_by_rule),
            "attach_producer": bool(args.attach_producer),
            "max_producer_depth": int(args.max_producer_depth),
            "skip_unknown": bool(args.skip_unknown),
            "enable_var_closure_check": bool(args.enable_var_closure_check),
            "stream_write": int(args.stream_write),
            "parallel_mining": int(args.parallel_mining),
            "mining_workers": int(args.mining_workers),
            "drop_unknown_in_schema": bool(args.drop_unknown_in_schema),
        },
        "timing_sec": {
            "parse_build": t1 - t0,
            "mining": t2 - t1,
            "total": t2 - t0,
        },
    "pattern_count": len(patterns_all),
        "patterns_summary_topN": patterns_summary,
    }

    # 始终写入全量模式（摘要+schema 格式）
    result_obj["patterns"] = patterns_all

    out_name = ("branched_mining.json" if args.mode=="fact_rule" else "rules_only_mining.json")
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, ensure_ascii=False, indent=2)
    print(f"results saved to: {out_path}")


if __name__ == "__main__":
    sys.exit(main())

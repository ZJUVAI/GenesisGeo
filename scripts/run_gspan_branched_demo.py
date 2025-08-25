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
import multiprocessing as mp
 


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "../src")
PROOF_GRAPH_PY = os.path.join(SRC_DIR, "newclid", "data_discovery", "proof_graph.py")
MINING_PIPELINE_PY = os.path.join(SRC_DIR, "newclid", "data_discovery", "mining_pipeline.py")
# 默认使用更小的 lil 数据集以保证快速稳定运行
DEFAULT_JSON = os.path.join(SRC_DIR, "newclid", "data_discovery", "r07_expanded_problems_results.json")

# 集中管理的默认超参（分组展示，便于快速修改）
CONFIG = {
    # ===== 常用：数据与运行模式 =====
    "json": DEFAULT_JSON,           # 输入结果 JSON
    "top_k": 5,                     # 结果预览 Top-K
    "quiet": True,                  # 安静输出
    "mode": "rule_only",            # fact_rule rule_only

    # ===== rule_only 专用 =====
    "rule_only_max_nodes_ratio": 0.1,  # 有效 max_nodes=ceil(max_rules_per_problem * ratio)；<=0 忽略

    # ===== 常用：核心阈值（结构/支持） =====
    "min_support": "5",             # 支持度（字符串，内部解析为 int/float）
    "max_nodes": 15,                # 搜索上限（节点数）
    "min_rule_nodes": 3,            # 最少规则节点
    "min_edges": 3,                 # 最少边数
    "min_rule_indeg2_count": 1,     # fact_rule：至少一个规则入度≥2（0 关闭）

    # ===== 常用：执行引擎与输出 =====
    "engine": "seeds_mproc",            # 执行引擎： single | seeds | seeds_mproc
    "workers": max(1, (os.cpu_count() or 2) // 2),  # 多进程 worker 数（仅 seeds_mproc 生效）
    "stream_write": 1,              # 1: 流式写入; 0: 汇总后写盘
    "stage_dump": 0,                # 1: dump 种子/扩展审计；0: 关闭

    # ===== 不常用：剪枝与过滤 =====
    "prune_low_support_labels": True,  # 标签级全局支持剪枝（F/R 均适用）
    "prune_by_rule": True,             # 规则级全局支持剪枝（R:code）
    "skip_unknown": True,              # 扩展时跳过 unknown 前提
    "enable_var_closure_check": True,  # 变量闭包检查（结论变量 ⊆ 前提变量）
    "drop_unknown_in_schema": 1,       # schema 中含 unknown() 则丢弃
    "enable_dependency_filter": 1,     # 依赖过滤（point_rely_on）

    # ===== 不常用：预算与日志 =====
    "sample_embeddings": 2,            # 代表性嵌入数
    "debug_limit_expansions": 1000000,  # 扩展步数上限（seeds_mproc 为全局限额）
    "debug_log_every": 5000,           # 调试日志频率
    "time_budget_seconds": None,       # 总时长预算（秒）
    "attach_producer": True,           # 允许接入生产者规则
    "max_producer_depth": 1,           # 生产者规则向上深度
}


def _load_proof_graph_module():
    spec = importlib.util.spec_from_file_location("proof_graph", PROOF_GRAPH_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from {PROOF_GRAPH_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_mining_pipeline_module():
    spec = importlib.util.spec_from_file_location("mining_pipeline", MINING_PIPELINE_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from {MINING_PIPELINE_PY}")
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

    # ----- 常用：数据与运行模式 -----
    parser.add_argument("--json", default=CONFIG["json"], help=f"Input results json (default: {CONFIG['json']})")
    default_out_dir = os.path.join(SRC_DIR, "newclid", "data_discovery", "data")
    parser.add_argument("--out-dir", default=default_out_dir, help="Directory to save input copy and mining results")
    parser.add_argument("--mode", choices=["fact_rule", "rule_only"], default=CONFIG["mode"], help="Mining on full graph (fact_rule) or rules-only graph (rule_only)")
    parser.add_argument("--top-k", default=CONFIG["top_k"], type=int)
    parser.add_argument("--quiet", action="store_true" if not CONFIG["quiet"] else "store_false")

    # ----- 常用：核心阈值（结构/支持） -----
    parser.add_argument("--min-support", default=CONFIG["min_support"], type=str)
    parser.add_argument("--max-nodes", default=CONFIG["max_nodes"], type=int)
    parser.add_argument("--min-rule-nodes", default=CONFIG["min_rule_nodes"], type=int)
    parser.add_argument("--min-edges", default=CONFIG["min_edges"], type=int)
    parser.add_argument("--min-rule-indeg2-count", default=CONFIG["min_rule_indeg2_count"], type=int)

    # ----- 常用：执行引擎与输出 -----
    parser.add_argument("--engine", choices=["single", "seeds", "seeds_mproc"], default=CONFIG["engine"],
                        help="Engine: single (default legacy) or seeds (single-process per-seed expansion)")
    parser.add_argument("--workers", default=CONFIG["workers"], type=int,
                        help="Workers for seeds_mproc engine (process count)")
    parser.add_argument("--stream-write", default=CONFIG["stream_write"], type=int,
                        help="1 to stream write (consume-and-write in main thread), 0 to aggregate then write")
    parser.add_argument("--stage-dump", default=CONFIG["stage_dump"], type=int,
                        help="1 to dump seeds and per-seed expansions audits; 0 to disable")

    # ----- 不常用：剪枝与过滤 -----
    parser.add_argument("--prune-low-support-labels", default=int(CONFIG["prune_low_support_labels"]), type=int,
                        help="1 to enable, 0 to disable (label-level global support pruning)")
    parser.add_argument("--prune-by-rule", default=int(CONFIG["prune_by_rule"]), type=int,
                        help="1 to enable, 0 to disable (rule-level global support pruning)")
    parser.add_argument("--skip-unknown", default=int(CONFIG["skip_unknown"]), type=int,
                        help="1 to skip F:unknown premises during expansion, 0 to include")
    parser.add_argument("--enable-var-closure-check", default=int(CONFIG["enable_var_closure_check"]), type=int,
                        help="1 to enforce conclusion vars subset of premise vars at finalize, 0 to skip")
    parser.add_argument("--drop-unknown-in-schema", default=CONFIG["drop_unknown_in_schema"], type=int,
                        help="1 to drop any pattern whose schema contains unknown()")
    parser.add_argument("--enable-dependency-filter", default=CONFIG["enable_dependency_filter"], type=int,
                        help="1 to enable dependency filter based on point_rely_on, 0 to disable")

    # ----- 不常用：预算与日志 -----
    parser.add_argument("--sample-embeddings", default=CONFIG["sample_embeddings"], type=int)
    parser.add_argument("--debug-limit-expansions", default=CONFIG["debug_limit_expansions"], type=int)
    parser.add_argument("--debug-log-every", default=CONFIG["debug_log_every"], type=int)
    parser.add_argument("--time-budget-seconds", default=CONFIG["time_budget_seconds"], type=float)
    parser.add_argument("--attach-producer", default=int(CONFIG["attach_producer"]), type=int,
                        help="1 to enable, 0 to disable (attach producer rule up to max_producer_depth)")
    parser.add_argument("--max-producer-depth", default=CONFIG["max_producer_depth"], type=int)

    # ----- rule_only 专用 -----
    parser.add_argument("--rule-only-max-nodes-ratio", default=CONFIG["rule_only_max_nodes_ratio"], type=float,
                        help="When mode=rule_only and ratio>0, set effective max_nodes=ceil(max rules-per-problem * ratio)")
    args = parser.parse_args(argv)
    dbg_every = max(1, int(args.debug_log_every))

    mod = _load_proof_graph_module()
    ProofGraph = getattr(mod, "ProofGraph")
    GSpanMiner = getattr(mod, "GSpanMiner")

    min_support = parse_min_support(args.min_support)

    # 计时：解析与构图
    t0 = time.time()
    print(f"[init] engine={args.engine} mode={args.mode} quiet={args.quiet} max_nodes={args.max_nodes} ratio={args.rule_only_max_nodes_ratio}", flush=True)
    pg = ProofGraph(verbose=not CONFIG["quiet"] if not hasattr(args, "quiet") else args.quiet is False,
                    log_level=logging.INFO)
    pg.from_results_json(args.json)
    t1 = time.time()
    print(f"[graph] loaded in {t1-t0:.3f}s from {args.json}", flush=True)

    # 当 mode=rule_only 且设置了比例时，计算有效 max_nodes
    effective_max_nodes = int(args.max_nodes)
    if args.mode == "rule_only" and args.rule_only_max_nodes_ratio and float(args.rule_only_max_nodes_ratio) > 0:
        try:
            import math
            max_rules_per_problem = 0
            for pid, step_map in pg.rule_step_map.items():
                max_rules_per_problem = max(max_rules_per_problem, len(step_map or {}))
            effective_max_nodes = max(1, int(math.ceil(max_rules_per_problem * float(args.rule_only_max_nodes_ratio))))
        except Exception as e:
            # 回退到显式 max_nodes
            effective_max_nodes = int(args.max_nodes)

    miner = GSpanMiner(
        pg,
        min_support=min_support,
        min_rule_nodes=args.min_rule_nodes,
        min_edges=args.min_edges,
        max_nodes=effective_max_nodes,
        sample_embeddings=args.sample_embeddings,
    )
    print(f"[miner] constructed with effective_max_nodes={effective_max_nodes}; support={args.min_support}; sample_embeddings={args.sample_embeddings}", flush=True)
    # 计时：挖掘
    patterns = []
    t2 = None  # 将在后续收集阶段结束后设置
    if args.engine == "single":
        print(f"[single] start mining mode={args.mode}", flush=True)
        if args.mode == "fact_rule":
            patterns = miner.run_branched(
                min_rule_indeg2_count=args.min_rule_indeg2_count,
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
                debug_limit_expansions=args.debug_limit_expansions,
                debug_log_every=args.debug_log_every,
                time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                prune_low_support_labels=bool(args.prune_low_support_labels),
                prune_by_rule=bool(args.prune_by_rule),
                enable_var_closure_check=bool(args.enable_var_closure_check),
            )
        t2 = time.time()
        print(f"[single] mined patterns={len(patterns)} in {t2-t1:.3f}s", flush=True)
    else:
        # seeds 引擎（单/多进程）：按种子扩展并即时处理
        audit_seeds = None
        audit_exp = None
        # 提前定义输出目录，避免在 stage_dump=0 时 out_dir 未定义
        out_dir = os.path.abspath(args.out_dir)
        if int(args.stage_dump) == 1:
            audit_seeds = open(os.path.join(out_dir, f"audit_{args.mode}_engine_seeds.ndjson"), "w", encoding="utf-8")
            audit_exp = open(os.path.join(out_dir, f"audit_{args.mode}_engine_expansions.ndjson"), "w", encoding="utf-8")

        # 构建种子
        if args.mode == "fact_rule":
            seeds = miner.build_branched_seeds(prune_by_rule=bool(args.prune_by_rule), skip_unknown=bool(args.skip_unknown))
        else:
            seeds = miner.build_rules_only_seeds(prune_by_rule=bool(args.prune_by_rule))
        print(f"[seeds] built seeds={len(seeds)}", flush=True)
        if audit_seeds:
            for idx, s in enumerate(seeds):
                try:
                    rec = {"seed_id": idx, "mode": args.mode, "pid": s.get("pid")}
                    audit_seeds.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        if args.engine == "seeds":
            # 单进程回调处理
            mp_mod = _load_mining_pipeline_module()
            MiningPipeline = getattr(mp_mod, "MiningPipeline")
            pipeline = MiningPipeline(pg, miner, args, out_dir)

            processed = 0
            t_emit0 = time.time()
            def emit_one(pobj: dict):
                nonlocal processed
                pipeline.process_one_pattern(pobj)
                processed += 1
                if processed % dbg_every == 0:
                    dt = time.time() - t_emit0
                    print(f"[seeds] processed={processed} elapsed={dt:.2f}s", flush=True)
                if audit_exp:
                    try:
                        audit_exp.write(json.dumps({
                            "support": pobj.get("support"),
                            "pids": list(pobj.get("pids", [])),
                            "labels": pobj.get("labels", []),
                        }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

            for s in seeds:
                if args.mode == "fact_rule":
                    miner.expand_branched_from_seed(
                        s,
                        min_rule_indeg2_count=args.min_rule_indeg2_count,
                        debug_limit_expansions=args.debug_limit_expansions,
                        debug_log_every=args.debug_log_every,
                        time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                        prune_low_support_labels=bool(args.prune_low_support_labels),
                        prune_by_rule=bool(args.prune_by_rule),
                        attach_producer=bool(args.attach_producer),
                        max_producer_depth=int(args.max_producer_depth),
                        skip_unknown=bool(args.skip_unknown),
                        enable_var_closure_check=bool(args.enable_var_closure_check),
                        emit=emit_one,
                    )
                else:
                    miner.expand_rules_only_from_seed(
                        s,
                        debug_limit_expansions=args.debug_limit_expansions,
                        debug_log_every=args.debug_log_every,
                        time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                        prune_low_support_labels=bool(args.prune_low_support_labels),
                        prune_by_rule=bool(args.prune_by_rule),
                        enable_var_closure_check=bool(args.enable_var_closure_check),
                        emit=emit_one,
                    )
            t2 = time.time()
            print(f"[seeds] expand+process done, processed_total={processed} time={t2-t1:.3f}s", flush=True)

            res = pipeline.finalize_results(
                problems=len(pg.fact_id_map),
                fact_cnt=sum(1 for nd in pg.nodes.values() if nd.get("type") == "fact"),
                rule_cnt=sum(1 for nd in pg.nodes.values() if nd.get("type") == "rule"),
                merged_nodes=len(miner.G.nodes),
                merged_edges=sum(len(vs) for vs in miner.G.out_edges.values()),
                t0=t0, t1=t1, t2=t2)

            if audit_seeds: audit_seeds.close()
            if audit_exp: audit_exp.close()
            return 0
        else:
            # seeds_mproc：多进程按种子扩展，主进程处理
            ctx = mp.get_context("fork")
            jobs = ctx.JoinableQueue()
            qout = ctx.Queue(maxsize=10000)
            # 全局扩展预算：限制所有 worker 的扩展总步数
            global_budget = None
            if args.debug_limit_expansions is not None:
                try:
                    limit = int(args.debug_limit_expansions)
                    if limit > 0:
                        global_budget = ctx.Semaphore(limit)
                except Exception:
                    global_budget = None

            mp_mod = _load_mining_pipeline_module()
            MiningPipeline = getattr(mp_mod, "MiningPipeline")
            pipeline = MiningPipeline(pg, miner, args, out_dir)

            def emit_fn(pobj: dict):
                # 非阻塞投递，队列满则直接丢弃，避免每次等待 1s 导致整体变慢
                try:
                    qout.put_nowait(pobj)
                except Exception:
                    # 可选：统计丢弃数量（略）
                    pass

            def worker_proc():
                # 阻塞式获取任务，依赖哨兵 None 退出，确保每个 put 都有对应 task_done
                while True:
                    seed = jobs.get()
                    if seed is None:
                        # 为哨兵调用 task_done 然后退出
                        jobs.task_done()
                        break
                    try:
                        if args.mode == "fact_rule":
                            miner.expand_branched_from_seed(
                                seed,
                                min_rule_indeg2_count=args.min_rule_indeg2_count,
                                debug_limit_expansions=args.debug_limit_expansions,
                                debug_log_every=args.debug_log_every,
                                time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                                prune_low_support_labels=bool(args.prune_low_support_labels),
                                prune_by_rule=bool(args.prune_by_rule),
                                attach_producer=bool(args.attach_producer),
                                max_producer_depth=int(args.max_producer_depth),
                                skip_unknown=bool(args.skip_unknown),
                                enable_var_closure_check=bool(args.enable_var_closure_check),
                                global_budget=global_budget,
                                emit=emit_fn,
                            )
                        else:
                            miner.expand_rules_only_from_seed(
                                seed,
                                debug_limit_expansions=args.debug_limit_expansions,
                                debug_log_every=args.debug_log_every,
                                time_budget_seconds=(None if args.time_budget_seconds is None else float(args.time_budget_seconds)),
                                prune_low_support_labels=bool(args.prune_low_support_labels),
                                prune_by_rule=bool(args.prune_by_rule),
                                enable_var_closure_check=bool(args.enable_var_closure_check),
                                global_budget=global_budget,
                                emit=emit_fn,
                            )
                    finally:
                        # 与 put 对应，防止 join 卡住
                        jobs.task_done()
                # 向主进程发送完成信号；必须确保送达，避免主进程等待未收到的完成信号
                while True:
                    try:
                        qout.put(None, timeout=0.5)
                        break
                    except Exception:
                        # 队列满则重试，直到放入成功
                        continue

            for s in seeds:
                jobs.put(s)
            n_workers = max(1, int(args.workers))
            procs = [ctx.Process(target=worker_proc, daemon=True) for _ in range(n_workers)]
            # 先投递哨兵，保证 worker 在处理完所有种子后能够顺利退出
            for _ in range(n_workers):
                jobs.put(None)
            for p_ in procs:
                p_.start()
            print(f"[mproc] started workers={n_workers}; queued seeds={len(seeds)}", flush=True)

            finished = 0
            processed = 0
            t_emit0 = time.time()
            while finished < n_workers:
                obj = qout.get()
                if obj is None:
                    finished += 1
                    print(f"[mproc] worker finished signals={finished}/{n_workers}", flush=True)
                    continue
                pipeline.process_one_pattern(obj)
                processed += 1
                if processed % dbg_every == 0:
                    dt = time.time() - t_emit0
                    print(f"[mproc] processed={processed} elapsed={dt:.2f}s", flush=True)
                if audit_exp:
                    try:
                        audit_exp.write(json.dumps({
                            "support": obj.get("support"),
                            "pids": list(obj.get("pids", [])),
                            "labels": obj.get("labels", []),
                        }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

            try:
                print("[mproc] waiting for jobs.join()", flush=True)
                jobs.join()
                print("[mproc] jobs.join() done", flush=True)
            except Exception:
                pass
            for p_ in procs:
                print(f"[mproc] joining worker pid={p_.pid}", flush=True)
                p_.join()
            print(f"[mproc] all workers joined; processed_total={processed}", flush=True)

            t2 = time.time()
            print(f"[mproc] expand+process done in {t2-t1:.3f}s", flush=True)
            res = pipeline.finalize_results(
                problems=len(pg.fact_id_map),
                fact_cnt=sum(1 for nd in pg.nodes.values() if nd.get("type") == "fact"),
                rule_cnt=sum(1 for nd in pg.nodes.values() if nd.get("type") == "rule"),
                merged_nodes=len(miner.G.nodes),
                merged_edges=sum(len(vs) for vs in miner.G.out_edges.values()),
                t0=t0, t1=t1, t2=t2)

            if audit_seeds: audit_seeds.close()
            if audit_exp: audit_exp.close()
            return 0

    # 打印 Top-K（仅在非并行挖掘路径下，先简单转换，不过滤），便于快速查看
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

    # 调用封装好的 MiningPipeline 完成后处理与写盘
    mp_mod = _load_mining_pipeline_module()
    MiningPipeline = getattr(mp_mod, "MiningPipeline")
    # 将有效 max_nodes 传入 pipeline 以便写入参数摘要
    args.max_nodes = effective_max_nodes
    pipeline = MiningPipeline(pg, miner, args, out_dir)
    pipeline.run(problems, fact_cnt, rule_cnt, merged_nodes, merged_edges, t0, t1, patterns)


if __name__ == "__main__":
    sys.exit(main())

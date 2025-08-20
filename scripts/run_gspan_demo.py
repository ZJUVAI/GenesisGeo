#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 gSpan（路径挖掘变体）一键运行脚本。
- 目的：开箱即用，直接 `python run_gspan_demo.py` 即可按脚本内推荐超参运行；无需再手写一堆命令行参数。
- 特点：将超参集中在 CONFIG 中；支持按需用命令行覆盖（可选）。
- 注意：为了避免导入 newclid 包时触发重量级依赖，这里用 importlib 直接按文件路径加载 proof_graph.py。
"""
import os
import sys
import argparse
import logging
import importlib.util

# 路径定位
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
PROOF_GRAPH_PY = os.path.join(SRC_DIR, "newclid", "data_discovery", "proof_graph.py")
DEFAULT_JSON = os.path.join(SRC_DIR, "newclid", "data_discovery", "r07_expanded_problems_results_lil.json")

# 集中管理的默认超参（不会卡住的一组）
CONFIG = {
    # 数据源：默认用较小的 lil 数据集，速度更稳
    "json": DEFAULT_JSON,
    # 支持度：绝对值或比例（字符串形式，内部会解析）
    "min_support": "2",  # "2" 或 "0.2"
    # 结构阈值（路径挖掘）
    "min_rule_nodes": 2,
    "min_edges": 3,
    # 搜索上限（越小越稳）
    "max_nodes": 9,
    # 每个模式展示的代表性嵌入数量
    "sample_embeddings": 1,
    # 打印前 K 个模式
    "top_k": 10,
    # 是否安静模式（减少 ProofGraph 解析日志）
    "quiet": True,
}


def _load_proof_graph_module():
    """按文件路径加载 proof_graph.py，避免触发包级别 __init__ 的重依赖。"""
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
    # 允许用命令行覆盖 CONFIG（可选），不传参也能运行
    parser = argparse.ArgumentParser(description="Run simplified gSpan path mining (one-click)")
    parser.add_argument("--json", default=CONFIG["json"])
    parser.add_argument("--min-support", default=CONFIG["min_support"], type=str)
    parser.add_argument("--min-rule-nodes", default=CONFIG["min_rule_nodes"], type=int)
    parser.add_argument("--min-edges", default=CONFIG["min_edges"], type=int)
    parser.add_argument("--max-nodes", default=CONFIG["max_nodes"], type=int)
    parser.add_argument("--sample-embeddings", default=CONFIG["sample_embeddings"], type=int)
    parser.add_argument("--top-k", default=CONFIG["top_k"], type=int)
    parser.add_argument("--quiet", action="store_true" if not CONFIG["quiet"] else "store_false")
    args = parser.parse_args(argv)

    mod = _load_proof_graph_module()
    ProofGraph = getattr(mod, "ProofGraph")
    GSpanMiner = getattr(mod, "GSpanMiner")

    min_support = parse_min_support(args.min_support)

    pg = ProofGraph(verbose=not CONFIG["quiet"] if not hasattr(args, "quiet") else args.quiet is False,
                    log_level=logging.INFO)
    pg.from_results_json(args.json)

    miner = GSpanMiner(
        pg,
        min_support=min_support,
        min_rule_nodes=args.min_rule_nodes,
        min_edges=args.min_edges,
        max_nodes=args.max_nodes,
        sample_embeddings=args.sample_embeddings,
    )
    patterns = miner.run()
    print(f"total patterns: {len(patterns)}")

    K = max(0, int(args.top_k))
    for i, p in enumerate(patterns[:K]):
        expr, vmap = miner.pattern_to_schema(p)
        labels = "->".join(p.get("labels", []))
        print(f"[{i}] support={p.get('support')} pids={p.get('pids')} labels={labels}")
        print(f"     schema: {expr}")
        if vmap:
            print(f"     var_map: {vmap}")


if __name__ == "__main__":
    sys.exit(main())

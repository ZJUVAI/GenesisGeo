#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量求解器一键脚本（从仓库根目录运行或在脚本所在目录运行均可）。
- 目的：python run_batch.py 即按脚本内集中超参运行，不必再传命令行参数。
- 直接调用 solver_utils.solve_problems_batch 并写出 outputs/<basename>.results.json。
"""
from __future__ import annotations
import os
import sys
import shutil
from typing import Optional

# 计算仓库相关路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))                 # .../Newclid
SRC_DIR = os.path.normpath(os.path.join(REPO_DIR, "src"))                   # .../Newclid/src
DATA_DISCOVERY_DIR = os.path.join(SRC_DIR, "newclid", "data_discovery")
PROBLEMS_PATH = os.path.join(REPO_DIR, "problems_datasets", "r07_problems.txt")
OUTPUTS_DIR = os.path.join(REPO_DIR, "outputs")
 
# 确保可以 import newclid.*
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 直接使用底层工具函数
from newclid.data_discovery.solver_utils import solve_problems_batch  # type: ignore


# 集中超参（每项均附注释说明用途）
CONFIG = {
    "problems_file": PROBLEMS_PATH,  # 待求解问题文件（两行一题的 jgex-231/AG 集合）

    "max_attempts": 100,               # 构建状态的最大尝试次数
    "timeout": 60,                     # 单题求解超时时间（秒）
    "limit": None,                     # 仅求解前 N 题；None 表示不限制
    "workers": 30,                      # 并行工作数；1 为串行，大于 1 则启用并行
    "backend": "process",            # 并行后端："process"（推荐）或 "thread"
}


def _write_results_json(staged_problems: str, stats: dict) -> str:
    base = os.path.splitext(os.path.basename(staged_problems))[0]
    out_json = os.path.join(OUTPUTS_DIR, f"{base}.results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return out_json


def _print_summary(stats: dict) -> None:
    total = int(stats.get("total", 0) or 0)
    solved = int(stats.get("solved", 0) or 0)
    rate = float(stats.get("solve_rate", 0.0) or 0.0)
    results = stats.get("results", []) or []
    fails = []
    for r in results:
        try:
            if not r.get("success"):
                fails.append((r.get("problem_id"), r.get("error")))
        except Exception:
            continue
    print("\n=== Solve Summary ===")
    print(f"Solved: {solved}/{total} ({rate:.2%})")
    print(f"Failed: {len(fails)}")
    if fails:
        print("Examples of failures (up to 10):")
        for pid, err in fails[:10]:
            if err:
                print(f"- {pid}: {err}")
            else:
                print(f"- {pid}")


def main(_: Optional[list[str]] = None) -> None:
    # 准备输出目录与题目文件副本（确保结果写入 outputs 下）
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    src_problems = os.path.abspath(CONFIG.get("problems_file") or PROBLEMS_PATH)
    staged_problems = os.path.join(OUTPUTS_DIR, os.path.basename(src_problems))
    try:
        shutil.copyfile(src_problems, staged_problems)
    except Exception as e:
        raise RuntimeError(f"无法复制题目文件到 outputs: {e}")

    # 审计打印（便于确认本次运行的关键超参）
    print("[run-batch] problems_file(src)=", src_problems)
    print("[run-batch] problems_file(run)=", staged_problems)
    print("[run-batch] outputs_dir         =", OUTPUTS_DIR)
    print("[run-batch] workers/backend/max_attempts/timeout/limit =",
          CONFIG["workers"], CONFIG["backend"], CONFIG["max_attempts"], CONFIG["timeout"], CONFIG["limit"]) 

    # 直接调用批量求解并落盘
    stats = solve_problems_batch(
        problems_file=staged_problems,
        rules_file="",
        max_attempts=int(CONFIG["max_attempts"]),
        timeout_sec=int(CONFIG["timeout"]),
    limit=CONFIG["limit"],
    workers=int(CONFIG["workers"]) if CONFIG.get("workers") else 1,
    backend=str(CONFIG["backend"]) if CONFIG.get("backend") else "process",
    )
    out_json = _write_results_json(staged_problems, stats)
    print("[run-batch] wrote results:", out_json)
    _print_summary(stats)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量求解器一键脚本（从仓库根目录运行或在脚本所在目录运行均可）。
- 目的：python run_batch.py 即按脚本内集中超参运行，不必再传命令行参数。
- 保持零侵入：不修改现有 src/newclid/data_discovery/run_batch.py 与其他模块。
"""
from __future__ import annotations
import os
import sys
import importlib.util
import shutil
from typing import Optional

# 计算仓库相关路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))                 # .../Newclid
SRC_DIR = os.path.normpath(os.path.join(REPO_DIR, "src"))                   # .../Newclid/src
DATA_DISCOVERY_DIR = os.path.join(SRC_DIR, "newclid", "data_discovery")
RUN_BATCH_PY = os.path.join(DATA_DISCOVERY_DIR, "run_batch.py")
PROBLEMS_PATH = os.path.join(REPO_DIR, "problems_datasets", "jgex_ag_231.txt")
OUTPUTS_DIR = os.path.join(REPO_DIR, "outputs")
 
# 确保可以 import newclid.*
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# 集中超参（每项均附注释说明用途）
CONFIG = {
    "problems_file": PROBLEMS_PATH,  # 待求解问题文件（两行一题的 jgex-231/AG 集合）

    "max_attempts": 100,               # 构建状态的最大尝试次数
    "timeout": 60,                     # 单题求解超时时间（秒）
    "limit": None,                     # 仅求解前 N 题；None 表示不限制

    "workers": 10,                      # 并行工作数；1 为串行，大于 1 则启用并行
    "backend": "process",            # 并行后端："process"（推荐）或 "thread"
}


def _ensure_paths_for_children() -> None:
    """确保父/子进程均可按模块名导入底层脚本与包。

    - 向 sys.path 与 PYTHONPATH 注入 SRC_DIR 与 DATA_DISCOVERY_DIR，便于子进程 spawn 时可 import。
    """
    # 父进程 sys.path 注入
    for p in (SRC_DIR, DATA_DISCOVERY_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    # 环境变量 PYTHONPATH 注入（供子进程继承）
    extra = os.pathsep.join([SRC_DIR, DATA_DISCOVERY_DIR])
    old = os.environ.get("PYTHONPATH", "")
    if old:
        if SRC_DIR not in old or DATA_DISCOVERY_DIR not in old:
            os.environ["PYTHONPATH"] = old + os.pathsep + extra
    else:
        os.environ["PYTHONPATH"] = extra


def _load_bottom_main():
    """加载底层批处理脚本的 main 函数（以模块名 'run_batch' 注册）。"""
    if not os.path.exists(RUN_BATCH_PY):
        raise RuntimeError(f"底层脚本不存在: {RUN_BATCH_PY}")
    # 确保路径对父/子进程可见
    _ensure_paths_for_children()
    # 使用与文件名一致的模块名，便于子进程可 import
    spec = importlib.util.spec_from_file_location("run_batch", RUN_BATCH_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 spec: {RUN_BATCH_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_batch"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "main")


def _build_argv(cfg: dict, problems_path: str) -> list[str]:
    """将集中超参转换为底层脚本 main(argv) 的参数列表。"""
    args: list[str] = []

    # 位置参数：problems_file（转绝对路径）
    problems_file_abs = os.path.abspath(problems_path)
    args.append(problems_file_abs)

    # 可选参数
    if cfg.get("max_attempts") is not None:
        args += ["--max-attempts", str(int(cfg["max_attempts"]))]
    if cfg.get("timeout") is not None:
        args += ["--timeout", str(int(cfg["timeout"]))]
    if cfg.get("limit") is not None:
        args += ["--limit", str(int(cfg["limit"]))]
    if cfg.get("workers") is not None:
        args += ["--workers", str(int(cfg["workers"]))]
    if cfg.get("backend"):
        args += ["--backend", str(cfg["backend"])]
    return args


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

    # 加载并调用底层 main
    bottom_main = _load_bottom_main()
    argv = _build_argv(CONFIG, staged_problems)
    # 底层 main() 不接收参数，使用 argparse 从 sys.argv 解析；这里临时覆写 sys.argv
    prev_argv = sys.argv[:]
    try:
        sys.argv = [RUN_BATCH_PY, *argv]
        bottom_main()
    finally:
        sys.argv = prev_argv


if __name__ == "__main__":
    main()

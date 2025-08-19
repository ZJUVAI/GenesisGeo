"""
求解器封装和批量处理
提供几何问题求解的工具函数
"""

import os
import sys
from typing import List, Dict, Optional
import time

# 添加项目根目录到 Python 路径以导入 newclid
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from newclid import GeometricSolverBuilder, GeometricSolver
from newclid import proof_writing


def solve_single_problem(
    problem_text: str,
    rules_file: str,
    max_attempts: int = 100,
    timeout_sec: int = 60,
) -> dict:
    """
    求解单个几何问题
    功能：使用指定规则文件求解单个几何问题
    参数：
        problem_text (str) - 问题文本描述
        rules_file (str) - 规则文件路径
        max_attempts (int) - 最大尝试次数，默认100
    返回值：dict - {"success": bool, "proof": list, "run_info": dict, "error": str}
    被调用：solve_problems_batch() 函数调用
    """
    try:
        # 创建求解器构建器
        solver_builder = GeometricSolverBuilder(123)
        
        # 设置自定义规则文件
        if os.path.exists(rules_file):
            # TODO: 需要确认如何设置自定义规则文件的方法
            pass
        
        # 加载问题并构建求解器
        solver_builder.load_problem_from_txt(problem_text)
        solver: GeometricSolver = solver_builder.build(max_attempts=max_attempts)

        # 运行求解，增加超时控制，防止单题长时间卡住
        success = solver.run(timeout=timeout_sec)

        return {
            "success": success,
            "proof": solver.proof if success else None,
            "run_info": solver.run_infos,
            "error": None,
        }
        
    except Exception as e:
        return {
            "success": False,
            "proof": None,
            "run_info": None,
            "error": str(e)
        }


def solve_problems_batch(
    problems_file: str,
    rules_file: str,
    max_attempts: int = 100,
    timeout_sec: int = 60,
    limit: Optional[int] = None,
) -> dict:
    """
    批量求解问题列表（集成了加载问题和统计功能）
    功能：从文件加载dev_jgex问题集并批量求解，同时计算统计数据
    参数：
        problems_file (str) - 问题文件路径（dev_jgex.txt格式）
        rules_file (str) - 规则文件路径
        max_attempts (int) - 最大尝试次数，默认100
    返回值：dict - {"total": int, "solved": int, "solve_rate": float, "results": list}
    被调用：iterative_rules_pipeline.py 中的 evaluate_current_performance() 调用
    """
    # 加载问题
    problems = _load_dev_jgex_problems(problems_file)
    if limit is not None:
        problems = problems[: max(limit, 0)]
    
    results = []
    solved_count = 0
    
    total = len(problems)
    print(f"开始求解 {total} 个问题... (max_attempts={max_attempts}, timeout={timeout_sec}s)")

    for idx, problem in enumerate(problems, start=1):
        t0 = time.time()
        print(f"[进度] {idx}/{total} 开始: {problem['problem_id']}")
        result = solve_single_problem(
            problem['problem_text'],
            rules_file,
            max_attempts=max_attempts,
            timeout_sec=timeout_sec,
        )
        
        result['problem_id'] = problem['problem_id']
        results.append(result)
        
        dur = time.time() - t0
        if result['success']:
            solved_count += 1
            print(f"[完成] {idx}/{total} 成功: {problem['problem_id']} 用时 {dur:.1f}s")
        else:
            err = result.get('error')
            err_msg = f" 失败: {err}" if err else " 失败"
            print(f"[完成] {idx}/{total}{err_msg}: {problem['problem_id']} 用时 {dur:.1f}s")
    
    # 计算统计数据
    solve_rate = solved_count / total if total > 0 else 0.0
    
    return {
        "total": total,
        "solved": solved_count,
        "solve_rate": solve_rate,
        "results": results
    }


def _load_dev_jgex_problems(file_path: str) -> List[Dict]:
    """
    加载dev_jgex问题集（内部函数）
    功能：解析dev_jgex.txt文件格式，提取问题ID和问题文本
    参数：file_path (str) - dev_jgex.txt文件路径
    返回值：List[Dict] - [{"problem_id": str, "problem_text": str}, ...]
    被调用：solve_problems_batch() 函数内部调用
    """
    problems = []
    
    if not os.path.exists(file_path):
        print(f"警告: 问题文件 {file_path} 不存在")
        return problems
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # dev_jgex.txt 格式: 偶数行是文件路径，奇数行是问题描述
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            problem_id = lines[i].strip()
            problem_text = lines[i + 1].strip()
            
            problems.append({
                "problem_id": problem_id,
                "problem_text": problem_text
            })
    
    return problems

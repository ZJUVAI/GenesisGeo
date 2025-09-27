"""
求解器封装和批量处理
提供几何问题求解的工具函数
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
import concurrent.futures as cf
import time

# 添加项目根目录到 Python 路径以导入 newclid
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from newclid import GeometricSolverBuilder, GeometricSolver
from newclid import proof_writing


def _extract_point_coords_from_solver(solver: GeometricSolver):
    """从求解器当前 proof 中提取所有点的坐标。

    返回:
        (point_lines, points_json)
        - point_lines: list[str]，如 "point a -0.1 0.2"，按名称排序
        - points_json: list[dict]，{"name": str, "x": float, "y": float}
    任一阶段失败则返回空列表。
    """
    try:
        # 延迟导入，避免在环境未就绪时硬依赖失败
        from newclid.dependencies.symbols import Point  # type: ignore
    except Exception:
        return [], []

    try:
        proof = getattr(solver, 'proof', None)
        if proof is None:
            return [], []
        dep_graph = getattr(proof, 'dep_graph', None)
        if dep_graph is None:
            return [], []
        symbols_graph = getattr(dep_graph, 'symbols_graph', None)
        if symbols_graph is None or not hasattr(symbols_graph, 'nodes_of_type'):
            return [], []
        points = symbols_graph.nodes_of_type(Point)
    except Exception:
        return [], []

    items = []
    for p in points:
        try:
            name = getattr(p, 'name', None)
            num = getattr(p, 'num', None)
            x = getattr(num, 'x', None) if num is not None else None
            y = getattr(num, 'y', None) if num is not None else None
            if name is None or x is None or y is None:
                continue
            # 强制转 float，避免后续 JSON 序列化问题
            items.append((str(name), float(x), float(y)))
        except Exception:
            continue

    items.sort(key=lambda t: t[0])
    point_lines = [f"point {name} {x} {y}" for name, x, y in items]
    points_json = [{"name": name, "x": x, "y": y} for name, x, y in items]
    return point_lines, points_json


def _extract_aux_from_solver(solver: GeometricSolver):
    """从求解器的证明状态中提取辅助构造信息。

    返回:
        (aux_points, aux_lines)
        - aux_points: list[str]，辅助构造中涉及的点名（按名称排序，去重）
        - aux_lines: list[str]，辅助构造的谓词行（含数值检查项），形如 "pred a b c [ID]"
    任一阶段失败则返回空列表。
    """
    try:
        proof = getattr(solver, 'proof', None)
        if proof is None:
            return [], []
        # 利用 proof_writing 的内部约定获取结构化分段
        # get_structured_proof 需要一个 id 映射；这里传入空 dict 由函数内部填充
        try:
            analysis, numerical_check, _ = proof_writing.get_structured_proof(proof, {})
        except Exception:
            analysis, numerical_check = "", ""

        # 直接从 dep_graph 获取细粒度对象再序列化，确保鲁棒
        dep_graph = getattr(proof, 'dep_graph', None)
        if dep_graph is None or not hasattr(dep_graph, 'get_proof_steps'):
            return [], []
        goals = [g for g in getattr(proof, 'goals', []) if getattr(g, 'check', lambda: False)()]
        (
            _points,
            _premises,
            _num_premises,
            aux_points_raw,
            aux_lines_raw,
            aux_num_lines_raw,
            _proof_steps,
        ) = dep_graph.get_proof_steps(goals)

        # 规范化输出
        aux_points = []
        try:
            # 使用符号的 canonical 名称（Point.name）以与 points[].name 保持大小写一致
            aux_points = sorted([str(getattr(p, 'name', getattr(p, 'pretty_name', p))) for p in aux_points_raw])
        except Exception:
            aux_points = []

        def _to_pred_str(dep_line) -> str:
            try:
                # 复用 get_structured_proof 的 pure_predicate 逻辑近似：statement [id]
                # 但这里没有 id，采用 to_str() 并省略 id，或尝试 [id] 若可用
                st = getattr(dep_line, 'statement', None)
                if st is None:
                    return ""
                # statement.to_str() 已是如 "pred args" 的格式
                s = st.to_str()
                # 如果构建时已有 id，可从临时映射中拿，但这里简化为无 id 版本
                return s
            except Exception:
                return ""

        aux_lines_txt = []
        try:
            aux_lines_txt = [_to_pred_str(x) for x in (aux_lines_raw or [])]
            aux_num_txt = [_to_pred_str(x) for x in (aux_num_lines_raw or [])]
            # 合并辅助构造与其数值校验
            aux_lines_txt = [s for s in (aux_lines_txt + aux_num_txt) if s]
        except Exception:
            aux_lines_txt = []


        return aux_points, aux_lines_txt
    except Exception:
        return [], []


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

        # 在求解前提取坐标，确保即使 run 失败也能获取
        point_lines, points_json = ([], [])
        try:
            point_lines, points_json = _extract_point_coords_from_solver(solver)
        except Exception:
            point_lines, points_json = ([], [])

        # 运行求解，增加超时控制，防止单题长时间卡住
        success = solver.run(timeout=timeout_sec)
        # 将 proof 转为可序列化的结构化文本，避免 JSON 序列化失败
        proof_obj = None
        if success and getattr(solver, 'proof', None) is not None:
            try:
                analysis, numerical_check, proof_text = proof_writing.get_structured_proof(solver.proof, {})
                proof_obj = {
                    "analysis": analysis,
                    "numerical_check": numerical_check,
                    "proof": proof_text,
                }
            except Exception:
                proof_obj = None

        res = {
            "success": success,
            "proof": proof_obj,
            "run_info": solver.run_infos,
            "error": None,
        }
        res["point_lines"] = point_lines
        res["points"] = points_json
        # 注入辅助构造信息（失败安全）
        try:
            aux_points, aux_lines = _extract_aux_from_solver(solver)
        except Exception:
            aux_points, aux_lines = ([], [])
        res["aux_points"] = aux_points
        res["aux_lines"] = aux_lines
        return res
        
    except Exception as e:
        res = {
            "success": False,
            "proof": None,
            "run_info": None,
            "error": str(e)
        }
        # 异常情况下不强制注入坐标字段
        return res


def _solve_problem_entry(args: Tuple[Dict, str, int, int]) -> Dict:
    """进程/线程池入口函数：独立求解一个问题（可被pickle）。"""
    problem, rules_file, max_attempts, timeout_sec = args
    result = solve_single_problem(
        problem['problem_text'],
        rules_file,
        max_attempts=max_attempts,
        timeout_sec=timeout_sec,
    )
    result['problem_id'] = problem['problem_id']
    return result


def solve_problems_batch(
    problems_file: str,
    rules_file: str,
    max_attempts: int = 100,
    timeout_sec: int = 60,
    limit: Optional[int] = None,
    workers: int = 1,
    backend: str = "process",
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
    
    results: List[Dict] = []
    solved_count = 0
    total = len(problems)
    print(f"开始求解 {total} 个问题... (max_attempts={max_attempts}, timeout={timeout_sec}s, workers={workers}, backend={backend})")

    # 串行模式
    if workers is None or workers <= 1:
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
    else:
        # 并行模式
        Executor = cf.ProcessPoolExecutor if backend == "process" else cf.ThreadPoolExecutor
        tasks = [(p, rules_file, max_attempts, timeout_sec) for p in problems]
        print(f"[并行] 使用 {backend} 池，workers={workers}")
        t_all = time.time()
        with Executor(max_workers=int(workers)) as ex:
            # 维持原顺序输出，可用 map；同时统计进度
            for idx, result in enumerate(ex.map(_solve_problem_entry, tasks), start=1):
                results.append(result)
                if result.get('success'):
                    solved_count += 1
                if idx % 10 == 1 or idx == total:
                    print(f"[并行进度] {idx}/{total} 已完成")
        print(f"[并行] 总用时 {time.time() - t_all:.1f}s")
    
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

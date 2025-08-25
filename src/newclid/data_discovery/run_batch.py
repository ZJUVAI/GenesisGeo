#!/usr/bin/env python3
"""
在 data_discovery 目录下批量调用 solve_problems_batch。
使用方法仅需提供一个问题文件路径（dev_jgex.txt 格式或两行一题的编号+题目格式）。
运行后会打印汇总，并把完整结果保存为与输入同名的 *_results.json。
"""

import os
import sys
import json
import importlib.util
import argparse
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# 优先常规导入，同目录导入失败则用路径动态加载
try:
    from solver_utils import solve_problems_batch  # type: ignore
    from solver_utils import solve_single_problem  # type: ignore
except Exception:
    _here = os.path.dirname(__file__)
    _mod_path = os.path.join(_here, 'solver_utils.py')
    _spec = importlib.util.spec_from_file_location('solver_utils', _mod_path)
    if _spec is None or _spec.loader is None:
        raise ImportError('无法加载 solver_utils')
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    solve_problems_batch = _mod.solve_problems_batch  # type: ignore[attr-defined]
    solve_single_problem = _mod.solve_single_problem  # type: ignore[attr-defined]

# 为了将 ProofState 等对象序列化为 JSON，优先正常导入 newclid.proof_writing
try:
    from newclid.proof_writing import get_structured_proof  # type: ignore
except Exception:
    # 退回到基于路径的动态加载
    _here = os.path.dirname(__file__)
    _pw_path = os.path.abspath(os.path.join(_here, os.pardir, 'proof_writing.py'))
    _spec_pw = importlib.util.spec_from_file_location('proof_writing', _pw_path)
    if _spec_pw is None or _spec_pw.loader is None:
        get_structured_proof = None  # type: ignore
    else:
        _pw_mod = importlib.util.module_from_spec(_spec_pw)
        _spec_pw.loader.exec_module(_pw_mod)  # type: ignore[attr-defined]
        get_structured_proof = _pw_mod.get_structured_proof  # type: ignore[attr-defined]


def _extract_point_rely_on_transitive(proof_state):
    """提取所有点的传递依赖(基于 Point.rely_on)。

    返回: dict[str, list[str]]，键为点名，值为传递依赖点名的去重有序列表(按字母排序稳定输出)。
    若解析失败，返回空字典。
    """

    # 延迟、健壮导入 Point
    try:
        from newclid.dependencies.symbols import Point  # type: ignore
    except Exception:
        # 动态路径加载兜底
        try:
            _here = os.path.dirname(__file__)
            _sym_path = os.path.abspath(os.path.join(_here, os.pardir, 'dependencies', 'symbols.py'))
            _spec_sym = importlib.util.spec_from_file_location('symbols', _sym_path)
            if _spec_sym is None or _spec_sym.loader is None:
                return {}
            _sym_mod = importlib.util.module_from_spec(_spec_sym)
            _spec_sym.loader.exec_module(_sym_mod)  # type: ignore[attr-defined]
            Point = getattr(_sym_mod, 'Point', None)  # type: ignore
            if Point is None:
                return {}
        except Exception:
            return {}

    try:
        dg = getattr(proof_state, 'dep_graph', None)
        if dg is None or not hasattr(dg, 'symbols_graph'):
            return {}
        sg = dg.symbols_graph
        if not hasattr(sg, 'nodes_of_type'):
            return {}
        points = sg.nodes_of_type(Point)
    except Exception:
        return {}

    # 构建依赖映射: name -> direct parents(list[Point])
    name2point = {}
    direct_deps = {}
    for p in points:
        try:
            name2point[p.name] = p
            parents = []
            for q in getattr(p, 'rely_on', []) or []:
                try:
                    parents.append(q)
                except Exception:
                    continue
            direct_deps[p.name] = parents
        except Exception:
            # 某个点异常时跳过
            direct_deps[p.name if hasattr(p, 'name') else str(p)] = []

    # 计算传递闭包: DFS 防环
    def ancestors(name: str, visiting: set[str] | None = None) -> set[str]:
        if visiting is None:
            visiting = set()
        res: set[str] = set()
        visiting.add(name)
        for parent in direct_deps.get(name, []):
            try:
                pname = parent.name
            except Exception:
                pname = str(parent)
            if pname in visiting:
                continue  # 避免环
            res.add(pname)
            res |= ancestors(pname, visiting.copy())
        return res

    result: dict[str, list[str]] = {}
    for name in name2point.keys():
        try:
            aset = ancestors(name)
            # 稳定输出，按字母排序
            result[name] = sorted(aset)
        except Exception:
            result[name] = []
    return result


def _to_jsonable(obj):
    """将任意对象转换为可 JSON 序列化的结构。

    - 对 ProofState：使用 get_structured_proof 生成三段文本；并在其上层 dict 注入 point_rely_on。
    - 对映射、序列、集合：递归转换。
    - 其它对象：转为字符串表示。
    """

    # 基础可序列化类型
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # 尝试识别 ProofState（鸭子类型判定以避免强依赖）
    try:
        is_proof_state_like = hasattr(obj, 'goals') and hasattr(obj, 'dep_graph')
    except Exception:
        is_proof_state_like = False

    if is_proof_state_like and callable(get_structured_proof):
        try:
            analysis, numerical_check, proof = get_structured_proof(obj, {})  # type: ignore[arg-type]
            return {
                "analysis": analysis,
                "numerical_check": numerical_check,
                "proof": proof,
            }
        except Exception:
            # 回退到字符串表示，避免整个保存失败
            return str(obj)

    # 映射类型：保证 key 为字符串；并在含 ProofState 的 dict 注入 point_rely_on
    if isinstance(obj, dict):
        # 在转换前尝试找到一个 ProofState-like 的值，用于提取依赖
        proof_state_obj = None
        try:
            for vv in obj.values():
                if hasattr(vv, 'goals') and hasattr(vv, 'dep_graph'):
                    proof_state_obj = vv
                    break
        except Exception:
            proof_state_obj = None

        # 若发现 ProofState，先提取传递依赖，填充到当前 dict 的副本中
        injected_point_rely_on = None
        if proof_state_obj is not None:
            try:
                injected_point_rely_on = _extract_point_rely_on_transitive(proof_state_obj)
            except Exception:
                injected_point_rely_on = None

        new_dict = {}
        for k, v in obj.items():
            try:
                key_str = k if isinstance(k, str) else str(k)
            except Exception:
                key_str = repr(k)
            new_dict[key_str] = _to_jsonable(v)
        if injected_point_rely_on is not None:
            # 压缩为 name -> "a,b,c" 的字典，减少 JSON 换行
            try:
                compact_map = {k: ",".join(v) if isinstance(v, list) else str(v)
                               for k, v in injected_point_rely_on.items()}
            except Exception:
                compact_map = injected_point_rely_on
            new_dict['point_rely_on'] = _to_jsonable(compact_map)
        return new_dict

    # 序列/集合
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # 其它复杂对象的常见可读接口
    for attr in ("to_str", "pretty", "__str__"):
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                text = val() if callable(val) else str(obj)
                if isinstance(text, str):
                    return text
            except Exception:
                pass

    # 最后兜底
    return str(obj)


def _load_dev_jgex_problems(file_path: str):
    """本地加载 dev_jgex.txt（两行一题：编号行 + 题目行）。"""
    problems = []
    if not os.path.exists(file_path):
        print(f"警告: 问题文件 {file_path} 不存在")
        return problems
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            problem_id = lines[i].strip()
            problem_text = lines[i + 1].strip()
            problems.append({"problem_id": problem_id, "problem_text": problem_text})
    return problems


def _solve_and_pack(args):
    """在子进程中执行单题求解并转为可序列化结果，避免跨进程序列化失败。

    args: (problem_id, problem_text, rules_file, max_attempts, timeout_sec)
    """
    problem_id, problem_text, rules_file, max_attempts, timeout_sec = args
    try:
        # 延迟导入，减少主进程状态对子进程的影响
        try:
            from solver_utils import solve_single_problem  # type: ignore
        except Exception:
            _here = os.path.dirname(__file__)
            _mod_path = os.path.join(_here, 'solver_utils.py')
            _spec = importlib.util.spec_from_file_location('solver_utils', _mod_path)
            if _spec is None or _spec.loader is None:
                raise ImportError('无法加载 solver_utils')
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
            solve_single_problem = _mod.solve_single_problem  # type: ignore[attr-defined]

        res = solve_single_problem(
            problem_text,
            rules_file,
            max_attempts=max_attempts,
            timeout_sec=timeout_sec,
        )
        res['problem_id'] = problem_id
    except Exception as e:
        res = {"success": False, "proof": None, "run_info": None, "error": str(e), "problem_id": problem_id}

    # 转为可 JSON 序列化，避免返回主进程时的 pickle/JSON 问题
    return _to_jsonable(res)


def main():
    parser = argparse.ArgumentParser(description="批量调用几何求解器")
    parser.add_argument("problems_file", help="问题文件路径 (两行一题的 dev_jgex 格式)")
    parser.add_argument("--max-attempts", type=int, default=100, help="构建状态时的最大尝试次数")
    parser.add_argument("--timeout", type=int, default=60, help="单题求解超时时间(秒)")
    parser.add_argument("--limit", type=int, default=None, help="仅求解前 N 题用于快速排查")
    parser.add_argument("--workers", type=int, default=1, help="并行工作数(>1 启用并行)")
    parser.add_argument("--backend", type=str, choices=["thread", "process"], default="process", help="并行后端：thread 或 process（默认 process 更稳定）")
    args = parser.parse_args()

    problems_file = args.problems_file
    if not os.path.isabs(problems_file):
        problems_file = os.path.abspath(problems_file)

    if not os.path.exists(problems_file):
        print(f"错误: 问题文件不存在 -> {problems_file}")
        sys.exit(2)

    # 若 workers==1 走原有串行流程；否则启用并行
    if args.workers <= 1:
        stats = solve_problems_batch(
            problems_file=problems_file,
            rules_file="",
            max_attempts=args.max_attempts,
            timeout_sec=args.timeout,
            limit=args.limit,
        )
    elif args.backend == "thread":
        problems = _load_dev_jgex_problems(problems_file)
        if args.limit is not None:
            problems = problems[: max(args.limit, 0)]
        total = len(problems)
        print(f"开始并行求解(线程) {total} 个问题... workers={args.workers}, max_attempts={args.max_attempts}, timeout={args.timeout}s")

        results = []
        solved_count = 0
        start_real = time.time()

        # 提交所有任务
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            future_map = {}
            for p in problems:
                st = time.time()
                fut = ex.submit(
                    solve_single_problem,
                    p['problem_text'],
                    "",  # 规则文件当前未使用
                    args.max_attempts,
                    args.timeout,
                )
                future_map[fut] = (p['problem_id'], st)

            processed = 0
            for fut in as_completed(future_map):
                pid, st = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"success": False, "proof": None, "run_info": None, "error": str(e)}
                res['problem_id'] = pid
                results.append(_to_jsonable(res))
                processed += 1
                dur = time.time() - st
                if res.get('success'):
                    solved_count += 1
                    print(f"[完成] {processed}/{total} 成功: {pid} 用时 {dur:.1f}s")
                else:
                    err = res.get('error')
                    err_msg = f" 失败: {err}" if err else " 失败"
                    print(f"[完成] {processed}/{total}{err_msg}: {pid} 用时 {dur:.1f}s")

        solve_rate = (solved_count / total) if total > 0 else 0.0
        stats = {"total": total, "solved": solved_count, "solve_rate": solve_rate, "results": results, "realtime_sec": time.time() - start_real}
    else:
        # 进程并行：更适合计算密集或线程不安全的库
        problems = _load_dev_jgex_problems(problems_file)
        if args.limit is not None:
            problems = problems[: max(args.limit, 0)]
        total = len(problems)
        print(f"开始并行求解(进程) {total} 个问题... workers={args.workers}, max_attempts={args.max_attempts}, timeout={args.timeout}s")

        results = []
        solved_count = 0
        start_real = time.time()

        # Linux/CUDA 更推荐 spawn，避免 fork 后 GPU/线程状态问题
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            future_map = {}
            for p in problems:
                st = time.time()
                fut = ex.submit(
                    _solve_and_pack,
                    (p['problem_id'], p['problem_text'], "", args.max_attempts, args.timeout)
                )
                future_map[fut] = (p['problem_id'], st)

            processed = 0
            for fut in as_completed(future_map):
                pid, st = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = _to_jsonable({"success": False, "proof": None, "run_info": None, "error": str(e), "problem_id": pid})
                # res 已经是 JSON-safe
                results.append(res)
                processed += 1
                dur = time.time() - st
                success = False
                try:
                    success = bool(res.get('success'))  # type: ignore[attr-defined]
                except Exception:
                    success = False
                if success:
                    solved_count += 1
                    print(f"[完成] {processed}/{total} 成功: {pid} 用时 {dur:.1f}s")
                else:
                    err = res.get('error') if isinstance(res, dict) else None
                    err_msg = f" 失败: {err}" if err else " 失败"
                    print(f"[完成] {processed}/{total}{err_msg}: {pid} 用时 {dur:.1f}s")

        solve_rate = (solved_count / total) if total > 0 else 0.0
        stats = {"total": total, "solved": solved_count, "solve_rate": solve_rate, "results": results, "realtime_sec": time.time() - start_real}

    print(f"总数: {stats['total']}, 解决: {stats['solved']}, 成功率: {stats['solve_rate']:.2%}")

    # 保存完整结果
    out_json = os.path.splitext(problems_file)[0] + "_results.json"
    try:
        json_safe = _to_jsonable(stats)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(json_safe, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {out_json}")
    except Exception as e:
        print(f"结果保存失败: {e}")


if __name__ == "__main__":
    main()

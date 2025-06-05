"""
Beam search routine for Newclid LLM辅助构造与推理流程。

本实现用于在Newclid几何自动证明框架中，结合语言模型输出和符号推理，进行辅助构造的beam search。
每个beam节点包含：(score, solver, problem_des, aux_history)
"""

import copy
import logging
from typing import Any, List, Tuple

from newclid.formulations.clause import Clause

def newclid_beam_search(
    model,
    solver,
    problem_des: str,
    search_depth: int,
    beam_size: int,
    out_file: str,
    run_ddar_fn,
    try_translate_lm_output_fn,
    insert_aux_to_premise_fn,
) -> bool:
    """
    Newclid框架下的beam search辅助构造与推理。

    Args:
        model: 具备beam_search(problem_des, beam_size)接口的LLM模型
        solver: GeometricSolver实例（初始状态）
        problem_des: 问题描述字符串
        search_depth: beam search深度
        beam_size: beam宽度
        out_file: 证明输出文件路径
        run_ddar_fn: 回调，符号推理函数，参数(solver, out_file) -> bool
        try_translate_lm_output_fn: 回调，翻译LLM输出为构造语句
        insert_aux_to_premise_fn: 回调，插入辅助构造到问题描述

    Returns:
        bool: 是否求解成功
    """
    beam_queue = []
    # (score, solver, problem_des, aux_history)
    beam_queue.append((0.0, solver, problem_des, []))

    for depth in range(search_depth):
        logging.info(f"[BeamSearch] Depth {depth}, beam size: {len(beam_queue)}")
        new_queue = []

        for prev_score, solver, problem_des, aux_history in beam_queue:
            # 1. LLM生成辅助构造
            outputs = model.beam_search(problem_des, beam_size=beam_size)
            # outputs: List[str]，每个为一个辅助点构造语句

            for lm_out in outputs:
                aux_pred = try_translate_lm_output_fn(lm_out, solver)
                if aux_pred.startswith("ERROR"):
                    continue

                # 2. 解析辅助构造为Clause并添加到proof
                try:
                    clauses = Clause.parse_line(aux_pred)
                except Exception as e:
                    logging.warning(f"[BeamSearch] Clause parse failed: {aux_pred}, error: {e}")
                    continue

                # 深拷贝solver，避免污染其他beam
                new_solver = copy.deepcopy(solver)
                try:
                    for clause in clauses:
                        new_solver.proof.add_construction(clause)
                except Exception as e:
                    logging.warning(f"[BeamSearch] Add construction failed: {aux_pred}, error: {e}")
                    continue

                # 3. 构造新的问题描述
                new_problem_des = insert_aux_to_premise_fn(problem_des, aux_pred)

                # 4. 再次尝试符号推理
                if run_ddar_fn(new_solver, out_file):
                    logging.info("[BeamSearch] Solved with auxiliary construction.")
                    return True

                # 5. 加入beam队列
                new_queue.append((prev_score, new_solver, new_problem_des, aux_history + [aux_pred]))

        # 只保留分数最高的beam_size个节点（此处score未用，可自定义打分策略）
        beam_queue = sorted(new_queue, key=lambda x: -x[0])[:beam_size]

    return False
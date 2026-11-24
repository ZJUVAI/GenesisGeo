#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SingleProofGraph: 单题目证明图

用途：
- 与现有 ProofGraph 保持 API 兼容（nodes/edges/point_coords/point_rely_on/aux_points 等）；
- 仅承载一题的数据，避免跨题合图，便于后续引入按题并行；
- 作为平滑替换层，先适配脚本 extract_aux_graph.py / prune_graphs.py / filter_and_prune.py。

不引入新依赖；仅继承并复用 ProofGraph 的解析逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .proof_graph import ProofGraph


class SingleProofGraph(ProofGraph):
    """仅包含单题目的 ProofGraph 包装。

    典型用法：
      spg = SingleProofGraph.build_from_result_record(result_dict)
      viz.render_problem(spg, spg.get_single_problem_id(), "out.png")
    """

    def __init__(self, *, verbose: bool = False) -> None:
        # 复用父类实现；当非 verbose 时，将日志级别提升为 WARNING，屏蔽 info 级输出
        from logging import INFO, WARNING

        super().__init__(verbose=verbose, log_level=(INFO if verbose else WARNING))
        self._single_problem_id: Optional[str] = None

    @classmethod
    def build_from_result_record(
        cls, result_obj: Dict[str, Any], *, verbose: bool = False
    ) -> "SingleProofGraph":
        """从 results[*] 的一条记录构建单题图。

        要求：result_obj 至少包含 problem_id 与 proof 三段文本（字典或字符串，沿用父类解析）。
        """
        inst = cls(verbose=verbose)
        # 父类方法会解析并填充 nodes/edges/point_* 等
        inst.from_single_result(result_obj)
        # 记录唯一题目 id
        pid = result_obj.get("problem_id")
        inst._single_problem_id = str(pid) if pid is not None else None
        return inst

    def get_single_problem_id(self) -> Optional[str]:
        """返回该图包含的唯一 problem_id。若未知则为 None。"""
        return self._single_problem_id


__all__ = ["SingleProofGraph"]

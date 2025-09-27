#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuxExtractor
- 仅负责：从结果 JSON 中筛选出 aux_points 非空的问题并输出过滤后的对象/文件。
- 不做：子图提取与其它改写；保持最小职责，便于在脚本中组合 ProofGraph 可视化。

使用方式：
  from newclid.data_discovery.aux_extractor import AuxExtractor
  stats = AuxExtractor().filter_results_with_aux(in_json, out_json)
"""
from __future__ import annotations

from typing import Any, Dict, List
import json


class AuxExtractor:
    """筛选 aux_points 非空的问题。

    输出对象结构与输入保持一致，仅替换/缩减 results。
    """

    def _has_non_empty_aux(self, res: Dict[str, Any]) -> bool:
        aux = res.get("aux_points")
        if not isinstance(aux, list):
            return False
        # 视为“非空”：至少包含一个非空字符串
        return any(str(x).strip() for x in aux)

    def filter_results_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """输入对象 -> 过滤后的对象（不落盘）。

        要求：obj 至少包含键 'results'（list）。若缺失则视为空集合。
        """
        results = obj.get("results")
        if not isinstance(results, list):
            filtered: List[Dict[str, Any]] = []
        else:
            filtered = [r for r in results if isinstance(r, dict) and self._has_non_empty_aux(r)]

        # 最小改动：浅拷贝 + 替换 results
        new_obj = dict(obj)
        new_obj["results"] = filtered
        return new_obj

    def filter_results_with_aux(self, input_json_path: str, output_json_path: str) -> Dict[str, int]:
        """读取 input_json -> 过滤 -> 写入 output_json，返回统计信息。"""
        with open(input_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        out_obj = self.filter_results_obj(obj if isinstance(obj, dict) else {})
        kept = len(out_obj.get("results", []) or [])
        total = len((obj or {}).get("results", []) or []) if isinstance(obj, dict) else 0
        dropped = max(0, total - kept)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"[aux] filtered: total={total} kept={kept} dropped={dropped} -> {output_json_path}")
        return {"total": total, "kept": kept, "dropped": dropped}

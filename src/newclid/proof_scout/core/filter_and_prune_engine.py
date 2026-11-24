#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FilterAndPruneEngine

职责：封装 scripts/filter_and_prune.py 中的完整流程为可复用类，便于脚本薄封装调用：
  - 读取"新格式"数据（仅支持 llm_input_renamed / llm_output_renamed）
  - 过滤 aux_points 非空的题目
  - 使用 SingleProofGraph + GraphPruner 进行图修剪
  - 基于修剪结果生成不含辅助点的命题与规则文本，去重并落盘 *_pruned_rules.txt
  - 可选并行渲染筛选前/修剪后对比图

不引入新依赖；复用 data_discovery 中现有类。
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# 仅在工作进程中延迟导入可视化相关库，避免非必须环境依赖

# ------------------------- 正则与通用解析工具 -------------------------
_TAG_RE = re.compile(r"</?\w+>")
# 放宽参数匹配字符集，允许包含 '.', '/', '-' 等（例如 rconst a b a c 1/2 [000]）
_FACT_SEG_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\s+(?:[A-Za-z0-9_./-]+\s+)*\[\d+\]")


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text or "")


def _extract_tag_content(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern = re.compile(rf"<{tag}>" r"(.*?)" rf"</{tag}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return ""


def _sanitize_clause_section(raw: str) -> str:
    if not raw:
        return ""
    content = _strip_tags(raw).replace("\n", " ")
    clauses = [seg.strip() for seg in content.split(";") if seg.strip()]
    return "; ".join(clauses)


def _norm_text_basic(s: str) -> str:
    """基本规范化：去标签、换行转空格、多空白压缩、去首尾空白。
    保持项的相对顺序，不做标点/分号重排，避免引入语义漂移。
    """
    if not s:
        return ""
    s = _strip_tags(s)
    s = s.replace("\n", " ")
    # 压缩多空白为单空格
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_problem_prefix_before_q(llm_input: str) -> str:
    """从 llm_input_renamed 提取 <problem> 标签内部问号前的内容；若无标签，则使用原始字符串。
    若找不到问号，则取全部。
    返回规范化后的文本（_norm_text_basic）。
    """
    if not isinstance(llm_input, str):
        return ""
    body = _extract_tag_content(llm_input, "problem") or llm_input
    # 仅取问号之前的部分
    qpos = body.find("?")
    if qpos >= 0:
        body = body[:qpos]
    return _norm_text_basic(body)


def _extract_aux_body_text(llm_output: str) -> str:
    """从 llm_output_renamed 中提取 <aux> 标签内部内容并规范化。缺失视为空。"""
    if not isinstance(llm_output, str):
        return ""
    aux_body = _extract_tag_content(llm_output, "aux")
    return _norm_text_basic(aux_body)


def _record_dedup_key(rec: Dict[str, Any]) -> Optional[str]:
    """生成用于去重的键：基于题干(问号前) + aux 内容。
    若关键字段缺失导致无法稳定抽取，则返回 None（放弃去重，保留该条）。
    """
    try:
        q = _extract_problem_prefix_before_q(rec.get("llm_input_renamed", ""))
        a = _extract_aux_body_text(rec.get("llm_output_renamed", ""))
        if not q and not a:
            return None
        combo = f"Q::{q}||A::{a}"
        return hashlib.sha1(combo.encode("utf-8")).hexdigest()
    except Exception:
        return None


def _dedup_by_question_and_aux(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """对 records 进行基于(题干+aux)的哈希去重。
    - 缺关键字段或解析异常的样本不参与去重（视为独特，直接保留）。
    - 返回 (去重后的列表, 统计信息)。
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    removed = 0
    removed_ids: List[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        key = _record_dedup_key(rec)
        if not key:
            out.append(rec)
            continue
        if key in seen:
            removed += 1
            pid = rec.get("problem_id")
            if pid is not None:
                removed_ids.append(str(pid))
            continue
        seen.add(key)
        out.append(rec)
    return out, {"removed": removed, "kept": len(out), "removed_ids": removed_ids}


def _extract_fact_segments(text: str) -> List[str]:
    if not text:
        return []
    content = _strip_tags(text).replace("\n", " ")
    return [seg.strip() for seg in _FACT_SEG_RE.findall(content) if seg.strip()]


def _extract_aux_points(aux_section: str) -> List[str]:
    content = _strip_tags(aux_section).replace("\n", " ")
    aux_points: List[str] = []
    for part in content.split(";"):
        tokens = [tok for tok in part.strip().split() if tok]
        if len(tokens) >= 2 and tokens[0].lower().startswith("x"):
            aux_points.append(tokens[1])
    return aux_points


def _extract_coords_from_fl_problem(fl_problem: str) -> dict:
    """从 fl_problem 中提取点坐标（新格式：point@x_y）。

    新格式示例：
      a@0.7177669039616846_-0.1290588213521465 b@-0.10503443932378137_-0.6472485204493951 = triangle a b c

    返回 dict[str, [float, float]]，与旧 point_coords 字段格式一致。
    """
    coords = {}
    # 匹配 word@float_float 格式（坐标可为负数和科学计数法）
    pattern = re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)"  # 点名
        r"@"
        r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"  # x 坐标
        r"_"
        r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"  # y 坐标
    )
    for m in pattern.finditer(fl_problem or ""):
        name, x, y = m.group(1), m.group(2), m.group(3)
        coords[name] = [float(x), float(y)]
    return coords


def _has_llm_format(record: Dict[str, Any]) -> bool:
    return isinstance(record, dict) and (
        "llm_input_renamed" in record or "llm_output_renamed" in record
    )


def _sanitize_basename(path: Path) -> str:
    stem = path.stem or "input"
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)


def _convert_llm_record(
    record: Dict[str, Any], base: str, index: int
) -> Dict[str, Any]:
    llm_input = record.get("llm_input_renamed", "")
    llm_output = record.get("llm_output_renamed", "")

    # Priority: pid (new format) > problem_id (legacy) > fallback line-number
    problem_id = record.get("pid") or record.get("problem_id")
    if not problem_id:
        problem_id = f"{base}:{index:06d}"

    problem_section = _extract_tag_content(llm_input, "problem") or llm_input
    analysis_segments = _extract_fact_segments(problem_section)
    aux_section = _extract_tag_content(llm_output, "aux")
    analysis_segments.extend(_extract_fact_segments(aux_section))
    analysis = "; ".join(analysis_segments)
    numerical = _sanitize_clause_section(
        _extract_tag_content(llm_output, "numerical_check")
    )
    trivial = _sanitize_clause_section(_extract_tag_content(llm_output, "trivial"))
    if trivial:
        numerical = f"{numerical}; {trivial}" if numerical else trivial
    proof_text = _sanitize_clause_section(_extract_tag_content(llm_output, "proof"))
    aux_points = _extract_aux_points(aux_section)
    existing_aux = record.get("aux_points")
    if isinstance(existing_aux, list) and existing_aux:
        aux_points = existing_aux

    converted = dict(record)
    converted["problem_id"] = problem_id
    converted["proof"] = {
        "analysis": analysis,
        "numerical_check": numerical,
        "proof": proof_text,
    }
    converted["aux_points"] = aux_points

    # Fallback: 如果 point_coords 缺失或为空，从 fl_problem 中解析坐标
    if not converted.get("point_coords"):
        fl_problem = record.get("fl_problem", "")
        if fl_problem:
            converted["point_coords"] = _extract_coords_from_fl_problem(fl_problem)

    return converted


def _read_input_payload(path: Path) -> Any:
    data = path.read_text(encoding="utf-8")
    if not data.strip():
        return {"results": []}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        results = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
        return {"results": results}


def _normalize_input_object(path: Path) -> Dict[str, Any]:
    payload = _read_input_payload(path)
    base = _sanitize_basename(path)

    def convert_list(raw_list: List[Any]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_list):
            if isinstance(item, dict) and _has_llm_format(item):
                converted.append(_convert_llm_record(item, base, idx))
            else:
                raise ValueError(
                    "filter_and_prune 仅支持包含 llm_input_renamed / llm_output_renamed 的新格式数据"
                )
        return converted

    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            normalized_results = convert_list(results)
        elif results is None:
            normalized_results = []
        else:
            raise ValueError("results 字段必须为列表")
        normalized = {k: v for k, v in payload.items() if k != "results"}
        normalized["results"] = normalized_results
        return normalized
    if isinstance(payload, list):
        normalized_results = convert_list(payload)
        return {"results": normalized_results}
    raise ValueError("输入文件必须为 JSON 对象或数组（新格式）")


def _combine_pngs_h(
    left_png: Path,
    right_png: Path,
    out_png: Path,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> str:
    """将两张 PNG 水平合并为一张，并在子图上方标注标题与副标题。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except Exception as e:
        raise RuntimeError(
            "缺少 matplotlib，无法进行合图保存。请安装 matplotlib 后重试。"
        ) from e

    img_l = mpimg.imread(str(left_png))
    img_r = mpimg.imread(str(right_png))

    fig, axes = plt.subplots(1, 2, figsize=(30, 20))
    axes[0].imshow(img_l)
    axes[0].axis("off")
    axes[0].set_title("Original", fontsize=10)

    axes[1].imshow(img_r)
    axes[1].axis("off")
    axes[1].set_title("Pruned", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)
    if subtitle:
        sub = subtitle if len(subtitle) <= 220 else (subtitle[:217] + "...")
        fig.text(
            0.5,
            0.015,
            "proposition w/o aux: " + sub,
            ha="center",
            va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.85),
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return str(out_png)


# ========================= Step 1d/1e helper functions =========================
# These module-level helpers are used by Steps 1d and 1e in the pipeline.
# - _parse_args_from_label: extract arguments from a fact label like "cong(A,B,C,D)"
# - _is_point_token / _alpha_seq: point name detection and a,b,c,... renaming
# - _parse_name_args: split "pred(arg1,arg2)" into (pred, [arg1, arg2])
# - _to_rule_text: (Step 1e) rename points to a,b,c,... and format rule string
# - _build_proposition_no_aux: (Step 1d) extract premises/conclusion from pruned graph
# - _worker_prune: (Step 1c) build + prune proof graph for one record
# ==============================================================================


def _parse_args_from_label(raw_label: str) -> List[str]:
    try:
        if "(" in raw_label and ")" in raw_label:
            inside = raw_label.split("(", 1)[1].rsplit(")", 1)[0]
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            return parts
    except Exception:
        pass
    return []


def _is_point_token(tok: str) -> bool:
    if not tok:
        return False
    return re.match(r"^[A-Za-z][A-Za-z0-9_]*$", tok) is not None


def _alpha_seq(n: int) -> str:
    letters: List[str] = []
    while True:
        n, r = divmod(n, 26)
        letters.append(chr(ord("a") + r))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(letters))


def _parse_name_args(raw_label: str) -> Tuple[str, List[str]]:
    name = raw_label
    args = _parse_args_from_label(raw_label)
    if "(" in raw_label:
        name = raw_label.split("(", 1)[0].strip()
    return name, args


def _to_rule_text(premises: List[str], conclusion: str) -> Tuple[str, Dict[str, str]]:
    """Step 1e helper: convert premises + conclusion into a canonical rule string.

    1. Collect all point names in first-appearance order from premises then conclusion.
    2. Rename: 1st point → a, 2nd → b, ...  (via _alpha_seq)
    3. Format: "pred1 a b, pred2 c d => pred3 a c"

    Returns (rule_text, rename_map).
    """
    seq: List[str] = []

    def _collect(args: List[str]):
        for a in args:
            if _is_point_token(a) and a not in seq:
                seq.append(a)

    parsed_prems: List[Tuple[str, List[str]]] = []
    for pl in premises:
        pn, pa = _parse_name_args(str(pl))
        parsed_prems.append((pn, pa))
        _collect(pa)
    cn, ca = _parse_name_args(str(conclusion))
    _collect(ca)

    rename: Dict[str, str] = {old: _alpha_seq(i) for i, old in enumerate(seq)}

    def _fmt(name: str, args: List[str]) -> str:
        out: List[str] = []
        for a in args:
            if _is_point_token(a) and a in rename:
                out.append(rename[a])
            else:
                out.append(a)
        return name + (" " + " ".join(out) if out else "")

    prem_txts = [_fmt(n, a) for (n, a) in parsed_prems]
    concl_txt = _fmt(cn, ca)
    rule_text = (
        (", ".join(prem_txts) + " => " + concl_txt)
        if prem_txts
        else ("=> " + concl_txt)
    )
    return rule_text, rename


def _split_clauses(text: str) -> List[str]:
    parts = [p.strip() for p in (text or "").split(",")]
    return [p for p in parts if p]


def _parse_clause(clause: str) -> Tuple[str, List[str]]:
    tokens = [t for t in (clause or "").strip().split() if t]
    if not tokens:
        return "", []
    return tokens[0], tokens[1:]


def _split_rule_text(
    rule_text: str,
) -> Tuple[List[Tuple[str, List[str]]], Tuple[str, List[str]]]:
    if not isinstance(rule_text, str):
        return [], ("", [])
    if "=>" in rule_text:
        left, right = re.split(r"\s*=>\s*", rule_text.strip(), maxsplit=1)
    else:
        left, right = rule_text.strip(), ""
    left_parsed = [_parse_clause(c) for c in _split_clauses(left)] if left else []
    right_parsed = _parse_clause(right) if right else ("", [])
    return left_parsed, right_parsed


def _collect_point_set(clauses: Iterable[Tuple[str, List[str]]]) -> set[str]:
    points: set[str] = set()
    for _, args in clauses:
        for arg in args:
            if _is_point_token(arg):
                points.add(arg)
    return points


def _collect_point_list(args: Iterable[str]) -> List[str]:
    return [a for a in args if _is_point_token(a)]


def _canonicalize_rule_text(rule_text: str) -> str:
    left_parsed, right_parsed = _split_rule_text(rule_text)

    seq: List[str] = []

    def collect(args: Iterable[str]) -> None:
        for a in args:
            if _is_point_token(a) and a not in seq:
                seq.append(a)

    for _, args in left_parsed:
        collect(args)
    collect(right_parsed[1])

    rename: Dict[str, str] = {old: _alpha_seq(i) for i, old in enumerate(seq)}

    def fmt(name: str, args: List[str]) -> str:
        mapped = [rename.get(a, a) if _is_point_token(a) else a for a in args]
        return (name + (" " + " ".join(mapped) if mapped else "")).strip()

    left_norm = ", ".join([fmt(n, a) for n, a in left_parsed if n])
    right_norm = fmt(*right_parsed) if right_parsed[0] else ""
    if left_norm and right_norm:
        return f"{left_norm} => {right_norm}"
    if right_norm:
        return f"=> {right_norm}"
    return left_norm


def _fmt_rule_id(idx: int) -> str:
    return f"r{idx:04d}"


def _pid_to_rid(pid: str) -> str:
    """Convert pid to rid by replacing 'p' prefix with 'r'.

    Examples: p000042 → r000042, p000042_0 → r000042_0
    For legacy pids without 'p' prefix, fall back to using the pid as-is with 'r' prefix.
    """
    pid_str = str(pid)
    if pid_str.startswith("p"):
        return "r" + pid_str[1:]
    # Legacy format (e.g., "base:000042") — keep as-is for backward compat
    return pid_str


def _build_proposition_no_aux(
    rendered: Dict[str, Any], aux_points: List[str]
) -> Optional[Dict[str, Any]]:
    """Step 1d helper: extract proposition (premises + conclusion) from a pruned proof graph.

    1. Identify fact nodes (type=="fact") in the pruned graph.
    2. Premises = fact nodes with in-degree 0 (root facts).
    3. Conclusion = fact node with out-degree 0 (must be exactly 1, else return None).
    4. Filter out premises whose arguments contain any aux point.
    5. Return {premises: [...], conclusion: "...", text: "..."}.
    """
    nodes = (rendered or {}).get("nodes") or []
    edges = (rendered or {}).get("edges") or []
    if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes:
        return None

    indeg = {n.get("idx"): 0 for n in nodes}
    outdeg = {n.get("idx"): 0 for n in nodes}
    for u, v in edges:
        outdeg[u] = outdeg.get(u, 0) + 1
        indeg[v] = indeg.get(v, 0) + 1

    fact_nodes = {n.get("idx"): n for n in nodes if n.get("type") == "fact"}
    premise_ids = [idx for idx, n in fact_nodes.items() if indeg.get(idx, 0) == 0]
    concl_ids = [idx for idx, n in fact_nodes.items() if outdeg.get(idx, 0) == 0]
    if len(concl_ids) != 1:
        return None

    aux_set = set(aux_points or [])
    kept_premises: List[str] = []
    for pid in premise_ids:
        raw_label = str(fact_nodes[pid].get("label", ""))
        args = _parse_args_from_label(raw_label)
        if not any(a in aux_set for a in args):
            kept_premises.append(raw_label)

    concl_label = str(fact_nodes[concl_ids[0]].get("label", ""))
    text = (
        (", ".join(kept_premises) + " -> " + concl_label)
        if kept_premises
        else ("-> " + concl_label)
    )
    return {"premises": kept_premises, "conclusion": concl_label, "text": text}


def _worker_prune(rec: Dict[str, Any]) -> Tuple[str, Any]:
    """Step 1c helper: build SingleProofGraph from one record and prune it.

    Called in parallel by ProcessPoolExecutor in run().
    Code path: SingleProofGraph.build_from_result_record(rec)
             → GraphPruner().prune_proof_graph(spg)
    Returns (pid, rendered_list) where rendered_list = [{nodes: [...], edges: [...]}, ...].
    """
    try:
        pid_val = rec.get("problem_id")
        if pid_val is None:
            return ("<none>", None)
        pid = str(pid_val)
        # 延迟导入，降低主进程依赖
        from newclid.proof_scout.core.graph_pruner import GraphPruner
        from newclid.proof_scout.core.single_proof_graph import SingleProofGraph

        pruner = GraphPruner()
        spg = SingleProofGraph.build_from_result_record(rec, verbose=False)
        rendered_list = pruner.prune_proof_graph(spg).get(pid, [])
        return (pid, rendered_list if rendered_list else None)
    except Exception:
        return (str(rec.get("problem_id")), None)


# ========================= Streaming helpers =========================


def _stream_jsonl_chunks(
    path: Path, chunk_size: int = 10000
) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
    """Yield (chunk_idx, records) by reading JSONL line-by-line.

    Each record is converted via _convert_llm_record.
    Memory: only one chunk (~chunk_size records) in memory at a time.
    """
    base = _sanitize_basename(path)
    buffer: List[Dict[str, Any]] = []
    chunk_idx = 0
    global_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict) or not _has_llm_format(rec):
                global_idx += 1
                continue
            converted = _convert_llm_record(rec, base, global_idx)
            buffer.append(converted)
            global_idx += 1
            if len(buffer) >= chunk_size:
                yield (chunk_idx, buffer)
                buffer = []
                chunk_idx += 1
    if buffer:
        yield (chunk_idx, buffer)


def _ray_prune_chunk_bounded(
    records: List[Dict[str, Any]],
    inflight_limit: int = 300,
) -> Dict[str, List[Dict[str, Any]]]:
    """Prune records using Ray with bounded in-flight tasks.

    Requires ray to be initialized before calling.
    Returns pruned_map: {pid: [rendered_dict, ...]}.
    """
    import ray

    @ray.remote(num_cpus=1, max_retries=0)
    def _ray_worker_prune(rec: Dict[str, Any]) -> Tuple[str, Any]:
        try:
            pid_val = rec.get("problem_id")
            if pid_val is None:
                return ("<none>", None)
            pid = str(pid_val)
            from newclid.proof_scout.core.graph_pruner import GraphPruner
            from newclid.proof_scout.core.single_proof_graph import SingleProofGraph

            pruner = GraphPruner()
            spg = SingleProofGraph.build_from_result_record(rec, verbose=False)
            rendered_list = pruner.prune_proof_graph(spg).get(pid, [])
            return (pid, rendered_list if rendered_list else None)
        except Exception:
            return (str(rec.get("problem_id")), None)

    pending: Dict[Any, str] = {}  # ray ref -> pid
    results: Dict[str, List[Dict[str, Any]]] = {}
    rec_iter = iter(
        r for r in records if isinstance(r, dict) and r.get("problem_id") is not None
    )

    # Initial fill
    for rec in rec_iter:
        ref = _ray_worker_prune.remote(rec)
        pending[ref] = str(rec.get("problem_id"))
        if len(pending) >= inflight_limit:
            break

    # Drain + refill loop
    while pending:
        done, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=30)
        for ref in done:
            try:
                pid, rendered_list = ray.get(ref)
                if rendered_list:
                    results[pid] = rendered_list
            except Exception:
                pass
            del pending[ref]

        # Refill
        for rec in rec_iter:
            ref = _ray_worker_prune.remote(rec)
            pending[ref] = str(rec.get("problem_id"))
            if len(pending) >= inflight_limit:
                break

    return results


def _worker_render_combined(
    rec: Dict[str, Any],
    rendered: Dict[str, Any],
    out_png: str,
    label_mode: str,
    single_figsize: Tuple[int, int],
    ranksep: float,
    nodesep: float,
    font_size: int,
) -> Tuple[str, str]:
    try:
        from pathlib import Path as _Path
        import tempfile as _tempfile
        from newclid.proof_scout.core.proof_graph_visualizer import ProofGraphVisualizer
        from newclid.proof_scout.core.single_proof_graph import SingleProofGraph

        pid = str(rec.get("problem_id"))
        viz_left = ProofGraphVisualizer()
        viz_left.LAYOUT_RANKSEP = ranksep
        viz_left.LAYOUT_NODESEP = nodesep
        spg = SingleProofGraph.build_from_result_record(rec, verbose=False)
        with _tempfile.TemporaryDirectory() as td:
            td_path = _Path(td)
            left_png = td_path / f"left_{pid}.png"
            right_png = td_path / f"right_{pid}.png"
            viz_left.render_problem(
                spg,
                pid,
                str(left_png),
                label_mode=label_mode,
                highlight=True,
                figsize=single_figsize,
            )
            # 修剪图（附 aux_points）
            rendered2 = dict(rendered)
            aux_points = rec.get("aux_points") or []
            if isinstance(aux_points, list):
                rendered2["aux_points"] = aux_points
            viz_right = ProofGraphVisualizer()
            viz_right.render_rendered(
                rendered2,
                str(right_png),
                label_mode=label_mode,
                highlight=True,
                figsize=single_figsize,
                font_size=font_size,
                show_direction_legend=True,
                layout_ranksep=ranksep,
                layout_nodesep=nodesep,
            )
            # 合图副标题优先使用规则文本
            prop = _build_proposition_no_aux(
                rendered2, list(rec.get("aux_points") or [])
            )
            subtitle = None
            if isinstance(prop, dict):
                try:
                    rule_text, _ = _to_rule_text(
                        prop.get("premises", []) or [], prop.get("conclusion", "")
                    )
                    subtitle = rule_text
                except Exception:
                    subtitle = prop.get("text")
            _combine_pngs_h(
                left_png,
                right_png,
                _Path(out_png),
                title=f"Problem {pid}",
                subtitle=subtitle,
            )
        return (pid, "ok")
    except Exception:
        return (str(rec.get("problem_id")), "failed")


class FilterAndPruneEngine:
    """一键筛选 + 修剪 + 规则提取（可并行绘图）的引擎类。"""

    def __init__(
        self,
        *,
        label_mode: str = "legend",
        figsize_single: Tuple[int, int] = (15, 20),
        figsize_combined: Tuple[int, int] = (30, 20),
        font_size: int = 9,
        ranksep: float = 2.8,
        nodesep: float = 1.2,
        overwrite: bool = True,
        progress_every: int = 10,
        max_workers: int = 50,
        # 若为 True，则在规则写出阶段打印并另存被跳过（无效）的规则，便于调试
        print_skipped_rules: bool = True,
        # 若为 True，则在绘图阶段打印并保存被跳过样本及原因
        print_render_skipped: bool = True,
        # 若为 True，则打印/保存去重阶段被移除的 pid 列表
        print_dedup_removed: bool = True,
        # 若为 True，则为所有参与处理的 pid 导出对应题目输入文本（JSONL），便于按 pid 检索题目
        save_pid_inputs: bool = True,
        # 新增：是否生成按规则编号的图片 rXXXX.png 及 rid_map.txt
        render_by_rule: bool = True,
        # 新增：是否保留/生成按 pid 命名的图片 proof_{pid}.png（默认关闭）
        keep_pid_images: bool = False,
        # 新增：谓词对称性配置文件路径（JSON），用于规则级去重的规范化
        symmetry_config_path: Optional[str] = None,
        # 需要跳过的谓词集合（若记录的 llm_output_renamed 包含这些谓词则过滤掉）
        skip_predicates: Optional[List[str]] = None,
        # 规则落盘时需要跳过的谓词集合（若规则文本包含这些谓词则不写入最终 rules.txt）
        rule_skip_predicates: Optional[List[str]] = None,
    ) -> None:
        self.label_mode = label_mode
        self.figsize_single = figsize_single
        self.figsize_combined = figsize_combined
        self.font_size = font_size
        self.ranksep = ranksep
        self.nodesep = nodesep
        self.overwrite = overwrite
        self.progress_every = progress_every
        self.max_workers = max_workers
        self.print_skipped_rules = print_skipped_rules
        self.print_render_skipped = print_render_skipped
        self.print_dedup_removed = print_dedup_removed
        self.save_pid_inputs = save_pid_inputs
        self.render_by_rule = render_by_rule
        self.keep_pid_images = keep_pid_images
        self.symmetry_config_path = symmetry_config_path
        self.skip_predicates = set()
        # self.skip_predicates = set(p.lower() for p in (skip_predicates if skip_predicates is not None else ["eqpoint", "constline"]))
        self.rule_skip_predicates = set(
            p.lower()
            for p in (
                rule_skip_predicates
                if rule_skip_predicates is not None
                else ["aconst", "rconst"]
            )
        )
        # 硬编码 19 个谓词的对称性配置（优先），外部 JSON 仅作覆盖
        self._symmetry_cfg: Dict[str, Any] = self._build_hardcoded_symmetry_config()
        if self.symmetry_config_path:
            try:
                p = Path(self.symmetry_config_path)
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        override = json.load(f)
                    self._symmetry_cfg.update(override)
                    print(f"[symmetry] merged overrides from {p}")
                else:
                    print(f"[symmetry] config not found: {p}")
            except Exception as e:
                print(f"[symmetry] failed to load config: {e}")

    # --- 硬编码 19 个谓词的对称性配置 ---
    @staticmethod
    def _build_hardcoded_symmetry_config() -> Dict[str, Any]:
        """Return hardcoded symmetry rules for 19 geometric predicates."""
        return {
            # [ab][cd], groups inter-swappable, intra-swappable
            "cong": {
                "kind": "swap-pairs",
                "groups": [[0, 1], [2, 3]],
                "allow_group_permutation": True,
            },
            "para": {
                "kind": "swap-pairs",
                "groups": [[0, 1], [2, 3]],
                "allow_group_permutation": True,
            },
            "perp": {
                "kind": "swap-pairs",
                "groups": [[0, 1], [2, 3]],
                "allow_group_permutation": True,
            },
            # [abc], fully unordered
            "coll": {"kind": "unordered"},
            "ncoll": {"kind": "unordered"},
            # [abcd], fully unordered
            "cyclic": {"kind": "unordered"},
            # [a][bc], a fixed, bc swappable
            "midp": {
                "kind": "swap-pairs",
                "groups": [[0], [1, 2]],
                "allow_group_permutation": False,
            },
            # complex symmetry, delegate to EqAngle.preparse()
            "eqangle": {"kind": "preparse"},
            # same logic as eqangle
            "eqratio": {"kind": "preparse"},
            # [abc][def], groups inter-swappable, intra same-direction rotation
            "contri": {
                "kind": "rotation",
                "groups": [[0, 1, 2], [3, 4, 5]],
                "allow_group_permutation": True,
                "rotation_mode": "same",
            },
            # [abc][def], groups inter-swappable, intra opposite-direction rotation
            "contrir": {
                "kind": "rotation",
                "groups": [[0, 1, 2], [3, 4, 5]],
                "allow_group_permutation": True,
                "rotation_mode": "opposite",
            },
            "simtri": {
                "kind": "rotation",
                "groups": [[0, 1, 2], [3, 4, 5]],
                "allow_group_permutation": True,
                "rotation_mode": "same",
            },
            "simtrir": {
                "kind": "rotation",
                "groups": [[0, 1, 2], [3, 4, 5]],
                "allow_group_permutation": True,
                "rotation_mode": "opposite",
            },
            "sameclock": {
                "kind": "rotation",
                "groups": [[0, 1, 2], [3, 4, 5]],
                "allow_group_permutation": True,
                "rotation_mode": "same",
            },
            # [a][bc][d][ef], pair (a,bc) <-> (d,ef) swappable, bc/ef swappable
            "sameside": {
                "kind": "sameside",
            },
            # [ab][cd][const], ab/cd groups swappable, constant stays
            "aconst": {
                "kind": "swap-pairs-const",
                "groups": [[0, 1], [2, 3]],
                "allow_group_permutation": True,
                "const_indices": [4],
            },
            "rconst": {
                "kind": "swap-pairs-const",
                "groups": [[0, 1], [2, 3]],
                "allow_group_permutation": True,
                "const_indices": [4],
            },
            # [ab], swappable
            "eqpoint": {"kind": "unordered"},
            # [abc], fully unordered
            "constline": {"kind": "unordered"},
        }

    # --- 谓词对称性与规范化 ---
    def _normalize_clause_by_symmetry(
        self, pred: str, args: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        spec = (
            self._symmetry_cfg.get(pred)
            if isinstance(self._symmetry_cfg, dict)
            else None
        )
        if not spec or not isinstance(spec, dict):
            return pred, tuple(args)
        kind = spec.get("kind")
        if kind == "unordered":
            return pred, tuple(sorted(args))
        if kind == "preparse":
            return self._normalize_by_preparse(pred, args)
        if kind == "rotation":
            return self._normalize_by_rotation(pred, args, spec)
        if kind == "sameside":
            return self._normalize_sameside(pred, args)
        if kind == "swap-pairs-const":
            return self._normalize_swap_pairs_const(pred, args, spec)
        groups = spec.get("groups")
        allow_perm = bool(spec.get("allow_group_permutation", False))
        if kind in ("swap-pairs", "swap-groups") and isinstance(groups, list):
            blocks: List[List[str]] = []
            try:
                for g in groups:
                    block = [args[i] for i in g if 0 <= i < len(args)]
                    block.sort()
                    blocks.append(block)
            except Exception:
                return pred, tuple(args)
            if allow_perm:
                blocks.sort(key=lambda b: ",".join(b))
            norm_args: List[str] = []
            for b in blocks:
                norm_args.extend(b)
            return pred, tuple(norm_args)
        return pred, tuple(args)

    def _normalize_by_preparse(
        self, pred: str, args: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        """Delegate to EqAngle.preparse() for eqangle/eqratio normalization."""
        if len(args) != 8:
            return pred, tuple(args)
        try:
            from newclid.predicates.equal_angles import EqAngle

            result = EqAngle.preparse(tuple(args))
            if result is None:
                return pred, tuple(args)
            return pred, tuple(result)
        except Exception:
            return pred, tuple(args)

    def _normalize_by_rotation(
        self, pred: str, args: List[str], spec: Dict[str, Any]
    ) -> Tuple[str, Tuple[str, ...]]:
        """Normalize predicates with rotation symmetry (contri, simtri, etc.)."""
        groups = spec.get("groups", [[0, 1, 2], [3, 4, 5]])
        allow_perm = bool(spec.get("allow_group_permutation", False))
        rotation_mode = spec.get("rotation_mode", "same")

        try:
            g1 = [args[i] for i in groups[0]]
            g2 = [args[i] for i in groups[1]]
        except (IndexError, TypeError):
            return pred, tuple(args)

        # Generate 3 rotations for g1
        g1_rots = [g1, [g1[1], g1[2], g1[0]], [g1[2], g1[0], g1[1]]]

        # Generate corresponding rotations for g2
        if rotation_mode == "same":
            g2_rots = [g2, [g2[1], g2[2], g2[0]], [g2[2], g2[0], g2[1]]]
        else:  # opposite
            g2_rev = g2[::-1]
            g2_rots = [
                g2_rev,
                [g2_rev[1], g2_rev[2], g2_rev[0]],
                [g2_rev[2], g2_rev[0], g2_rev[1]],
            ]

        candidates: List[Tuple[str, ...]] = []
        for r1 in g1_rots:
            for r2 in g2_rots:
                candidates.append(tuple(r1 + r2))
                if allow_perm:
                    candidates.append(tuple(r2 + r1))

        return pred, min(candidates)

    def _normalize_sameside(
        self, pred: str, args: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        """Normalize sameside: [a][bc][d][ef], pair (a,bc)<->(d,ef) swappable, bc/ef swappable."""
        if len(args) != 6:
            return pred, tuple(args)
        a, b, c, d, e, f = args
        # Sort within bc and ef
        bc = sorted([b, c])
        ef = sorted([e, f])
        # Two candidates: (a, bc, d, ef) vs (d, ef, a, bc)
        cand1 = tuple([a] + bc + [d] + ef)
        cand2 = tuple([d] + ef + [a] + bc)
        return pred, min(cand1, cand2)

    def _normalize_swap_pairs_const(
        self, pred: str, args: List[str], spec: Dict[str, Any]
    ) -> Tuple[str, Tuple[str, ...]]:
        """Normalize aconst/rconst: [ab][cd][const], ab/cd groups swappable, constant stays."""
        groups = spec.get("groups", [[0, 1], [2, 3]])
        allow_perm = bool(spec.get("allow_group_permutation", False))
        const_indices = spec.get("const_indices", [4])

        try:
            blocks: List[List[str]] = []
            for g in groups:
                block = [args[i] for i in g if 0 <= i < len(args)]
                block.sort()
                blocks.append(block)
            consts = [args[i] for i in const_indices if 0 <= i < len(args)]
        except (IndexError, TypeError):
            return pred, tuple(args)

        if allow_perm:
            blocks.sort(key=lambda b: ",".join(b))

        norm_args: List[str] = []
        for b in blocks:
            norm_args.extend(b)
        norm_args.extend(consts)
        return pred, tuple(norm_args)

    # --- Step 3.5a: Simplify trivial eqangle/eqratio predicates ---
    @staticmethod
    def _simplify_clause(pred: str, args: List[str]) -> Tuple[str, List[str], bool]:
        """Simplify a single clause if it's a trivial eqangle/eqratio.

        Rules:
          eqangle X1 X2 X3 X4 X5 X6 X7 X8:
            - if {X5,X6} == {X7,X8} → para X1 X2 X3 X4
            - if {X1,X2} == {X3,X4} → para X5 X6 X7 X8
          eqratio X1 X2 X3 X4 X5 X6 X7 X8:
            - if {X5,X6} == {X7,X8} → cong X1 X2 X3 X4
            - if {X1,X2} == {X3,X4} → cong X5 X6 X7 X8

        Returns: (new_pred, new_args, was_simplified)
        """
        if len(args) != 8:
            return pred, args, False

        if pred == "eqangle":
            edge1 = frozenset(args[0:2])
            edge2 = frozenset(args[2:4])
            edge3 = frozenset(args[4:6])
            edge4 = frozenset(args[6:8])
            if edge3 == edge4:
                return "para", list(args[0:4]), True
            if edge1 == edge2:
                return "para", list(args[4:8]), True
        elif pred == "eqratio":
            edge1 = frozenset(args[0:2])
            edge2 = frozenset(args[2:4])
            edge3 = frozenset(args[4:6])
            edge4 = frozenset(args[6:8])
            if edge3 == edge4:
                return "cong", list(args[0:4]), True
            if edge1 == edge2:
                return "cong", list(args[4:8]), True

        return pred, args, False

    def _simplify_trivial_predicates(self, rule_text: str) -> Tuple[str, bool]:
        """Simplify trivial eqangle/eqratio predicates in a rule text.

        Returns: (simplified_rule_text, was_simplified)
        """
        left_parsed, right_parsed = _split_rule_text(rule_text)
        any_simplified = False

        new_left: List[Tuple[str, List[str]]] = []
        for pred, args in left_parsed:
            if not pred:
                continue
            new_pred, new_args, simplified = self._simplify_clause(pred, list(args))
            if simplified:
                any_simplified = True
            new_left.append((new_pred, new_args))

        rn, ra = right_parsed
        new_rn, new_ra = rn, list(ra)
        if rn:
            new_rn, new_ra, simplified = self._simplify_clause(rn, list(ra))
            if simplified:
                any_simplified = True

        if not any_simplified:
            return rule_text, False

        left_str = ", ".join(
            f"{n} " + " ".join(a) if a else n for n, a in new_left if n
        )
        right_str = f"{new_rn} " + " ".join(new_ra) if new_rn else ""
        if left_str and right_str:
            return f"{left_str} => {right_str}", True
        if right_str:
            return f"=> {right_str}", True
        return left_str, True

    # --- Step 3.5b: eqpoint extraction and analysis ---
    @staticmethod
    def _extract_eqpoint_from_proof(llm_output: str) -> List[Tuple[str, str]]:
        """Extract eqpoint facts from the <proof> section of llm_output_renamed.

        Returns: list of (point1, point2) pairs.
        """
        proof_text = _extract_tag_content(llm_output, "proof")
        if not proof_text:
            return []

        eqpoint_pairs: List[Tuple[str, str]] = []
        pattern = re.compile(
            r"\beqpoint\s+([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\s+\[\d+\]"
        )
        for match in pattern.finditer(proof_text):
            p1, p2 = match.group(1), match.group(2)
            eqpoint_pairs.append((p1, p2))

        return eqpoint_pairs

    @staticmethod
    def _map_eqpoint_to_rule(
        eqpoint_pairs: List[Tuple[str, str]],
        rename_map: Dict[str, str],
    ) -> List[Tuple[str, str]]:
        """Map eqpoint pairs from original point names to rule's canonical names."""
        mapped: List[Tuple[str, str]] = []
        for p1, p2 in eqpoint_pairs:
            r1, r2 = rename_map.get(p1), rename_map.get(p2)
            if r1 is not None and r2 is not None and r1 != r2:
                mapped.append((r1, r2))
        return mapped

    @staticmethod
    def _build_equivalence_classes(
        eqpoint_pairs: List[Tuple[str, str]],
    ) -> Dict[str, str]:
        """Build point equivalence classes via Union-Find.

        Returns: mapping from every point to its representative (lexicographically smallest).
        """
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py

        for p1, p2 in eqpoint_pairs:
            union(p1, p2)

        return {pt: find(pt) for pt in parent}

    def _generate_merged_rule(
        self, rule_text: str, eqpoint_pairs: List[Tuple[str, str]]
    ) -> str:
        """Generate a rule variant with eqpoint-equivalent points merged."""
        if not eqpoint_pairs:
            return rule_text

        equiv_map = self._build_equivalence_classes(eqpoint_pairs)
        left_parsed, right_parsed = _split_rule_text(rule_text)

        def replace_args(args: List[str]) -> List[str]:
            return [equiv_map.get(a, a) if _is_point_token(a) else a for a in args]

        new_left = [(n, replace_args(list(a))) for n, a in left_parsed]
        new_right = (right_parsed[0], replace_args(list(right_parsed[1])))

        left_str = ", ".join(
            f"{n} " + " ".join(a) if a else n for n, a in new_left if n
        )
        right_str = f"{new_right[0]} " + " ".join(new_right[1]) if new_right[0] else ""
        if left_str and right_str:
            return f"{left_str} => {right_str}"
        if right_str:
            return f"=> {right_str}"
        return left_str

    def _canonicalize_rule_with_symmetry(self, rule_text: str) -> str:
        """Step 1e Phase 2 helper: normalize a rule string for dedup.

        1. Rename points to a,b,c,... by first-appearance order
        2. Normalize argument order per predicate symmetry config
           (e.g. cong(a,b,c,d) has group_size=2, allow_perm=True)
        3. Sort premise clauses lexicographically
        4. Final point renaming based on sorted predicate order
        """
        left_parsed, right_parsed = _split_rule_text(rule_text)
        # 收集点出现顺序并重命名（稳定）
        seq: List[str] = []

        def collect(args: Iterable[str]) -> None:
            for a in args:
                if _is_point_token(a) and a not in seq:
                    seq.append(a)

        for n, a in left_parsed:
            collect(a)
        collect(right_parsed[1])
        rename: Dict[str, str] = {old: _alpha_seq(i) for i, old in enumerate(seq)}

        def map_args(args: List[str]) -> List[str]:
            return [rename.get(a, a) if _is_point_token(a) else a for a in args]

        # 子句级：先重命名，再按对称性规范化
        left_norm_clauses: List[Tuple[str, Tuple[str, ...]]] = []
        for n, a in left_parsed:
            if not n:
                continue
            mapped = map_args(a)
            left_norm_clauses.append(self._normalize_clause_by_symmetry(n, mapped))
        rn, ra = right_parsed
        right_norm_clause: Tuple[str, Tuple[str, ...]] = ("", tuple())
        if rn:
            right_norm_clause = self._normalize_clause_by_symmetry(rn, map_args(ra))

        # 前提子句排序（字典序）
        left_norm_clauses.sort(
            key=lambda nc: f"{nc[0]} " + " ".join(nc[1]) if nc[1] else nc[0]
        )

        # Step 3.5: 根据排序后的点出现顺序重新命名
        final_seq: List[str] = []

        def collect_final(args: Iterable[str]) -> None:
            for a in args:
                if _is_point_token(a) and a not in final_seq:
                    final_seq.append(a)

        for n, a in left_norm_clauses:
            collect_final(a)
        collect_final(right_norm_clause[1])

        final_rename: Dict[str, str] = {
            old: _alpha_seq(i) for i, old in enumerate(final_seq)
        }

        def final_map_args(args: Tuple[str, ...]) -> Tuple[str, ...]:
            return tuple(
                final_rename.get(a, a) if _is_point_token(a) else a for a in args
            )

        left_norm_clauses = [(n, final_map_args(a)) for n, a in left_norm_clauses]
        right_norm_clause = (right_norm_clause[0], final_map_args(right_norm_clause[1]))

        left_texts = [f"{n} " + " ".join(a) if a else n for (n, a) in left_norm_clauses]
        right_text = (
            f"{right_norm_clause[0]} " + " ".join(right_norm_clause[1])
            if right_norm_clause[0]
            else ""
        )
        if left_texts and right_text:
            return ", ".join(left_texts) + " => " + right_text
        if right_text:
            return "=> " + right_text
        return ", ".join(left_texts)

    # --- Step 4: Normalization ---
    def _normalize_rules(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int, List[Tuple[str, List[str], str]]]:
        """Step 4: Normalize rules — missing_points check + canonicalize + signature.

        Returns: (normalized_rules, n_skipped, skipped_entries)
        """
        normalized_rules: List[Dict[str, Any]] = []
        skipped = 0
        skipped_entries: List[Tuple[str, List[str], str]] = []

        for rec in results:
            if not isinstance(rec, dict):
                continue
            rule = rec.get("proposition_rule")
            if not isinstance(rule, str) or not rule.strip():
                continue

            # Check missing_points
            left_parsed, right_parsed = _split_rule_text(rule)
            left_points = _collect_point_set(left_parsed)
            right_points = set(_collect_point_list(right_parsed[1]))
            missing = right_points - left_points

            if missing:
                pid = rec.get("problem_id", "<unknown>")
                print(f"[Step 4] skip pid={pid} missing_points={sorted(missing)}")
                if self.print_skipped_rules:
                    skipped_entries.append((str(pid), sorted(missing), rule))
                skipped += 1
                continue

            # Normalize
            norm = self._canonicalize_rule_with_symmetry(rule)
            if not norm:
                continue

            normalized_rules.append(
                {
                    "pid": str(rec.get("problem_id", "<unknown>")),
                    "seed": rec.get("seed"),
                    "original": rule,
                    "normalized": norm,
                    "llm_input_renamed": rec.get("llm_input_renamed", ""),
                    "llm_output_renamed": rec.get("llm_output_renamed", ""),
                    "point_coords": rec.get("point_coords", {}),
                    "eqpoint_original": rec.get("eqpoint_original"),
                    "eqpoint_mapped": rec.get("eqpoint_mapped"),
                    "proposition_rule_merged": rec.get("proposition_rule_merged"),
                }
            )

        # Generate signatures
        for item in normalized_rules:
            item["signature"] = self._signature_from_normalized_rule(item["normalized"])

        print(
            f"[Step 4: normalization] input={skipped + len(normalized_rules)} "
            f"normalized={len(normalized_rules)} skipped={skipped}"
        )
        return normalized_rules, skipped, skipped_entries

    # --- Step 5: Deduplication ---
    def _dedup_rules(
        self,
        normalized_rules: List[Dict[str, Any]],
        *,
        external_seen_hashes: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
        """Step 5: Deduplicate rules — sort by signature + SHA256 hash dedup.

        Args:
            external_seen_hashes: If provided, use this set for cross-chunk dedup
                (hashes are added in-place). Internal dup tracking still works.

        Returns: (deduped_entries, norm_dup_map)
        """
        # Sort by signature
        normalized_rules.sort(key=lambda x: x["signature"])

        # Dedup by full normalized string hash
        seen_hashes: Dict[str, Dict[str, Any]] = {}
        entries: List[Dict[str, Any]] = []
        norm_dup_map: Dict[str, List[Tuple[str, str]]] = {}

        for item in normalized_rules:
            norm_hash = hashlib.sha256(item["normalized"].encode("utf-8")).hexdigest()

            # Cross-chunk dedup: skip if already seen in a previous chunk
            if external_seen_hashes is not None:
                if norm_hash in external_seen_hashes:
                    continue
                external_seen_hashes.add(norm_hash)

            if norm_hash not in seen_hashes:
                seen_hashes[norm_hash] = item
                entries.append(
                    {
                        "rid": _pid_to_rid(item["pid"]),
                        "norm_rule": item["normalized"],
                        "rule": item["original"],
                        "pid": item["pid"],
                        "seed": item.get("seed"),
                        "llm_input_renamed": item.get("llm_input_renamed", ""),
                        "llm_output_renamed": item.get("llm_output_renamed", ""),
                        "point_coords": item.get("point_coords", {}),
                        "eqpoint_original": item.get("eqpoint_original"),
                        "eqpoint_mapped": item.get("eqpoint_mapped"),
                        "proposition_rule_merged": item.get("proposition_rule_merged"),
                    }
                )
            else:
                # Track duplicates
                if norm_hash not in norm_dup_map:
                    first = seen_hashes[norm_hash]
                    norm_dup_map[norm_hash] = [(first["pid"], first["original"])]
                norm_dup_map[norm_hash].append((item["pid"], item["original"]))

        print(
            f"[Step 5: deduplication] input={len(normalized_rules)} "
            f"unique={len(entries)} duplicates={len(normalized_rules) - len(entries)}"
        )
        return entries, norm_dup_map

    # --- Step 6: Rule dump ---
    def _dump_rules(
        self,
        entries: List[Dict[str, Any]],
        norm_dup_map: Dict[str, List[Tuple[str, str]]],
        skipped_entries: List[Tuple[str, List[str], str]],
        pruned_path: Path,
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Step 6: Dump rules — filter by rule_skip_predicates + write output files.

        Returns: (n_rules, n_skipped_predicates, final_entries)
        """
        # Filter by rule_skip_predicates
        final_entries: List[Dict[str, Any]] = []
        skipped_by_pred: List[Dict[str, Any]] = []
        if self.rule_skip_predicates:
            for e in entries:
                rule_lower = e["norm_rule"].lower()
                matched = [p for p in self.rule_skip_predicates if p in rule_lower]
                if matched:
                    skipped_by_pred.append(
                        {
                            "rid": e["rid"],
                            "rule": e["norm_rule"],
                            "matched_predicates": matched,
                        }
                    )
                else:
                    final_entries.append(e)
        else:
            final_entries = entries

        # rid is already aligned with pid via _pid_to_rid(), no re-numbering needed

        # Write rules.txt
        rules_path = pruned_path.with_name(pruned_path.stem + "_rules.txt")
        lines: List[str] = []
        for e in final_entries:
            lines.append(e["rid"])
            lines.append(e["norm_rule"])
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rules_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        # Write duplicated_rules.txt
        if norm_dup_map:
            norm_dup_lines: List[str] = []
            for norm_hash, lst in norm_dup_map.items():
                norm_dup_lines.append(f"norm={lst[0][1]}")
                for pid, rule in lst:
                    norm_dup_lines.append(f"  pid={pid} :: {rule}")
            dup_path = pruned_path.with_name(pruned_path.stem + "_duplicated_rules.txt")
            with open(dup_path, "w", encoding="utf-8") as f:
                f.write("\n".join(norm_dup_lines) + "\n")

        # Write skipped rules (missing points)
        if self.print_skipped_rules and skipped_entries:
            skipped_path = pruned_path.with_name(
                pruned_path.stem + "_rules_skipped.txt"
            )
            out_lines: List[str] = []
            for spid, miss, srule in skipped_entries:
                out_lines.append(f"pid={spid} missing_points={miss}")
                out_lines.append(srule)
            skipped_path.parent.mkdir(parents=True, exist_ok=True)
            with open(skipped_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

        print(
            f"[Step 6: rule dump] wrote {len(final_entries)} rules to {rules_path}"
            f" (skipped_predicates={len(skipped_by_pred)}"
            f"{' rule_skip_predicates=' + str(sorted(self.rule_skip_predicates)) if skipped_by_pred else ''})"
        )

        # Write eqpoint-related outputs
        rules_with_eqpoint = [
            e
            for e in final_entries
            if e.get("eqpoint_mapped") and len(e.get("eqpoint_mapped", [])) > 0
        ]
        if rules_with_eqpoint:
            eqpoint_path = pruned_path.with_name(
                pruned_path.stem + "_rules_with_eqpoint.json"
            )
            with open(eqpoint_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "rid": e["rid"],
                            "pid": e["pid"],
                            "rule": e["norm_rule"],
                            "eqpoint_mapped": e.get("eqpoint_mapped", []),
                            "rule_merged": e.get("proposition_rule_merged", ""),
                        }
                        for e in rules_with_eqpoint
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Write merged rules text
            merged_path = pruned_path.with_name(pruned_path.stem + "_rules_merged.txt")
            merged_lines: List[str] = []
            for e in rules_with_eqpoint:
                merged_lines.append(e["rid"])
                merged_lines.append(e.get("proposition_rule_merged", e["norm_rule"]))
            with open(merged_path, "w", encoding="utf-8") as f:
                f.write("\n".join(merged_lines) + ("\n" if merged_lines else ""))

            print(
                f"[Step 6: eqpoint output] wrote {len(rules_with_eqpoint)} rules with eqpoint to {eqpoint_path}"
            )

        return len(final_entries), len(skipped_by_pred), final_entries

    def _signature_from_normalized_rule(self, normalized_rule: str) -> str:
        """Generate signature from normalized rule for sorting purposes only."""
        try:
            left_parsed, right_parsed = _split_rule_text(normalized_rule)
            n_prem = len([n for n, _ in left_parsed if n])
            n_concl = 1 if right_parsed[0] else 0
            first_pred = left_parsed[0][0] if left_parsed else ""
            return f"prem:{n_prem}|concl:{n_concl}|{first_pred}"
        except Exception:
            return "prem:0|concl:0|"

    def run(
        self,
        input_json: str | Path,
        output_dir: str | Path,
        save_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete filter-and-prune pipeline.

        Pipeline steps:
          Step 1: Input filter — aux filter + predicate filter (skip_predicates)
          Step 2: Graph prune — build SingleProofGraph + GraphPruner for each record
          Step 3: Proposition extraction — _build_proposition_no_aux() from pruned graph
          Step 4: Normalization — point rename + predicate symmetry + premise sort
          Step 5: Deduplication — SHA256 hash dedup on normalized rule text
          Step 6: Rule dump — predicate filter (rule_skip_predicates) + write rules.txt
        """
        in_path = Path(input_json)
        if not in_path.exists():
            raise FileNotFoundError(f"input not found: {in_path}")

        # Setup intermediates directory for --save_intermediates
        intermediates_dir = None
        if save_intermediates:
            intermediates_dir = Path(output_dir) / "intermediates"
            intermediates_dir.mkdir(parents=True, exist_ok=True)

        # ================================================================
        # Step 1: Input filter
        # Keep only records with non-empty aux_points, and optionally
        # remove records whose llm_output_renamed contains skip_predicates.
        # ================================================================
        normalized_obj = _normalize_input_object(in_path)
        src_results_all = normalized_obj.get("results", []) or []
        total_results = len(src_results_all)

        from newclid.proof_scout.core.aux_extractor import AuxExtractor

        AuxExtractor()

        src_results_kept = []
        dropped_no_aux = []
        dropped_predicate = []
        for rec in src_results_all:
            # Aux filter: keep only records with non-empty aux_points
            aux_points = rec.get("aux_points") if isinstance(rec, dict) else None
            if not aux_points:
                dropped_no_aux.append(rec)
                continue
            # Predicate filter: skip records whose llm_output_renamed contains skip_predicates
            if self.skip_predicates:
                llm_out = (rec.get("llm_output_renamed") or "").lower()
                matched_preds = [p for p in self.skip_predicates if p in llm_out]
                if matched_preds:
                    dropped_predicate.append(
                        {"record": rec, "matched_predicates": matched_preds}
                    )
                    continue
            src_results_kept.append(rec)

        kept = len(src_results_kept)
        print(
            f"[Step 1: input filter] total={total_results} kept={kept} "
            f"dropped_no_aux={len(dropped_no_aux)} dropped_predicate={len(dropped_predicate)}"
            f"{' skip_predicates=' + str(sorted(self.skip_predicates)) if self.skip_predicates else ''}"
        )

        # Save intermediate: step 1
        if intermediates_dir:
            kept_records = [
                {
                    "problem_id": str(r.get("problem_id", "")),
                    "aux_points": r.get("aux_points", []),
                    "fl_problem": r.get("fl_problem", "")[:200],
                }
                for r in src_results_kept
            ]
            no_aux_records = [
                {
                    "problem_id": str(r.get("problem_id", "")),
                    "aux_points": r.get("aux_points", []),
                    "fl_problem": r.get("fl_problem", "")[:200],
                }
                for r in dropped_no_aux
            ]
            pred_records = [
                {
                    "problem_id": str(d["record"].get("problem_id", "")),
                    "matched_predicates": d["matched_predicates"],
                }
                for d in dropped_predicate
            ]
            with open(
                intermediates_dir / "step1_input_filter.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "total": total_results,
                        "kept": kept,
                        "dropped_no_aux": len(dropped_no_aux),
                        "dropped_predicate": len(dropped_predicate),
                        "skip_predicates": sorted(self.skip_predicates)
                        if self.skip_predicates
                        else [],
                        "kept_records": kept_records,
                        "dropped_no_aux_records": no_aux_records,
                        "dropped_predicate_records": pred_records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if kept <= 0:
            return {
                "total": total_results,
                "kept": 0,
                "dropped": total_results - kept,
                "rendered": 0,
                "skipped": 0,
                "failed": 0,
            }

        # Update normalized_obj with filtered results for downstream steps
        aux_obj = dict(normalized_obj)
        aux_obj["results"] = src_results_kept
        src_results_for_prune = src_results_kept

        pids = [
            str(r.get("problem_id"))
            for r in src_results_for_prune
            if isinstance(r, dict) and r.get("problem_id") is not None
        ]

        # Build pid → source record index for later steps
        idx_src: Dict[str, Any] = {}
        for r in src_results_for_prune:
            if isinstance(r, dict) and r.get("problem_id") is not None:
                idx_src[str(r["problem_id"])] = r

        # ================================================================
        # Step 2: Graph prune
        # Build SingleProofGraph from each record, then prune with GraphPruner.
        # Code: _worker_prune() → SingleProofGraph.build_from_result_record()
        #                        → GraphPruner().prune_proof_graph()
        # Output: pruned_map[pid] = [rendered_dict, ...] (list of sub-graphs)
        # ================================================================
        pruned_map: Dict[str, List[Dict[str, Any]]] = {}
        mw = int(self.max_workers or 0)
        if mw > 1:
            print(f"[Step 2: graph prune] parallel pruning with max_workers={mw}")
            with ProcessPoolExecutor(max_workers=mw) as ex:
                futs = [
                    ex.submit(_worker_prune, rec)
                    for rec in src_results_for_prune
                    if isinstance(rec, dict) and rec.get("problem_id") is not None
                ]
                for pid, rendered_list in (f.result() for f in as_completed(futs)):
                    if rendered_list:
                        pruned_map[pid] = rendered_list
        else:
            print("[Step 2: graph prune] sequential pruning")
            for rec in src_results_for_prune:
                if not isinstance(rec, dict) or rec.get("problem_id") is None:
                    continue
                pid, rendered_list = _worker_prune(rec)
                if rendered_list:
                    pruned_map[pid] = rendered_list

        # Build flat rendered lookup: sub_pid → rendered (for rendering steps)
        # Also build sub_pid → original pid mapping
        flat_rendered_map: Dict[str, Dict[str, Any]] = {}
        sub_pid_to_orig: Dict[str, str] = {}
        for pid, rendered_list in pruned_map.items():
            if len(rendered_list) == 1:
                flat_rendered_map[pid] = rendered_list[0]
                sub_pid_to_orig[pid] = pid
            else:
                for sub_idx, rendered in enumerate(rendered_list):
                    sub_pid = f"{pid}_{sub_idx}"
                    flat_rendered_map[sub_pid] = rendered
                    sub_pid_to_orig[sub_pid] = pid

        # Save intermediate: step 1c
        if intermediates_dir:
            pruned_count = len(pruned_map)
            total_subgraphs = sum(len(rl) for rl in pruned_map.values())
            failed_count = len(pids) - pruned_count
            failed_pids = [p for p in pids if p not in pruned_map]
            pruned_records = []
            failed_records = []
            for pid, rendered_list in pruned_map.items():
                for sub_idx, rendered in enumerate(rendered_list):
                    sub_pid = f"{pid}_{sub_idx}" if len(rendered_list) > 1 else pid
                    nodes = rendered.get("nodes", [])
                    edges = rendered.get("edges", [])
                    fact_nodes = [n for n in nodes if n.get("type") == "fact"]
                    thm_nodes = [n for n in nodes if n.get("type") == "theorem"]
                    pruned_records.append(
                        {
                            "problem_id": sub_pid,
                            "orig_pid": pid,
                            "n_subgraphs": len(rendered_list),
                            "n_nodes": len(nodes),
                            "n_edges": len(edges),
                            "n_fact_nodes": len(fact_nodes),
                            "n_theorem_nodes": len(thm_nodes),
                        }
                    )
            # Collect failed records with details
            for pid in failed_pids:
                src_rec = idx_src.get(pid, {})
                fl_problem = src_rec.get("fl_problem", "")
                failed_records.append(
                    {
                        "problem_id": pid,
                        "fl_problem": fl_problem[:300],
                        "aux_points": src_rec.get("aux_points", []),
                        "reason": "Graph pruning failed (likely no valid proof structure)",
                    }
                )
            with open(
                intermediates_dir / "step2_graph_prune.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "input": len(pids),
                        "pruned": pruned_count,
                        "total_subgraphs": total_subgraphs,
                        "failed": failed_count,
                        "pruned_records": pruned_records,
                        "failed_pids": failed_pids,
                        "failed_records": failed_records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        print(
            f"[Step 2: graph prune] input={len(pids)} pruned={len(pruned_map)} subgraphs={sum(len(rl) for rl in pruned_map.values())} failed={len(pids) - len(pruned_map)}"
        )

        # ================================================================
        # Step 3: Proposition extraction
        # From each pruned graph, extract premise facts (in-degree=0) and
        # conclusion fact (out-degree=0, must be exactly 1).
        # Remove premises that reference aux points.
        # For split sub-graphs, iterate with pid_0, pid_1 suffixes.
        # Code: _build_proposition_no_aux(rendered, aux_points)
        #       → _to_rule_text(premises, conclusion)
        # Output: out_results[i]["proposition_no_aux"], out_results[i]["proposition_rule"]
        # ================================================================
        out_results: List[Dict[str, Any]] = []
        for pid in pids:
            rendered_list = pruned_map.get(pid)
            if not rendered_list:
                continue
            for sub_idx, rendered in enumerate(rendered_list):
                # pid 命名：多子图用后缀，单子图保持原 pid
                if len(rendered_list) > 1:
                    sub_pid = f"{pid}_{sub_idx}"
                else:
                    sub_pid = pid
                base: Dict[str, Any] = {"problem_id": sub_pid, "rendered": rendered}
                src = idx_src.get(pid, {})  # 用原始 pid 查找源数据
                for k in ("aux_points", "point_lines", "points", "point_rely_on"):
                    if k in src:
                        base[k] = src[k]
                if "aux_points" not in base and isinstance(src.get("aux_points"), list):
                    base["aux_points"] = src["aux_points"]
                # 保留原始 llm_input_renamed / llm_output_renamed（不修改 goal）
                base["llm_input_renamed"] = src.get("llm_input_renamed", "")
                base["llm_output_renamed"] = src.get("llm_output_renamed", "")
                base["point_coords"] = src.get("point_coords", {})
                base["seed"] = src.get("seed")
                prop = _build_proposition_no_aux(
                    rendered, list(base.get("aux_points", []) or [])
                )
                if prop:
                    base["proposition_no_aux"] = prop
                    try:
                        rule_text, rename_map = _to_rule_text(
                            prop.get("premises", []) or [], prop.get("conclusion", "")
                        )
                        base["proposition_rule"] = rule_text
                        base["rename_map"] = rename_map
                        base["proposition_no_aux"]["rule_text"] = rule_text
                    except Exception:
                        base["proposition_no_aux"].setdefault(
                            "rule_text", prop.get("text", "")
                        )
                out_results.append(base)

        # Save intermediate: step 1d proposition extraction
        if intermediates_dir:
            has_prop = sum(1 for r in out_results if r.get("proposition_no_aux"))
            has_rule = sum(1 for r in out_results if r.get("proposition_rule"))
            no_fact = 0
            multi_concl = 0
            # Build detailed records for successes and failures
            success_records = []
            fail_no_fact_records = []
            fail_multi_concl_records = []
            out_pid_set = {
                r.get("problem_id") for r in out_results if r.get("proposition_no_aux")
            }
            # Iterate over all sub_pids in flat_rendered_map
            for sub_pid, rendered in flat_rendered_map.items():
                nodes = rendered.get("nodes", [])
                edges = rendered.get("edges", [])
                fact_nodes_d = [n for n in nodes if n.get("type") == "fact"]
                if sub_pid in out_pid_set:
                    rec_d = next(
                        (r for r in out_results if r.get("problem_id") == sub_pid), {}
                    )
                    prop = rec_d.get("proposition_no_aux", {})
                    success_records.append(
                        {
                            "problem_id": sub_pid,
                            "aux_points": rec_d.get("aux_points", []),
                            "rule": rec_d.get("proposition_rule", ""),
                            "premises": prop.get("premises", []),
                            "conclusion": prop.get("conclusion", ""),
                        }
                    )
                elif not fact_nodes_d:
                    no_fact += 1
                    thm_labels = [
                        n.get("label", "") for n in nodes if n.get("type") == "theorem"
                    ]
                    fail_no_fact_records.append(
                        {
                            "problem_id": sub_pid,
                            "n_nodes": len(nodes),
                            "theorem_labels": thm_labels[:10],
                        }
                    )
                else:
                    indeg_c = {n.get("idx"): 0 for n in nodes}
                    outdeg_c = {n.get("idx"): 0 for n in nodes}
                    for u, v in edges:
                        outdeg_c[u] = outdeg_c.get(u, 0) + 1
                        indeg_c[v] = indeg_c.get(v, 0) + 1
                    concl_ids = [
                        n.get("idx")
                        for n in fact_nodes_d
                        if outdeg_c.get(n.get("idx"), 0) == 0
                    ]
                    concl_labels = (
                        [
                            str(fact_nodes_d[i].get("label", ""))
                            for i, n in enumerate(fact_nodes_d)
                            if n.get("idx") in concl_ids
                        ]
                        if len(concl_ids) != 1
                        else []
                    )
                    if len(concl_ids) != 1:
                        multi_concl += 1
                        fail_multi_concl_records.append(
                            {
                                "problem_id": sub_pid,
                                "n_conclusions": len(concl_ids),
                                "conclusion_labels": concl_labels[:5],
                            }
                        )
            with open(
                intermediates_dir / "step3_propositions.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "input": len(out_results),
                        "has_proposition": has_prop,
                        "has_rule": has_rule,
                        "fail_no_fact_nodes": no_fact,
                        "fail_multi_conclusion": multi_concl,
                        "success_records": success_records,
                        "fail_no_fact_records": fail_no_fact_records,
                        "fail_multi_concl_records": fail_multi_concl_records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        out_obj = {k: v for k, v in normalized_obj.items() if k != "results"}
        out_obj["results"] = out_results

        pruned_json = in_path.with_name(in_path.stem + "_pruned.json")
        with open(pruned_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[filter+prune] pruned json written: {pruned_json}")

        # ================================================================
        # Step 3.5a: Simplify Trivial Predicates
        # ================================================================
        simplified_count = 0
        simplification_records: List[Dict[str, Any]] = []

        for rec in out_results:
            rule = rec.get("proposition_rule")
            if not rule:
                continue

            simplified_rule, was_simplified = self._simplify_trivial_predicates(rule)
            if was_simplified:
                simplified_count += 1
                simplification_records.append(
                    {
                        "pid": rec.get("problem_id"),
                        "original": rule,
                        "simplified": simplified_rule,
                    }
                )
                rec["proposition_rule"] = simplified_rule
                rec["was_simplified"] = True

        print(
            f"[Step 3.5a: simplify predicates] simplified={simplified_count}/{len(out_results)}"
        )

        if intermediates_dir and simplification_records:
            with open(
                intermediates_dir / "step3.5a_simplified_predicates.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(simplification_records, f, ensure_ascii=False, indent=2)

        # ================================================================
        # Step 3.5b: Extract and Map eqpoint
        # ================================================================
        eqpoint_count = 0
        eqpoint_records: List[Dict[str, Any]] = []

        for rec in out_results:
            llm_output = rec.get("llm_output_renamed", "")
            rename_map = rec.get("rename_map", {})
            rule = rec.get("proposition_rule")

            if not llm_output or not rename_map or not rule:
                continue

            # Extract eqpoint from proof
            eqpoint_original = self._extract_eqpoint_from_proof(llm_output)
            if not eqpoint_original:
                continue

            # Map to rule point names
            eqpoint_mapped = self._map_eqpoint_to_rule(eqpoint_original, rename_map)
            if not eqpoint_mapped:
                continue

            # Generate merged rule
            merged_rule = self._generate_merged_rule(rule, eqpoint_mapped)

            rec["eqpoint_original"] = eqpoint_original
            rec["eqpoint_mapped"] = eqpoint_mapped
            rec["proposition_rule_merged"] = merged_rule

            eqpoint_count += 1
            eqpoint_records.append(
                {
                    "pid": rec.get("problem_id"),
                    "eqpoint_original": eqpoint_original,
                    "eqpoint_mapped": eqpoint_mapped,
                    "rule": rule,
                    "rule_merged": merged_rule,
                }
            )

        print(
            f"[Step 3.5b: eqpoint analysis] rules_with_eqpoint={eqpoint_count}/{len(out_results)}"
        )

        if intermediates_dir and eqpoint_records:
            with open(
                intermediates_dir / "step3.5b_eqpoint_analysis.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(eqpoint_records, f, ensure_ascii=False, indent=2)

        # ================================================================
        # Step 4: Normalization
        # ================================================================
        normalized_rules, skipped_rules, skipped_entries = self._normalize_rules(
            out_results
        )

        # Save intermediate: step 4
        if intermediates_dir:
            all_rules_raw = [
                r.get("proposition_rule", "")
                for r in out_results
                if r.get("proposition_rule")
            ]

            # Normalized rules JSON
            normalized_rules_json = []
            for item in normalized_rules:
                normalized_rules_json.append(
                    {
                        "pid": item["pid"],
                        "original": item["original"],
                        "normalized": item["normalized"],
                        "signature": item.get("signature", ""),
                    }
                )
            with open(
                intermediates_dir / "step4_normalized_rules.json", "w", encoding="utf-8"
            ) as f:
                json.dump(normalized_rules_json, f, ensure_ascii=False, indent=2)

            # Signature distribution
            sig_dist: Dict[str, int] = {}
            for item in normalized_rules:
                sig = item.get("signature", "")
                sig_dist[sig] = sig_dist.get(sig, 0) + 1
            with open(
                intermediates_dir / "step4_signature_distribution.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(sig_dist, f, indent=2)

            # Predicate normalization samples
            pred_samples: Dict[str, List[Dict[str, Any]]] = {}
            sample_limit = 5
            for rec in out_results:
                rule = rec.get("proposition_rule")
                if not rule:
                    continue
                left_parsed, right_parsed = _split_rule_text(rule)
                for pred_name, pred_args in left_parsed:
                    if (
                        not pred_name
                        or pred_name in pred_samples
                        and len(pred_samples[pred_name]) >= sample_limit
                    ):
                        continue
                    _, norm_args = self._normalize_clause_by_symmetry(
                        pred_name, list(pred_args)
                    )
                    if list(pred_args) != list(norm_args):
                        pred_samples.setdefault(pred_name, []).append(
                            {"input": list(pred_args), "output": list(norm_args)}
                        )
                rn, ra = right_parsed
                if rn and (
                    rn not in pred_samples
                    or len(pred_samples.get(rn, [])) < sample_limit
                ):
                    _, norm_args = self._normalize_clause_by_symmetry(rn, list(ra))
                    if list(ra) != list(norm_args):
                        pred_samples.setdefault(rn, []).append(
                            {"input": list(ra), "output": list(norm_args)}
                        )
            with open(
                intermediates_dir / "step4_predicate_normalization_samples.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(pred_samples, f, ensure_ascii=False, indent=2)

        # ================================================================
        # Step 5: Deduplication
        # ================================================================
        rule_entries, norm_dup_map = self._dedup_rules(normalized_rules)

        # Save intermediate: step 5
        if intermediates_dir:
            # Build dedup groups for JSON output
            dedup_groups = []
            for norm_hash, lst in norm_dup_map.items():
                # Find the normalized form from the first entry
                norm_form = ""
                for e in rule_entries:
                    h = hashlib.sha256(
                        e.get("norm_rule", "").encode("utf-8")
                    ).hexdigest()
                    if h == norm_hash:
                        norm_form = e.get("norm_rule", "")
                        break
                dedup_groups.append(
                    {
                        "hash": norm_hash[:16],
                        "normalized": norm_form,
                        "count": len(lst),
                        "rules": [{"pid": pid, "original": rule} for pid, rule in lst],
                    }
                )
            with open(
                intermediates_dir / "step5_dedup.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "input": len(normalized_rules),
                        "unique": len(rule_entries),
                        "duplicates": len(normalized_rules) - len(rule_entries),
                        "dedup_groups": dedup_groups,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # ================================================================
        # Step 6: Rule dump
        # ================================================================
        n_rules, n_skipped_pred, rule_entries = self._dump_rules(
            rule_entries, norm_dup_map, skipped_entries, pruned_json
        )

        # Save intermediate: step 6
        if intermediates_dir:
            all_rules_raw = [
                r.get("proposition_rule", "")
                for r in out_results
                if r.get("proposition_rule")
            ]
            skipped_entries_json = []
            for rec in out_results:
                rule = rec.get("proposition_rule")
                if not rule:
                    continue
                left_parsed, right_parsed = _split_rule_text(rule)
                left_points = _collect_point_set(left_parsed)
                right_points = set(_collect_point_list(right_parsed[1]))
                missing = right_points - left_points
                if missing:
                    skipped_entries_json.append(
                        {
                            "pid": str(rec.get("problem_id", "")),
                            "missing_points": sorted(missing),
                            "rule": rule,
                        }
                    )

            with open(
                intermediates_dir / "step6_rules_stats.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "input_rules_raw": len(all_rules_raw),
                        "output_rules_deduped": n_rules,
                        "skipped_rules_missing_points": skipped_rules,
                        "skipped_rules_predicates": n_skipped_pred,
                        "rule_skip_predicates": sorted(self.rule_skip_predicates)
                        if self.rule_skip_predicates
                        else [],
                        "skipped_entries": skipped_entries_json,
                        "entries": [
                            {
                                "rid": e.get("rid"),
                                "pid": e.get("pid"),
                                "seed": e.get("seed"),
                                "rule": e.get("norm_rule"),
                                "rule_original": e.get("rule"),
                                "llm_input_renamed": e.get("llm_input_renamed", ""),
                                "llm_output_renamed": e.get("llm_output_renamed", ""),
                                "point_coords": e.get("point_coords", {}),
                            }
                            for e in rule_entries
                        ],
                    },
                    f,
                    indent=2,
                )

        # 合并图渲染输出目录
        combo_stem = in_path.stem + "_aux"
        out_base_dir = Path(output_dir) / (combo_stem + "_combo")
        out_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[filter+prune] writing combined images to: {out_base_dir}")

        # 渲染：按 pid 的图片（可选，默认关闭）
        pid_done = pid_skipped = pid_failed = 0
        if self.keep_pid_images:
            all_sub_pids = sorted(flat_rendered_map.keys())
            total = len(all_sub_pids)
            print(f"[filter+prune] start rendering by pid: total={total}")
            skipped_render_entries: List[Tuple[str, str]] = []  # (pid, reason)

            def submit_all(executor=None):
                tasks = []
                for sub_pid in all_sub_pids:
                    orig_pid = sub_pid_to_orig.get(sub_pid, sub_pid)
                    out_png = out_base_dir / f"proof_{sub_pid}.png"
                    if out_png.exists() and not self.overwrite:
                        if self.print_render_skipped:
                            skipped_render_entries.append(
                                (sub_pid, "exists_overwrite_false")
                            )
                        yield (sub_pid, "skipped")
                        continue
                    rendered = flat_rendered_map.get(sub_pid)
                    if not isinstance(rendered, dict):
                        if self.print_render_skipped:
                            skipped_render_entries.append(
                                (sub_pid, "no_pruned_rendered")
                            )
                        yield (sub_pid, "skipped")
                        continue
                    rec = idx_src.get(orig_pid)
                    if rec is None:
                        if self.print_render_skipped:
                            skipped_render_entries.append((sub_pid, "no_source_record"))
                        yield (sub_pid, "skipped")
                        continue
                    if executor is None:
                        yield _worker_render_combined(
                            rec,
                            rendered,
                            str(out_png),
                            self.label_mode,
                            self.figsize_single,
                            self.ranksep,
                            self.nodesep,
                            self.font_size,
                        )
                    else:
                        tasks.append(
                            executor.submit(
                                _worker_render_combined,
                                rec,
                                rendered,
                                str(out_png),
                                self.label_mode,
                                self.figsize_single,
                                self.ranksep,
                                self.nodesep,
                                self.font_size,
                            )
                        )
                if executor is not None:
                    for fut in as_completed(tasks):
                        yield fut.result()

            mw = int(self.max_workers or 0)
            if mw > 1:
                print(f"[filter+prune] parallel rendering by pid with max_workers={mw}")
                with ProcessPoolExecutor(max_workers=mw) as ex:
                    for idx, (pid, st) in enumerate(submit_all(ex), start=1):
                        if st == "ok":
                            pid_done += 1
                        elif st == "skipped":
                            pid_skipped += 1
                        else:
                            pid_failed += 1
                        if self.progress_every and (idx % self.progress_every == 0):
                            print(
                                f"[filter+prune] (pid) {idx}/{total} done={pid_done} skipped={pid_skipped} failed={pid_failed}"
                            )
            else:
                print("[filter+prune] sequential rendering by pid")
                for idx, (pid, st) in enumerate(submit_all(None), start=1):
                    if st == "ok":
                        pid_done += 1
                    elif st == "skipped":
                        pid_skipped += 1
                    else:
                        pid_failed += 1
                    if self.progress_every and (idx % self.progress_every == 0):
                        print(
                            f"[filter+prune] (pid) {idx}/{total} done={pid_done} skipped={pid_skipped} failed={pid_failed}"
                        )

            if self.print_render_skipped and (pid_skipped > 0):
                skipped_render_path = out_base_dir / "render_skipped.txt"
                with open(skipped_render_path, "w", encoding="utf-8") as f:
                    for spid, reason in skipped_render_entries:
                        f.write(f"pid={spid}\treason={reason}\n")
                print(f"[render] skipped list saved to {skipped_render_path}")

        # 渲染：按规则编号（默认开启）
        rid_done = rid_skipped = rid_failed = 0
        rid_total = len(rule_entries)
        if self.render_by_rule and rid_total > 0:
            print(f"[filter+prune] start rendering by rule: total={rid_total}")
            rid_skipped_entries: List[Tuple[str, str]] = []  # (rid, reason)

            def submit_rules_all(executor=None):
                tasks = []
                for e in rule_entries:
                    rid = str(e.get("rid"))
                    pid = str(e.get("pid"))
                    orig_pid = sub_pid_to_orig.get(pid, pid)
                    out_png = out_base_dir / f"{rid}.png"
                    if out_png.exists() and not self.overwrite:
                        if self.print_render_skipped:
                            rid_skipped_entries.append((rid, "exists_overwrite_false"))
                        yield (rid, "skipped")
                        continue
                    rendered = flat_rendered_map.get(pid)
                    if not isinstance(rendered, dict):
                        if self.print_render_skipped:
                            rid_skipped_entries.append((rid, "no_pruned_rendered"))
                        yield (rid, "skipped")
                        continue
                    rec = idx_src.get(orig_pid)
                    if rec is None:
                        if self.print_render_skipped:
                            rid_skipped_entries.append((rid, "no_source_record"))
                        yield (rid, "skipped")
                        continue
                    if executor is None:
                        yield _worker_render_combined(
                            rec,
                            rendered,
                            str(out_png),
                            self.label_mode,
                            self.figsize_single,
                            self.ranksep,
                            self.nodesep,
                            self.font_size,
                        )
                    else:
                        tasks.append(
                            executor.submit(
                                _worker_render_combined,
                                rec,
                                rendered,
                                str(out_png),
                                self.label_mode,
                                self.figsize_single,
                                self.ranksep,
                                self.nodesep,
                                self.font_size,
                            )
                        )
                if executor is not None:
                    for fut in as_completed(tasks):
                        yield fut.result()

            mw = int(self.max_workers or 0)
            if mw > 1:
                print(
                    f"[filter+prune] parallel rendering by rule with max_workers={mw}"
                )
                with ProcessPoolExecutor(max_workers=mw) as ex:
                    for idx, (rid, st) in enumerate(submit_rules_all(ex), start=1):
                        if st == "ok":
                            rid_done += 1
                        elif st == "skipped":
                            rid_skipped += 1
                        else:
                            rid_failed += 1
                        if self.progress_every and (idx % self.progress_every == 0):
                            print(
                                f"[filter+prune] (rule) {idx}/{rid_total} done={rid_done} skipped={rid_skipped} failed={rid_failed}"
                            )
            else:
                print("[filter+prune] sequential rendering by rule")
                for idx, (rid, st) in enumerate(submit_rules_all(None), start=1):
                    if st == "ok":
                        rid_done += 1
                    elif st == "skipped":
                        rid_skipped += 1
                    else:
                        rid_failed += 1
                    if self.progress_every and (idx % self.progress_every == 0):
                        print(
                            f"[filter+prune] (rule) {idx}/{rid_total} done={rid_done} skipped={rid_skipped} failed={rid_failed}"
                        )

            # 保存 rid_map.txt（rid -> pid 与规则文本）
            rid_map_path = out_base_dir / "rid_map.txt"
            with open(rid_map_path, "w", encoding="utf-8") as f:
                for e in rule_entries:
                    rid = str(e.get("rid"))
                    pid = str(e.get("pid"))
                    rule = str(e.get("rule"))
                    f.write(f"{rid} pid={pid} rule={rule}\n")
            print(f"[rules] rid_map saved to {rid_map_path}")

            if self.print_render_skipped and (rid_skipped > 0):
                rid_skipped_path = out_base_dir / "rid_render_skipped.txt"
                with open(rid_skipped_path, "w", encoding="utf-8") as f:
                    for rrid, reason in rid_skipped_entries:
                        f.write(f"rid={rrid}\treason={reason}\n")
                print(f"[render] rid skipped list saved to {rid_skipped_path}")

        # 总结输出
        print("-" * 30)
        print(f"Total problems: {len(pids)}")
        if self.keep_pid_images:
            print(
                f"Combined images (by pid): {pid_done}, Skipped: {pid_skipped}, Failed: {pid_failed}"
            )
        print(
            f"Combined images (by rule): {rid_done}, Skipped: {rid_skipped}, Failed: {rid_failed}"
        )
        print(f"Output directory: {out_base_dir}")
        print(f"JSON saved: {pruned_json}")

        # 导出 pid -> 输入题目文本 的映射（JSONL），便于按 pid 检索题目
        if self.save_pid_inputs and pids:
            pid_inputs_path = out_base_dir / "pid_inputs.jsonl"
            with open(pid_inputs_path, "w", encoding="utf-8") as f:
                for pid in pids:
                    src_rec = idx_src.get(pid, {})
                    llm_input = (
                        src_rec.get("llm_input_renamed", "")
                        if isinstance(src_rec, dict)
                        else ""
                    )
                    problem_body = (
                        _extract_tag_content(llm_input, "problem") or llm_input
                    )
                    problem_text = _norm_text_basic(problem_body)
                    json.dump({"pid": pid, "text": problem_text}, f, ensure_ascii=False)
                    f.write("\n")
            print(f"[pid-inputs] mapping saved to {pid_inputs_path}")

        return {
            "total": total_results,
            "kept": kept,
            "dropped": total_results - kept,
            "rendered": (pid_done if self.keep_pid_images else 0) + rid_done,
            "skipped": (pid_skipped if self.keep_pid_images else 0) + rid_skipped,
            "failed": (pid_failed if self.keep_pid_images else 0) + rid_failed,
            "rules": n_rules,
            "skipped_rules": n_skipped_pred,
            "json": str(pruned_json),
            "images_dir": str(out_base_dir),
        }

    # ================================================================
    # Part-based API: run_part1_filter() and run_part2_extract()
    # ================================================================

    def run_part1_filter(
        self,
        input_jsonl: Path,
        output_jsonl: Path,
    ) -> Dict[str, Any]:
        """Execute Step 1 (input filter) only.

        Reads input_jsonl (JSONL format), filters records:
        - Drops records with empty aux_points
        - Drops records whose llm_output_renamed contains skip_predicates

        Writes passing records to output_jsonl in JSONL format (one JSON per line).
        Returns statistics dict.
        """
        input_jsonl = Path(input_jsonl)
        output_jsonl = Path(output_jsonl)
        if not input_jsonl.exists():
            raise FileNotFoundError(f"input not found: {input_jsonl}")

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        base = _sanitize_basename(input_jsonl)
        total = 0
        n_kept = 0
        n_no_aux = 0
        n_pred = 0

        with (
            open(input_jsonl, "r", encoding="utf-8") as fin,
            open(output_jsonl, "w", encoding="utf-8") as fout,
        ):
            for idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_no_aux += 1  # treat malformed as dropped
                    continue
                if not isinstance(rec, dict):
                    n_no_aux += 1
                    continue
                # Convert to internal format (adds problem_id, aux_points, etc.)
                if _has_llm_format(rec):
                    rec = _convert_llm_record(rec, base, idx)
                # Aux filter
                aux_points = rec.get("aux_points")
                if not aux_points:
                    n_no_aux += 1
                    continue
                # Predicate filter
                if self.skip_predicates:
                    llm_out = (rec.get("llm_output_renamed") or "").lower()
                    if any(p in llm_out for p in self.skip_predicates):
                        n_pred += 1
                        continue
                json.dump(rec, fout, ensure_ascii=False)
                fout.write("\n")
                n_kept += 1

        print(
            f"[Part 1: input filter] total={total} kept={n_kept} "
            f"dropped_no_aux={n_no_aux} dropped_predicate={n_pred}"
            + (
                f" skip_predicates={sorted(self.skip_predicates)}"
                if self.skip_predicates
                else ""
            )
        )
        return {
            "total": total,
            "kept": n_kept,
            "dropped_no_aux": n_no_aux,
            "dropped_predicate": n_pred,
            "output_jsonl": str(output_jsonl),
        }

    def run_part2_extract(
        self,
        input_jsonl: Path,
        output_dir: Path,
        save_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Execute Steps 2-6 (graph prune → rule dump), skipping Step 1.

        input_jsonl can be Part 1's filtered.jsonl or any raw JSONL file.
        Equivalent to run() but reads pre-filtered JSONL instead of running Step 1.

        Returns statistics dict with 'rules_file' and 'source_data_file' keys.
        """
        input_jsonl = Path(input_jsonl)
        output_dir = Path(output_dir)
        if not input_jsonl.exists():
            raise FileNotFoundError(f"input not found: {input_jsonl}")

        # Setup intermediates directory
        intermediates_dir = None
        if save_intermediates:
            intermediates_dir = output_dir / "intermediates"
            intermediates_dir.mkdir(parents=True, exist_ok=True)

        # Read JSONL records
        base = _sanitize_basename(input_jsonl)
        src_results_kept = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                # Ensure record is in internal format
                if _has_llm_format(rec) and "proof" not in rec:
                    rec = _convert_llm_record(rec, base, idx)
                src_results_kept.append(rec)

        kept = len(src_results_kept)
        print(f"[Part 2: extract] loaded {kept} records from {input_jsonl}")

        if kept <= 0:
            return {"rules": 0, "kept": 0, "rules_file": None, "source_data_file": None}

        pids = [
            str(r.get("problem_id"))
            for r in src_results_kept
            if isinstance(r, dict) and r.get("problem_id") is not None
        ]
        idx_src: Dict[str, Any] = {}
        for r in src_results_kept:
            if isinstance(r, dict) and r.get("problem_id") is not None:
                idx_src[str(r["problem_id"])] = r

        # ================================================================
        # Step 2: Graph prune
        # ================================================================
        pruned_map: Dict[str, List[Dict[str, Any]]] = {}
        mw = int(self.max_workers or 0)
        if mw > 1:
            print(
                f"[Part 2 / Step 2: graph prune] parallel pruning with max_workers={mw}"
            )
            with ProcessPoolExecutor(max_workers=mw) as ex:
                futs = [
                    ex.submit(_worker_prune, rec)
                    for rec in src_results_kept
                    if isinstance(rec, dict) and rec.get("problem_id") is not None
                ]
                for pid, rendered_list in (f.result() for f in as_completed(futs)):
                    if rendered_list:
                        pruned_map[pid] = rendered_list
        else:
            print("[Part 2 / Step 2: graph prune] sequential pruning")
            for rec in src_results_kept:
                if not isinstance(rec, dict) or rec.get("problem_id") is None:
                    continue
                pid, rendered_list = _worker_prune(rec)
                if rendered_list:
                    pruned_map[pid] = rendered_list

        flat_rendered_map: Dict[str, Dict[str, Any]] = {}
        sub_pid_to_orig: Dict[str, str] = {}
        for pid, rendered_list in pruned_map.items():
            if len(rendered_list) == 1:
                flat_rendered_map[pid] = rendered_list[0]
                sub_pid_to_orig[pid] = pid
            else:
                for sub_idx, rendered in enumerate(rendered_list):
                    sub_pid = f"{pid}_{sub_idx}"
                    flat_rendered_map[sub_pid] = rendered
                    sub_pid_to_orig[sub_pid] = pid

        print(
            f"[Part 2 / Step 2: graph prune] input={len(pids)} "
            f"pruned={len(pruned_map)} "
            f"subgraphs={sum(len(rl) for rl in pruned_map.values())} "
            f"failed={len(pids) - len(pruned_map)}"
        )

        # ================================================================
        # Step 3: Proposition extraction
        # ================================================================
        out_results: List[Dict[str, Any]] = []
        for pid in pids:
            rendered_list = pruned_map.get(pid)
            if not rendered_list:
                continue
            for sub_idx, rendered in enumerate(rendered_list):
                sub_pid = f"{pid}_{sub_idx}" if len(rendered_list) > 1 else pid
                base_rec: Dict[str, Any] = {"problem_id": sub_pid, "rendered": rendered}
                src = idx_src.get(pid, {})
                for k in ("aux_points", "point_lines", "points", "point_rely_on"):
                    if k in src:
                        base_rec[k] = src[k]
                if "aux_points" not in base_rec and isinstance(
                    src.get("aux_points"), list
                ):
                    base_rec["aux_points"] = src["aux_points"]
                base_rec["llm_input_renamed"] = src.get("llm_input_renamed", "")
                base_rec["llm_output_renamed"] = src.get("llm_output_renamed", "")
                base_rec["point_coords"] = src.get("point_coords", {})
                base_rec["seed"] = src.get("seed")
                prop = _build_proposition_no_aux(
                    rendered, list(base_rec.get("aux_points", []) or [])
                )
                if prop:
                    base_rec["proposition_no_aux"] = prop
                    try:
                        rule_text, rename_map = _to_rule_text(
                            prop.get("premises", []) or [], prop.get("conclusion", "")
                        )
                        base_rec["proposition_rule"] = rule_text
                        base_rec["rename_map"] = rename_map
                        base_rec["proposition_no_aux"]["rule_text"] = rule_text
                    except Exception:
                        base_rec["proposition_no_aux"].setdefault(
                            "rule_text", prop.get("text", "")
                        )
                out_results.append(base_rec)

        # Step 3.5a: Simplify trivial predicates
        simplified_count = 0
        for rec in out_results:
            rule = rec.get("proposition_rule")
            if not rule:
                continue
            simplified_rule, was_simplified = self._simplify_trivial_predicates(rule)
            if was_simplified:
                simplified_count += 1
                rec["proposition_rule"] = simplified_rule
                rec["was_simplified"] = True
        print(
            f"[Part 2 / Step 3.5a: simplify predicates] simplified={simplified_count}/{len(out_results)}"
        )

        # Step 3.5b: Extract eqpoint
        eqpoint_count = 0
        for rec in out_results:
            llm_output = rec.get("llm_output_renamed", "")
            rename_map = rec.get("rename_map", {})
            rule = rec.get("proposition_rule")
            if not llm_output or not rename_map or not rule:
                continue
            eqpoint_original = self._extract_eqpoint_from_proof(llm_output)
            if not eqpoint_original:
                continue
            eqpoint_mapped = self._map_eqpoint_to_rule(eqpoint_original, rename_map)
            if not eqpoint_mapped:
                continue
            merged_rule = self._generate_merged_rule(rule, eqpoint_mapped)
            rec["eqpoint_original"] = eqpoint_original
            rec["eqpoint_mapped"] = eqpoint_mapped
            rec["proposition_rule_merged"] = merged_rule
            eqpoint_count += 1
        print(
            f"[Part 2 / Step 3.5b: eqpoint analysis] rules_with_eqpoint={eqpoint_count}/{len(out_results)}"
        )

        # Write pruned json (needed by _dump_rules)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_obj: Dict[str, Any] = {"results": out_results}
        pruned_json = output_dir / (input_jsonl.stem + "_pruned.json")
        with open(pruned_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        # ================================================================
        # Step 4: Normalization
        # ================================================================
        normalized_rules, skipped_rules, skipped_entries = self._normalize_rules(
            out_results
        )

        # ================================================================
        # Step 5: Deduplication
        # ================================================================
        rule_entries, norm_dup_map = self._dedup_rules(normalized_rules)

        # ================================================================
        # Step 6: Rule dump
        # ================================================================
        n_rules, n_skipped_pred, rule_entries = self._dump_rules(
            rule_entries, norm_dup_map, skipped_entries, pruned_json
        )

        # Determine output paths (mirrors _dump_rules naming convention)
        rules_file = pruned_json.with_name(pruned_json.stem + "_rules.txt")
        source_data_file = None
        if save_intermediates and intermediates_dir:
            source_data_file = intermediates_dir / "step6_rules_stats.json"
            # Save step6 stats for Part 4 consumption
            with open(source_data_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input_rules_raw": len(normalized_rules),
                        "output_rules_deduped": n_rules,
                        "skipped_rules_missing_points": skipped_rules,
                        "skipped_rules_predicates": n_skipped_pred,
                        "rule_skip_predicates": sorted(self.rule_skip_predicates)
                        if self.rule_skip_predicates
                        else [],
                        "entries": [
                            {
                                "rid": e.get("rid"),
                                "pid": e.get("pid"),
                                "seed": e.get("seed"),
                                "rule": e.get("norm_rule"),
                                "rule_original": e.get("rule"),
                                "llm_input_renamed": e.get("llm_input_renamed", ""),
                                "llm_output_renamed": e.get("llm_output_renamed", ""),
                                "point_coords": e.get("point_coords", {}),
                            }
                            for e in rule_entries
                        ],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        print(f"[Part 2: extract] done — {n_rules} rules written to {rules_file}")
        return {
            "kept": kept,
            "rules": n_rules,
            "skipped_rules": n_skipped_pred,
            "rules_file": str(rules_file),
            "source_data_file": str(source_data_file) if source_data_file else None,
            "json": str(pruned_json),
        }

    # ================================================================
    # Streaming mode: chunk-level processing for large datasets
    # ================================================================

    def _filter_chunk(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """Step 1 per-chunk: filter by aux_points and skip_predicates.

        Returns: (kept_records, n_dropped_no_aux, n_dropped_predicate)
        """
        kept: List[Dict[str, Any]] = []
        n_no_aux = 0
        n_pred = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            aux_points = rec.get("aux_points")
            if not aux_points:
                n_no_aux += 1
                continue
            if self.skip_predicates:
                llm_out = (rec.get("llm_output_renamed") or "").lower()
                if any(p in llm_out for p in self.skip_predicates):
                    n_pred += 1
                    continue
            kept.append(rec)
        return kept, n_no_aux, n_pred

    def _extract_propositions_chunk(
        self,
        pruned_map: Dict[str, List[Dict[str, Any]]],
        idx_src: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Steps 3 + 3.5a + 3.5b per-chunk: extract propositions, simplify, eqpoint."""
        out_results: List[Dict[str, Any]] = []
        for pid, rendered_list in pruned_map.items():
            src = idx_src.get(pid, {})
            for sub_idx, rendered in enumerate(rendered_list):
                sub_pid = f"{pid}_{sub_idx}" if len(rendered_list) > 1 else pid
                base_rec: Dict[str, Any] = {"problem_id": sub_pid, "rendered": rendered}
                for k in ("aux_points", "point_lines", "points", "point_rely_on"):
                    if k in src:
                        base_rec[k] = src[k]
                if "aux_points" not in base_rec and isinstance(
                    src.get("aux_points"), list
                ):
                    base_rec["aux_points"] = src["aux_points"]
                base_rec["llm_input_renamed"] = src.get("llm_input_renamed", "")
                base_rec["llm_output_renamed"] = src.get("llm_output_renamed", "")
                base_rec["point_coords"] = src.get("point_coords", {})
                base_rec["seed"] = src.get("seed")

                prop = _build_proposition_no_aux(
                    rendered, list(base_rec.get("aux_points", []) or [])
                )
                if prop:
                    base_rec["proposition_no_aux"] = prop
                    try:
                        rule_text, rename_map = _to_rule_text(
                            prop.get("premises", []) or [], prop.get("conclusion", "")
                        )
                        base_rec["proposition_rule"] = rule_text
                        base_rec["rename_map"] = rename_map
                    except Exception:
                        pass
                out_results.append(base_rec)

        # Step 3.5a: Simplify trivial predicates
        for rec in out_results:
            rule = rec.get("proposition_rule")
            if not rule:
                continue
            simplified_rule, was_simplified = self._simplify_trivial_predicates(rule)
            if was_simplified:
                rec["proposition_rule"] = simplified_rule

        # Step 3.5b: Extract and map eqpoint
        for rec in out_results:
            llm_output = rec.get("llm_output_renamed", "")
            rename_map = rec.get("rename_map", {})
            rule = rec.get("proposition_rule")
            if not llm_output or not rename_map or not rule:
                continue
            eqpoint_original = self._extract_eqpoint_from_proof(llm_output)
            if not eqpoint_original:
                continue
            eqpoint_mapped = self._map_eqpoint_to_rule(eqpoint_original, rename_map)
            if not eqpoint_mapped:
                continue
            merged_rule = self._generate_merged_rule(rule, eqpoint_mapped)
            rec["eqpoint_original"] = eqpoint_original
            rec["eqpoint_mapped"] = eqpoint_mapped
            rec["proposition_rule_merged"] = merged_rule

        return out_results

    def _apply_rule_skip_predicates(
        self, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter entries by rule_skip_predicates (Step 6 predicate filter)."""
        if not self.rule_skip_predicates:
            return entries
        kept: List[Dict[str, Any]] = []
        for e in entries:
            rule_lower = e["norm_rule"].lower()
            if not any(p in rule_lower for p in self.rule_skip_predicates):
                kept.append(e)
        return kept

    def run_streaming(
        self,
        input_json: str | Path,
        output_dir: str | Path,
        *,
        chunk_size: int = 10000,
        inflight_limit: int = 300,
        save_intermediates: bool = False,
        save_pruned_json: bool = False,
        seed_reducer_queue: Optional[Queue] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline in streaming mode for large datasets.

        Processes JSONL input in chunks, using Ray for parallel graph pruning
        and incremental dedup/dump. Memory usage is bounded by chunk_size.

        Args:
            input_json: Path to input JSONL file.
            output_dir: Output directory.
            chunk_size: Records per chunk (default: 10000).
            inflight_limit: Max in-flight Ray tasks (default: 300).
            save_intermediates: Save step-level intermediate files.
            save_pruned_json: Write _pruned.json (default: False for large datasets).
            seed_reducer_queue: Optional queue for incremental Stage 2 reduction.
                When provided, completed seed groups are pushed as
                ("seed_done", seed_val, rules_list) messages.
        """
        import ray

        in_path = Path(input_json)
        if not in_path.exists():
            raise FileNotFoundError(f"input not found: {in_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        intermediates_dir = None
        if save_intermediates:
            intermediates_dir = out_dir / "intermediates"
            intermediates_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Ray
        mw = int(self.max_workers or 100)
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=mw,
                _temp_dir="/tmp/ray",
            )
            logger.info(f"[streaming] Ray initialized with num_cpus={mw}")

        # Output file paths
        rules_path = out_dir / (in_path.stem + "_pruned_rules.txt")

        # Global state across chunks
        global_seen_hashes: Set[str] = set()
        global_total = 0
        global_kept = 0
        global_dropped_no_aux = 0
        global_dropped_predicate = 0
        global_pruned = 0
        global_prune_failed = 0
        global_propositions = 0
        global_rules_written = 0
        global_skipped_missing = 0
        global_skipped_pred = 0

        # Seed tracking for incremental group reduction
        current_seed: Optional[int] = None
        seed_accumulator: Dict[int, List[Dict[str, Any]]] = {}
        no_seed_rules: List[Dict[str, Any]] = []

        all_entries_for_stats: List[Dict[str, Any]] = []

        t_start = time.time()

        print(f"\n{'=' * 60}")
        print("Stage 1: Rule Extraction (STREAMING)")
        print(f"{'=' * 60}")
        print(f"  Input: {in_path}")
        print(f"  Output: {out_dir}")
        print(f"  chunk_size={chunk_size}, inflight_limit={inflight_limit}")
        print(f"  max_workers={mw}")

        # Per-step cumulative timing accumulators
        t_step1_total = t_step2_total = t_step3_total = t_step4_total = (
            t_step5_total
        ) = t_step6_total = 0.0

        with open(rules_path, "w", encoding="utf-8") as rules_fh:
            for chunk_idx, chunk in _stream_jsonl_chunks(in_path, chunk_size):
                t_chunk = time.time()
                chunk_total = len(chunk)
                global_total += chunk_total

                # Step 1: Filter
                t0 = time.time()
                kept_records, n_no_aux, n_pred = self._filter_chunk(chunk)
                t_step1_total += time.time() - t0
                global_kept += len(kept_records)
                global_dropped_no_aux += n_no_aux
                global_dropped_predicate += n_pred

                if not kept_records:
                    print(
                        f"  [chunk {chunk_idx}] {chunk_total} records, 0 kept, skipping"
                    )
                    continue

                # Build idx_src for this chunk
                idx_src: Dict[str, Any] = {}
                for r in kept_records:
                    if isinstance(r, dict) and r.get("problem_id") is not None:
                        idx_src[str(r["problem_id"])] = r

                # Step 2: Graph prune (Ray bounded parallel)
                t0 = time.time()
                pruned_map = _ray_prune_chunk_bounded(kept_records, inflight_limit)
                t_step2_total += time.time() - t0
                n_pruned = len(pruned_map)
                n_failed = len(kept_records) - n_pruned
                global_pruned += n_pruned
                global_prune_failed += n_failed

                if not pruned_map:
                    print(
                        f"  [chunk {chunk_idx}] {chunk_total} records, {len(kept_records)} kept, 0 pruned"
                    )
                    continue

                # Steps 3 + 3.5a + 3.5b: Proposition extraction
                t0 = time.time()
                out_results = self._extract_propositions_chunk(pruned_map, idx_src)
                t_step3_total += time.time() - t0
                global_propositions += len(out_results)

                # Step 4: Normalization
                t0 = time.time()
                normalized_rules, n_skipped, _ = self._normalize_rules(out_results)
                t_step4_total += time.time() - t0
                global_skipped_missing += n_skipped

                if not normalized_rules:
                    print(
                        f"  [chunk {chunk_idx}] {chunk_total} records, 0 normalized rules"
                    )
                    continue

                # Step 5: Dedup (incremental with global hash set)
                t0 = time.time()
                entries, _ = self._dedup_rules(
                    normalized_rules, external_seen_hashes=global_seen_hashes
                )
                t_step5_total += time.time() - t0

                if not entries:
                    print(
                        f"  [chunk {chunk_idx}] {chunk_total} records, 0 unique rules"
                    )
                    continue

                # Step 6: Predicate filter + incremental append to rules.txt
                t0 = time.time()
                final_entries = self._apply_rule_skip_predicates(entries)
                global_skipped_pred += len(entries) - len(final_entries)
                for e in final_entries:
                    rules_fh.write(e["rid"] + "\n")
                    rules_fh.write(e["norm_rule"] + "\n")
                    global_rules_written += 1
                rules_fh.flush()
                t_step6_total += time.time() - t0

                # Track entries for stats output
                all_entries_for_stats.extend(final_entries)

                # Seed tracking for incremental group reduction
                if seed_reducer_queue is not None:
                    for e in final_entries:
                        seed_val = e.get("seed")
                        if seed_val is None:
                            no_seed_rules.append(e)
                            continue
                        if current_seed is not None and seed_val != current_seed:
                            # Seed boundary detected
                            if current_seed in seed_accumulator:
                                seed_reducer_queue.put(
                                    (
                                        "seed_done",
                                        current_seed,
                                        seed_accumulator.pop(current_seed),
                                    )
                                )
                        current_seed = seed_val
                        seed_accumulator.setdefault(seed_val, []).append(e)

                chunk_elapsed = time.time() - t_chunk
                print(
                    f"  [chunk {chunk_idx}] total={chunk_total} kept={len(kept_records)} "
                    f"pruned={n_pruned} rules={len(final_entries)} "
                    f"cumulative_unique={global_rules_written} ({chunk_elapsed:.1f}s)"
                )

        # Flush remaining seed groups
        if seed_reducer_queue is not None:
            for seed_val, rules in seed_accumulator.items():
                seed_reducer_queue.put(("seed_done", seed_val, rules))
            seed_accumulator.clear()
            # Signal completion
            seed_reducer_queue.put(("all_done", no_seed_rules))

        # Write step6_rules_stats.json for Stage 2 compatibility
        if save_intermediates and intermediates_dir:
            stats_path = intermediates_dir / "step6_rules_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input_rules_raw": global_propositions,
                        "output_rules_deduped": global_rules_written,
                        "skipped_rules_missing_points": global_skipped_missing,
                        "skipped_rules_predicates": global_skipped_pred,
                        "rule_skip_predicates": sorted(self.rule_skip_predicates)
                        if self.rule_skip_predicates
                        else [],
                        "entries": [
                            {
                                "rid": e.get("rid"),
                                "pid": e.get("pid"),
                                "seed": e.get("seed"),
                                "rule": e.get("norm_rule"),
                                "rule_original": e.get("rule"),
                                "llm_input_renamed": e.get("llm_input_renamed", ""),
                                "llm_output_renamed": e.get("llm_output_renamed", ""),
                                "point_coords": e.get("point_coords", {}),
                            }
                            for e in all_entries_for_stats
                        ],
                    },
                    f,
                    indent=2,
                )
            print(f"[streaming] step6_rules_stats.json saved to {stats_path}")

        ray.shutdown()

        total_elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"Stage 1 (STREAMING) complete in {total_elapsed:.1f}s")
        print(f"  Total records: {global_total}")
        print(f"  Kept after filter: {global_kept}")
        print(f"  Dropped (no aux): {global_dropped_no_aux}")
        print(f"  Dropped (predicate): {global_dropped_predicate}")
        print(f"  Pruned: {global_pruned}, Failed: {global_prune_failed}")
        print(f"  Rules written: {global_rules_written}")
        print(f"  Global unique hashes: {len(global_seen_hashes)}")
        print("  Step timing breakdown:")
        print(f"    Step 1 (filter):      {t_step1_total:6.1f}s")
        print(f"    Step 2 (prune/Ray):   {t_step2_total:6.1f}s")
        print(f"    Step 3 (propositions):{t_step3_total:6.1f}s")
        print(f"    Step 4 (normalize):   {t_step4_total:6.1f}s")
        print(f"    Step 5 (dedup):       {t_step5_total:6.1f}s")
        print(f"    Step 6 (dump):        {t_step6_total:6.1f}s")
        print(
            f"    Other (Ray init/IO):  {total_elapsed - t_step1_total - t_step2_total - t_step3_total - t_step4_total - t_step5_total - t_step6_total:6.1f}s"
        )
        print(f"{'=' * 60}")

        return {
            "total": global_total,
            "kept": global_kept,
            "dropped": global_total - global_kept,
            "rules": global_rules_written,
            "skipped_rules": global_skipped_pred,
            "skipped_missing_points": global_skipped_missing,
            "json": str(rules_path),
            "elapsed_seconds": total_elapsed,
            "step_timing": {
                "step1_filter": round(t_step1_total, 2),
                "step2_prune_ray": round(t_step2_total, 2),
                "step3_propositions": round(t_step3_total, 2),
                "step4_normalize": round(t_step4_total, 2),
                "step5_dedup": round(t_step5_total, 2),
                "step6_dump": round(t_step6_total, 2),
            },
            "source_data_file": str(intermediates_dir / "step6_rules_stats.json")
            if intermediates_dir
            else None,
        }


__all__ = ["FilterAndPruneEngine"]


def _combine_pngs_h(
    left_png: Path,
    right_png: Path,
    out_png: Path,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> str:
    """将两张 PNG 水平合并为一张，并在子图上方标注标题与副标题。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except Exception as e:
        raise RuntimeError(
            "缺少 matplotlib，无法进行合图保存。请安装 matplotlib 后重试。"
        ) from e

    img_l = mpimg.imread(str(left_png))
    img_r = mpimg.imread(str(right_png))

    fig, axes = plt.subplots(1, 2, figsize=(30, 20))
    axes[0].imshow(img_l)
    axes[0].axis("off")
    axes[0].set_title("Original", fontsize=10)

    axes[1].imshow(img_r)
    axes[1].axis("off")
    axes[1].set_title("Pruned", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)
    if subtitle:
        sub = subtitle if len(subtitle) <= 220 else (subtitle[:217] + "...")
        fig.text(
            0.5,
            0.015,
            "proposition w/o aux: " + sub,
            ha="center",
            va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.85),
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return str(out_png)

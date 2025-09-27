"""
证明图构建器 ProofGraph

功能概述（当前实现）：
- 从题目解析结果(JSON: results[*].proof.analysis / numerical_check / proof 及附带元数据)构建分层有向图。
- 节点两类：
    - fact 节点(奇数层)：谓词与点参数（args）所对应的事实；可带 produced_by/used_by 关联。
    - rule 节点(偶数层)：一次推理步骤；带 premises/conclusion 关联。
- 边为有向且无类型：前提 fact → rule，rule → 结论 fact。
- 题级元数据：
    - point_coords: {problem_id: {point_lines: [str], points: [{name,x,y}]}}
    - point_rely_on: {problem_id: {point: {deps...}}}
    - aux_points: {problem_id: [str]}  辅助构造点名列表（若上游提供）。

兼容性说明：
- 保留原有公开 API（from_results_json/obj、nodes/edges 等）。
- 仅新增字段，不改变既有字段含义与挖掘算法行为。

元数据字段（节点级）：
- Fact 节点：
    - node_id: "F:{problem_id}:{id_local}"
    - type: "fact"
    - label: 谓词名
    - args: [str]
    - id_local: 局部 ID
    - problem_id: str
    - layer: int
    - produced_by: Optional[str]  生成该 fact 的规则节点（若存在）
    - used_by: list[str]         消费该 fact 的规则节点列表
- Rule 节点：
    - node_id: "R:{problem_id}:{step_index}:{code}"
    - type: "rule"
    - code: 规则代码
    - step_index: int
    - problem_id: str
    - layer: int
    - premises: list[str]      前提 fact 节点 ID
    - conclusion: Optional[str] 结论 fact 节点 ID

保留 FSM 导出接口占位：to_gspan / to_gspan_files（未实现）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set


# 回滚开关：若设为 False，将关闭“新增元数据的填充”（保留图结构不变）
ENABLE_PG_METADATA_REWRITE: bool = True



class ProofGraph:
    """
    分层有向证明图。

    公有成员：
    - nodes: Dict[node_id -> dict]  每个节点包含属性（fact 或 rule）
    - edges: List[Tuple[src_id, dst_id]] 有向边，不区分类型
    - fact_id_map: {problem_id: {id_local: fact_node_id}}
    - rule_step_map: {problem_id: {step_index: rule_node_id}}
    """

    # 正则：用于解析带方括号的局部ID，如 [004]
    _BRACKET_ID_RE = re.compile(r"\[(\d+)\]")
    # 正则：解析 fact 片段，如 "cong a d a e [000]"
    _FACT_SEG_RE = re.compile(r"^\s*(?P<pred>\w+)\s+(?P<args>.*?)\s*\[(?P<id>\d+)\]\s*$")
    # 正则：剥离 <analysis> .. </analysis> / <proof> .. </proof> 等标签
    _TAGS_RE = re.compile(r"<\/?\w+>")

    def __init__(self, *, verbose: bool = True, log_level: int = logging.INFO) -> None:
        self.nodes = {}
        self.edges = []
        self.fact_id_map = {}
        self.rule_step_map = {}
        # 每题的点依赖映射：{problem_id: {point: set(deps)}}
        self.point_rely_on = {}
        # 每题的点坐标缓存：{problem_id: {"point_lines": list[str], "points": list[dict]}}
        self.point_coords = {}
        # 每题的辅助点列表：{problem_id: [str]}
        self.aux_points = {}

        # 统计/诊断
        self.stats = {
            "warnings": 0,
            "placeholders": 0,
            "duplicate_fact_same": 0,
            "duplicate_fact_conflict": 0,
            "unparsed_fact_segments": 0,
            "bad_proof_steps": 0,
        }

        # 日志
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)

    # -------------------------------
    # 日志辅助
    # -------------------------------
    def _log(self, level: int, msg: str) -> None:
        self.logger.log(level, msg)
        if self.verbose and level >= logging.WARNING:
            # 同步在控制台简单输出关键问题
            print(f"[ProofGraph][WARN] {msg}")

    # -------------------------------
    # 工具方法
    # -------------------------------
    @staticmethod
    def _strip_tags(text: str) -> str:
        return ProofGraph._TAGS_RE.sub("", text or "")

    @staticmethod
    def _mk_fact_node_id(problem_id: str, id_local: str) -> str:
        return f"F:{problem_id}:{id_local}"

    @staticmethod
    def _mk_rule_node_id(problem_id: str, step_index: int, code: str) -> str:
        return f"R:{problem_id}:{step_index}:{code}"

    # -------------------------------
    # Fact 解析（analysis / numerical_check）
    # -------------------------------
    def parse_facts_from_text(self, problem_id: str, text: Optional[str]) -> None:
        """
        解析一个文本块，提取若干 "pred args [NNN]" 子句，生成 fact 节点(layer=1)。
        不区分来源（analysis / numerical_check）。
        """
        if not text:
            return
        content = self._strip_tags(text)
        # 按分号切分子句
        for raw_seg in content.split(";"):
            seg = raw_seg.strip()
            if not seg:
                continue
            m = self._FACT_SEG_RE.match(seg)
            if not m:
                # 不是标准 fact 结构，发出 warning
                self.stats["unparsed_fact_segments"] += 1
                self._log(logging.WARNING, f"unparsed_fact_segment (ignored as fact): '{seg}' (problem {problem_id})")
                continue

            pred = m.group("pred")
            args_str = m.group("args").strip()
            id_local = m.group("id")
            args = [tok for tok in args_str.split() if tok]

            # 尝试登记
            self._add_fact(problem_id, id_local, pred, args, layer=1, source_hint="fact_block")

    # 内部：添加 fact 节点（处理重复/冲突）
    def _add_fact(
        self,
        problem_id: str,
        id_local: str,
        label: str,
        args: List[str],
        layer: int,
        source_hint: str,
    ) -> str:
        if problem_id not in self.fact_id_map:
            self.fact_id_map[problem_id] = {}

        key_map = self.fact_id_map[problem_id]
        if id_local in key_map:
            # 已存在：一致/冲突检查
            node_id = key_map[id_local]
            exist = self.nodes[node_id]
            if exist["label"] == label and exist.get("args", []) == args:
                self.stats["duplicate_fact_same"] += 1
                self.logger.info(
                    f"duplicate_fact_same: keep existing fact {node_id} ({label} {args}) from {source_hint}"
                )
                return node_id
            else:
                # 冲突，忽略新内容
                self.stats["duplicate_fact_conflict"] += 1
                self._log(
                    logging.WARNING,
                    f"duplicate_fact_conflict: id_local [{id_local}] already mapped to {exist['node_id']}\n"
                    f"  existing: {exist['label']} {exist.get('args', [])}\n  new: {label} {args} (ignored)",
                )
                return node_id

        node_id = self._mk_fact_node_id(problem_id, id_local)
        self.nodes[node_id] = {
            "node_id": node_id,
            "type": "fact",
            "label": label,
            "args": list(args),
            "id_local": id_local,
            "problem_id": problem_id,
            "layer": int(layer),
            # 新增：默认的元数据字段
            "produced_by": None,
            "used_by": [],
        }
        key_map[id_local] = node_id
        self.logger.info(
            f"add_fact: {node_id} label={label} args={args} layer={layer} src={source_hint}"
        )
        return node_id

    # -------------------------------
    # proof 步骤解析与入图
    # -------------------------------
    def parse_proof_step(self, line: str) -> Optional[Tuple[str, List[str], str, str, List[str]]]:
        """
        解析一条 proof 语句：
        形如："pred a b c [004] rule [000] [001]"。
        返回： (concl_pred, concl_args, concl_id, rule_code, premise_ids)
        无法解析则返回 None（并在调用方记录与日志）。
        """
        s = line.strip()
        if not s:
            return None

        # 找到第一个 [NNN] 作为结论 ID 的锚点
        first = self._BRACKET_ID_RE.search(s)
        if not first:
            return None

        concl_id = first.group(1)
        left = s[: first.start()].strip()
        right = s[first.end():].strip()

        # left: "pred args..."，拆出 pred 与 args
        if not left:
            return None
        tokens_left = left.split()
        if len(tokens_left) < 1:
            return None
        concl_pred = tokens_left[0]
        concl_args = tokens_left[1:]

        # right: 以 rule code 开头，随后若干 [NNN]
        if not right:
            return None
        # rule code 是第一个非空 token
        parts = right.split()
        rule_code = parts[0]
        rest = " ".join(parts[1:])
        premise_ids = self._BRACKET_ID_RE.findall(rest)

        return concl_pred, concl_args, concl_id, rule_code, premise_ids

    def add_rule_step(
        self,
        problem_id: str,
        step_index: int,
        concl_pred: str,
        concl_args: List[str],
        concl_id: str,
        rule_code: str,
        premise_ids: List[str],
    ) -> Tuple[str, str]:
        """
        将一条解析好的规则步加入图：
        - 为缺失前提创建占位 fact（label='unknown', args=[], layer=1）并 warning。
        - 计算 rule.layer = max(premises.layer)+1（若全部占位或空，则=2）。
        - 连接所有 premise→rule 边，rule→结论 边。
        返回：(rule_node_id, conclusion_fact_node_id)
        """
        # 准备前提 fact 节点
        premise_node_ids: List[str] = []
        max_layer = 1
        if problem_id not in self.fact_id_map:
            self.fact_id_map[problem_id] = {}

        for pid in premise_ids:
            fmap = self.fact_id_map[problem_id]
            if pid not in fmap:
                # 直接抛错：不再创建 unknown 占位 fact
                raise ValueError(
                    f"Missing premise fact id [{pid}] for problem {problem_id} at step {step_index} using rule {rule_code}"
                )
            node_id = fmap[pid]
            premise_node_ids.append(node_id)
            max_layer = max(max_layer, int(self.nodes[node_id]["layer"]))

        rule_layer = max_layer + 1 if premise_node_ids else 2

        # 创建 rule 节点
        if problem_id not in self.rule_step_map:
            self.rule_step_map[problem_id] = {}
        rule_node_id = self._mk_rule_node_id(problem_id, step_index, rule_code)
        self.nodes[rule_node_id] = {
            "node_id": rule_node_id,
            "type": "rule",
            "code": rule_code,
            "step_index": int(step_index),
            "problem_id": problem_id,
            "layer": int(rule_layer),
            # 新增：结构元数据
            "premises": list(premise_node_ids) if ENABLE_PG_METADATA_REWRITE else [],
            "conclusion": None,
        }
        self.rule_step_map[problem_id][step_index] = rule_node_id
        self.logger.info(
            f"add_rule: {rule_node_id} code={rule_code} layer={rule_layer} premises={premise_ids}"
        )

        # 前提 → 规则 边
        for pnid in premise_node_ids:
            self.edges.append((pnid, rule_node_id))
            if ENABLE_PG_METADATA_REWRITE:
                try:
                    # 记录消费该 fact 的规则
                    self.nodes[pnid].setdefault("used_by", [])
                    if rule_node_id not in self.nodes[pnid]["used_by"]:
                        self.nodes[pnid]["used_by"].append(rule_node_id)
                except Exception:
                    pass

        # 结论 fact
        concl_fact_node_id = self._ensure_conclusion_fact(
            problem_id, concl_id, concl_pred, concl_args, produced_by=rule_node_id, layer=rule_layer + 1
        )

        # 规则 → 结论 边
        self.edges.append((rule_node_id, concl_fact_node_id))

        # 回填 rule 的结论引用
        if ENABLE_PG_METADATA_REWRITE:
            try:
                self.nodes[rule_node_id]["conclusion"] = concl_fact_node_id
            except Exception:
                pass

        return rule_node_id, concl_fact_node_id

    def _ensure_conclusion_fact(
        self,
        problem_id: str,
        id_local: str,
        label: str,
        args: List[str],
        produced_by: str,
        layer: int,
    ) -> str:
        fmap = self.fact_id_map.setdefault(problem_id, {})
        if id_local in fmap:
            node_id = fmap[id_local]
            exist = self.nodes[node_id]
            if exist["label"] == label and exist.get("args", []) == args:
                self.logger.info(
                    f"conclusion_already_exists: {node_id} matches, produced_by={produced_by}"
                )
                if ENABLE_PG_METADATA_REWRITE:
                    # 若未登记生成者，则补记
                    if exist.get("produced_by") in (None, ""):
                        exist["produced_by"] = produced_by
                return node_id
            else:
                self.stats["warnings"] += 1
                self._log(
                    logging.WARNING,
                    f"conclusion_conflict_ignore_new: [{id_local}] existing={exist['label']} {exist.get('args', [])} vs new={label} {args}",
                )
                return node_id

        # 创建新结论 fact
        node_id = self._add_fact(
            problem_id, id_local, label=label, args=args, layer=layer, source_hint=f"produced_by:{produced_by}"
        )
        if ENABLE_PG_METADATA_REWRITE:
            try:
                self.nodes[node_id]["produced_by"] = produced_by
            except Exception:
                pass
        return node_id

    # -------------------------------
    # 构建流程入口
    # -------------------------------
    def from_single_result(self, result_obj: Dict[str, Any]) -> None:
        """
        从单个题目的 result 字典构建子图（并合入本图）。
        仅使用 result_obj['problem_id'] 与 result_obj['proof'] 三段文本。
        """
        problem_id = str(result_obj.get("problem_id", "unknown"))
        proof_block = result_obj.get("proof", {}) or {}
        # 兼容 proof 为 dict 或 str 的两种结构
        if isinstance(proof_block, dict):
            analysis_text = proof_block.get("analysis", "")
            numerical_text = proof_block.get("numerical_check", "")
            proof_text = proof_block.get("proof", "")
        else:
            # 当 proof 为纯字符串（如异常信息或仅有步骤文本）时
            analysis_text = ""
            numerical_text = ""
            proof_text = str(proof_block)

        # 可选：缓存该题的点坐标信息（如果上游已提供）
        try:
            coords: Dict[str, Any] = {}
            pls = result_obj.get("point_lines")
            pts = result_obj.get("points")
            if isinstance(pls, list) and pls and all(isinstance(s, str) for s in pls):
                coords["point_lines"] = list(pls)
            if isinstance(pts, list) and pts and all(isinstance(d, dict) for d in pts):
                coords["points"] = list(pts)
            if coords:
                self.point_coords[str(problem_id)] = coords
        except Exception:
            pass

        # 坐标缓存已在上方 try 块处理

        # 解析点依赖 point_rely_on（支持放在 proof_block 或顶层 result_obj）
        # 注意：proof_block 可能是 dict 或 str，只有在为 dict 时才可 .get
        if isinstance(proof_block, dict):
            rely_src = proof_block.get("point_rely_on") or result_obj.get("point_rely_on") or {}
        else:
            rely_src = result_obj.get("point_rely_on") or {}
        if isinstance(rely_src, dict):
            mapping: Dict[str, Set[str]] = {}
            for pt, deps in rely_src.items():
                if isinstance(deps, str):
                    vals = [x.strip() for x in deps.split(",")] if deps else []
                elif isinstance(deps, list):
                    vals = [str(x).strip() for x in deps]
                else:
                    vals = []
                mapping[str(pt).strip()] = {v for v in vals if v}
            self.point_rely_on[str(problem_id)] = mapping
        else:
            # 若无依赖信息，记录空映射
            self.point_rely_on[str(problem_id)] = {}

        # 解析辅助构造点列表（若存在）
        try:
            aux = result_obj.get("aux_points")
            if isinstance(aux, list) and all(isinstance(a, str) for a in aux):
                self.aux_points[str(problem_id)] = list(aux)
        except Exception:
            pass

        # 初始事实（layer=1）
        self.parse_facts_from_text(problem_id, analysis_text)
        self.parse_facts_from_text(problem_id, numerical_text)

        # proof 步骤
        content = self._strip_tags(proof_text)
        step_index = 1
        for raw_seg in content.split(";"):
            seg = raw_seg.strip()
            if not seg:
                continue
            parsed = self.parse_proof_step(seg)
            if not parsed:
                self.stats["bad_proof_steps"] += 1
                self._log(logging.WARNING, f"bad_proof_step_skip: '{seg}' (problem {problem_id})")
                continue
            concl_pred, concl_args, concl_id, rule_code, premise_ids = parsed
            self.logger.info(
                f"parse_step_ok: problem={problem_id} step={step_index} rule={rule_code} concl=[{concl_id}] premises={premise_ids}"
            )
            self.add_rule_step(
                problem_id,
                step_index,
                concl_pred,
                concl_args,
                concl_id,
                rule_code,
                premise_ids,
            )
            step_index += 1

        # 汇总
        self.logger.info(
            f"problem {problem_id} summary: nodes={len(self.nodes)} edges={len(self.edges)} "
            f"placeholders={self.stats['placeholders']} unparsed_fact_segments={self.stats['unparsed_fact_segments']} "
            f"bad_proof_steps={self.stats['bad_proof_steps']} dup_same={self.stats['duplicate_fact_same']} dup_conflict={self.stats['duplicate_fact_conflict']}"
        )

    def from_results_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.from_results_obj(obj)

    def from_results_obj(self, obj: Dict[str, Any]) -> None:
        results = obj.get("results", []) or []
        for res in results:
            self.from_single_result(res)
        self.logger.info(
            f"all problems processed: total_nodes={len(self.nodes)} total_edges={len(self.edges)}"
        )
    # 追加：按题目统计平均节点/边数量，便于配置 max_nodes
        try:
            problem_ids: Set[str] = {nd.get("problem_id") for nd in self.nodes.values() if nd.get("problem_id") is not None}
            num_problems = len(problem_ids) if problem_ids else 0
            if num_problems > 0:
                avg_nodes = len(self.nodes) / num_problems
                avg_edges = len(self.edges) / num_problems
                self.logger.info(
                    f"per-problem averages: problems={num_problems} avg_nodes={avg_nodes:.2f} avg_edges={avg_edges:.2f}"
                )
            else:
                self.logger.info("per-problem averages: problems=0 (no data)")
        except Exception as e:
            self.logger.warning(f"failed to compute per-problem averages: {e}")

    # -------------------------------
    # 便捷构造与说明接口
    # -------------------------------
    @classmethod
    def build_from_results_json(cls, path: str, *, verbose: bool = True, log_level: int = logging.INFO) -> "ProofGraph":
        """类构造器：从 JSON 路径直接构建 ProofGraph 实例。"""
        inst = cls(verbose=verbose, log_level=log_level)
        inst.from_results_json(path)
        return inst

    def describe_schema(self) -> Dict[str, Any]:
        """返回当前图节点/边与题级元数据的字段说明（用于自描述/调试/文档）。"""
        return {
            "fact_node": {
                "node_id": "F:{problem_id}:{id_local}",
                "type": "fact",
                "label": "predicate",
                "args": ["A", "B", "C", ...],
                "id_local": "local id in brackets",
                "problem_id": "string",
                "layer": "int, odd",
                "produced_by": "Optional[str] rule node id",
                "used_by": ["rule node id", ...],
            },
            "rule_node": {
                "node_id": "R:{problem_id}:{step_index}:{code}",
                "type": "rule",
                "code": "rule code",
                "step_index": "int",
                "problem_id": "string",
                "layer": "int, even",
                "premises": ["fact node id", ...],
                "conclusion": "fact node id",
            },
            "edge": ["(src_node_id, dst_node_id) directed"],
            "problem_meta": {
                "point_coords": {"point_lines": ["point a x y"], "points": [{"name": "a", "x": 0.0, "y": 0.0}]},
                "point_rely_on": {"p": ["deps..."]},
                "aux_points": ["m", "n", ...],
            },
            "rendered": {
                "nodes": [{"idx": 0, "type": "fact|rule", "label": "cong(a,b,c,...)"}],
                "edges": [[0,1],[1,2]],
            },
        }

    def get_problem_summary(self, problem_id: str) -> Dict[str, Any]:
        """汇总单题的节点/边统计与元数据（便于快速审计）。"""
        pid = str(problem_id)
        facts = [nd for nd in self.nodes.values() if nd.get("type") == "fact" and nd.get("problem_id") == pid]
        rules = [nd for nd in self.nodes.values() if nd.get("type") == "rule" and nd.get("problem_id") == pid]
        edges = [(u, v) for (u, v) in self.edges if self.nodes.get(u, {}).get("problem_id") == pid]
        return {
            "problem_id": pid,
            "facts": len(facts),
            "rules": len(rules),
            "edges": len(edges),
            "point_coords": self.point_coords.get(pid),
            "point_rely_on": {k: sorted(list(v)) for k, v in self.point_rely_on.get(pid, {}).items()},
            "aux_points": self.aux_points.get(pid, []),
        }

    # -------------------------------
    # FSM 导出占位
    # -------------------------------
    def to_gspan(self, *_, **__):
        """占位：导出为 gSpan 格式的图数据（未实现）。"""
        self._log(logging.ERROR, "to_gspan called but not implemented yet.")
        raise NotImplementedError("to_gspan not implemented yet")

    def to_gspan_files(self, *_, **__):
        """占位：按需将图拆分/写入文件（未实现）。"""
        self._log(logging.ERROR, "to_gspan_files called but not implemented yet.")
        raise NotImplementedError("to_gspan_files not implemented yet")


# ==========================================
# 初版 gSpan（路径挖掘变体，合并大图，输出阶段过滤）
# 为可读性牺牲部分通用性：仅挖掘从 fact 开始到 fact 结束的有向路径模式。
# 仍可发现多步推导（F->R->F->R->F...）。
# 注：后续可扩展为完整 gSpan（含 backward 扩展与最小DFS码）。

class MergedGraph:
    """将 ProofGraph 的多题子图合为一张大图G*，节点保留 problem_id，不跨题连边。

    为了简化：
    - 节点标签：F:{predicate} / R:{code}
    - edges: 有向，来自 ProofGraph.edges
    - 保留 gid -> 原节点ID 映射，便于后续注入 args

    性能注释：可按标签分桶、建立反向邻接以加速扩展；初版保持简单。
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.out_edges: Dict[int, List[int]] = {}
        self.in_edges: Dict[int, List[int]] = {}
        self._next_gid = 0

    def add_node(self, label: str, type_: str, problem_id: str, orig_node_id: str) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self.nodes[gid] = {
            "gid": gid,
            "label": label,  # 例如 F:cong / R:r54
            "type": type_,   # fact / rule
            "problem_id": problem_id,
            "orig_node_id": orig_node_id,
        }
        self.out_edges.setdefault(gid, [])
        self.in_edges.setdefault(gid, [])
        return gid

    def add_edge(self, u: int, v: int) -> None:
        self.out_edges.setdefault(u, []).append(v)
        self.in_edges.setdefault(v, []).append(u)


class GSpanMiner:
    """简化版 gSpan：在合并大图上挖掘“有向路径”频繁子图。

    限制与选择：
    - 仅 forward 扩展，模式为简单路径（不包含回边），满足 fact 开始、fact 结束的输出约束。
    - 结构约束只在输出阶段过滤（不会阻碍扩展）。
    - 频繁度：按覆盖的 problem_id 数量计数。

    重要参数：
    - min_support: int 或 float(0~1)，表示最小题目覆盖数或覆盖比例。
    - min_rule_nodes: 最少规则节点数（避免单步推导）。
    - min_edges: 最少边数（进一步避免平凡模式）。
    - max_nodes: 扩展上限，避免爆炸（性能安全阈值）。
    - sample_embeddings: 每个模式保留多少条代表嵌入用于可读化。
    """

    def __init__(
        self,
        proof_graph: ProofGraph,
        *,
        min_support: float | int = 2,
        min_rule_nodes: int = 1,
        min_edges: int = 2,
        max_nodes: int = 9,
        sample_embeddings: int = 1,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.pg = proof_graph
        self.min_support_param = min_support
        self.min_rule_nodes = int(min_rule_nodes)
        self.min_edges = int(min_edges)
        self.max_nodes = int(max_nodes)
        self.sample_embeddings = int(sample_embeddings)
        self.logger = log or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.G = self._build_merged_graph()
        # 延迟构建：规则-规则邻接（仅规则节点）
        self._rule_out_rules: Optional[Dict[int, List[int]]]=None
        self._rule_in_rules: Optional[Dict[int, List[int]]]=None

    def _build_rule_adjacency(self) -> None:
        """基于合并图 G 构建仅规则节点的邻接：r1 -> r2 若存在 r1->F 且 F->r2（同题）。"""
        if self._rule_out_rules is not None and self._rule_in_rules is not None:
            return
        rule_out: Dict[int, List[int]] = {}
        rule_in: Dict[int, List[int]] = {}
        for gid_r1, nd_r1 in self.G.nodes.items():
            if nd_r1["type"] != "rule":
                continue
            pid = nd_r1["problem_id"]
            for gid_f in self.G.out_edges.get(gid_r1, []):
                nd_f = self.G.nodes[gid_f]
                if nd_f["type"] != "fact" or nd_f["problem_id"] != pid:
                    continue
                for gid_r2 in self.G.out_edges.get(gid_f, []):
                    nd_r2 = self.G.nodes[gid_r2]
                    if nd_r2["type"] != "rule" or nd_r2["problem_id"] != pid:
                        continue
                    if gid_r2 == gid_r1:
                        continue
                    rule_out.setdefault(gid_r1, [])
                    if gid_r2 not in rule_out[gid_r1]:
                        rule_out[gid_r1].append(gid_r2)
                    rule_in.setdefault(gid_r2, [])
                    if gid_r1 not in rule_in[gid_r2]:
                        rule_in[gid_r2].append(gid_r1)
        # 确保每个规则节点都有键
        for gid, nd in self.G.nodes.items():
            if nd["type"] == "rule":
                rule_out.setdefault(gid, [])
                rule_in.setdefault(gid, [])
        self._rule_out_rules = rule_out
        self._rule_in_rules = rule_in

    # ------------------------- 构建合并大图 -------------------------
    def _build_merged_graph(self) -> MergedGraph:
        G = MergedGraph()
        # 原节点ID -> gid
        id2gid: Dict[str, int] = {}
        for node_id, nd in self.pg.nodes.items():
            if nd["type"] == "fact":
                label = f"F:{nd['label']}"
            else:
                label = f"R:{nd['code']}"
            gid = G.add_node(label=label, type_=nd["type"], problem_id=nd["problem_id"], orig_node_id=node_id)
            id2gid[node_id] = gid

        for (u, v) in self.pg.edges:
            if u in id2gid and v in id2gid:
                G.add_edge(id2gid[u], id2gid[v])
        return G

    # 计算 min_support 的绝对阈值
    def _min_sup_abs(self) -> int:
        pids: Set[str] = {nd["problem_id"] for nd in self.G.nodes.values()}
        if isinstance(self.min_support_param, float):
            return max(1, int((len(pids) * self.min_support_param + 0.9999)))
        return int(self.min_support_param)

    # ------------------------- 主入口：挖掘 -------------------------
    def run(self) -> List[Dict[str, Any]]:
        """
        返回模式列表：每个模式包含
        - labels: 路径上的节点标签序列，例如 [F:coll, R:r54, F:midp]
        - edges: [(0,1), (1,2), ...]
        - support: 覆盖题目数量
        - pids: 覆盖题目ID集合
        - embeddings: 若干代表嵌入（每条：{pid, mapping: pattern_idx->gid}）

        注：为简化，将“模式签名”定义为 labels 序列，避免同构去重复杂性。
        局限：不同结构但同标签序列的模式会被折叠（对路径而言通常可接受）。
        """
        min_sup = self._min_sup_abs()
        self.logger.info(f"gSpan path-mining start: min_support={min_sup}, min_rule_nodes={self.min_rule_nodes}, min_edges={self.min_edges}")

        # 初始化：从所有 F->R 边作为起始路径（保证从 fact 开始）
        initial: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for u, nd_u in self.G.nodes.items():
            if nd_u["type"] != "fact":
                continue
            for v in self.G.out_edges.get(u, []):
                nd_v = self.G.nodes[v]
                if nd_v["type"] != "rule":
                    continue
                key = (nd_u["label"], nd_v["label"])  # (F:*, R:*)
                entry = {"pids": {nd_u["problem_id"]}, "emb": [{"pid": nd_u["problem_id"], "mapping": {0: u, 1: v}}]}
                initial.setdefault(key, []).append(entry)

        # 聚合初始模式（去重 pid，拼接嵌入）
        patterns: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        def add_or_update(labels: Tuple[str, ...], entries: List[Dict[str, Any]]):
            p = patterns.setdefault(labels, {"labels": list(labels), "edges": [(i, i+1) for i in range(len(labels)-1)], "pids": set(), "embeddings": []})
            for e in entries:
                p["pids"].update(e["pids"])
                p["embeddings"].extend(e["emb"])  # 可截断

        for (l0, l1), entries in initial.items():
            labels = (l0, l1)
            add_or_update(labels, entries)

        # 过滤支持
        patterns = {k: v for k, v in patterns.items() if len(v["pids"]) >= min_sup}

        # 递归扩展：仅在路径尾添加一个新节点（forward）
        results: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        def support_ok(pobj: Dict[str, Any]) -> bool:
            return len(pobj["pids"]) >= min_sup

        def expand(labels: Tuple[str, ...], pobj: Dict[str, Any]):
            # 剪枝：长度上限
            if len(labels) >= self.max_nodes:
                finalize(labels, pobj)
                return
            # 扩展：从尾节点出发到新节点
            last_idx = len(labels) - 1
            # 准备扩展所需的嵌入
            new_entries: Dict[str, Any] = {"pids": set(), "emb": []}

            for emb in pobj["embeddings"]:
                pid = emb["pid"]
                mapping = emb["mapping"]
                gid_tail = mapping[last_idx]
                for w in self.G.out_edges.get(gid_tail, []):
                    # 不允许重用节点，保持简单路径
                    if w in mapping.values():
                        continue
                    label_w = self.G.nodes[w]["label"]
                    # 生成新嵌入
                    new_mapping = dict(mapping)
                    new_mapping[last_idx + 1] = w
                    new_entries["pids"].add(pid)
                    new_entries["emb"].append({"pid": pid, "mapping": new_mapping})

            if not new_entries["emb"]:
                finalize(labels, pobj)
                return

            # 把扩展按“新标签”分桶，形成不同新模式
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for e in new_entries["emb"]:
                new_gid = e["mapping"][last_idx + 1]
                new_label = self.G.nodes[new_gid]["label"]
                # 合并同label的新路径
                buckets.setdefault(new_label, []).append(e)

            for new_label, emb_list in buckets.items():
                new_labels = labels + (new_label,)
                # 复制并更新模式对象（避免原 pobj 被污染）
                new_pobj = {
                    "labels": list(new_labels),
                    "edges": [(i, i+1) for i in range(len(new_labels)-1)],
                    # 注意：这里保留全部嵌入以便在后续扩展阶段正确聚合支持度；
                    # 代表性截断仅在输出阶段进行。
                    "pids": set(e["pid"] for e in emb_list),
                    "embeddings": emb_list,
                }
                # 扩展策略：若结构尚未达到输出下限（规则/边数），继续扩展以尝试满足结构；
                # 一旦结构满足，再依据支持度决定是否继续扩展，否则交由 finalize 处理。
                rule_cnt_new = sum(1 for lb in new_labels if lb.startswith("R:"))
                edges_new = len(new_labels) - 1
                struct_not_met = (rule_cnt_new < self.min_rule_nodes) or (edges_new < self.min_edges)
                if struct_not_met:
                    expand(new_labels, new_pobj)
                else:
                    if support_ok(new_pobj):
                        expand(new_labels, new_pobj)
                    else:
                        # 支持不足，作为候选交由 finalize（其中会再次做支持过滤）
                        finalize(new_labels, new_pobj)

        def finalize(labels: Tuple[str, ...] | List[str], pobj_like: Dict[str, Any]):
            # 输出阶段过滤结构约束：起点/终点是 fact；最少规则节点数；最少边数
            labels_t = tuple(labels)
            if len(labels_t) < 2:
                return
            start_label = labels_t[0]
            end_label = labels_t[-1]
            if not (start_label.startswith("F:") and end_label.startswith("F:")):
                return
            # 规则节点数统计
            rule_cnt = sum(1 for lb in labels_t if lb.startswith("R:"))
            if rule_cnt < self.min_rule_nodes:
                return
            if (len(labels_t) - 1) < self.min_edges:
                return
            # 支持度过滤（输出阶段也必须满足最小支持度）
            if len(pobj_like.get("pids", set())) < min_sup:
                return
            # 记录结果（按 labels 作为签名）
            rec = results.get(labels_t)
            if rec is None:
                results[labels_t] = {
                    "labels": list(labels_t),
                    "edges": [(i, i+1) for i in range(len(labels_t)-1)],
                    "pids": set(pobj_like.get("pids", set())),
                    # 仅在输出阶段截断代表嵌入数量，避免结果过大
                    "embeddings": list(pobj_like.get("embeddings", []))[: self.sample_embeddings],
                }
            else:
                rec["pids"].update(pobj_like.get("pids", set()))
                # 代表嵌入按需追加（受 sample_embeddings 限制）
                for e in pobj_like.get("embeddings", []):
                    if len(rec["embeddings"]) < self.sample_embeddings:
                        rec["embeddings"].append(e)

        # 从初始模式开始扩展
        for labels_t, pobj in patterns.items():
            # 不要在此处截断嵌入，以免影响支持度在扩展阶段的聚合。
            expand(labels_t, pobj)

        # 收尾：附加支持计数与排序
        out: List[Dict[str, Any]] = []
        for labels_t, rec in results.items():
            out.append({
                "labels": rec["labels"],
                "edges": rec["edges"],
                "support": len(rec["pids"]),
                "pids": sorted(list(rec["pids"])),
                "embeddings": rec["embeddings"],
            })

        # 简单排序：优先规则数多、长度长、支持高
        out.sort(key=lambda r: (sum(1 for lb in r["labels"] if lb.startswith("R:")), len(r["labels"]), r["support"]), reverse=True)

        self.logger.info(f"gSpan path-mining done: patterns={len(out)}")
        return out

    # ------------------------- 可读化：前提->结论 -------------------------
    def pattern_to_schema(self, pattern: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """
        将一个路径模式（带嵌入）转换为“前提->结论”的字符串，同时返回变量映射：真实点名 -> 变量名。
        规则：
        - 取第一条代表嵌入，将 pattern 索引节点映射回 G* 节点，再回到 ProofGraph 原节点。
        - 收集路径两端的 fact（起点/终点）；若希望包含中间 fact 也可扩展（当前仅两端）。
        - 对涉及的 args 做变量化：首次出现分配 X1,X2,...，同名点复用同一变量。
        """
        if not pattern.get("embeddings"):
            return "", {}
        emb = pattern["embeddings"][0]
        mapping = emb["mapping"]  # pattern_idx -> gid

        # 收集两端 fact 的 args
        labels: List[str] = pattern["labels"]
        start_idx, end_idx = 0, len(labels) - 1
        start_gid = mapping.get(start_idx)
        end_gid = mapping.get(end_idx)
        if start_gid is None or end_gid is None:
            return "", {}

        start_node = self.G.nodes[start_gid]
        end_node = self.G.nodes[end_gid]
        start_orig = self.pg.nodes[start_node["orig_node_id"]]
        end_orig = self.pg.nodes[end_node["orig_node_id"]]

        # 变量化
        var_map: Dict[str, str] = {}
        def vget(x: str) -> str:
            if x not in var_map:
                var_map[x] = f"X{len(var_map)+1}"
            return var_map[x]

        def fact_to_term(nd: Dict[str, Any]) -> str:
            # nd 是 ProofGraph 中的 fact 节点
            pred = nd["label"]
            args = nd.get("args", [])
            var_args = ",".join(vget(a) for a in args)
            return f"{pred}({var_args})"

        premise = fact_to_term(start_orig)
        conclusion = fact_to_term(end_orig)
        expr = f"{premise} => {conclusion}"
        return expr, var_map

    # ------------------------- 分叉子图挖掘（新增，不替换现有路径挖掘） -------------------------
    def run_branched(
        self,
        *,
        min_rule_indeg2_count: int = 0,
        debug_limit_expansions: Optional[int] = None,
        debug_log_every: int = 10000,
        time_budget_seconds: Optional[float] = None,
        prune_low_support_labels: bool = True,
        prune_by_rule: bool = True,
        attach_producer: bool = True,
        max_producer_depth: int = 1,
        skip_unknown: bool = True,
        enable_var_closure_check: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        在合并图上挖掘允许分叉/汇合的连通子图（受限变体）。
        """
        import time
        start_time = time.time()

        min_sup = self._min_sup_abs()
        # 边数不再单独受限，由 max_nodes 与其他预算控制

        def make_signature(labels: List[str], edges: List[Tuple[int, int]]):
            return (tuple(labels), tuple(sorted(edges)))

        # Pre-compute global label support (for both facts and rules)

        results: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[int,int], ...]], Dict[str, Any]] = {}
        visited: Set[Tuple[Tuple[str, ...], Tuple[Tuple[int,int], ...]]] = set()
        expansions = 0

        label_pid_support: Dict[str, int] = {}
        if prune_low_support_labels or prune_by_rule:
            tmp_map: Dict[str, Set[str]] = {}
            for gid, nd in self.G.nodes.items():
                lb = nd["label"]
                tmp_map.setdefault(lb, set()).add(nd["problem_id"])
            label_pid_support = {lb: len(pids) for lb, pids in tmp_map.items()}

        def support_ok(pobj: Dict[str, Any]) -> bool:
            return len(pobj.get("pids", set())) >= min_sup

        def compute_pids(emb_list: List[Dict[str, Any]]) -> Set[str]:
            return {e["pid"] for e in emb_list}

        def degrees(labels: List[str], edges: List[Tuple[int,int]]):
            indeg = [0]*len(labels)
            outdeg = [0]*len(labels)
            for a,b in edges:
                outdeg[a]+=1; indeg[b]+=1
            return indeg, outdeg

        def eligible_output(labels: List[str], edges: List[Tuple[int,int]], pids: Set[str]) -> bool:
            if len(edges) < self.min_edges:
                return False
            if sum(1 for lb in labels if lb.startswith("R:")) < self.min_rule_nodes:
                return False
            if len(pids) < min_sup:
                return False
            indeg, outdeg = degrees(labels, edges)
            src_facts = [i for i,lb in enumerate(labels) if lb.startswith("F:") and indeg[i]==0]
            # 新增：所有出度为0的节点必须是 fact，不允许 rule 成为汇点
            sink_all = [i for i,_ in enumerate(labels) if outdeg[i]==0]
            if any(labels[i].startswith("R:") for i in sink_all):
                return False
            sink_facts = [i for i,lb in enumerate(labels) if lb.startswith("F:") and outdeg[i]==0]
            if not src_facts or len(sink_facts) != 1:
                return False
            if min_rule_indeg2_count>0:
                cnt = 0
                for i,lb in enumerate(labels):
                    if lb.startswith("R:") and indeg[i] >= 2:
                        cnt += 1
                if cnt < min_rule_indeg2_count:
                    return False
            return True

        def filter_embs_var_closure(labels_loc: List[str], edges_loc: List[Tuple[int,int]], embs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not enable_var_closure_check:
                return embs
            if not labels_loc or not edges_loc or not embs:
                return embs
            indeg, outdeg = degrees(labels_loc, edges_loc)
            premise_idx = [i for i,lb in enumerate(labels_loc) if lb.startswith("F:") and indeg[i]==0]
            sink_idx = [i for i,lb in enumerate(labels_loc) if lb.startswith("F:") and outdeg[i]==0]
            if len(sink_idx) != 1:
                return embs
            good: List[Dict[str, Any]] = []
            for emb in embs:
                mapping = emb.get("mapping", {})
                sink_gid = mapping.get(sink_idx[0])
                if sink_gid is None:
                    continue
                concl_orig = self.pg.nodes[self.G.nodes[sink_gid]["orig_node_id"]]
                concl_args = set(concl_orig.get("args", []))
                premise_args: Set[str] = set()
                for i in premise_idx:
                    gid = mapping.get(i)
                    if gid is None:
                        continue
                    nd = self.G.nodes[gid]
                    orig = self.pg.nodes[nd["orig_node_id"]]
                    premise_args.update(orig.get("args", []))
                if concl_args.issubset(premise_args):
                    good.append(emb)
            return good

        def finalize(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any]):
            if not eligible_output(labels, edges, pobj.get("pids", set())):
                return
            # 变量闭包：对每条嵌入逐一过滤，并据此更新支持
            embs = pobj.get("embeddings", [])
            embs = filter_embs_var_closure(labels, edges, embs)
            if len({e.get("pid") for e in embs}) < min_sup:
                return
            sig = make_signature(labels, edges)
            rec = results.get(sig)
            if rec is None:
                results[sig] = {
                    "nodes": [{"idx": i, "label": lb} for i, lb in enumerate(labels)],
                    "labels": list(labels),
                    "edges": list(edges),
                    "pids": {e.get("pid") for e in embs},
                    "embeddings": list(embs)[: self.sample_embeddings],
                    "support": len({e.get("pid") for e in embs}),
                }
            else:
                rec["pids"].update({e.get("pid") for e in embs})
                rec["support"] = len(rec["pids"])
                for e in embs:
                    if len(rec["embeddings"]) < self.sample_embeddings:
                        rec["embeddings"].append(e)

        def expand(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any], *, producer_layers_used: int = 0):
            nonlocal expansions
            if time_budget_seconds is not None and (time.time() - start_time) >= time_budget_seconds:
                finalize(labels, edges, pobj)
                return
            if debug_limit_expansions is not None and expansions >= debug_limit_expansions:
                finalize(labels, edges, pobj)
                return
            if len(labels) >= self.max_nodes:
                finalize(labels, edges, pobj)
                return
            sig = make_signature(labels, edges)
            if sig in visited:
                return
            visited.add(sig)

            # 预先输出一次：当且仅当“所有已包含的规则在所有嵌入上都是完整的”（避免丢失可用结果）
            def all_rules_complete() -> bool:
                for r_idx, lb in enumerate(labels):
                    if not lb.startswith("R:"):
                        continue
                    for emb in pobj.get("embeddings", []):
                        mapping = emb.get("mapping", {})
                        gid_r = mapping.get(r_idx)
                        if gid_r is None:
                            return False
                        for fu in self.G.in_edges.get(gid_r, []):
                            nd_fu = self.G.nodes[fu]
                            if nd_fu["type"] != "fact":
                                continue
                            if skip_unknown and nd_fu["label"] == "F:unknown":
                                continue
                            if fu not in mapping.values():
                                return False
                return True
            if all_rules_complete():
                finalize(labels, edges, pobj)
            # Helper: complete a single rule r_idx by adding all missing premise facts (grouped by identical label sequences)
            def try_complete_rule_once(labels0: List[str], edges0: List[Tuple[int,int]], pobj0: Dict[str, Any], r_idx: int) -> bool:
                lb_r = labels0[r_idx]
                if not lb_r.startswith("R:"):
                    return False
                # Group embeddings by the sequence of missing-premise labels
                buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
                for emb in pobj0["embeddings"]:
                    pid = emb["pid"]
                    mapping = emb["mapping"]
                    gid_r = mapping.get(r_idx)
                    if gid_r is None:
                        continue
                    # collect all fact premises of this rule
                    missing: List[int] = []
                    for fu in self.G.in_edges.get(gid_r, []):
                        nd_fu = self.G.nodes[fu]
                        if nd_fu["type"] != "fact":
                            continue
                        if skip_unknown and nd_fu["label"] == "F:unknown":
                            continue
                        if fu in mapping.values():
                            continue
                        if nd_fu["problem_id"] != pid:
                            continue
                        missing.append(fu)
                    # stable order by (label, gid)
                    missing_sorted = sorted(missing, key=lambda g: (self.G.nodes[g]["label"], g))
                    key = tuple(self.G.nodes[g]["label"] for g in missing_sorted)
                    # apply mapping update for this embedding (temporarily)
                    new_map = dict(mapping)
                    next_idx = len(labels0)
                    for fu in missing_sorted:
                        new_map[next_idx] = fu
                        next_idx += 1
                    buckets.setdefault(key, []).append({"pid": pid, "mapping": new_map})
                progressed = False
                for key_labels, emb_list in buckets.items():
                    if not key_labels:
                        # nothing to add for this group
                        continue
                    # prune by labels support if needed
                    if prune_low_support_labels:
                        low = False
                        for lb_add in key_labels:
                            if label_pid_support.get(lb_add, 0) < min_sup:
                                low = True; break
                        if low:
                            continue
                    # capacity check
                    added_nodes = len(key_labels)
                    if len(labels0) + added_nodes > self.max_nodes:
                        continue
                    # build new pattern
                    new_labels = list(labels0) + list(key_labels)
                    new_edges = list(edges0)
                    # new edges: each new premise -> r_idx
                    base = len(labels0)
                    for i in range(added_nodes):
                        new_edges.append((base + i, r_idx))
                    expansions += 1
                    if debug_limit_expansions is not None and expansions % max(1, debug_log_every) == 0:
                        self.logger.info(f"branched-debug: expansions={expansions} complete-rule r_idx={r_idx} add={list(key_labels)}")
                    new_pobj = {"labels": new_labels, "edges": new_edges, "embeddings": emb_list, "pids": compute_pids(emb_list)}
                    expand(new_labels, new_edges, new_pobj, producer_layers_used=producer_layers_used)
                    progressed = True
                return progressed

            # Phase A: ensure all rules in current pattern are complete (rules integrity)
            for r_idx, lb in enumerate(labels):
                if not lb.startswith("R:"):
                    continue
                # detect if any embedding has missing premises for this rule
                need_complete = False
                for emb in pobj["embeddings"]:
                    mapping = emb["mapping"]
                    gid_r = mapping.get(r_idx)
                    if gid_r is None:
                        continue
                    for fu in self.G.in_edges.get(gid_r, []):
                        if self.G.nodes[fu]["type"] != "fact":
                            continue
                        if skip_unknown and self.G.nodes[fu]["label"] == "F:unknown":
                            continue
                        if fu not in mapping.values():
                            need_complete = True
                            break
                    if need_complete:
                        break
                if need_complete:
                    progressed = try_complete_rule_once(labels, edges, pobj, r_idx)
                    # 若确实完成了补全（生成了子分支），则交由子分支继续扩展；
                    # 若由于剪枝/容量无法补全，则不要提前返回，允许进入后续阶段尝试其它扩展。
                    if progressed:
                        return

            # Phase B: FRF atomic expansion (fact -> rule -> conclusion fact) with full premises of the rule
            # Only after all existing rules are complete
            for f_idx, lb in enumerate(labels):
                if not lb.startswith("F:"):
                    continue
                # group embeddings by (rule_label, concl_label, other_premise_labels...)
                buckets_frf: Dict[Tuple[str, str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
                for emb in pobj["embeddings"]:
                    pid = emb["pid"]
                    mapping = emb["mapping"]
                    gid_f = mapping.get(f_idx)
                    if gid_f is None:
                        continue
                    for rv in self.G.out_edges.get(gid_f, []):
                        nd_rv = self.G.nodes[rv]
                        if nd_rv["type"] != "rule":
                            continue
                        if prune_by_rule and label_pid_support.get(nd_rv["label"], 0) < min_sup:
                            continue
                        if rv in mapping.values():
                            continue
                        if nd_rv["problem_id"] != pid:
                            continue
                        # conclusion fact(s) of this rule (expected 1)
                        fouts = [fv for fv in self.G.out_edges.get(rv, []) if self.G.nodes[fv]["type"] == "fact" and self.G.nodes[fv]["problem_id"] == pid]
                        if not fouts:
                            continue
                        # choose all possible conclusions (usually 1)
                        for fv in fouts:
                            if fv in mapping.values():
                                # it's okay if conclusion already present? we avoid duplicates
                                continue
                            # collect other premises (excluding current gid_f)
                            others: List[int] = []
                            for fu in self.G.in_edges.get(rv, []):
                                nd_fu = self.G.nodes[fu]
                                if nd_fu["type"] != "fact":
                                    continue
                                if fu == gid_f:
                                    continue
                                if skip_unknown and nd_fu["label"] == "F:unknown":
                                    continue
                                if fu in mapping.values():
                                    continue
                                if nd_fu["problem_id"] != pid:
                                    continue
                                others.append(fu)
                            others_sorted = sorted(others, key=lambda g: (self.G.nodes[g]["label"], g))
                            key = (nd_rv["label"], self.G.nodes[fv]["label"], tuple(self.G.nodes[g]["label"] for g in others_sorted))
                            new_map = dict(mapping)
                            # append rule
                            idx_r = len(labels)
                            new_map[idx_r] = rv
                            # append other premises
                            idx_next = idx_r + 1
                            for fu in others_sorted:
                                new_map[idx_next] = fu
                                idx_next += 1
                            # append conclusion
                            idx_f = idx_next
                            new_map[idx_f] = fv
                            buckets_frf.setdefault(key, []).append({"pid": pid, "mapping": new_map})
                for key, emb_list in buckets_frf.items():
                    r_lb, f_lb, prem_lbs = key
                    # capacity check
                    add_count = 1 + len(prem_lbs) + 1
                    if len(labels) + add_count > self.max_nodes:
                        continue
                    if prune_low_support_labels:
                        low = False
                        if label_pid_support.get(r_lb, 0) < min_sup or label_pid_support.get(f_lb, 0) < min_sup:
                            low = True
                        else:
                            for lbx in prem_lbs:
                                if label_pid_support.get(lbx, 0) < min_sup:
                                    low = True; break
                        if low:
                            continue
                    # edges check
                    # 不再单独限制边数
                    # build new pattern
                    base = len(labels)
                    new_labels = list(labels) + [r_lb] + list(prem_lbs) + [f_lb]
                    new_edges = list(edges)
                    # f_idx -> r
                    new_edges.append((f_idx, base))
                    # premises -> r
                    for i in range(len(prem_lbs)):
                        new_edges.append((base + 1 + i, base))
                    # r -> concl
                    new_edges.append((base, base + 1 + len(prem_lbs)))
                    expansions += 1
                    if debug_limit_expansions is not None and expansions % max(1, debug_log_every) == 0:
                        self.logger.info(f"branched-debug: expansions={expansions} FRF from f_idx={f_idx} add={[r_lb, *prem_lbs, f_lb]}")
                    new_pobj = {"labels": new_labels, "edges": new_edges, "embeddings": emb_list, "pids": compute_pids(emb_list)}
                    expand(new_labels, new_edges, new_pobj, producer_layers_used=producer_layers_used)

            # Phase C: attach producer (rule -> fact) up to max_producer_depth
            if attach_producer and producer_layers_used < max_producer_depth:
                for f_idx, lb in enumerate(labels):
                    if not lb.startswith("F:"):
                        continue
                    buckets_prod: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
                    for emb in pobj["embeddings"]:
                        pid = emb["pid"]
                        mapping = emb["mapping"]
                        gid_f = mapping.get(f_idx)
                        if gid_f is None:
                            continue
                        for rv in self.G.in_edges.get(gid_f, []):
                            nd_rv = self.G.nodes[rv]
                            if nd_rv["type"] != "rule":
                                continue
                            if prune_by_rule and label_pid_support.get(nd_rv["label"], 0) < min_sup:
                                continue
                            if rv in mapping.values():
                                continue
                            if nd_rv["problem_id"] != pid:
                                continue
                            # premises of rv
                            prem: List[int] = []
                            for fu in self.G.in_edges.get(rv, []):
                                nd_fu = self.G.nodes[fu]
                                if nd_fu["type"] != "fact":
                                    continue
                                if skip_unknown and nd_fu["label"] == "F:unknown":
                                    continue
                                if fu in mapping.values():
                                    continue
                                if nd_fu["problem_id"] != pid:
                                    continue
                                prem.append(fu)
                            prem_sorted = sorted(prem, key=lambda g: (self.G.nodes[g]["label"], g))
                            key = (nd_rv["label"], tuple(self.G.nodes[g]["label"] for g in prem_sorted))
                            new_map = dict(mapping)
                            idx_r = len(labels)
                            new_map[idx_r] = rv
                            idx_next = idx_r + 1
                            for fu in prem_sorted:
                                new_map[idx_next] = fu
                                idx_next += 1
                            buckets_prod.setdefault(key, []).append({"pid": pid, "mapping": new_map})
                    for key, emb_list in buckets_prod.items():
                        r_lb, prem_lbs = key
                        add_count = 1 + len(prem_lbs)
                        if len(labels) + add_count > self.max_nodes:
                            continue
                        if prune_low_support_labels:
                            low = False
                            if label_pid_support.get(r_lb, 0) < min_sup:
                                low = True
                            else:
                                for lbx in prem_lbs:
                                    if label_pid_support.get(lbx, 0) < min_sup:
                                        low = True; break
                            if low:
                                continue
                        # 不再单独限制边数
                        base = len(labels)
                        new_labels = list(labels) + [r_lb] + list(prem_lbs)
                        new_edges = list(edges)
                        # premises -> r
                        for i in range(len(prem_lbs)):
                            new_edges.append((base + 1 + i, base))
                        # r -> f_idx
                        new_edges.append((base, f_idx))
                        expansions += 1
                        if debug_limit_expansions is not None and expansions % max(1, debug_log_every) == 0:
                            self.logger.info(f"branched-debug: expansions={expansions} attach-producer f_idx={f_idx} add={[r_lb, *prem_lbs]}")
                        new_pobj = {"labels": new_labels, "edges": new_edges, "embeddings": emb_list, "pids": compute_pids(emb_list)}
                        expand(new_labels, new_edges, new_pobj, producer_layers_used=producer_layers_used + 1)

            # Finally, try to finalize current pattern (might already satisfy constraints)
            finalize(labels, edges, pobj)

        # Build initial seeds using FRF with full premises (rules integrity from the start)
        initial_by_sig: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[int,int], ...]], Dict[str, Any]] = {}
        for u, nd_u in self.G.nodes.items():
            if nd_u["type"] != "fact":
                continue
            for r in self.G.out_edges.get(u, []):
                nd_r = self.G.nodes[r]
                if nd_r["type"] != "rule":
                    continue
                if prune_by_rule and label_pid_support.get(nd_r["label"], 0) < min_sup:
                    continue
                for w in self.G.out_edges.get(r, []):
                    nd_w = self.G.nodes[w]
                    if nd_w["type"] != "fact":
                        continue
                    pid = nd_u["problem_id"]
                    if nd_r["problem_id"] != pid or nd_w["problem_id"] != pid:
                        continue
                    # collect other premises
                    others: List[int] = []
                    for fu in self.G.in_edges.get(r, []):
                        nd_fu = self.G.nodes[fu]
                        if nd_fu["type"] != "fact":
                            continue
                        if fu == u:
                            continue
                        if skip_unknown and nd_fu["label"] == "F:unknown":
                            continue
                        if nd_fu["problem_id"] != pid:
                            continue
                        others.append(fu)
                    others_sorted = sorted(others, key=lambda g: (self.G.nodes[g]["label"], g))
                    labels0 = [nd_u["label"], nd_r["label"], *[self.G.nodes[g]["label"] for g in others_sorted], nd_w["label"]]
                    edges0: List[Tuple[int,int]] = []
                    # u->r
                    edges0.append((0,1))
                    # others -> r (indices start at 2)
                    for i in range(len(others_sorted)):
                        edges0.append((2 + i, 1))
                    # r -> w (index = 2 + len(others))
                    edges0.append((1, 2 + len(others_sorted)))
                    mapping0 = {0: u, 1: r}
                    for i, fu in enumerate(others_sorted):
                        mapping0[2 + i] = fu
                    mapping0[2 + len(others_sorted)] = w
                    pobj0 = {"labels": labels0, "edges": edges0, "embeddings": [{"pid": pid, "mapping": mapping0}], "pids": {pid}}
                    # support pruning by min_sup is deferred to aggregation below
                    sig0 = make_signature(labels0, edges0)
                    if sig0 not in initial_by_sig:
                        initial_by_sig[sig0] = {"labels": list(labels0), "edges": list(edges0), "pids": set(), "embeddings": []}
                    initial_by_sig[sig0]["pids"].add(pid)
                    initial_by_sig[sig0]["embeddings"].append({"pid": pid, "mapping": mapping0})

        # Filter seeds by support
        patterns_list: List[Dict[str, Any]] = []
        for sig, pobj in initial_by_sig.items():
            if len(pobj["pids"]) >= min_sup:
                patterns_list.append(pobj)

        for pobj in patterns_list:
            expand(list(pobj["labels"]), list(pobj["edges"]), pobj, producer_layers_used=0)

        out: List[Dict[str, Any]] = []
        for _, rec in results.items():
            out.append({
                "nodes": rec["nodes"],
                "labels": rec["labels"],
                "edges": rec["edges"],
                "support": len(rec["pids"]),
                "pids": sorted(list(rec["pids"])),
                "embeddings": rec["embeddings"],
            })

        out.sort(key=lambda r: (sum(1 for lb in r["labels"] if lb.startswith("R:")), len(r["edges"]), r["support"]), reverse=True)
        self.logger.info(f"branched-mining done: patterns={len(out)}")
        return out

    # ===============================
    # 可视化：按题目生成 rendered 与 PNG
    # ===============================
    def _viz__parse_pred_args(self, label: str):
        """解析形如 'pred a b c' 或 'pred(a,b,c)' 的标签为 (pred, [args])。
        ProofGraph 内部事实节点的 label 是谓词名，args 在节点中；但为了通用性，这里也支持 pred(xxx) 文本。
        """
        try:
            if "(" in label and label.endswith(")"):
                pred, rest = label.split("(", 1)
                args = rest[:-1]
                parts = [a.strip() for a in args.split(",") if a.strip()]
                return pred.strip(), parts
            # 空格分割的简易兜底
            toks = label.split()
            if toks:
                return toks[0].strip(), toks[1:]
            return label, []
        except Exception:
            return label, []

    def _viz__fact_points(self, fact_node: Dict[str, Any]) -> List[str]:
        """从 fact 节点获取参与点集合：优先使用节点 args；否则从 label 文本解析。
        对 aconst/rconst 等尾常量做忽略处理。
        """
        args = list(fact_node.get("args", []) or [])
        if not args:
            pred, parts = self._viz__parse_pred_args(str(fact_node.get("label", "")))
            args = parts
        pred = str(fact_node.get("label", ""))
        # 若 label 本身是谓词名，则 pred 不含括号；此处仅在 pred 需要时处理
        p0 = pred.split("(", 1)[0].strip()
        if p0 in {"aconst", "rconst"} and len(args) >= 1:
            return [a for a in args[:-1]]
        return args

    def build_rendered_for_problem(self, problem_id: str, *, label_mode: str = "legend") -> Dict[str, Any]:
        """基于当前图构建单题目的 rendered 结构：{"nodes": [{idx,type,label}], "edges": [(u,v),...]}

        - label_mode："full" | "short" | "legend"（legend 模式只影响 draw 阶段的显示，rendered 仍存 raw label）。
        - 事实节点 label 采用 "pred(a,b,...)" 形式，规则节点 label 为其 code。
        - 仅包含该 problem_id 的节点与边，节点 idx 重新编号（0..N-1）。
        """
        pid = str(problem_id)
        # 收集节点
        node_ids = [nid for nid, nd in self.nodes.items() if nd.get("problem_id") == pid]
        if not node_ids:
            return {"nodes": [], "edges": []}
        # 建立映射
        nid2idx: Dict[str, int] = {}
        nodes_out: List[Dict[str, Any]] = []
        for idx, nid in enumerate(sorted(node_ids)):
            nd = self.nodes[nid]
            ntype = nd.get("type")
            if ntype == "fact":
                pred = str(nd.get("label", ""))
                args = nd.get("args", []) or []
                if args:
                    label = f"{pred}({','.join(str(a) for a in args)})"
                else:
                    label = pred
            else:
                # rule: 使用 code
                label = str(nd.get("code", nd.get("label", "rule")))
            nodes_out.append({"idx": idx, "type": ntype, "label": label, "_orig": nid})
            nid2idx[nid] = idx

        # 收集边（仅保留同题边）
        edges_out: List[Tuple[int, int]] = []
        for (u, v) in self.edges:
            u_nd = self.nodes.get(u)
            v_nd = self.nodes.get(v)
            if not u_nd or not v_nd:
                continue
            if u_nd.get("problem_id") != pid or v_nd.get("problem_id") != pid:
                continue
            if u not in nid2idx or v not in nid2idx:
                continue
            edges_out.append((nid2idx[u], nid2idx[v]))

        return {"nodes": [{k: v for k, v in nd.items() if k != "_orig"} for nd in nodes_out], "edges": edges_out}

    def draw_problem_png(
        self,
        problem_id: str,
        out_png: str,
        *,
        label_mode: str = "legend",
        style_opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将指定题目的证明图绘制为 PNG。

        视觉规则：
        - 前提节点（入度=0 的 fact）边框加粗；
        - 结论节点（出度=0 的 fact）蓝色底、蓝色边框、边框加粗；
        - 含辅助点的 fact 节点（与 aux_points 交集非空）橙色底；
        - 其它 fact 节点绿色底；
        - 规则节点浅灰底；
        - 若结论与辅助冲突，以结论蓝优先。
        """
    # Visualization always enabled

        # 依赖按需导入
        try:
            import networkx as nx  # type: ignore
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "缺少依赖：需要 networkx 与 matplotlib。请安装后重试。"\
                f" 导入错误：{type(e).__name__}: {e}"
            )
        try:
            import pydot  # noqa: F401, type: ignore
        except Exception as e:
            # 允许回退
            pydot = None  # type: ignore

        rendered = self.build_rendered_for_problem(problem_id, label_mode=label_mode)
        nodes = rendered.get("nodes") or []
        edges = rendered.get("edges") or []
        if not nodes:
            raise ValueError(f"problem {problem_id} has no nodes to render")

        # 计算入度/出度
        indeg: Dict[int, int] = {n.get("idx"): 0 for n in nodes}
        outdeg: Dict[int, int] = {n.get("idx"): 0 for n in nodes}
        for (u, v) in edges:
            outdeg[u] = outdeg.get(u, 0) + 1
            indeg[v] = indeg.get(v, 0) + 1

        # 组织 union of aux points（按题）
        aux_set = set(self.aux_points.get(str(problem_id), []) or [])

        # 构图
        import networkx as nx  # type: ignore
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["idx"], ntype=n.get("type"), raw_label=str(n.get("label", "")))
        for (u, v) in edges:
            G.add_edge(u, v)

        # 布局：优先 graphviz
        try:
            if pydot is not None:
                pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
            else:
                raise RuntimeError("pydot not available")
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        # 样式参数
        COLOR_FACT_OK = "#e0ffe0"       # 绿色
        COLOR_FACT_AUX = "#ffe4b5"      # 橙色（Moccasin）
        COLOR_FACT_CONCL = "#e0f2ff"    # 蓝色
        COLOR_RULE = "#f0f0f0"          # 浅灰
        COLOR_BORDER = "#333333"
        COLOR_CONCL_BORDER = "#007bff"

        # 生成每个节点的显示标签（根据 label_mode）
        def short_label(raw: str) -> str:
            try:
                if "(" in raw and raw.endswith(")"):
                    return raw.split("(", 1)[0].strip()
                return raw.split()[0]
            except Exception:
                return raw

        legend_lines: List[str] = []
        fact_counter = 0
        rule_counter = 0
        display_label: Dict[int, str] = {}
        # 用于含辅助点判定：需要从原节点取 args，因此先用 rendered 的 raw_label 与 PG 节点结合
        # 建立 idx -> 原节点对象映射
        idx2orig: Dict[int, Dict[str, Any]] = {}
        # 构建 orig 查找：依赖 build_rendered_for_problem 的顺序与 fields
        # 为避免在 rendered 中暴露 _orig，我们通过重建：按 label 匹配可能不唯一，改为直接从 self.nodes 过滤该 problem 的顺序映射
        pid = str(problem_id)
        # 重新建立 nid 列表与 idx 映射（与 build_rendered_for_problem 使用相同规则：按 node_id 排序）
        node_ids = [nid for nid, nd in self.nodes.items() if nd.get("problem_id") == pid]
        node_ids_sorted = sorted(node_ids)
        nid2idx_local = {nid: i for i, nid in enumerate(node_ids_sorted)}
        for nid, nd in self.nodes.items():
            if nd.get("problem_id") != pid:
                continue
            idx = nid2idx_local[nid]
            idx2orig[idx] = nd

        for n in nodes:
            idx = n.get("idx")
            ntype = n.get("type")
            raw = str(n.get("label", ""))
            if label_mode == "short":
                lab = short_label(raw)
            elif label_mode == "legend":
                if ntype == "rule":
                    rule_counter += 1
                    lab = f"R{rule_counter}"
                    legend_lines.append(f"{lab}: {raw}")
                else:
                    fact_counter += 1
                    # 多结论时不单独标 C，这里统一使用编号
                    lab = f"F{fact_counter}"
                    legend_lines.append(f"{lab}: {raw}")
            else:
                lab = raw
            display_label[idx] = lab

        # 颜色/形状/线宽计算
        node_colors: Dict[int, str] = {}
        node_edge_colors: Dict[int, str] = {}
        node_linewidths: Dict[int, float] = {}
        node_sizes: Dict[int, int] = {}
        node_shapes: Dict[int, str] = {}

        for n in nodes:
            idx = n.get("idx")
            ntype = n.get("type")
            node_edge_colors[idx] = COLOR_BORDER
            node_linewidths[idx] = 1.0
            if ntype == "rule":
                node_shapes[idx] = 's'
                node_colors[idx] = COLOR_RULE
                node_sizes[idx] = 1300
            else:
                node_shapes[idx] = 'o'
                node_sizes[idx] = 1600
                # 判断是否结论（出度为 0）
                is_concl = outdeg.get(idx, 0) == 0
                # 判断是否前提（入度为 0）
                is_prem = indeg.get(idx, 0) == 0
                # 判断是否含辅助点
                has_aux = False
                try:
                    orig = idx2orig.get(idx)
                    pts = set(self._viz__fact_points(orig or {}))
                    if aux_set and pts and (pts & aux_set):
                        has_aux = True
                except Exception:
                    has_aux = False

                if is_concl:
                    node_colors[idx] = COLOR_FACT_CONCL
                    node_edge_colors[idx] = COLOR_CONCL_BORDER
                    node_linewidths[idx] = 2.0
                elif has_aux:
                    node_colors[idx] = COLOR_FACT_AUX
                else:
                    node_colors[idx] = COLOR_FACT_OK
                if is_prem:
                    node_linewidths[idx] = 2.0

        # 分形状绘制
        import matplotlib.pyplot as plt  # type: ignore
        fig_size = (12, 9)
        if isinstance(style_opts, dict) and isinstance(style_opts.get("figsize"), (list, tuple)):
            try:
                w, h = style_opts.get("figsize")
                fig_size = (float(w), float(h))
            except Exception:
                pass
        fig, ax = plt.subplots(figsize=fig_size)
        ax.axis("off")

        import networkx as nx  # type: ignore
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=15, width=1.2, edge_color="#777777")

        # 聚合同形状节点
        shape_map: Dict[str, List[int]] = {}
        for n in nodes:
            shape_map.setdefault(node_shapes[n["idx"]], []).append(n["idx"])
        for shape, nodelist in shape_map.items():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodelist, node_shape=shape,
                node_color=[node_colors[nid] for nid in nodelist],
                edgecolors=[node_edge_colors[nid] for nid in nodelist],
                linewidths=[node_linewidths[nid] for nid in nodelist],
                node_size=[node_sizes[nid] for nid in nodelist],
                ax=ax,
            )

        # 标签
        font_size = 8
        if isinstance(style_opts, dict) and isinstance(style_opts.get("font_size"), (int, float)):
            try:
                font_size = int(style_opts.get("font_size"))
            except Exception:
                pass
        nx.draw_networkx_labels(G, pos, labels=display_label, font_size=font_size, ax=ax)

        # legend
        if label_mode == "legend" and legend_lines:
            legend_text = "Legend:\n" + "\n".join(legend_lines)
            ax.text(0.99, 0.99, legend_text, transform=ax.transAxes, ha='right', va='top',
                    fontsize=max(7, font_size-1),
                    bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))

        # 保存
        from pathlib import Path as _Path
        outp = _Path(out_png)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outp, format="png", dpi=200)
        plt.close(fig)

    def render_all_problems_to_dir(
        self,
        out_dir: str,
        *,
        label_mode: str = "legend",
        overwrite: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """批量渲染当前图中所有题目，保存为 PNG：{out_dir}/proof_{problem_id}.png
        返回统计信息。
        """
        from pathlib import Path as _Path
        out_base = _Path(out_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        # 收集题目 ID
        pids: List[str] = sorted({str(nd.get("problem_id")) for nd in self.nodes.values() if nd.get("problem_id") is not None})
        total = 0
        done = 0
        skipped = 0
        failed = 0
        for pid in pids:
            out_png = out_base / f"proof_{pid}.png"
            if out_png.exists() and not overwrite:
                skipped += 1
                continue
            try:
                self.draw_problem_png(pid, str(out_png), label_mode=label_mode)
                done += 1
            except Exception:
                failed += 1
            total += 1
            if show_progress and total % 10 == 0:
                print(f"[render] {total}/{len(pids)} done={done} skipped={skipped} failed={failed}")
        if show_progress:
            print(f"[render] finished: total={len(pids)} done={done} skipped={skipped} failed={failed}")
        return {"total": total, "done": done, "skipped": skipped, "failed": failed, "out_dir": str(out_base)}

    # ------------------------- 并行支持：按种子构建与从种子扩展（branched） -------------------------
    def build_branched_seeds(
        self,
        *,
        prune_by_rule: bool = True,
        skip_unknown: bool = True,
    ) -> List[Dict[str, Any]]:
        """构建分叉挖掘的初始种子（FRF + 完整前提），返回满足最小支持的种子列表。
        每个 seed 形如 {labels, edges, embeddings, pids}。
        """
        min_sup = self._min_sup_abs()

        # 全局标签支持（用于规则标签剪枝）
        label_pid_support: Dict[str, int] = {}
        if prune_by_rule:
            tmp_map: Dict[str, Set[str]] = {}
            for gid, nd in self.G.nodes.items():
                lb = nd["label"]
                tmp_map.setdefault(lb, set()).add(nd["problem_id"])
            label_pid_support = {lb: len(pids) for lb, pids in tmp_map.items()}

        # 初始化 FRF 种子
        initial_by_sig: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[int,int], ...]], Dict[str, Any]] = {}
        for u, nd_u in self.G.nodes.items():
            if nd_u["type"] != "fact":
                continue
            for r in self.G.out_edges.get(u, []):
                nd_r = self.G.nodes[r]
                if nd_r["type"] != "rule":
                    continue
                if prune_by_rule and label_pid_support.get(nd_r["label"], 0) < min_sup:
                    continue
                for w in self.G.out_edges.get(r, []):
                    nd_w = self.G.nodes[w]
                    if nd_w["type"] != "fact":
                        continue
                    pid = nd_u["problem_id"]
                    if nd_r["problem_id"] != pid or nd_w["problem_id"] != pid:
                        continue
                    # 其他前提（排除当前 u）
                    others: List[int] = []
                    for fu in self.G.in_edges.get(r, []):
                        nd_fu = self.G.nodes[fu]
                        if nd_fu["type"] != "fact":
                            continue
                        if fu == u:
                            continue
                        if skip_unknown and nd_fu["label"] == "F:unknown":
                            continue
                        if nd_fu["problem_id"] != pid:
                            continue
                        others.append(fu)
                    others_sorted = sorted(others, key=lambda g: (self.G.nodes[g]["label"], g))
                    labels0 = [nd_u["label"], nd_r["label"], *[self.G.nodes[g]["label"] for g in others_sorted], nd_w["label"]]
                    edges0: List[Tuple[int,int]] = []
                    # u->r
                    edges0.append((0,1))
                    # others -> r (indices start at 2)
                    for i in range(len(others_sorted)):
                        edges0.append((2 + i, 1))
                    # r -> w (index = 2 + len(others))
                    edges0.append((1, 2 + len(others_sorted)))
                    mapping0 = {0: u, 1: r}
                    for i, fu in enumerate(others_sorted):
                        mapping0[2 + i] = fu
                    mapping0[2 + len(others_sorted)] = w
                    pobj0 = {"labels": labels0, "edges": edges0, "embeddings": [{"pid": pid, "mapping": mapping0}], "pids": {pid}}
                    sig0 = (tuple(labels0), tuple(sorted(edges0)))
                    if sig0 not in initial_by_sig:
                        initial_by_sig[sig0] = {"labels": list(labels0), "edges": list(edges0), "pids": set(), "embeddings": []}
                    initial_by_sig[sig0]["pids"].add(pid)
                    initial_by_sig[sig0]["embeddings"].append({"pid": pid, "mapping": mapping0})

        # 过滤支持
        seeds: List[Dict[str, Any]] = []
        for _sig, pobj in initial_by_sig.items():
            if len(pobj["pids"]) >= min_sup:
                seeds.append(pobj)
        return seeds

    def expand_branched_from_seed(
        self,
    seed: Dict[str, Any],
    *,
    min_rule_indeg2_count: int = 0,
        debug_limit_expansions: Optional[int] = None,
        debug_log_every: int = 10000,
        time_budget_seconds: Optional[float] = None,
        prune_low_support_labels: bool = True,
        prune_by_rule: bool = True,
        attach_producer: bool = True,
        max_producer_depth: int = 1,
        skip_unknown: bool = True,
        enable_var_closure_check: bool = False,
        global_budget: Optional[Any] = None,
        emit: Optional[Any] = None,
    ) -> None:
        """从一个 seed 开始扩展（分叉挖掘变体），在满足 finalize 条件时通过 emit 回传模式对象。
        线程内维护独立 visited/expansions，避免与其他线程共享状态。
        """
        import time as _time
        start_time = _time.time()

        min_sup = self._min_sup_abs()
    # 边数不再单独受限，由 max_nodes 与其他预算控制

        def make_signature(labels: List[str], edges: List[Tuple[int, int]]):
            return (tuple(labels), tuple(sorted(edges)))

        visited: Set[Tuple[Tuple[str, ...], Tuple[Tuple[int,int], ...]]] = set()
        expansions = 0

        # 全局标签支持（用于剪枝）
        label_pid_support: Dict[str, int] = {}
        if prune_low_support_labels or prune_by_rule:
            tmp_map: Dict[str, Set[str]] = {}
            for gid, nd in self.G.nodes.items():
                lb = nd["label"]
                tmp_map.setdefault(lb, set()).add(nd["problem_id"])
            label_pid_support = {lb: len(pids) for lb, pids in tmp_map.items()}

        def compute_pids(emb_list: List[Dict[str, Any]]) -> Set[str]:
            return {e["pid"] for e in emb_list}

        # 全局预算：跨进程扩展总步数上限（seeds_mproc 中启用）
        def try_consume_budget() -> bool:
            if global_budget is None:
                return True
            try:
                return bool(global_budget.acquire(block=False))
            except Exception:
                return False

        def degrees(labels: List[str], edges: List[Tuple[int,int]]):
            indeg = [0]*len(labels)
            outdeg = [0]*len(labels)
            for a,b in edges:
                outdeg[a]+=1; indeg[b]+=1
            return indeg, outdeg

        def filter_embs_var_closure(labels_loc: List[str], edges_loc: List[Tuple[int,int]], embs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not enable_var_closure_check:
                return embs
            if not labels_loc or not edges_loc or not embs:
                return embs
            indeg, outdeg = degrees(labels_loc, edges_loc)
            premise_idx = [i for i,lb in enumerate(labels_loc) if lb.startswith("F:") and indeg[i]==0]
            sink_idx = [i for i,lb in enumerate(labels_loc) if lb.startswith("F:") and outdeg[i]==0]
            if len(sink_idx) != 1:
                return embs
            good: List[Dict[str, Any]] = []
            for emb in embs:
                mapping = emb.get("mapping", {})
                sink_gid = mapping.get(sink_idx[0])
                if sink_gid is None:
                    continue
                concl_orig = self.pg.nodes[self.G.nodes[sink_gid]["orig_node_id"]]
                concl_args = set(concl_orig.get("args", []))
                premise_args: Set[str] = set()
                for i in premise_idx:
                    gid = mapping.get(i)
                    if gid is None:
                        continue
                    nd = self.G.nodes[gid]
                    orig = self.pg.nodes[nd["orig_node_id"]]
                    premise_args.update(orig.get("args", []))
                if concl_args.issubset(premise_args):
                    good.append(emb)
            return good

        def eligible_output(labels: List[str], edges: List[Tuple[int,int]], pids: Set[str]) -> bool:
            if len(edges) < self.min_edges:
                return False
            if sum(1 for lb in labels if lb.startswith("R:")) < self.min_rule_nodes:
                return False
            if len(pids) < min_sup:
                return False
            indeg, outdeg = degrees(labels, edges)
            src_facts = [i for i,lb in enumerate(labels) if lb.startswith("F:") and indeg[i]==0]
            sink_all = [i for i,_ in enumerate(labels) if outdeg[i]==0]
            if any(labels[i].startswith("R:") for i in sink_all):
                return False
            sink_facts = [i for i,lb in enumerate(labels) if lb.startswith("F:") and outdeg[i]==0]
            if not src_facts or len(sink_facts) != 1:
                return False
            if min_rule_indeg2_count>0:
                cnt = 0
                for i,lb in enumerate(labels):
                    if lb.startswith("R:") and degrees(labels, edges)[0][i] >= 2:
                        cnt += 1
                if cnt < min_rule_indeg2_count:
                    return False
            return True

        def finalize(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any]):
            if not eligible_output(labels, edges, pobj.get("pids", set())):
                return
            embs = pobj.get("embeddings", [])
            embs = filter_embs_var_closure(labels, edges, embs)
            if len({e.get("pid") for e in embs}) < min_sup:
                return
            if emit is not None:
                pids = {e.get("pid") for e in embs}
                rec = {
                    "nodes": [{"idx": i, "label": lb} for i, lb in enumerate(labels)],
                    "labels": list(labels),
                    "edges": list(edges),
                    "pids": pids,
                    "embeddings": list(embs)[: self.sample_embeddings],
                    "support": len(pids),
                }
                emit(rec)

        def expand(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any], *, producer_layers_used: int = 0):
            nonlocal expansions
            if time_budget_seconds is not None and (_time.time() - start_time) >= time_budget_seconds:
                finalize(labels, edges, pobj)
                return
            if debug_limit_expansions is not None and expansions >= debug_limit_expansions:
                finalize(labels, edges, pobj)
                return
            if len(labels) >= self.max_nodes:
                finalize(labels, edges, pobj)
                return
            sig = make_signature(labels, edges)
            if sig in visited:
                return
            visited.add(sig)

            # 若当前所有规则在所有嵌入上已完整，可先尝试输出
            def all_rules_complete() -> bool:
                for r_idx, lb in enumerate(labels):
                    if not lb.startswith("R:"):
                        continue
                    for emb in pobj.get("embeddings", []):
                        mapping = emb.get("mapping", {})
                        gid_r = mapping.get(r_idx)
                        if gid_r is None:
                            return False
                        for fu in self.G.in_edges.get(gid_r, []):
                            nd_fu = self.G.nodes[fu]
                            if nd_fu["type"] != "fact":
                                continue
                            if skip_unknown and nd_fu["label"] == "F:unknown":
                                continue
                            if fu not in mapping.values():
                                return False
                return True
            if all_rules_complete():
                finalize(labels, edges, pobj)

            # Phase A: 完整化规则
            def try_complete_rule_once(labels0: List[str], edges0: List[Tuple[int,int]], pobj0: Dict[str, Any], r_idx: int) -> bool:
                lb_r = labels0[r_idx]
                if not lb_r.startswith("R:"):
                    return False
                buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
                for emb in pobj0.get("embeddings", []):
                    pid = emb["pid"]
                    mapping = emb["mapping"]
                    gid_r = mapping.get(r_idx)
                    if gid_r is None:
                        continue
                    missing: List[int] = []
                    for fu in self.G.in_edges.get(gid_r, []):
                        nd_fu = self.G.nodes[fu]
                        if nd_fu["type"] != "fact":
                            continue
                        if skip_unknown and nd_fu["label"] == "F:unknown":
                            continue
                        if fu in mapping.values():
                            continue
                        if nd_fu["problem_id"] != pid:
                            continue
                        missing.append(fu)
                    missing_sorted = sorted(missing, key=lambda g: (self.G.nodes[g]["label"], g))
                    key = tuple(self.G.nodes[g]["label"] for g in missing_sorted)
                    new_map = dict(mapping)
                    next_idx = len(labels0)
                    for fu in missing_sorted:
                        new_map[next_idx] = fu
                        next_idx += 1
                    buckets.setdefault(key, []).append({"pid": pid, "mapping": new_map})
                progressed = False
                for key_labels, emb_list in buckets.items():
                    if not key_labels:
                        continue
                    if prune_low_support_labels:
                        low = False
                        for lb_add in key_labels:
                            # label_pid_support 仅对 fact/rule 标签统计题目覆盖度
                            if label_pid_support.get(lb_add, 0) < min_sup:
                                low = True; break
                        if low:
                            continue
                    added_nodes = len(key_labels)
                    if len(labels0) + added_nodes > self.max_nodes:
                        continue
                    new_labels = list(labels0) + list(key_labels)
                    new_edges = list(edges0)
                    base = len(labels0)
                    for i in range(added_nodes):
                        new_edges.append((base + i, r_idx))
                    # 预算检查
                    if not try_consume_budget():
                        continue
                    expansions += 1
                    if debug_limit_expansions is not None and expansions % max(1, debug_log_every) == 0:
                        self.logger.info(f"branched-debug: expansions={expansions} complete-rule r_idx={r_idx} add={list(key_labels)}")
                    new_pobj = {"labels": new_labels, "edges": new_edges, "embeddings": emb_list, "pids": compute_pids(emb_list)}
                    expand(new_labels, new_edges, new_pobj, producer_layers_used=producer_layers_used)
                    progressed = True
                return progressed

            for r_idx, lb in enumerate(labels):
                if not lb.startswith("R:"):
                    continue
                need_complete = False
                for emb in pobj.get("embeddings", []):
                    mapping = emb["mapping"]
                    gid_r = mapping.get(r_idx)
                    if gid_r is None:
                        continue
                    for fu in self.G.in_edges.get(gid_r, []):
                        if self.G.nodes[fu]["type"] != "fact":
                            continue
                        if skip_unknown and self.G.nodes[fu]["label"] == "F:unknown":
                            continue
                        if fu not in mapping.values():
                            need_complete = True
                            break
                    if need_complete:
                        break
                if need_complete:
                    progressed = try_complete_rule_once(labels, edges, pobj, r_idx)
                    if progressed:
                        return

            # Phase B: FRF 扩展
            for f_idx, lb in enumerate(labels):
                if not lb.startswith("F:"):
                    continue
                buckets_frf: Dict[Tuple[str, str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
                for emb in pobj.get("embeddings", []):
                    pid = emb["pid"]
                    mapping = emb["mapping"]
                    gid_f = mapping.get(f_idx)
                    if gid_f is None:
                        continue
                    for rv in self.G.out_edges.get(gid_f, []):
                        nd_rv = self.G.nodes[rv]
                        if nd_rv["type"] != "rule":
                            continue
                        if prune_by_rule and label_pid_support.get(nd_rv["label"], 0) < min_sup:
                            continue
                        if rv in mapping.values():
                            continue
                        if nd_rv["problem_id"] != pid:
                            continue
                        fouts = [fv for fv in self.G.out_edges.get(rv, []) if self.G.nodes[fv]["type"] == "fact" and self.G.nodes[fv]["problem_id"] == pid]
                        if not fouts:
                            continue
                        for fv in fouts:
                            if fv in mapping.values():
                                continue
                            others: List[int] = []
                            for fu in self.G.in_edges.get(rv, []):
                                nd_fu = self.G.nodes[fu]
                                if nd_fu["type"] != "fact":
                                    continue
                                if fu == gid_f:
                                    continue
                                if skip_unknown and nd_fu["label"] == "F:unknown":
                                    continue
                                if fu in mapping.values():
                                    continue
                                if nd_fu["problem_id"] != pid:
                                    continue
                                others.append(fu)
                            others_sorted = sorted(others, key=lambda g: (self.G.nodes[g]["label"], g))
                            key = (nd_rv["label"], self.G.nodes[fv]["label"], tuple(self.G.nodes[g]["label"] for g in others_sorted))
                            new_map = dict(mapping)
                            idx_r = len(labels)
                            new_map[idx_r] = rv
                            idx_next = idx_r + 1
                            for fu in others_sorted:
                                new_map[idx_next] = fu
                                idx_next += 1
                            idx_f = idx_next
                            new_map[idx_f] = fv
                            buckets_frf.setdefault(key, []).append({"pid": pid, "mapping": new_map})
                for key, emb_list in buckets_frf.items():
                    r_lb, f_lb, prem_lbs = key
                    add_count = 1 + len(prem_lbs) + 1
                    if len(labels) + add_count > self.max_nodes:
                        continue
                    if prune_low_support_labels:
                        low = False
                        if label_pid_support.get(r_lb, 0) < min_sup or label_pid_support.get(f_lb, 0) < min_sup:
                            low = True
                        else:
                            for lbx in prem_lbs:
                                if label_pid_support.get(lbx, 0) < min_sup:
                                    low = True; break
                        if low:
                            continue
                    # 不再单独限制边数
                    base = len(labels)
                    new_labels = list(labels) + [r_lb] + list(prem_lbs) + [f_lb]
                    new_edges = list(edges)
                    new_edges.append((f_idx, base))
                    for i in range(len(prem_lbs)):
                        new_edges.append((base + 1 + i, base))
                    new_edges.append((base, base + 1 + len(prem_lbs)))
                    # 预算检查
                    if not try_consume_budget():
                        continue
                    expansions += 1
                    if debug_limit_expansions is not None and expansions % max(1, debug_log_every) == 0:
                        self.logger.info(f"branched-debug: expansions={expansions} FRF from f_idx={f_idx} add={[r_lb, *prem_lbs, f_lb]}")
                    new_pobj = {"labels": new_labels, "edges": new_edges, "embeddings": emb_list, "pids": compute_pids(emb_list)}
                    expand(new_labels, new_edges, new_pobj, producer_layers_used=producer_layers_used)

            # Phase C: 连接生产者规则
            if attach_producer and producer_layers_used < max_producer_depth:
                for f_idx, lb in enumerate(labels):
                    if not lb.startswith("F:"):
                        continue
                    buckets_prod: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
                    for emb in pobj.get("embeddings", []):
                        pid = emb["pid"]
                        mapping = emb["mapping"]
                        gid_f = mapping.get(f_idx)
                        if gid_f is None:
                            continue
                        for rv in self.G.in_edges.get(gid_f, []):
                            nd_rv = self.G.nodes[rv]
                            if nd_rv["type"] != "rule":
                                continue
                            if prune_by_rule and label_pid_support.get(nd_rv["label"], 0) < min_sup:
                                continue
                            if rv in mapping.values():
                                continue
                            if nd_rv["problem_id"] != pid:
                                continue
                            prem: List[int] = []
                            for fu in self.G.in_edges.get(rv, []):
                                nd_fu = self.G.nodes[fu]
                                if nd_fu["type"] != "fact":
                                    continue
                                if skip_unknown and nd_fu["label"] == "F:unknown":
                                    continue
                                if fu in mapping.values():
                                    continue
                                if nd_fu["problem_id"] != pid:
                                    continue
                                prem.append(fu)
                            prem_sorted = sorted(prem, key=lambda g: (self.G.nodes[g]["label"], g))
                            key = (nd_rv["label"], tuple(self.G.nodes[g]["label"] for g in prem_sorted))
                            new_map = dict(mapping)
                            idx_r = len(labels)
                            new_map[idx_r] = rv
                            idx_next = idx_r + 1
                            for fu in prem_sorted:
                                new_map[idx_next] = fu
                                idx_next += 1
                            buckets_prod.setdefault(key, []).append({"pid": pid, "mapping": new_map})
                    for key, emb_list in buckets_prod.items():
                        r_lb, prem_lbs = key
                        add_count = 1 + len(prem_lbs)
                        if len(labels) + add_count > self.max_nodes:
                            continue
                        if prune_low_support_labels:
                            low = False
                            if label_pid_support.get(r_lb, 0) < min_sup:
                                low = True
                            else:
                                for lbx in prem_lbs:
                                    if label_pid_support.get(lbx, 0) < min_sup:
                                        low = True; break
                            if low:
                                continue
                        # 不再单独限制边数
                        base = len(labels)
                        new_labels = list(labels) + [r_lb] + list(prem_lbs)
                        new_edges = list(edges)
                        for i in range(len(prem_lbs)):
                            new_edges.append((base + 1 + i, base))
                        new_edges.append((base, f_idx))
                        # 预算检查
                        if not try_consume_budget():
                            continue
                        expansions += 1
                        if debug_limit_expansions is not None and expansions % max(1, debug_log_every) == 0:
                            self.logger.info(f"branched-debug: expansions={expansions} attach-producer f_idx={f_idx} add={[r_lb, *prem_lbs]}")
                        new_pobj = {"labels": new_labels, "edges": new_edges, "embeddings": emb_list, "pids": compute_pids(emb_list)}
                        expand(new_labels, new_edges, new_pobj, producer_layers_used=producer_layers_used + 1)

            finalize(labels, edges, pobj)

        expand(list(seed.get("labels", [])), list(seed.get("edges", [])), seed, producer_layers_used=0)

    # ------------------------- 规则图挖掘（仅规则节点） -------------------------
    def run_rules_only(
        self,
        *,
        debug_limit_expansions: Optional[int] = None,
        debug_log_every: int = 10000,
        time_budget_seconds: Optional[float] = None,
        prune_low_support_labels: bool = True,
        prune_by_rule: bool = True,
        enable_var_closure_check: bool = False,
    ) -> List[Dict[str, Any]]:
        """在规则-规则邻接图上挖掘连通子图（允许分叉/汇合），节点均为规则。"""
        import time as _time
        start_time = _time.time()
        self._build_rule_adjacency()
        assert self._rule_out_rules is not None and self._rule_in_rules is not None

        min_sup = self._min_sup_abs()
    # 边数不再单独受限，由 max_nodes 与其他预算控制

        # 全局标签支持（仅规则标签）
        label_pid_support: Dict[str, int] = {}
        if prune_low_support_labels or prune_by_rule:
            tmp_map: Dict[str, Set[str]] = {}
            for gid, nd in self.G.nodes.items():
                if nd["type"] != "rule":
                    continue
                lb = f"R:{nd['code']}" if 'code' in nd else self.G.nodes[gid]["label"]
                tmp_map.setdefault(lb, set()).add(nd["problem_id"])
            label_pid_support = {lb: len(pids) for lb, pids in tmp_map.items()}

        def degrees(edges: List[Tuple[int,int]], n: int) -> Tuple[List[int], List[int]]:
            indeg = [0]*n; outdeg=[0]*n
            for a,b in edges:
                outdeg[a]+=1; indeg[b]+=1
            return indeg, outdeg

        def signature(labels: List[str], edges: List[Tuple[int,int]]):
            return (tuple(labels), tuple(sorted(edges)))

        results: Dict[Tuple[Tuple[str,...], Tuple[Tuple[int,int], ...]], Dict[str, Any]] = {}
        visited: Set[Tuple[Tuple[str,...], Tuple[Tuple[int,int], ...]]] = set()
        expansions = 0

        # 初始化种子：所有 r1->r2（同题）
        seeds_by_label: Dict[Tuple[str,str], List[Dict[str, Any]]] = {}
        for r1, outs in self._rule_out_rules.items():
            lb1 = self.G.nodes[r1]["label"]
            pid1 = self.G.nodes[r1]["problem_id"]
            if prune_by_rule and label_pid_support.get(lb1, 0) < min_sup:
                continue
            for r2 in outs:
                if self.G.nodes[r2]["problem_id"] != pid1:
                    continue
                lb2 = self.G.nodes[r2]["label"]
                if prune_by_rule and label_pid_support.get(lb2, 0) < min_sup:
                    continue
                key = (lb1, lb2)
                seeds_by_label.setdefault(key, []).append({
                    "pids": {pid1},
                    "emb": [{"pid": pid1, "mapping": {0: r1, 1: r2}}],
                })

        # 聚合初始模式
        patterns: Dict[Tuple[str,...], Dict[str, Any]] = {}
        def add_or_update(labels_t: Tuple[str,...], entries: List[Dict[str, Any]]):
            pobj = patterns.setdefault(labels_t, {"labels": list(labels_t), "edges": [(i,i+1) for i in range(len(labels_t)-1)], "pids": set(), "embeddings": []})
            for e in entries:
                pobj["pids"].update(e["pids"])
                pobj["embeddings"].extend(e["emb"])

        for key, entries in seeds_by_label.items():
            add_or_update(tuple(key), entries)

        # 过滤支持
        patterns = {k:v for k,v in patterns.items() if len(v["pids"]) >= min_sup}

        def support_ok(pobj: Dict[str, Any]) -> bool:
            return len(pobj.get("pids", set())) >= min_sup

        def finalize(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any]):
            # 结构过滤
            if len(labels) < self.min_rule_nodes or len(edges) < self.min_edges:
                return
            # 依据“能导出唯一结论”的嵌入过滤支持
            valid_pids: Set[str] = set()
            kept_embs: List[Dict[str, Any]] = []
            for emb in pobj.get("embeddings", []):
                pid = emb["pid"]
                rule_gids = [emb["mapping"].get(i) for i in range(len(labels))]
                if any(gid is None for gid in rule_gids):
                    continue
                # 计算外部前提与结论候选
                Rset = set(rule_gids)
                produced: Set[int] = set()
                consumed: Set[int] = set()
                for rg in Rset:
                    # rg -> facts (produced)
                    for f in self.G.out_edges.get(rg, []):
                        if self.G.nodes[f]["type"] != "fact":
                            continue
                        if self.G.nodes[f]["problem_id"] != pid:
                            continue
                        produced.add(f)
                        # fact -> rule (consumed by rules inside Rset)
                        for r2 in self.G.out_edges.get(f, []):
                            if r2 in Rset and self.G.nodes[r2]["problem_id"] == pid:
                                consumed.add(f)
                conclusion = produced - consumed
                # 仅接受唯一结论
                if len(conclusion) != 1:
                    continue
                valid_pids.add(pid)
                kept_embs.append(emb)
            if len(valid_pids) < min_sup:
                return
            sig = signature(labels, edges)
            rec = results.get(sig)
            if rec is None:
                results[sig] = {
                    "nodes": [{"idx": i, "label": lb} for i, lb in enumerate(labels)],
                    "labels": list(labels),
                    "edges": list(edges),
                    "pids": set(valid_pids),
                    "embeddings": kept_embs[: self.sample_embeddings],
                    "support": len(valid_pids),
                }
            else:
                rec["pids"].update(valid_pids)
                rec["support"] = len(rec["pids"])
                for e in kept_embs:
                    if len(rec["embeddings"]) < self.sample_embeddings:
                        rec["embeddings"].append(e)

        def expand(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any]):
            nonlocal expansions
            if time_budget_seconds is not None and (_time.time() - start_time) >= time_budget_seconds:
                finalize(labels, edges, pobj)
                return
            if debug_limit_expansions is not None and expansions >= debug_limit_expansions:
                finalize(labels, edges, pobj)
                return
            if len(labels) >= self.max_nodes:
                finalize(labels, edges, pobj)
                return
            sig = signature(labels, edges)
            if sig in visited:
                return
            visited.add(sig)
            expansions += 1
            if debug_log_every and expansions % debug_log_every == 0:
                self.logger.info(f"rules-only expand step={expansions}, cur_nodes={len(labels)}, cur_edges={len(edges)}, results={len(results)}")

            # 从任一已有规则节点尝试向外扩一条边并加入新规则节点
            n = len(labels)
            # 为每个候选扩展，收集新的嵌入
            # 候选结构：列表项为 (new_labels, new_edges, new_embeddings)
            candidates: Dict[Tuple[Tuple[str,...], Tuple[Tuple[int,int],...]], Dict[str, Any]] = {}

            for anchor_idx in range(n):
                # 依据每条嵌入，从对应的规则 gid 找到可扩展的相邻规则
                bucket_out: Dict[str, List[Dict[str, Any]]] = {}
                bucket_in: Dict[str, List[Dict[str, Any]]] = {}
                for emb in pobj.get("embeddings", []):
                    pid = emb["pid"]
                    gid_anchor = emb["mapping"].get(anchor_idx)
                    if gid_anchor is None:
                        continue
                    # 向外扩展：anchor -> w
                    for w in self._rule_out_rules.get(gid_anchor, []):
                        if self.G.nodes[w]["problem_id"] != pid:
                            continue
                        # 不重复节点，保持简单
                        if w in emb["mapping"].values():
                            continue
                        new_label = self.G.nodes[w]["label"]
                        # 组装新嵌入
                        mapping2 = dict(emb["mapping"]) ; mapping2[n] = w
                        bucket_out.setdefault(new_label, []).append({"pid": pid, "mapping": mapping2})
                    # 向内扩展：u -> anchor
                    for u in self._rule_in_rules.get(gid_anchor, []):
                        if self.G.nodes[u]["problem_id"] != pid:
                            continue
                        if u in emb["mapping"].values():
                            continue
                        new_label = self.G.nodes[u]["label"]
                        mapping2 = dict(emb["mapping"]) ; mapping2[n] = u
                        bucket_in.setdefault(new_label, []).append({"pid": pid, "mapping": mapping2})

                # 生成候选：anchor -> new_n
                for new_label, emb_list in bucket_out.items():
                    new_labels = labels + [new_label]
                    new_edges = edges + [(anchor_idx, n)]
                    sig2 = signature(new_labels, new_edges)
                    cand = candidates.setdefault(sig2, {"labels": new_labels, "edges": new_edges, "embeddings": []})
                    cand["embeddings"].extend(emb_list)
                # 生成候选：new_n -> anchor
                for new_label, emb_list in bucket_in.items():
                    new_labels = labels + [new_label]
                    new_edges = edges + [(n, anchor_idx)]
                    sig2 = signature(new_labels, new_edges)
                    cand = candidates.setdefault(sig2, {"labels": new_labels, "edges": new_edges, "embeddings": []})
                    cand["embeddings"].extend(emb_list)

            # 对每个候选进行支持聚合与剪枝后递归
            for (_sig, cand) in candidates.items():
                # 聚合 pids
                pids = {e["pid"] for e in cand["embeddings"]}
                cand_pobj = {"pids": pids, "embeddings": cand["embeddings"]}
                # 标签全局剪枝
                if prune_low_support_labels or prune_by_rule:
                    for lb in cand["labels"]:
                        if not lb.startswith("R:"):
                            continue
                        if prune_by_rule and label_pid_support.get(lb, 0) < min_sup:
                            break
                    else:
                        pass
                    # 如果 break 未触发则继续；若触发则跳过
                    if any((prune_by_rule and label_pid_support.get(lb, 0) < min_sup) for lb in cand["labels"] if lb.startswith("R:")):
                        continue
                # 结构限界
                if len(cand["labels"]) > self.max_nodes:
                    finalize(cand["labels"], cand["edges"], cand_pobj)
                else:
                    # 若支持已达阈值，继续扩展以寻更大结构；否则也可尝试扩展，但会在 finalize 处过滤
                    if support_ok(cand_pobj):
                        expand(cand["labels"], cand["edges"], cand_pobj)
                    else:
                        finalize(cand["labels"], cand["edges"], cand_pobj)

        # 从初始模式开始扩展
        for labels_t, pobj in patterns.items():
            expand(list(labels_t), [(i,i+1) for i in range(len(labels_t)-1)], pobj)

        # 收尾：输出组装
        out: List[Dict[str, Any]] = []
        for sig, rec in results.items():
            out.append({
                "nodes": rec["nodes"],
                "labels": rec["labels"],
                "edges": rec["edges"],
                "support": len(rec["pids"]),
                "pids": sorted(list(rec["pids"])),
                "embeddings": rec["embeddings"],
            })
        # 排序：优先节点多、边多、支持高
        out.sort(key=lambda r: (len(r["labels"]), len(r["edges"]), r["support"]), reverse=True)
        self.logger.info(f"rules-only mining done: patterns={len(out)}")
        return out

    # ------------------------- 并行支持：按种子构建与从种子扩展（rules-only） -------------------------
    def build_rules_only_seeds(
        self,
        *,
        prune_by_rule: bool = True,
    ) -> List[Dict[str, Any]]:
        """构建规则图挖掘的初始种子（r1->r2），并按标签序列聚合，过滤最小支持后返回。"""
        self._build_rule_adjacency()
        assert self._rule_out_rules is not None and self._rule_in_rules is not None
        min_sup = self._min_sup_abs()

        # 全局标签支持（仅规则标签）
        label_pid_support: Dict[str, int] = {}
        if prune_by_rule:
            tmp_map: Dict[str, Set[str]] = {}
            for gid, nd in self.G.nodes.items():
                if nd["type"] != "rule":
                    continue
                lb = self.G.nodes[gid]["label"]
                tmp_map.setdefault(lb, set()).add(nd["problem_id"])
            label_pid_support = {lb: len(pids) for lb, pids in tmp_map.items()}

        seeds_by_label: Dict[Tuple[str,str], List[Dict[str, Any]]] = {}
        for r1, outs in self._rule_out_rules.items():
            lb1 = self.G.nodes[r1]["label"]
            pid1 = self.G.nodes[r1]["problem_id"]
            if prune_by_rule and label_pid_support.get(lb1, 0) < min_sup:
                continue
            for r2 in outs:
                if self.G.nodes[r2]["problem_id"] != pid1:
                    continue
                lb2 = self.G.nodes[r2]["label"]
                if prune_by_rule and label_pid_support.get(lb2, 0) < min_sup:
                    continue
                key = (lb1, lb2)
                seeds_by_label.setdefault(key, []).append({
                    "pids": {pid1},
                    "emb": [{"pid": pid1, "mapping": {0: r1, 1: r2}}],
                })

        # 聚合到模式
        patterns: Dict[Tuple[str,...], Dict[str, Any]] = {}
        def add_or_update(labels_t: Tuple[str,...], entries: List[Dict[str, Any]]):
            pobj = patterns.setdefault(labels_t, {"labels": list(labels_t), "edges": [(i,i+1) for i in range(len(labels_t)-1)], "pids": set(), "embeddings": []})
            for e in entries:
                pobj["pids"].update(e["pids"])
                pobj["embeddings"].extend(e["emb"])

        for key, entries in seeds_by_label.items():
            add_or_update(tuple(key), entries)

        # 过滤最小支持
        patterns = {k:v for k,v in patterns.items() if len(v["pids"]) >= min_sup}
        return list(patterns.values())

    def expand_rules_only_from_seed(
        self,
        seed: Dict[str, Any],
        *,
        debug_limit_expansions: Optional[int] = None,
        debug_log_every: int = 10000,
        time_budget_seconds: Optional[float] = None,
        prune_low_support_labels: bool = True,
        prune_by_rule: bool = True,
        enable_var_closure_check: bool = False,  # 仅用于后续 schema 层过滤，这里保持与 run_rules_only 一致
        global_budget: Optional[Any] = None,
        emit: Optional[Any] = None,
    ) -> None:
        """从一个 rules-only 种子开始扩展，达到 finalize 条件时 emit 模式对象。"""
        import time as _time
        start_time = _time.time()
        self._build_rule_adjacency()
        assert self._rule_out_rules is not None and self._rule_in_rules is not None

        min_sup = self._min_sup_abs()
    # 边数不再单独受限，由 max_nodes 与其他预算控制

        # 标签支持
        label_pid_support: Dict[str, int] = {}
        if prune_low_support_labels or prune_by_rule:
            tmp_map: Dict[str, Set[str]] = {}
            for gid, nd in self.G.nodes.items():
                if nd["type"] != "rule":
                    continue
                lb = self.G.nodes[gid]["label"]
                tmp_map.setdefault(lb, set()).add(nd["problem_id"])
            label_pid_support = {lb: len(pids) for lb, pids in tmp_map.items()}

        def degrees(edges: List[Tuple[int,int]], n: int) -> Tuple[List[int], List[int]]:
            indeg = [0]*n; outdeg=[0]*n
            for a,b in edges:
                outdeg[a]+=1; indeg[b]+=1
            return indeg, outdeg

        def signature(labels: List[str], edges: List[Tuple[int,int]]):
            return (tuple(labels), tuple(sorted(edges)))

        visited: Set[Tuple[Tuple[str,...], Tuple[Tuple[int,int], ...]]] = set()
        expansions = 0

        def try_consume_budget() -> bool:
            if global_budget is None:
                return True
            try:
                return bool(global_budget.acquire(block=False))
            except Exception:
                return False

        def support_ok(pobj: Dict[str, Any]) -> bool:
            return len(pobj.get("pids", set())) >= min_sup

        def finalize(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any]):
            if len(labels) < self.min_rule_nodes or len(edges) < self.min_edges:
                return
            # 依据“能导出唯一结论”的嵌入过滤支持
            valid_pids: Set[str] = set()
            kept_embs: List[Dict[str, Any]] = []
            for emb in pobj.get("embeddings", []):
                pid = emb["pid"]
                rule_gids = [emb["mapping"].get(i) for i in range(len(labels))]
                if any(gid is None for gid in rule_gids):
                    continue
                Rset = set(rule_gids)
                produced: Set[int] = set()
                consumed: Set[int] = set()
                for rg in Rset:
                    for f in self.G.out_edges.get(rg, []):
                        if self.G.nodes[f]["type"] != "fact" or self.G.nodes[f]["problem_id"] != pid:
                            continue
                        produced.add(f)
                        for r2 in self.G.out_edges.get(f, []):
                            if r2 in Rset and self.G.nodes[r2]["problem_id"] == pid:
                                consumed.add(f)
                conclusion = produced - consumed
                if len(conclusion) != 1:
                    continue
                valid_pids.add(pid)
                kept_embs.append(emb)
            if len(valid_pids) < min_sup:
                return
            if emit is not None:
                rec = {
                    "nodes": [{"idx": i, "label": lb} for i, lb in enumerate(labels)],
                    "labels": list(labels),
                    "edges": list(edges),
                    "pids": set(valid_pids),
                    "embeddings": kept_embs[: self.sample_embeddings],
                    "support": len(valid_pids),
                }
                emit(rec)

        def expand(labels: List[str], edges: List[Tuple[int,int]], pobj: Dict[str, Any]):
            nonlocal expansions
            if time_budget_seconds is not None and (_time.time() - start_time) >= time_budget_seconds:
                finalize(labels, edges, pobj)
                return
            if debug_limit_expansions is not None and expansions >= debug_limit_expansions:
                finalize(labels, edges, pobj)
                return
            if len(labels) >= self.max_nodes:
                finalize(labels, edges, pobj)
                return
            sig = signature(labels, edges)
            if sig in visited:
                return
            visited.add(sig)
            if not try_consume_budget():
                # 没有预算，直接 finalize 当前结构
                finalize(labels, edges, pobj)
                return
            expansions += 1
            if debug_log_every and expansions % max(1, debug_log_every) == 0:
                self.logger.info(f"rules-only expand step={expansions}, cur_nodes={len(labels)}, cur_edges={len(edges)}")

            n = len(labels)
            candidates: Dict[Tuple[Tuple[str,...], Tuple[Tuple[int,int],...]], Dict[str, Any]] = {}
            for anchor_idx in range(n):
                bucket_out: Dict[str, List[Dict[str, Any]]] = {}
                bucket_in: Dict[str, List[Dict[str, Any]]] = {}
                for emb in pobj.get("embeddings", []):
                    pid = emb["pid"]
                    gid_anchor = emb["mapping"].get(anchor_idx)
                    if gid_anchor is None:
                        continue
                    for w in self._rule_out_rules.get(gid_anchor, []):
                        if self.G.nodes[w]["problem_id"] != pid:
                            continue
                        if w in emb["mapping"].values():
                            continue
                        new_label = self.G.nodes[w]["label"]
                        mapping2 = dict(emb["mapping"]) ; mapping2[n] = w
                        bucket_out.setdefault(new_label, []).append({"pid": pid, "mapping": mapping2})
                    for u in self._rule_in_rules.get(gid_anchor, []):
                        if self.G.nodes[u]["problem_id"] != pid:
                            continue
                        if u in emb["mapping"].values():
                            continue
                        new_label = self.G.nodes[u]["label"]
                        mapping2 = dict(emb["mapping"]) ; mapping2[n] = u
                        bucket_in.setdefault(new_label, []).append({"pid": pid, "mapping": mapping2})
                for new_label, emb_list in bucket_out.items():
                    new_labels = labels + [new_label]
                    new_edges = edges + [(anchor_idx, n)]
                    sig2 = signature(new_labels, new_edges)
                    cand = candidates.setdefault(sig2, {"labels": new_labels, "edges": new_edges, "embeddings": []})
                    cand["embeddings"].extend(emb_list)
                for new_label, emb_list in bucket_in.items():
                    new_labels = labels + [new_label]
                    new_edges = edges + [(n, anchor_idx)]
                    sig2 = signature(new_labels, new_edges)
                    cand = candidates.setdefault(sig2, {"labels": new_labels, "edges": new_edges, "embeddings": []})
                    cand["embeddings"].extend(emb_list)

            for (_sig, cand) in candidates.items():
                pids = {e["pid"] for e in cand["embeddings"]}
                cand_pobj = {"pids": pids, "embeddings": cand["embeddings"]}
                if prune_low_support_labels or prune_by_rule:
                    if any((prune_by_rule and label_pid_support.get(lb, 0) < min_sup) for lb in cand["labels"] if lb.startswith("R:")):
                        continue
                if len(cand["labels"]) > self.max_nodes:
                    finalize(cand["labels"], cand["edges"], cand_pobj)
                else:
                    if support_ok(cand_pobj):
                        expand(cand["labels"], cand["edges"], cand_pobj)
                    else:
                        finalize(cand["labels"], cand["edges"], cand_pobj)

        expand(list(seed.get("labels", [])), list(seed.get("edges", [])), seed)

    def pattern_to_schema_rules_only(self, pattern: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """将规则子图模式映射回 G，恢复‘外部前提 => 唯一结论’的 schema。取第一条代表嵌入。"""
        if not pattern.get("embeddings"):
            return "", {}
        emb = pattern["embeddings"][0]
        mapping = emb["mapping"]  # pattern_idx -> rule_gid (in G)
        pid = emb["pid"]
        rule_gids = [mapping.get(i) for i in range(len(pattern.get("labels", [])))]
        if any(gid is None for gid in rule_gids):
            return "", {}
        Rset = set(rule_gids)
        produced: Set[int] = set()
        consumed: Set[int] = set()
        premises_ext: Set[int] = set()
        for rg in Rset:
            # 输入前提
            for fu in self.G.in_edges.get(rg, []):
                if self.G.nodes[fu]["type"] != "fact" or self.G.nodes[fu]["problem_id"] != pid:
                    continue
                premises_ext.add(fu)
            # 输出事实
            for fv in self.G.out_edges.get(rg, []):
                if self.G.nodes[fv]["type"] != "fact" or self.G.nodes[fv]["problem_id"] != pid:
                    continue
                produced.add(fv)
                for r2 in self.G.out_edges.get(fv, []):
                    if r2 in Rset and self.G.nodes[r2]["problem_id"] == pid:
                        consumed.add(fv)
        # 外部前提 = 所有前提 - produced（内部生成）
        premises_ext -= produced
        conclusion = produced - consumed
        if len(conclusion) != 1:
            return "", {}
        concl_gid = next(iter(conclusion))

        # 变量化与渲染
        def vmap_builder():
            return {}
        var_map: Dict[str,str] = vmap_builder()
        def vget(x: str) -> str:
            if x not in var_map:
                var_map[x] = f"X{len(var_map)+1}"
            return var_map[x]
        def fact_term_from_gid(fid: int) -> str:
            orig = self.pg.nodes[self.G.nodes[fid]["orig_node_id"]]
            pred = orig["label"]
            args = orig.get("args", [])
            if pred in {"aconst", "rconst"} and len(args) >= 1:
                head = ",".join(vget(a) for a in args[:-1])
                tail = str(args[-1])
                return f"{pred}(" + (head + "," if head else "") + tail + ")"
            return f"{pred}({','.join(vget(a) for a in args)})"

        premise_terms = [fact_term_from_gid(fid) for fid in sorted(list(premises_ext))]
        concl_term = fact_term_from_gid(concl_gid)
        expr = (" ∧ ".join(premise_terms) + (" => " if premise_terms else "=> ") + concl_term)
        return expr, var_map

    def pattern_to_schema_branched(self, pattern: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """
        将分叉子图模式转换为“合取 => 单一结论”表达：
        - premise = 所有 in-degree=0 的 fact 的合取
        - conclusion = 唯一 out-degree=0 的 fact（若不唯一则返回空表达）
        """
        labels: List[str] = pattern.get("labels", [])
        edges: List[Tuple[int,int]] = pattern.get("edges", [])
        embs = pattern.get("embeddings", [])
        if not labels or not edges or not embs:
            return "", {}
        emb = embs[0]
        mapping = emb.get("mapping", {})

        indeg = [0]*len(labels)
        outdeg = [0]*len(labels)
        for a,b in edges:
            outdeg[a]+=1; indeg[b]+=1

        # 变量化
        var_map: Dict[str, str] = {}
        def vget(x: str) -> str:
            if x not in var_map:
                var_map[x] = f"X{len(var_map)+1}"
            return var_map[x]

        def fact_term(idx: int) -> str:
            gid = mapping.get(idx)
            if gid is None:
                return ""
            node = self.G.nodes[gid]
            orig = self.pg.nodes[node["orig_node_id"]]
            pred = orig["label"]
            args = orig.get("args", [])
            if pred in {"aconst", "rconst"} and len(args) >= 1:
                head = ",".join(vget(a) for a in args[:-1])
                tail = str(args[-1])
                return f"{pred}(" + (head + "," if head else "") + tail + ")"
            return f"{pred}(" + ",".join(vget(a) for a in args) + ")"

        premises = [fact_term(i) for i,lb in enumerate(labels) if lb.startswith("F:") and indeg[i]==0]
        premises = [p for p in premises if p]
        # 只允许唯一结论
        sink_idxs = [i for i,lb in enumerate(labels) if lb.startswith("F:") and outdeg[i]==0]
        if len(sink_idxs) != 1:
            return "", {}
        concl = fact_term(sink_idxs[0])
        if not concl:
            return "", {}
        expr = " ∧ ".join(premises) + " => " + concl
        return expr, var_map


__all__ = ["ProofGraph", "GSpanMiner", "MergedGraph"]
 

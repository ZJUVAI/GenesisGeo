"""证明图构建与图约简。

- 把一条原始合成数据记录解析为内存中的完整证明图（点坐标 + fact/rule 节点 + 边）。
- 图约简：单前提推导折叠（simplify_single_step）。按辅助点拆解见 decomposer.py。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Iterator

from tqdm import tqdm

from newclid.discovery.data_models import (
    FactNode,
    PredicateInstance,
    Point,
    RuleNode,
)
from newclid.discovery.extraction import parsing


# ---------------------------------------------------------------------------
# 证明图
# ---------------------------------------------------------------------------

class SingleProofGraph:
    """从一条 result record 构建的有向证明图（DAG）。

    节点两类：
    - fact 节点（几何命题）：题设前提 / numerical_check（produced_by=None），
      以及 proof 步骤推出的结论（produced_by=对应 rule id）。
    - rule 节点（推理步骤）：premises → rule → conclusion。

    边：(premise_fact_id → rule_id) 与 (rule_id → conclusion_fact_id)。

    另附题级元数据：点坐标、辅助点、goal，以及 seed / index_in_seed 溯源信息。
    """

    def __init__(
        self,
        *,
        problem_id: str,
        seed: int | None,
        index_in_seed: int,
        points: tuple[Point, ...] = (),
        aux_points: tuple[str, ...] = (),
        goal: PredicateInstance | None = None,
    ) -> None:
        self.problem_id = problem_id
        self.seed = seed
        self.index_in_seed = index_in_seed
        self.points = points
        self.aux_points = aux_points
        self.goal = goal
        self.facts: dict[str, FactNode] = {}
        self.rules: list[RuleNode] = []
        self.edges: list[tuple[str, str]] = []
        # 原始 JSONL 记录，便于溯源 / 导出原数据
        self.raw_record: dict[str, Any] | None = None

    # -- 构建辅助 --------------------------------------------------------

    def _add_given_fact(self, pred: str, args: tuple[str, ...], id_local: str) -> None:
        """登记一个给定前提 fact（produced_by=None）。id 重复且内容一致则忽略。"""
        exist = self.facts.get(id_local)
        if exist is not None:
            return  # 首次登记优先，重复 id 忽略
        self.facts[id_local] = FactNode(
            id=id_local, predicate=pred, args=args, produced_by=None,
        )

    def _ensure_conclusion_fact(
        self, pred: str, args: tuple[str, ...], id_local: str, rule_id: str,
    ) -> None:
        """登记 / 回填由 rule 生成的结论 fact。"""
        exist = self.facts.get(id_local)
        if exist is None:
            self.facts[id_local] = FactNode(
                id=id_local, predicate=pred, args=args, produced_by=rule_id,
            )
        elif exist.produced_by is None:
            # 已作为前提登记过，补记生成者
            self.facts[id_local] = dataclasses.replace(exist, produced_by=rule_id)

    # -- 顶层构建 --------------------------------------------------------

    @classmethod
    def build_from_result_record(
        cls,
        record: dict[str, Any],
        *,
        seed: int | None,
        index_in_seed: int,
    ) -> "SingleProofGraph":
        """从一条原始合成数据记录构建完整证明图。

        Parameters
        ----------
        record : dict
            原始 JSONL 记录，需含 ``fl_problem`` / ``llm_input_renamed`` /
            ``llm_output_renamed``。
        seed : int | None
            该记录的 seed。
        index_in_seed : int
            该记录在同一 seed 内的序号（从 0 开始）。

        Returns
        -------
        SingleProofGraph

        Raises
        ------
        ValueError
            proof 步骤引用了不存在的前提 fact id。
        """
        llm_input = record.get("llm_input_renamed", "") or ""
        llm_output = record.get("llm_output_renamed", "") or ""
        fl_problem = record.get("fl_problem", "") or ""

        problem_section = parsing.extract_tag_content(llm_input, "problem") or llm_input
        aux_section = parsing.extract_tag_content(llm_output, "aux")

        graph = cls(
            problem_id=f"{seed}:{index_in_seed}",
            seed=seed,
            index_in_seed=index_in_seed,
            points=parsing.extract_points(fl_problem),
            aux_points=parsing.extract_aux_points(aux_section),
            goal=parsing.extract_goal(problem_section),
        )
        graph.raw_record = record

        # 给定前提 fact：题设 + aux + numerical_check + trivial
        given_sections = [
            problem_section,
            aux_section,
            parsing.extract_tag_content(llm_output, "numerical_check"),
        ]
        for section in given_sections:
            for pred, args, id_local in parsing.parse_fact_segments(section):
                graph._add_given_fact(pred, args, id_local)

        trivial_section = parsing.extract_tag_content(llm_output, "trivial")
        trivial_fact_ids: set[str] = set()
        for pred, args, id_local in parsing.parse_fact_segments(trivial_section):
            trivial_fact_ids.add(id_local)

        # proof 步骤 → rule 节点 + 结论 fact + 边
        proof_text = parsing.extract_tag_content(llm_output, "proof")
        step = 0
        for raw in parsing.strip_tags(proof_text).split(";"):
            if not raw.strip():
                continue
            parsed = parsing.parse_proof_step(raw)
            if parsed is None:
                continue
            step += 1
            concl_pred, concl_args, concl_id, code, premise_ids = parsed
            rule_id = f"R{step}"

            # 去除trivial前提
            filtered_premise_ids = [
                pid for pid in premise_ids
                if pid not in trivial_fact_ids
            ]

            for pid in filtered_premise_ids:
                if pid not in graph.facts:
                    raise ValueError(
                        f"missing premise fact [{pid}] at step {step} "
                        f"(rule {code}) in problem {graph.problem_id}"
                    )
                graph.edges.append((pid, rule_id))

            graph._ensure_conclusion_fact(concl_pred, concl_args, concl_id, rule_id)
            graph.edges.append((rule_id, concl_id))
            graph.rules.append(
                RuleNode(
                    id=rule_id,
                    step=step,
                    code=code,
                    premises=tuple(filtered_premise_ids),
                    conclusion=concl_id,
                )
            )

        return graph

    # -- goal 定位 -------------------------------------------------------

    def find_goal_fact_id(self) -> str | None:
        """定位结论 fact：优先精确匹配 goal 谓词+参数，退化为出度为 0 的推导结论。"""
        goal = self.goal
        if goal is not None:
            for fid, fact in self.facts.items():
                if fact.predicate == goal.predicate and tuple(fact.args) == tuple(goal.args):
                    return fid
        used_as_premise = {u for (u, _v) in self.edges}
        candidates = [
            fid for fid, f in self.facts.items()
            if f.produced_by is not None and fid not in used_as_premise
        ]
        return candidates[-1] if candidates else None

    # -- 单前提推导折叠 --------------------------------------------------

    def simplify_single_step(self) -> int:
        """折叠「单前提直接推出结论」的平凡步骤，返回折叠的步数。

        对某条规则 R：premise 恰为一个 a、conclusion 为 b（a→b），且 b 不是 goal：
        把所有把 b 当前提的规则中的 b 替换为 a（a→b 与 b,x→y 合并为 a,x→y），
        删掉 R 与 fact b。

        只处理「消费 b 的规则」，不扫描整张图，因此不会影响与 b 无关的规则
        （包括前提数为 0 的规则）。

        迭代至不动点；每次至少移除一条规则，保证终止。
        """
        goal_id = self.find_goal_fact_id()

        collapsed = 0
        while True:
            r = next(
                (r for r in self.rules
                 if len(r.premises) == 1 and r.premises[0] != r.conclusion
                 and r.conclusion != goal_id),
                None,
            )
            if r is None:
                break
            a, b = r.premises[0], r.conclusion
            self.rules.remove(r)
            for consumer in self.rules:
                if b in consumer.premises:
                    new_premises = tuple(
                        dict.fromkeys(a if p == b else p for p in consumer.premises)
                    )
                    self.rules[self.rules.index(consumer)] = dataclasses.replace(
                        consumer, premises=new_premises,
                    )
            self.facts.pop(b, None)
            collapsed += 1

        self._rebuild_edges()
        return collapsed

    def _rebuild_edges(self) -> None:
        """根据当前 rules 重建 edges（premise→rule, rule→conclusion）。"""
        self.edges = []
        for r in self.rules:
            for p in r.premises:
                self.edges.append((p, r.id))
            self.edges.append((r.id, r.conclusion))

    # -- 便捷统计 --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """返回该图的规模统计（调试 / 校验用）。"""
        return {
            "problem_id": self.problem_id,
            "seed": self.seed,
            "index_in_seed": self.index_in_seed,
            "n_points": len(self.points),
            "n_facts": len(self.facts),
            "n_rules": len(self.rules),
            "n_edges": len(self.edges),
            "n_given_facts": sum(1 for f in self.facts.values() if f.produced_by is None),
            "goal": None if self.goal is None else self.goal.predicate,
        }


# ---------------------------------------------------------------------------
# 输入读取 + 批量构图
# ---------------------------------------------------------------------------

def _iter_jsonl(input_path: str) -> Iterator[dict[str, Any]]:
    """逐行流式读取 JSONL，跳过空行。"""
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def select_records(
    input_path: str,
    *,
    limit: int | None = None,
    sample: int | None = None,
    random_seed: int | None = None,
) -> list[tuple[int, Any, int, dict[str, Any]]]:
    """读取原始 JSONL 并按 (limit, sample) 规则选出要处理的记录。

    为每条记录分配 (seed, index_in_seed)：index_in_seed 为该记录在同一 seed 内
    出现的序号（从 0 开始），配合 seed 构成文件内唯一的 problem_id。

    采样方式（互斥；都不设则处理全部）：
    - ``limit``：取前 N 条（顺序，流式）。
    - ``sample``：从整个文件中随机取 N 条（需先读完整文件以保证 index_in_seed 一致）。

    Returns
    -------
    list[(line_no, seed, index_in_seed, record)]
        按行号排序，供串行 / 并行构图共用。
    """
    records: list[tuple[int, Any, int, dict[str, Any]]] = []
    per_seed_counter: dict[Any, int] = {}
    for line_no, record in enumerate(
        tqdm(_iter_jsonl(input_path), desc="[part1] 读取数据", unit="行")
    ):
        if sample is None and limit is not None and line_no >= limit:
            break
        seed = record.get("seed")
        index_in_seed = per_seed_counter.get(seed, 0)
        per_seed_counter[seed] = index_in_seed + 1
        records.append((line_no, seed, index_in_seed, record))

    if sample is not None and sample < len(records):
        import random

        rng = random.Random(random_seed)
        chosen = rng.sample(records, sample)
        chosen.sort(key=lambda t: t[0])  # 按行号排序，输出稳定
        return chosen
    return records


def build_proof_graphs(
    input_path: str,
    *,
    limit: int | None = None,
    sample: int | None = None,
    random_seed: int | None = None,
) -> tuple[list[SingleProofGraph], list[dict[str, Any]]]:
    """读取原始 JSONL 数据，逐条构建完整证明图（串行）。

    Parameters
    ----------
    input_path : str
        原始合成数据 JSONL 路径。
    limit : int | None
        仅处理前 limit 条。
    sample : int | None
        随机抽取 sample 条。与 limit 同时给出时，sample 优先。
    random_seed : int | None
        随机采样的随机种子，便于复现。

    Returns
    -------
    (graphs, failures)
        graphs: 成功构建的 SingleProofGraph 列表。
        failures: 构建失败记录 [{seed, index_in_seed, line, error}, ...]。
    """
    chosen = select_records(
        input_path, limit=limit, sample=sample, random_seed=random_seed,
    )

    graphs: list[SingleProofGraph] = []
    failures: list[dict[str, Any]] = []
    for line_no, seed, index_in_seed, record in tqdm(
        chosen, desc="[part1] 构建证明图", unit="条"
    ):
        try:
            graph = SingleProofGraph.build_from_result_record(
                record, seed=seed, index_in_seed=index_in_seed,
            )
            graphs.append(graph)
        except Exception as exc:  # 单条失败不影响整批
            failures.append(
                {
                    "seed": seed,
                    "index_in_seed": index_in_seed,
                    "line": line_no,
                    "error": str(exc),
                }
            )

    return graphs, failures


def graph_contains_predicates(graph: SingleProofGraph, skip_predicates: set[str]) -> bool:
    """判断该图（或子图）的 fact 节点中是否出现 skip_predicates 中的任一谓词。

    用于 Part 1 提取后按 rule_skip_predicates 整图丢弃：只要子证明图中任意一个
    fact（前提或中间结论）命中该类别，就应整张丢弃，而不仅是命题的前提/结论。
    """
    if not skip_predicates:
        return False
    return any(f.predicate in skip_predicates for f in graph.facts.values())


def graph_has_bare_eqpoint(graph: SingleProofGraph) -> bool:
    """判断图中是否存在"裸" eqpoint —— 两个参数都不是辅助点的 eqpoint fact。

    eqpoint 只允许作为辅助点之间的关系出现：``proposition_extractor`` 收集前提时
    排除一切「参数含辅助点的 fact」(见 extract_proposition 的 aux 过滤)，所以带
    辅助点的 eqpoint 天然进不了命题；而两个参数都是普通点的 eqpoint 一旦是给定
    fact（produced_by 为 None）就会被当作前提收进去，泄漏进最终规则。结论侧不会
    出问题：goal 取自原始记录的 ``?`` 子句，不可能是 eqpoint。
    """
    aux = set(graph.aux_points)
    return any(
        f.predicate == "eqpoint" and not (set(f.args) & aux)
        for f in graph.facts.values()
    )


def graph_skip_reason(graph: SingleProofGraph, skip_predicates: set[str]) -> str | None:
    """Part 1 整图丢弃判定：返回丢弃原因（``"predicate"`` / ``"eqpoint"``），保留则 None。

    两条独立规则：

    1. ``"predicate"``：``skip_predicates``（config: ``rule_skip_predicates``）里的
       谓词只要在图中任一 fact 出现就整图丢弃 —— 前提、中间结论、带辅助点的、
       没被最终规则用到的都算。
    2. ``"eqpoint"``：恒定生效、不可配置。见 graph_has_bare_eqpoint —— 裸 eqpoint
       有资格成为命题前提、进而泄漏进提取出的规则，整图丢弃；含辅助点的 eqpoint
       是证明内部的正常中间步骤，保留。

    合并成一个入口是为了让串行/并行两条 part1 路径共用同一套判定，不会各写一份。
    """
    if graph_contains_predicates(graph, skip_predicates):
        return "predicate"
    if graph_has_bare_eqpoint(graph):
        return "eqpoint"
    return None


# ---------------------------------------------------------------------------
# 图约简（剪枝）已由 simplify_single_step（单步折叠）+ decomposer（按辅助点拆解）实现，
# 见 SingleProofGraph.simplify_single_step 与 extraction/decomposer.py。
# ---------------------------------------------------------------------------

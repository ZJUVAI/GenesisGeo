"""按辅助点拆解证明图（Part 1 第二子步）。

节点二分：
- ``aux``：fact 参数中含辅助点（aux_points）。
- ``normal``：不含辅助点。

对每个 normal 节点当根，向 premise 方向反向 BFS：遇 aux 继续向上，遇 normal 停
（该 normal 作为边界叶子/前提），无前序也停。得到的子图仅保留「含 aux 节点」者，
全 normal 的直接忽略。

每个子图仍是一张 SingleProofGraph（可直接复用 visualizer 绘制），
problem_id 形如 ``{父problem_id}#{根节点id}``。
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from newclid.discovery.data_models import PredicateInstance

if TYPE_CHECKING:
    from newclid.discovery.extraction.graph_manager import SingleProofGraph


def _is_aux(fact, aux_set: set[str]) -> bool:
    return bool(set(fact.args) & aux_set)


def decompose_by_aux(graph: "SingleProofGraph") -> list["SingleProofGraph"]:
    """把一张证明图按 aux/normal 拆解为若干子证明图。

    Returns
    -------
    list[SingleProofGraph]
        每个 normal 根节点对应一张子图（仅含 aux 者被保留）。
        无辅助点时返回空列表。
    """
    from newclid.discovery.extraction.graph_manager import SingleProofGraph

    aux_set = set(graph.aux_points)
    if not aux_set:
        return []

    # 结论 fact id -> 产生它的规则（可能多条）
    producers: dict[str, list] = {}
    for r in graph.rules:
        producers.setdefault(r.conclusion, []).append(r)

    subgraphs: list[SingleProofGraph] = []
    for root_id, root_fact in graph.facts.items():
        if _is_aux(root_fact, aux_set):
            continue  # 只对 normal 节点作根

        # 反向 BFS：只穿过 aux 节点继续向上
        inc_facts: set[str] = {root_id}
        inc_rules: list = []
        rule_ids_seen: set[str] = set()
        expanded: set[str] = {root_id}
        to_expand: list[str] = [root_id]
        while to_expand:
            f = to_expand.pop()
            for r in producers.get(f, []):
                if r.id in rule_ids_seen:
                    continue
                rule_ids_seen.add(r.id)
                inc_rules.append(r)
                for p in r.premises:
                    inc_facts.add(p)
                    pf = graph.facts.get(p)
                    if pf is not None and _is_aux(pf, aux_set) and p not in expanded:
                        expanded.add(p)
                        to_expand.append(p)

        # 结构闭包：把「可由子图已有节点推出」的前提吸收为中间节点，消除冗余前提。
        # 吸收要求该节点某条产生规则的前提全部已在集合里 —— 不新增节点。
        derived = {r.conclusion for r in inc_rules}
        changed = True
        while changed:
            changed = False
            for fid in list(inc_facts):
                if fid in derived:
                    continue  # 已是推导结论
                for r in producers.get(fid, []):
                    if r.id in rule_ids_seen:
                        continue
                    if all(p in inc_facts for p in r.premises):
                        inc_rules.append(r)
                        rule_ids_seen.add(r.id)
                        derived.add(fid)
                        changed = True
                        break

        # 仅保留含 aux 节点的子图
        has_aux = any(
            _is_aux(graph.facts[fid], aux_set)
            for fid in inc_facts
            if fid in graph.facts
        )
        if not has_aux:
            continue

        # 组装子证明图（复用 SingleProofGraph）
        # 子图内实际产生者：决定每个 fact 是前提(None)还是中间节点(rule id)
        producer_id: dict[str, str] = {}
        for r in inc_rules:
            producer_id.setdefault(r.conclusion, r.id)
        sub = SingleProofGraph(
            problem_id=f"{graph.problem_id}#{root_id}",
            seed=graph.seed,
            index_in_seed=graph.index_in_seed,
            points=graph.points,
            aux_points=graph.aux_points,
            goal=PredicateInstance(predicate=root_fact.predicate, args=root_fact.args),
        )
        for fid in inc_facts:
            of = graph.facts.get(fid)
            if of is None:
                continue
            sub.facts[fid] = dataclasses.replace(of, produced_by=producer_id.get(fid))
        sub.rules = list(inc_rules)
        sub.raw_record = graph.raw_record
        sub._rebuild_edges()
        subgraphs.append(sub)

    return subgraphs

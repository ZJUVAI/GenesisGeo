#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPruner

职责：按指定规则对证明图（按题目 problem_id 切分）进行迭代修剪。

修剪规则（迭代执行直至稳定）：
1) 对任一规则节点 R：
   - R 的所有前驱 fact 节点都是题目的前提（当前子图中入度为 0 的 fact）；且
   - 与 R 相连的所有 fact 节点（包括它的前提与它的结论）均不包含辅助点（aux）。
   则删除 R 及其相连的边；
   删除后，被 R 指向的结论 fact 将自然成为新的前提（入度为 0）。
2) 在每轮删除后，若存在“前提” fact 节点与任何规则节点都不相连（即度为 0），则删除该 fact 节点。

输出：每题一个 rendered 结构（兼容 ProofGraphVisualizer.render_rendered）：
  {
    "nodes": [{"idx":int, "type":"fact|rule", "label":str}],
    "edges": [[u,v], ...],
    "aux_points": ["m","n",...]
  }

不引入新依赖；仅依赖 ProofGraph 的节点/边与 aux_points、节点的 label/args 元数据。
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Set


class GraphPruner:
    def __init__(self) -> None:
        pass

    # ----------------------------- 公共入口 -----------------------------
    def prune_proof_graph(self, pg: Any) -> Dict[str, Dict[str, Any]]:
        """
        输入：ProofGraph 实例（newclid.data_discovery.proof_graph.ProofGraph）
        输出：{problem_id: rendered_dict}
        """
        # 收集题目 ID
        pids: Set[str] = set()
        for nd in pg.nodes.values():
            pid = nd.get("problem_id")
            if pid is not None:
                pids.add(str(pid))

        out: Dict[str, Dict[str, Any]] = {}
        for pid in sorted(pids):
            rendered = self._prune_single_problem(pg, pid)
            out[pid] = rendered
        return out

    # ----------------------------- 单题修剪 -----------------------------
    def _prune_single_problem(self, pg: Any, problem_id: str) -> Dict[str, Any]:
        pid = str(problem_id)

        # 过滤该题的节点
        nodes_all: Dict[str, Dict[str, Any]] = {
            nid: nd for nid, nd in pg.nodes.items() if nd.get("problem_id") == pid
        }
        # 过滤该题的边
        edges_all: List[Tuple[str, str]] = [
            (u, v)
            for (u, v) in pg.edges
            if u in nodes_all and v in nodes_all
        ]
        # 辅助点集合
        aux_set: Set[str] = set()
        try:
            aux_set = set((pg.aux_points or {}).get(pid, []) or [])
        except Exception:
            aux_set = set()

        # 复制到可变结构
        nodes_alive: Set[str] = set(nodes_all.keys())
        edges_alive: Set[Tuple[str, str]] = set(edges_all)

        def recompute_degrees() -> Tuple[Dict[str, int], Dict[str, int]]:
            indeg = {nid: 0 for nid in nodes_alive}
            outdeg = {nid: 0 for nid in nodes_alive}
            for (u, v) in list(edges_alive):
                if u not in nodes_alive or v not in nodes_alive:
                    # 清理悬挂边
                    try:
                        edges_alive.remove((u, v))
                    except KeyError:
                        pass
                    continue
                outdeg[u] = outdeg.get(u, 0) + 1
                indeg[v] = indeg.get(v, 0) + 1
            return indeg, outdeg

        def is_fact(nid: str) -> bool:
            return nodes_all[nid].get("type") == "fact"

        def is_rule(nid: str) -> bool:
            return nodes_all[nid].get("type") == "rule"

        def fact_contains_aux(nid: str) -> bool:
            if not is_fact(nid):
                return False
            try:
                args = nodes_all[nid].get("args") or []
                return bool(aux_set and any(str(a) in aux_set for a in args))
            except Exception:
                return False

        changed = True
        while changed:
            changed = False
            indeg, outdeg = recompute_degrees()

            # 当前“题目前提”：入度为 0 的 fact
            current_premises: Set[str] = {nid for nid in nodes_alive if is_fact(nid) and indeg.get(nid, 0) == 0}

            # 找到需要删除的规则节点
            rules_to_delete: List[str] = []
            for nid in list(nodes_alive):
                if not is_rule(nid):
                    continue
                # R 的前驱 facts
                preds = [u for (u, v) in edges_alive if v == nid and is_fact(u)]
                # 要求所有前驱为“前提”
                if any(p not in current_premises for p in preds):
                    continue
                # 与 R 相连的全部 fact（前提 + 结论）
                succ_facts = [v for (u, v) in edges_alive if u == nid and is_fact(v)]
                adj_facts = set(preds) | set(succ_facts)
                if not adj_facts:
                    # 无 fact 相连则不处理
                    continue
                # 所有相连 fact 均不含辅助点
                if any(fact_contains_aux(f) for f in adj_facts):
                    continue
                rules_to_delete.append(nid)

            if not rules_to_delete:
                break

            # 删除规则节点及与之相连的边
            for r in rules_to_delete:
                if r not in nodes_alive:
                    continue
                # 删除边
                for (u, v) in list(edges_alive):
                    if u == r or v == r:
                        try:
                            edges_alive.remove((u, v))
                        except KeyError:
                            pass
                # 删除规则节点
                nodes_alive.discard(r)
                changed = True

            # 删除孤立的前提 fact（度为 0）
            indeg, outdeg = recompute_degrees()
            for f in list(nodes_alive):
                if not is_fact(f):
                    continue
                if indeg.get(f, 0) == 0 and outdeg.get(f, 0) == 0:
                    nodes_alive.discard(f)
                    # 连带清边
                    for (u, v) in list(edges_alive):
                        if u == f or v == f:
                            try:
                                edges_alive.remove((u, v))
                            except KeyError:
                                pass
                    changed = True

        # 生成 rendered 结构：重排 idx，组装 label
        alive_sorted = sorted(nodes_alive)
        idx_map: Dict[str, int] = {nid: i for i, nid in enumerate(alive_sorted)}

        def fact_label(nid: str) -> str:
            nd = nodes_all[nid]
            pred = str(nd.get("label", ""))
            args = [str(a) for a in (nd.get("args") or [])]
            return f"{pred}(" + ",".join(args) + ")" if args else pred

        nodes_out: List[Dict[str, Any]] = []
        for nid in alive_sorted:
            nd = nodes_all[nid]
            ntype = nd.get("type")
            if ntype == "fact":
                label = fact_label(nid)
            else:
                label = str(nd.get("code", nd.get("label", "rule")))
            nodes_out.append({"idx": idx_map[nid], "type": ntype, "label": label})

        edges_out: List[List[int]] = []
        for (u, v) in edges_alive:
            if u in idx_map and v in idx_map:
                edges_out.append([idx_map[u], idx_map[v]])

        rendered = {
            "nodes": nodes_out,
            "edges": edges_out,
            # 供可视化着色用
            "aux_points": sorted(list(aux_set)),
        }
        return rendered


__all__ = ["GraphPruner"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPruner

职责：按指定规则对证明图（按题目 problem_id 切分）进行迭代修剪。

修剪流程：
  ① 删除孤立 fact（indeg=0, outdeg=0）
  ② 自底向上递归删减：反复删除不含 aux 的结论层及其前驱规则
  ③ 再删一次孤立 fact
  ④ 提取连通子图（每个 outdeg=0 的 fact 为一个独立子图）
  ⑤ 对每个子图做自顶向下修剪（含兄弟规则保护）
  ⑥ 返回 List[rendered_dict]

自顶向下修剪规则（迭代执行直至稳定）：
1) 对任一规则节点 R：
   - R 的所有前驱 fact 节点都是题目的前提（当前子图中入度为 0 的 fact）；且
   - 与 R 相连的所有 fact 节点（包括它的前提与它的结论）均不包含辅助点（aux）。
   则删除 R 及其相连的边；
   删除后，被 R 指向的结论 fact 将自然成为新的前提（入度为 0）。
2) 在每轮删除后，若存在"前提" fact 节点与任何规则节点都不相连（即度为 0），则删除该 fact 节点。

输出：每题一个 rendered 结构列表（兼容 ProofGraphVisualizer.render_rendered）：
  [{
    "nodes": [{"idx":int, "type":"fact|rule", "label":str}],
    "edges": [[u,v], ...],
    "aux_points": ["m","n",...]
  }, ...]

不引入新依赖；仅依赖 ProofGraph 的节点/边与 aux_points、节点的 label/args 元数据。
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Any, List, Tuple, Set


class GraphPruner:
    def __init__(self) -> None:
        pass

    # ----------------------------- 公共入口 -----------------------------
    def prune_proof_graph(self, pg: Any) -> Dict[str, List[Dict[str, Any]]]:
        """
        输入：ProofGraph 实例（newclid.data_discovery.proof_graph.ProofGraph）
        输出：{problem_id: [rendered_dict, ...]}
        """
        # 收集题目 ID
        pids: Set[str] = set()
        for nd in pg.nodes.values():
            pid = nd.get("problem_id")
            if pid is not None:
                pids.add(str(pid))

        out: Dict[str, List[Dict[str, Any]]] = {}
        for pid in sorted(pids):
            rendered_list = self._prune_single_problem(pg, pid)
            out[pid] = rendered_list
        return out

    # ----------------------------- 单题修剪 -----------------------------
    def _prune_single_problem(self, pg: Any, problem_id: str) -> List[Dict[str, Any]]:
        pid = str(problem_id)

        # 过滤该题的节点
        nodes_all: Dict[str, Dict[str, Any]] = {
            nid: nd for nid, nd in pg.nodes.items() if nd.get("problem_id") == pid
        }
        # 过滤该题的边
        edges_all: List[Tuple[str, str]] = [
            (u, v) for (u, v) in pg.edges if u in nodes_all and v in nodes_all
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

        # ---- 局部工具函数 ----
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

        def _recompute_degrees(
            n_alive: Set[str], e_alive: Set[Tuple[str, str]]
        ) -> Tuple[Dict[str, int], Dict[str, int]]:
            indeg = {nid: 0 for nid in n_alive}
            outdeg = {nid: 0 for nid in n_alive}
            to_remove = []
            for u, v in e_alive:
                if u not in n_alive or v not in n_alive:
                    to_remove.append((u, v))
                    continue
                outdeg[u] = outdeg.get(u, 0) + 1
                indeg[v] = indeg.get(v, 0) + 1
            for e in to_remove:
                e_alive.discard(e)
            return indeg, outdeg

        def _remove_orphan_facts(
            n_alive: Set[str], e_alive: Set[Tuple[str, str]]
        ) -> None:
            """删除度为 0 的孤立 fact 节点。"""
            indeg, outdeg = _recompute_degrees(n_alive, e_alive)
            for f in list(n_alive):
                if not is_fact(f):
                    continue
                if indeg.get(f, 0) == 0 and outdeg.get(f, 0) == 0:
                    n_alive.discard(f)
                    for u, v in list(e_alive):
                        if u == f or v == f:
                            e_alive.discard((u, v))

        def _remove_node(
            nid: str, n_alive: Set[str], e_alive: Set[Tuple[str, str]]
        ) -> None:
            """删除节点及其所有关联边。"""
            n_alive.discard(nid)
            for u, v in list(e_alive):
                if u == nid or v == nid:
                    e_alive.discard((u, v))

        # ================================================================
        # 步骤 ①：初始清理 — 删除孤立 fact
        # ================================================================
        _remove_orphan_facts(nodes_alive, edges_alive)

        # ================================================================
        # 步骤 ②：自底向上递归删减
        # 每轮迭代删除不含 aux 的结论层（outdeg=0 fact）及其前驱规则，
        # 前提是该结论和其所有前驱 fact 都不含 aux。
        # 删除后暴露新的结论候选，继续迭代。
        # ================================================================
        bu_changed = True
        while bu_changed:
            bu_changed = False
            indeg, outdeg = _recompute_degrees(nodes_alive, edges_alive)
            # 结论候选：outdeg=0 的 fact 节点
            conclusion_candidates = [
                f for f in nodes_alive if is_fact(f) and outdeg.get(f, 0) == 0
            ]

            for concl in conclusion_candidates:
                if concl not in nodes_alive:
                    continue
                if fact_contains_aux(concl):
                    continue  # 含 aux 的结论不删

                # 找直接前驱 rule 节点
                parent_rules = {
                    r
                    for (r, v) in edges_alive
                    if v == concl and r in nodes_alive and is_rule(r)
                }

                if not parent_rules:
                    # 无前驱 rule 的非 aux 结论 → 孤立结论，删除
                    _remove_node(concl, nodes_alive, edges_alive)
                    bu_changed = True
                    continue

                # 找这些 rule 的直接前驱 fact 节点
                parent_facts: Set[str] = set()
                for r in parent_rules:
                    for u, v in edges_alive:
                        if v == r and u in nodes_alive and is_fact(u):
                            parent_facts.add(u)

                # 所有直接前驱 fact 都不含 aux → 可删该结论及其前驱 rule
                if parent_facts and all(not fact_contains_aux(f) for f in parent_facts):
                    _remove_node(concl, nodes_alive, edges_alive)
                    for r in parent_rules:
                        _remove_node(r, nodes_alive, edges_alive)
                    bu_changed = True

        # ================================================================
        # 步骤 ③：再删一次孤立 fact
        # ================================================================
        _remove_orphan_facts(nodes_alive, edges_alive)

        # ================================================================
        # 步骤 ④：提取连通子图
        # 每个 outdeg=0 的 fact 为一个子图的结论，通过反向 BFS 收集所有可达祖先。
        # ================================================================
        indeg, outdeg = _recompute_degrees(nodes_alive, edges_alive)
        conclusion_nodes = [
            f for f in nodes_alive if is_fact(f) and outdeg.get(f, 0) == 0
        ]

        if not conclusion_nodes:
            # 无结论节点 → 返回空
            return []

        subgraphs: List[Tuple[Set[str], Set[Tuple[str, str]]]] = []
        for concl in conclusion_nodes:
            # 反向 BFS
            visited: Set[str] = set()
            queue: deque[str] = deque([concl])
            visited.add(concl)
            sub_edges: Set[Tuple[str, str]] = set()
            while queue:
                cur = queue.popleft()
                for u, v in edges_alive:
                    if v == cur and u in nodes_alive and u not in visited:
                        visited.add(u)
                        sub_edges.add((u, v))
                        queue.append(u)
                    elif v == cur and u in visited:
                        sub_edges.add((u, v))
            subgraphs.append((visited, sub_edges))

        # ================================================================
        # 步骤 ⑤：对每个子图做自顶向下修剪
        # ================================================================
        rendered_list: List[Dict[str, Any]] = []
        for sub_nodes, sub_edges in subgraphs:
            # 复制为独立可变集合
            sn = set(sub_nodes)
            se = set(sub_edges)
            self._top_down_prune(
                sn,
                se,
                nodes_all,
                aux_set,
                is_fact,
                is_rule,
                fact_contains_aux,
                _recompute_degrees,
                _remove_orphan_facts,
            )
            # 生成 rendered
            rendered = self._build_rendered(sn, se, nodes_all, aux_set)
            if rendered and rendered.get("nodes"):
                rendered_list.append(rendered)

        return rendered_list

    # ----------------------------- 自顶向下修剪 -----------------------------
    @staticmethod
    def _top_down_prune(
        nodes_alive: Set[str],
        edges_alive: Set[Tuple[str, str]],
        nodes_all: Dict[str, Dict[str, Any]],
        aux_set: Set[str],
        is_fact,
        is_rule,
        fact_contains_aux,
        _recompute_degrees,
        _remove_orphan_facts,
    ) -> None:
        """自顶向下迭代修剪（原有逻辑，含兄弟规则保护）。就地修改 nodes_alive / edges_alive。"""
        changed = True
        while changed:
            changed = False
            indeg, outdeg = _recompute_degrees(nodes_alive, edges_alive)

            current_premises: Set[str] = {
                nid for nid in nodes_alive if is_fact(nid) and indeg.get(nid, 0) == 0
            }

            rules_to_delete: List[str] = []
            for nid in list(nodes_alive):
                if not is_rule(nid):
                    continue
                preds = [u for (u, v) in edges_alive if v == nid and is_fact(u)]
                if any(p not in current_premises for p in preds):
                    continue
                succ_facts = [v for (u, v) in edges_alive if u == nid and is_fact(v)]
                adj_facts = set(preds) | set(succ_facts)
                if not adj_facts:
                    continue
                if any(fact_contains_aux(f) for f in adj_facts):
                    continue

                # 兄弟规则-辅助点牵连保护
                blocked_by_sibling = False
                for p in preds:
                    sibling_rules = [
                        v
                        for (u, v) in edges_alive
                        if u == p and is_rule(v) and v != nid
                    ]
                    if not sibling_rules:
                        continue
                    for r2 in sibling_rules:
                        r2_preds = [
                            u for (u, v) in edges_alive if v == r2 and is_fact(u)
                        ]
                        if any(fact_contains_aux(fp) for fp in r2_preds):
                            blocked_by_sibling = True
                            break
                    if blocked_by_sibling:
                        break
                if blocked_by_sibling:
                    continue

                rules_to_delete.append(nid)

            if not rules_to_delete:
                break

            for r in rules_to_delete:
                if r not in nodes_alive:
                    continue
                for u, v in list(edges_alive):
                    if u == r or v == r:
                        edges_alive.discard((u, v))
                nodes_alive.discard(r)
                changed = True

            # 删除孤立前提 fact
            _remove_orphan_facts(nodes_alive, edges_alive)

    # ----------------------------- 生成 rendered 结构 -----------------------------
    @staticmethod
    def _build_rendered(
        nodes_alive: Set[str],
        edges_alive: Set[Tuple[str, str]],
        nodes_all: Dict[str, Dict[str, Any]],
        aux_set: Set[str],
    ) -> Dict[str, Any]:
        """从存活节点/边生成 rendered 结构。"""
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
        for u, v in edges_alive:
            if u in idx_map and v in idx_map:
                edges_out.append([idx_map[u], idx_map[v]])

        return {
            "nodes": nodes_out,
            "edges": edges_out,
            "aux_points": sorted(list(aux_set)),
        }


__all__ = ["GraphPruner"]

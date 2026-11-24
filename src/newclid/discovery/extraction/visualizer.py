"""证明图可视化（Part 1 调试辅助）。

把 SingleProofGraph 画成一张有向图 PNG：
- fact 节点按角色着色：
    * 前提（给定、不含辅助点）        绿色
    * 辅助前提（给定、含辅助点）      红色
    * 结论（goal 对应 fact）          黄色
    * 中间节点，含辅助点              粉色
    * 中间节点，不含辅助点            蓝色
- rule 节点画成小灰方块，作为前提→结论的连接。

布局：按 DAG 最长路径分层，自顶向下（前提在上、结论在下），不依赖 graphviz。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newclid.discovery.extraction.graph_manager import SingleProofGraph


# 配色
_COLOR_PREMISE = "#66bb6a"  # 绿
_COLOR_AUX_PREMISE = "#ef5350"  # 红
_COLOR_CONCLUSION = "#ffee58"  # 黄
_COLOR_MID_AUX = "#f48fb1"  # 粉
_COLOR_MID = "#42a5f5"  # 蓝
_COLOR_RULE = "#d0d0d0"  # 灰（rule 节点）
_COLOR_EDGE = "#888888"


def _find_goal_fact_id(graph: "SingleProofGraph") -> str | None:
    """定位结论 fact（委托给图自身的方法）。"""
    return graph.find_goal_fact_id()


def _classify(fact, aux_set: set[str], goal_fact_id: str | None) -> str:
    """返回 fact 节点的角色类别。"""
    if fact.id == goal_fact_id:
        return "conclusion"
    has_aux = bool(set(fact.args) & aux_set)
    if fact.produced_by is None:
        return "aux_premise" if has_aux else "premise"
    return "mid_aux" if has_aux else "mid"


_CATEGORY_COLOR = {
    "premise": _COLOR_PREMISE,
    "aux_premise": _COLOR_AUX_PREMISE,
    "conclusion": _COLOR_CONCLUSION,
    "mid_aux": _COLOR_MID_AUX,
    "mid": _COLOR_MID,
}


def _compute_layers(
    node_ids: list[str], edges: list[tuple[str, str]]
) -> dict[str, int]:
    """按最长路径给每个节点分层（源节点层 0）。"""
    preds: dict[str, list[str]] = {n: [] for n in node_ids}
    for u, v in edges:
        if v in preds:
            preds[v].append(u)

    layer: dict[str, int] = {}

    def depth(n: str, stack: set[str]) -> int:
        if n in layer:
            return layer[n]
        if not preds[n] or n in stack:
            layer[n] = 0
            return 0
        stack.add(n)
        d = 1 + max(depth(p, stack) for p in preds[n])
        stack.discard(n)
        layer[n] = d
        return d

    for n in node_ids:
        depth(n, set())
    return layer


def draw_proof_graph(graph: "SingleProofGraph", out_path: str) -> str:
    """把一张证明图画成 PNG，返回输出路径。

    只画 fact（谓词）节点；边直接从各前提谓词连到结论谓词（rule 节点被折叠，
    不关心用的是哪条规则）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from pathlib import Path

    aux_set = set(graph.aux_points)
    goal_fact_id = _find_goal_fact_id(graph)

    # 折叠 rule 节点：premise fact -> conclusion fact
    fact_ids = list(graph.facts.keys())
    fact_edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for r in graph.rules:
        for pid in r.premises:
            e = (pid, r.conclusion)
            if e[0] != e[1] and e not in seen_edges:
                seen_edges.add(e)
                fact_edges.append(e)

    # 分层布局
    layer = _compute_layers(fact_ids, fact_edges)
    by_layer: dict[int, list[str]] = {}
    for n in fact_ids:
        by_layer.setdefault(layer[n], []).append(n)

    pos: dict[str, tuple[float, float]] = {}
    max_layer = max(by_layer) if by_layer else 0
    max_width = max((len(v) for v in by_layer.values()), default=1)
    for lyr, nodes in by_layer.items():
        count = len(nodes)
        for i, n in enumerate(sorted(nodes)):
            x = (i - (count - 1) / 2.0) * 2.0
            y = float(max_layer - lyr)  # 源在上，结论在下
            pos[n] = (x, y)

    # 标签与颜色
    labels: dict[str, str] = {}
    fact_color: dict[str, str] = {}
    for fid, fact in graph.facts.items():
        cat = _classify(fact, aux_set, goal_fact_id)
        fact_color[fid] = _CATEGORY_COLOR[cat]
        args = " ".join(fact.args)
        labels[fid] = f"{fact.predicate}\n{args}" if args else fact.predicate

    # 构 networkx 图
    G = nx.DiGraph()
    G.add_nodes_from(fact_ids)
    G.add_edges_from(fact_edges)

    # 画布尺寸随规模自适应
    fig_w = max(10.0, max_width * 2.2)
    fig_h = max(6.0, (max_layer + 1) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        width=1.1,
        edge_color=_COLOR_EDGE,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=fact_ids,
        node_shape="o",
        node_color=[fact_color[f] for f in fact_ids],
        edgecolors="#333333",
        linewidths=1.2,
        node_size=2200,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    # 图例
    from matplotlib.patches import Patch

    legend = [
        Patch(facecolor=_COLOR_PREMISE, edgecolor="#333", label="premise"),
        Patch(facecolor=_COLOR_AUX_PREMISE, edgecolor="#333", label="aux premise"),
        Patch(facecolor=_COLOR_CONCLUSION, edgecolor="#333", label="conclusion"),
        Patch(facecolor=_COLOR_MID_AUX, edgecolor="#333", label="mid (has aux point)"),
        Patch(facecolor=_COLOR_MID, edgecolor="#333", label="mid"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"proof graph: {graph.problem_id}", fontsize=10)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return str(out)

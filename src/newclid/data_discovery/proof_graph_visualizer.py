#!/usr/bin/env python3
"""
ProofGraph 可视化工具：将题目内的证明图或子图渲染为 PNG。

- 不引入新依赖：使用 networkx + matplotlib + pydot（Graphviz）
- 节点：fact(圆) / rule(方)；前提(入度0)绿色、结论(出度0且唯一)蓝色描边；其余白色
- 提供按题目渲染与按节点子集渲染两种入口
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import pydot

# 颜色与样式
FIG_BASE_SIZE = (15, 20)
FIG_DPI = 200
# 结论节点（唯一出度为 0 的 fact）填充与描边
COLOR_CONCLUSION = "#e0f2ff"
COLOR_CONCLUSION_BORDER = "#007bff"
# 其它 fact 节点：包含辅助点 → 橙色；否则 → 绿色
COLOR_FACT_AUX = "#ffe6cc"       # 浅橙色填充（含辅助点）
COLOR_FACT_NORMAL = "#e0ffe0"    # 浅绿色填充（不含辅助点）
# rule 节点与边框颜色
COLOR_RULE = "#f0f0f0"
COLOR_BORDER = "#333333"


class ProofGraphVisualizer:
    def __init__(self) -> None:
        self.SHOW_DIRECTION_LEGEND = True
        self.DIRECTION_TEXT = "Premises -> Conclusion"
        self.LAYOUT_RANKDIR = 'TB'  # 新增：使布局方向可配置
        self.LAYOUT_RANKSEP = 2.5
        self.LAYOUT_NODESEP = 1.0
        self.SPRING_K = 1.2
        self.SPRING_ITER = 100

    def _ensure_libs(self):
        try:
            import networkx as nx
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError(
                "缺少依赖：需要 networkx 与 matplotlib。请执行\n"
                "  pip install --user networkx matplotlib\n"
                f"导入错误：{type(e).__name__}: {e}"
            )
        try:
            import pydot
        except Exception as e:
            raise RuntimeError(
                "缺少 pydot 库，无法使用 Graphviz 布局。请执行\n"
                "  pip install --user pydot\n"
                "并确保系统已安装 Graphviz（如 'sudo apt-get install graphviz'）。\n"
                f"导入错误：{type(e).__name__}: {e}"
            )

    def _build_nx_for_problem(self, pg, problem_id: str):
        import networkx as nx
        nodes = {nid: nd for nid, nd in pg.nodes.items() if nd.get("problem_id") == str(problem_id)}
        edges = [(u, v) for (u, v) in pg.edges if u in nodes and v in nodes]
        if not nodes:
            raise ValueError(f"problem_id not found or empty graph: {problem_id}")
        
        G = nx.DiGraph()
        # 修正：应用层次化布局指令
        G.graph['graph'] = {
            'rankdir': self.LAYOUT_RANKDIR,
            'ranksep': str(self.LAYOUT_RANKSEP),
            'nodesep': str(self.LAYOUT_NODESEP)
        }
        
        id_map = {}
        # 收集辅助点集合（若上游未提供，则为空集）
        aux_points = set()
        try:
            aux_points = set((pg.aux_points or {}).get(str(problem_id), []) or [])
        except Exception:
            aux_points = set()
        for i, (nid, nd) in enumerate(nodes.items()):
            id_map[nid] = i
            ntype = nd.get("type")
            label = self._fact_label(pg, nd) if ntype == "fact" else f"R:{nd.get('code')}"
            contains_aux = False
            if ntype == "fact":
                try:
                    args = [str(a) for a in (nd.get("args") or [])]
                    contains_aux = any(a in aux_points for a in args)
                except Exception:
                    contains_aux = False
            G.add_node(i, ntype=ntype, label=label, contains_aux=contains_aux)
        for u, v in edges:
            G.add_edge(id_map[u], id_map[v])
        return G

    def _fact_label(self, pg, nd) -> str:
        pred = str(nd.get("label", ""))
        args = [str(a) for a in (nd.get("args") or [])]
        return f"{pred}(" + ",".join(args) + ")"

    def _draw(self, G, out_png: Path, *, label_mode: str = "short", highlight: bool = True, figsize=None, font_size: int = 8):
        import networkx as nx
        import matplotlib.pyplot as plt
        
        indeg = {n: d for n, d in G.in_degree()}
        outdeg = {n: d for n, d in G.out_degree()}
        fact_sinks = [n for n, d in G.nodes(data=True) if d.get("ntype") == "fact" and outdeg.get(n, 0) == 0]
        unique_concl = fact_sinks[0] if len(fact_sinks) == 1 else None

        # 修正：移除 args 参数，依赖图属性
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=self.SPRING_K, iterations=self.SPRING_ITER)

        #... (其余绘图逻辑保持不变)...
        node_colors, node_edge_colors, node_linewidths, node_sizes, node_shapes, labels = {}, {}, {}, {}, {}, {}
        legend_lines: list[str] = []
        # 为 legend 模式准备稳定编号
        rule_counter = 0
        fact_counter = 0
        for n, d in G.nodes(data=True):
            ntype = d.get("ntype")
            raw_label = str(d.get("label", ""))
            if ntype == "rule":
                node_shapes[n], node_sizes[n], node_colors[n], node_edge_colors[n], node_linewidths[n] = 's', 1300, COLOR_RULE, COLOR_BORDER, 1.0
                if label_mode == "legend":
                    rule_counter += 1
                    disp = f"R{rule_counter}"
                    labels[n] = disp
                    legend_lines.append(f"{disp}: {raw_label}")
                else:
                    labels[n] = raw_label if label_mode == "full" else raw_label.split("(", 1)[0]
            else:
                node_shapes[n], node_sizes[n] = 'o', 1600
                # 新规则：结论节点蓝色；其余 fact：含辅助点橙色，否则绿色；前提（入度0）仅加粗边框
                if highlight:
                    if unique_concl is not None and n == unique_concl:
                        node_colors[n], node_edge_colors[n], node_linewidths[n] = COLOR_CONCLUSION, COLOR_CONCLUSION_BORDER, 2.0 if indeg.get(n, 0) == 0 else 1.0
                    else:
                        contains_aux = bool(d.get("contains_aux", False))
                        fill = COLOR_FACT_AUX if contains_aux else COLOR_FACT_NORMAL
                        lw = 2.0 if indeg.get(n, 0) == 0 else 1.0
                        node_colors[n], node_edge_colors[n], node_linewidths[n] = fill, COLOR_BORDER, lw
                else:
                    contains_aux = bool(d.get("contains_aux", False))
                    fill = COLOR_FACT_AUX if contains_aux else COLOR_FACT_NORMAL
                    node_colors[n], node_edge_colors[n], node_linewidths[n] = fill, COLOR_BORDER, 1.0
                if label_mode == "legend":
                    # 结论节点优先用 C
                    if unique_concl is not None and n == unique_concl:
                        disp = "C"
                    else:
                        fact_counter += 1
                        disp = f"F{fact_counter}"
                    labels[n] = disp
                    legend_lines.append(f"{disp}: {raw_label}")
                else:
                    labels[n] = raw_label if label_mode == "full" else raw_label.split("(", 1)[0]

        shape_map = {}
        for idx, shp in node_shapes.items():
            shape_map.setdefault(shp, []).append(idx)

        # legend 模式下默认略微加宽
        eff_figsize = figsize or ( (FIG_BASE_SIZE[0] * 1.25, FIG_BASE_SIZE[1]) if label_mode == "legend" else FIG_BASE_SIZE )
        fig, ax = plt.subplots(figsize=eff_figsize)
        ax.axis("off")
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=15, width=1.2, edge_color="#777777")
        for shp, nodes in shape_map.items():
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shp, node_color=[node_colors[n] for n in nodes], edgecolors=[node_edge_colors[n] for n in nodes], linewidths=[node_linewidths[n] for n in nodes], node_size=[node_sizes[n] for n in nodes], ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, ax=ax)

        # 右上角图例（legend 模式）
        if label_mode == "legend" and legend_lines:
            legend_text = "Legend:\n" + "\n".join(legend_lines)
            ax.text(0.99, 0.99, legend_text, transform=ax.transAxes, ha='right', va='top', fontsize=max(7, font_size-1), bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))

        if self.SHOW_DIRECTION_LEGEND:
            premise_idxs = [n for n, d in G.nodes(data=True) if d.get("ntype") == "fact" and indeg.get(n, 0) == 0]
            premise_labels = [str(labels.get(i, "P")) for i in premise_idxs]
            concl_label = labels.get(unique_concl, "C") if unique_concl is not None else "C"
            txt = "{" + ", ".join(premise_labels) + "} -> " + concl_label
            if len(txt) > 120: txt = txt[:117] + "..."
            ax.text(0.01, 0.99, txt, transform=ax.transAxes, ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))
        
        ax.margins(0.12)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=FIG_DPI)
        plt.close(fig)

    def render_problem(self, pg, problem_id: str, out_png: str | Path, *, label_mode: str = "short", layout: str = "dot", highlight: bool = True, figsize=None):
        self._ensure_libs()
        G = self._build_nx_for_problem(pg, str(problem_id))
        self._draw(G, Path(out_png), label_mode=label_mode, highlight=highlight, figsize=figsize)
        return str(out_png)

    def render_subgraph(self, pg, node_ids: Iterable[str], out_png: str | Path, *, label_mode: str = "short", highlight: bool = True, figsize=None):
        self._ensure_libs()
        import networkx as nx
        node_ids = list(node_ids)
        if not node_ids: raise ValueError("empty node_ids for subgraph rendering")
        pid = pg.nodes[node_ids[0]]["problem_id"]
        full = self._build_nx_for_problem(pg, pid)
        self._draw(full, Path(out_png), label_mode=label_mode, highlight=highlight, figsize=figsize)
        return str(out_png)

    def render_rendered(self, rendered: dict, out_png: str | Path, *, label_mode: str = "short", highlight: bool = True, figsize=None, font_size: int = 8, show_direction_legend: Optional[bool] = None, layout_ranksep: Optional[float] = None, layout_nodesep: Optional[float] = None):
        self._ensure_libs()
        import networkx as nx
        import matplotlib.pyplot as plt

        nodes = (rendered or {}).get("nodes") or []
        edges = (rendered or {}).get("edges") or []
        if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes:
            raise ValueError("rendered 缺少 nodes/edges 或为空")

        # 修正：使用正确、简洁的度计算逻辑
        indeg = {n.get("idx"): 0 for n in nodes}
        outdeg = {n.get("idx"): 0 for n in nodes}
        for u, v in edges:
            outdeg[u] = outdeg.get(u, 0) + 1
            indeg[v] = indeg.get(v, 0) + 1
        fact_sinks = [n for n in nodes if n.get("type") == "fact" and outdeg.get(n.get("idx"), 0) == 0]
        unique_concl_idx = (fact_sinks[0]["idx"]) if len(fact_sinks) == 1 else None

        G = nx.DiGraph()
        # 修正：应用层次化布局指令
        G.graph['graph'] = {
            'rankdir': self.LAYOUT_RANKDIR,
            'ranksep': str(layout_ranksep if layout_ranksep is not None else self.LAYOUT_RANKSEP),
            'nodesep': str(layout_nodesep if layout_nodesep is not None else self.LAYOUT_NODESEP)
        }
        # 渲染路径的辅助点集合（若提供）
        aux_points = set((rendered or {}).get("aux_points") or [])
        
        for n in nodes:
            idx, ntype, raw_label = n.get("idx"), n.get("type"), str(n.get("label", ""))
            lbl = raw_label if label_mode == "full" else raw_label.split("(", 1)[0]
            contains_aux = False
            if ntype == "fact":
                # 1) 优先使用节点自带标记
                contains_aux = bool(n.get("contains_aux") or n.get("aux") or n.get("is_aux"))
                # 2) 尝试根据 args 或 label 解析
                if not contains_aux:
                    args = n.get("args")
                    if isinstance(args, list):
                        try:
                            contains_aux = any(str(a) in aux_points for a in args)
                        except Exception:
                            contains_aux = False
                    else:
                        # 从 label 中解析 pred(a,b,...) 形式
                        try:
                            if "(" in raw_label and ")" in raw_label:
                                inside = raw_label.split("(", 1)[1].rsplit(")", 1)[0]
                                parts = [p.strip() for p in inside.split(",") if p.strip()]
                                contains_aux = any(p in aux_points for p in parts)
                        except Exception:
                            contains_aux = False
            # 同时保留 raw_label 以供 legend 使用
            G.add_node(idx, ntype=ntype, label=lbl, contains_aux=contains_aux, raw_label=raw_label)
        for u, v in edges:
            G.add_edge(u, v)

        # 修正：移除 args 参数，依赖图属性
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=self.SPRING_K, iterations=self.SPRING_ITER)

        #... (其余绘图逻辑保持不变)...
        node_colors, node_edge_colors, node_linewidths, node_sizes, node_shapes = {}, {}, {}, {}, {}
        legend_lines: list[str] = []
        rule_counter = 0
        fact_counter = 0
        for n, d in G.nodes(data=True):
            ntype = d.get("ntype")
            if ntype == "rule":
                node_shapes[n], node_sizes[n], node_colors[n], node_edge_colors[n], node_linewidths[n] = 's', 1300, COLOR_RULE, COLOR_BORDER, 1.0
                if label_mode == "legend":
                    rule_counter += 1
                    disp = f"R{rule_counter}"
                    G.nodes[n]["label"] = disp
                    legend_lines.append(f"{disp}: {d.get('raw_label', d.get('label',''))}")
            else:
                node_shapes[n], node_sizes[n] = 'o', 1600
                if highlight:
                    if unique_concl_idx is not None and n == unique_concl_idx:
                        node_colors[n], node_edge_colors[n], node_linewidths[n] = COLOR_CONCLUSION, COLOR_CONCLUSION_BORDER, 2.0 if indeg.get(n, 0) == 0 else 1.0
                    else:
                        contains_aux = bool(d.get("contains_aux", False))
                        fill = COLOR_FACT_AUX if contains_aux else COLOR_FACT_NORMAL
                        lw = 2.0 if indeg.get(n, 0) == 0 else 1.0
                        node_colors[n], node_edge_colors[n], node_linewidths[n] = fill, COLOR_BORDER, lw
                else:
                    contains_aux = bool(d.get("contains_aux", False))
                    fill = COLOR_FACT_AUX if contains_aux else COLOR_FACT_NORMAL
                    node_colors[n], node_edge_colors[n], node_linewidths[n] = fill, COLOR_BORDER, 1.0
                if label_mode == "legend":
                    if unique_concl_idx is not None and n == unique_concl_idx:
                        disp = "C"
                    else:
                        fact_counter += 1
                        disp = f"F{fact_counter}"
                    G.nodes[n]["label"] = disp
                    legend_lines.append(f"{disp}: {d.get('raw_label', d.get('label',''))}")

        shape_map = {}
        for idx, shp in node_shapes.items():
            shape_map.setdefault(shp, []).append(idx)

        eff_figsize = figsize or ( (FIG_BASE_SIZE[0] * 1.25, FIG_BASE_SIZE[1]) if label_mode == "legend" else FIG_BASE_SIZE )
        fig, ax = plt.subplots(figsize=eff_figsize)
        ax.axis("off")
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=15, width=1.2, edge_color="#777777")
        for shp, nodelist in shape_map.items():
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_shape=shp, node_color=[node_colors[n] for n in nodelist], edgecolors=[node_edge_colors[n] for n in nodelist], linewidths=[node_linewidths[n] for n in nodelist], node_size=[node_sizes[n] for n in nodelist], ax=ax)
        labels = {n: d.get("label", "") for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, ax=ax)

        # 右上角图例（legend 模式）
        if label_mode == "legend" and legend_lines:
            legend_text = "Legend:\n" + "\n".join(legend_lines)
            ax.text(0.99, 0.99, legend_text, transform=ax.transAxes, ha='right', va='top', fontsize=max(7, font_size-1), bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))

        sd = self.SHOW_DIRECTION_LEGEND if show_direction_legend is None else bool(show_direction_legend)
        if sd:
            premise_idxs = [n.get("idx") for n in nodes if n.get("type") == "fact" and indeg.get(n.get("idx"), 0) == 0]
            premise_labels = [labels.get(i, "P") for i in premise_idxs]
            concl_label = labels.get(unique_concl_idx, "C") if unique_concl_idx is not None else "C"
            txt = "{" + ", ".join(premise_labels) + "} -> " + concl_label
            if len(txt) > 120: txt = txt[:117] + "..."
            ax.text(0.01, 0.99, txt, transform=ax.transAxes, ha='left', va='top', fontsize=max(8, font_size), bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))

        ax.margins(0.12)
        fig.tight_layout()
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=FIG_DPI)
        plt.close(fig)
        return str(out_png)
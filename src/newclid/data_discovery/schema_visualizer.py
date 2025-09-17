#!/usr/bin/env python3
"""
Schema visualizer: 将 schema 的 rendered 图结构批量渲染为 PNG。

功能要点：
- 复用现有 scripts/visualize_schemas.py 的 networkx + matplotlib + pydot(Graphviz dot) 渲染风格；
- 提供两类入口：
  1) render_patterns: 直接从挖掘产物的 patterns[*].rendered 批量出图；
  2) render_from_rely_on_file: 读取二次筛选产物 partition_by_rely_on.json 的分组，分别出图；
- 不新增依赖；缺依赖时给出清晰提示；dot 失败回退 spring_layout；
- 以最小改动满足“在 filt_schemas.py 末尾接入绘图”的需求。
"""

import json
from pathlib import Path

# 可调整的默认样式
FIG_BASE_SIZE = (12, 9)
FIG_DPI = 200
COLOR_PREMISE = "#e0ffe0"         # Light Green
COLOR_CONCLUSION = "#e0f2ff"      # Light Blue
COLOR_INTERMEDIATE_FACT = "#FFFFFF"  # White
COLOR_RULE = "#f0f0f0"            # Very Light Grey
COLOR_BORDER = "#333333"
COLOR_CONCLUSION_BORDER = "#007bff"  # Blue


class SchemaVisualizer:
    """批量渲染 schema 的工具类。"""

    def __init__(self):
        pass

    # ---------------- 内部工具 ----------------
    def _predicate_name(self, label):
        try:
            p = label.split("(", 1)[0].strip()
            return p if p else label
        except Exception:
            return label

    def _save_png_nx(self, rendered, out_png, *, label_mode="full", highlight_kind=True):
        """使用 networkx + matplotlib 渲染 PNG。优先 Graphviz dot 布局，失败退回 spring_layout。"""
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
            import pydot  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "缺少 pydot 库，无法使用 Graphviz 布局。请执行\n"
                "  pip install --user pydot\n"
                "并确保系统已安装 Graphviz（如 'sudo apt-get install graphviz'）。\n"
                f"导入错误：{type(e).__name__}: {e}"
            )

        nodes = rendered.get("nodes") or []
        edges = rendered.get("edges") or []
        if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes or not edges:
            raise ValueError("rendered 缺少 nodes/edges 或为空")

        # 统计入度/出度，用于前提/结论高亮
        indeg = {n.get("idx"): 0 for n in nodes}
        outdeg = {n.get("idx"): 0 for n in nodes}
        for u, v in edges:
            outdeg[u] = outdeg.get(u, 0) + 1
            indeg[v] = indeg.get(v, 0) + 1
        fact_sinks = [n for n in nodes if n.get("type") == "fact" and outdeg.get(n.get("idx", -1), 0) == 0]
        unique_concl_idx = fact_sinks[0]["idx"] if len(fact_sinks) == 1 else None

        import networkx as nx  # type: ignore
        G = nx.DiGraph()
        G.graph['graph'] = {'rankdir': 'TB', 'ranksep': '0.7'}
        G.graph['node'] = {'shape': 'circle'}
        G.graph['edges'] = {'arrowsize': '0.8'}

        # 节点
        for n in nodes:
            idx = n.get("idx")
            ntype = n.get("type")
            raw_label = str(n.get("label", ""))
            label = self._predicate_name(raw_label) if label_mode == "short" else raw_label
            G.add_node(idx, label=label, ntype=ntype)
        # 边
        for u, v in edges:
            G.add_edge(u, v)

        # 优先使用 graphviz 布局
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        # 风格
        node_colors, node_edge_colors, node_linewidths, node_sizes, node_shapes = {}, {}, {}, {}, {}
        for n_idx, n_data in G.nodes(data=True):
            ntype = n_data.get("ntype")
            if ntype == "rule":
                node_shapes[n_idx] = 's'
                node_colors[n_idx] = COLOR_RULE
                node_edge_colors[n_idx] = COLOR_BORDER
                node_linewidths[n_idx] = 1.0
                node_sizes[n_idx] = 1300
            else:
                node_shapes[n_idx] = 'o'
                node_sizes[n_idx] = 1600
                if highlight_kind:
                    if indeg.get(n_idx, 0) == 0:
                        node_colors[n_idx] = COLOR_PREMISE
                        node_edge_colors[n_idx] = COLOR_BORDER
                        node_linewidths[n_idx] = 1.0
                    elif n_idx == unique_concl_idx:
                        node_colors[n_idx] = COLOR_CONCLUSION
                        node_edge_colors[n_idx] = COLOR_CONCLUSION_BORDER
                        node_linewidths[n_idx] = 2.0
                    else:
                        node_colors[n_idx] = COLOR_INTERMEDIATE_FACT
                        node_edge_colors[n_idx] = COLOR_BORDER
                        node_linewidths[n_idx] = 1.0
                else:
                    node_colors[n_idx] = COLOR_INTERMEDIATE_FACT
                    node_edge_colors[n_idx] = COLOR_BORDER
                    node_linewidths[n_idx] = 1.0

        shape_map = {}
        for idx, shape in node_shapes.items():
            shape_map.setdefault(shape, []).append(idx)

        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=FIG_BASE_SIZE)
        ax.axis("off")

        # 边
        import networkx as nx  # type: ignore
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=15, width=1.2, edge_color="#777777")

        # 节点
        for shape, nodelist in shape_map.items():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodelist, node_shape=shape,
                node_color=[node_colors[n] for n in nodelist],
                edgecolors=[node_edge_colors[n] for n in nodelist],
                linewidths=[node_linewidths[n] for n in nodelist],
                node_size=[node_sizes[n] for n in nodelist],
                ax=ax,
            )

        # 标签
        labels = {n_idx: n_data["label"] for n_idx, n_data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        ax.margins(0.1)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=FIG_DPI)
        plt.close(fig)

    # ---------------- 公共接口 ----------------
    def render_patterns(self, patterns, out_dir, *, max_items=0, overwrite=True, label_mode="full", highlight_kind=True):
        """从包含 rendered 的 patterns 批量渲染。返回统计信息。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        done = 0
        skipped = 0
        failed = 0
        for idx, p in enumerate(patterns):
            if max_items and done >= max_items:
                break
            rendered = (p or {}).get("rendered") or {}
            nodes = rendered.get("nodes")
            edges = rendered.get("edges")
            if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes or not edges:
                skipped += 1
                continue
            out_png = out_dir / f"schema_{idx:04d}.png"
            if out_png.exists() and not overwrite:
                skipped += 1
                continue
            try:
                self._save_png_nx(rendered, out_png, label_mode=label_mode, highlight_kind=highlight_kind)
                done += 1
            except Exception:
                failed += 1
            total += 1
        return {"total": total, "done": done, "skipped": skipped, "failed": failed, "out_dir": str(out_dir)}

    def render_from_rely_on_file(self, partition_json_path, buckets, base_out_dir, *, max_items_per_bucket=0, overwrite=True, label_mode="full", highlight_kind=True):
        """读取 partition_by_rely_on.json 的指定分组并分别渲染到子目录。返回各桶统计。"""
        partition_json_path = Path(partition_json_path)
        base_out_dir = Path(base_out_dir)
        obj = json.loads(partition_json_path.read_text(encoding="utf-8"))
        stats = {}
        for b in buckets:
            arr = obj.get(b)
            if not isinstance(arr, list):
                stats[b] = {"total": 0, "done": 0, "skipped": 0, "failed": 0, "out_dir": str(base_out_dir / b)}
                continue
            # 兼容两种位置的 rendered：顶层或 item.rendered
            patterns_like = []
            for rec in arr:
                if isinstance(rec, dict):
                    rendered = rec.get("rendered")
                    if not isinstance(rendered, dict) or not rendered:
                        item = rec.get("item") if isinstance(rec.get("item"), dict) else None
                        if item and isinstance(item.get("rendered"), dict):
                            rendered = item.get("rendered")
                    if isinstance(rendered, dict):
                        patterns_like.append({"rendered": rendered})
            out_dir = base_out_dir / b
            st = self.render_patterns(
                patterns_like, out_dir,
                max_items=max_items_per_bucket,
                overwrite=overwrite,
                label_mode=label_mode,
                highlight_kind=highlight_kind,
            )
            stats[b] = st
        return stats

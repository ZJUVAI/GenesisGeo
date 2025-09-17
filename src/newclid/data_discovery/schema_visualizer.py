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
import textwrap
from pathlib import Path

# 可调整的默认样式
FIG_BASE_SIZE = (12, 9)
FIG_DPI = 200
COLOR_PREMISE = "#e0ffe0"         # Light Green (premise in union)
COLOR_PREMISE_OUT = "#ffe4b5"     # Light Orange/Moccasin (premise not in union)
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

    def _parse_pred_args(self, label: str):
        """解析形如 'pred(a,b,...)' 的标签，返回 (pred, [args])；失败返回 (label, [])."""
        try:
            if "(" in label and label.endswith(")"):
                pred, rest = label.split("(", 1)
                args = rest[:-1]
                parts = [a.strip() for a in args.split(",") if a.strip()]
                return pred.strip(), parts
            return label, []
        except Exception:
            return label, []

    def _fact_points_from_label(self, label: str):
        """从 fact 文本标签中提取参与点（忽略 aconst/rconst 的尾常量）。"""
        pred, args = self._parse_pred_args(str(label))
        if pred in {"aconst", "rconst"} and len(args) >= 1:
            return [a for a in args[:-1]]
        return list(args)

    def _save_png_nx(self, rendered, out_png, *, label_mode="full", highlight_kind=True,
                      enable_union_rely_styling: bool = False, viz_meta: dict | None = None,
                      style_opts: dict | None = None):
        """使用 networkx + matplotlib 渲染 PNG。优先 Graphviz dot 布局，失败退回 spring_layout。

        参数：
        - label_mode: "full" | "short" | "legend"
        - enable_union_rely_styling: 是否基于 union_rely 对前提进行分色
        - viz_meta: 可包含 {"union_rely": List[str], "relation": str}
        - style_opts: 可覆写颜色/字号/尺寸等，如 {"figsize": (w,h), "font_size": 8}
        """
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

    # indeg/outdeg 已计算（见上）

        # union_rely 元数据（变量名域）
        union_rely_set = set()
        relation_str = None
        schema_text = None
        if isinstance(viz_meta, dict):
            ur = viz_meta.get("union_rely")
            if isinstance(ur, (list, tuple, set)):
                union_rely_set = set(str(x) for x in ur)
            rel = viz_meta.get("relation")
            if isinstance(rel, str):
                relation_str = rel
            sch = viz_meta.get("schema")
            if isinstance(sch, str) and sch.strip():
                schema_text = sch.strip()

        # legend 模式准备：为节点创建短编号与图例条目
        legend_lines: list[str] = []
        fact_counter = 0
        rule_counter = 0
        displayed_label_by_idx: dict[int, str] = {}

        # 节点
        for n in nodes:
            idx = n.get("idx")
            ntype = n.get("type")
            raw_label = str(n.get("label", ""))
            if label_mode == "short":
                label = self._predicate_name(raw_label)
            elif label_mode == "legend":
                if ntype == "rule":
                    rule_counter += 1
                    label = f"R{rule_counter}"
                    legend_lines.append(f"{label}: {raw_label}")
                else:
                    fact_counter += 1
                    # 结论节点使用 C 更直观
                    if unique_concl_idx is not None and idx == unique_concl_idx:
                        label = "C"
                    else:
                        label = f"F{fact_counter}"
                    legend_lines.append(f"{label}: {raw_label}")
                displayed_label_by_idx[idx] = label
            else:
                label = raw_label
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
                    # premise?
                    if indeg.get(n_idx, 0) == 0:
                        if enable_union_rely_styling and union_rely_set:
                            # 检查该前提 fact 的点是否包含于 union_rely
                            # 通过标签解析点名（与 schema 变量域对应）
                            raw_lab = next((nn.get("label") for nn in nodes if nn.get("idx") == n_idx), None)
                            pts = set(self._fact_points_from_label(str(raw_lab)))
                            node_colors[n_idx] = COLOR_PREMISE if pts.issubset(union_rely_set) else COLOR_PREMISE_OUT
                        else:
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
        # 尺寸与字体
        figsize = FIG_BASE_SIZE
        if label_mode == "legend":
            # 略微加宽画布，为图例留白
            figsize = (FIG_BASE_SIZE[0] * 1.35, FIG_BASE_SIZE[1])
        if isinstance(style_opts, dict) and isinstance(style_opts.get("figsize"), (list, tuple)):
            try:
                w, h = style_opts.get("figsize")
                figsize = (float(w), float(h))
            except Exception:
                pass
        fig, ax = plt.subplots(figsize=figsize)
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
        font_size = 8
        if isinstance(style_opts, dict) and isinstance(style_opts.get("font_size"), (int, float)):
            try:
                font_size = int(style_opts.get("font_size"))
            except Exception:
                pass
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, ax=ax)

        # 图例（legend 模式）
        if label_mode == "legend" and legend_lines:
            legend_text = "Legend:\n" + "\n".join(legend_lines)
            ax.text(0.99, 0.99, legend_text, transform=ax.transAxes, ha='right', va='top',
                    fontsize=max(7, font_size-1),
                    bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))

        # schema 文本（左上角）
        if schema_text:
            wrapped = textwrap.fill(f"schema: {schema_text}", width=100)
            ax.text(0.01, 0.99, wrapped, transform=ax.transAxes, ha='left', va='top',
                    fontsize=max(7, font_size-1),
                    bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))

        # union_rely 注释框
        if union_rely_set:
            info = f"union_rely = {{{', '.join(sorted(list(union_rely_set)))}}}"
            ax.text(0.01, 0.01, info, transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=max(7, font_size-1),
                    bbox=dict(boxstyle='round', fc='white', ec='#999999', alpha=0.85))

        # 适当边距，避免裁剪
        ax.margins(0.12 if label_mode == "legend" else 0.1)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=FIG_DPI)
        plt.close(fig)

    # ---------------- 公共接口 ----------------
    def render_patterns(self, patterns, out_dir, *, max_items=0, overwrite=True, label_mode="full", highlight_kind=True,
                        enable_union_rely_styling: bool = False, style_opts: dict | None = None,
                        show_progress: bool = True, progress_every: int = 5, progress_prefix: str = "render"):
        """从包含 rendered 的 patterns 批量渲染。返回统计信息。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        done = 0
        skipped = 0
        failed = 0
        planned = len(patterns) if isinstance(patterns, (list, tuple)) else 0
        if max_items and planned:
            planned = min(planned, max_items)
        if show_progress and planned:
            print(f"[{progress_prefix}] start: planned={planned}, out={out_dir}")
        for idx, p in enumerate(patterns):
            if max_items and done >= max_items:
                break
            rendered = (p or {}).get("rendered") or {}
            nodes = rendered.get("nodes")
            edges = rendered.get("edges")
            if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes or not edges:
                skipped += 1
                if show_progress and progress_every and ((idx + 1) % progress_every == 0):
                    print(f"[{progress_prefix}] {idx+1}/{planned or (idx+1)} done={done} skipped={skipped} failed={failed}")
                continue
            out_png = out_dir / f"schema_{idx:04d}.png"
            if out_png.exists() and not overwrite:
                skipped += 1
                if show_progress and progress_every and ((idx + 1) % progress_every == 0):
                    print(f"[{progress_prefix}] {idx+1}/{planned or (idx+1)} done={done} skipped={skipped} failed={failed}")
                continue
            try:
                self._save_png_nx(rendered, out_png, label_mode=label_mode, highlight_kind=highlight_kind,
                                   enable_union_rely_styling=enable_union_rely_styling, viz_meta=None,
                                   style_opts=style_opts)
                done += 1
            except Exception:
                failed += 1
            total += 1
            if show_progress and progress_every and ((idx + 1) % progress_every == 0):
                print(f"[{progress_prefix}] {idx+1}/{planned or (idx+1)} done={done} skipped={skipped} failed={failed}")
        if show_progress and (planned or total):
            print(f"[{progress_prefix}] finished: total={planned or total} done={done} skipped={skipped} failed={failed}")
        return {"total": total, "done": done, "skipped": skipped, "failed": failed, "out_dir": str(out_dir)}

    def render_from_rely_on_file(self, partition_json_path, buckets, base_out_dir, *,
                                 max_items_per_bucket=0, overwrite=True, label_mode="full", highlight_kind=True,
                                 enable_union_rely_styling: bool = True, style_opts: dict | None = None,
                                 show_progress: bool = True, progress_every: int = 5):
        """读取 partition_by_rely_on.json 的指定分组并分别渲染到子目录。返回各桶统计。

        将每条记录中的 union_rely / relation 以 viz_meta 形式传递以支持前提分色与注释。
        """
        partition_json_path = Path(partition_json_path)
        base_out_dir = Path(base_out_dir)
        obj = json.loads(partition_json_path.read_text(encoding="utf-8"))
        stats = {}
        for b in buckets:
            arr = obj.get(b)
            if not isinstance(arr, list):
                stats[b] = {"total": 0, "done": 0, "skipped": 0, "failed": 0, "out_dir": str(base_out_dir / b)}
                continue
            out_dir = base_out_dir / b
            out_dir.mkdir(parents=True, exist_ok=True)

            total = 0; done = 0; skipped = 0; failed = 0
            planned = len(arr)
            if max_items_per_bucket:
                planned = min(planned, max_items_per_bucket)
            if show_progress:
                print(f"[viz:{b}] start: planned={planned}, out={out_dir}")
            for idx, rec in enumerate(arr):
                if max_items_per_bucket and done >= max_items_per_bucket:
                    break
                if not isinstance(rec, dict):
                    skipped += 1; continue
                # 兼容两种位置的 rendered：顶层或 item.rendered
                rendered = rec.get("rendered") if isinstance(rec.get("rendered"), dict) else None
                if not rendered:
                    item = rec.get("item") if isinstance(rec.get("item"), dict) else None
                    if item and isinstance(item.get("rendered"), dict):
                        rendered = item.get("rendered")
                if not isinstance(rendered, dict) or not rendered:
                    skipped += 1
                    if show_progress and progress_every and ((idx + 1) % progress_every == 0):
                        print(f"[viz:{b}] {idx+1}/{planned} done={done} skipped={skipped} failed={failed}")
                    continue

                viz_meta = {}
                # 透传 union_rely 与 relation（若存在）
                ur = rec.get("union_rely")
                if isinstance(ur, (list, tuple, set)):
                    viz_meta["union_rely"] = list(ur)
                rel = rec.get("relation")
                if isinstance(rel, str):
                    viz_meta["relation"] = rel
                sch = rec.get("schema")
                if isinstance(sch, str) and sch.strip():
                    viz_meta["schema"] = sch.strip()

                out_png = out_dir / f"schema_{idx:04d}.png"
                if out_png.exists() and not overwrite:
                    skipped += 1
                    if show_progress and progress_every and ((idx + 1) % progress_every == 0):
                        print(f"[viz:{b}] {idx+1}/{planned} done={done} skipped={skipped} failed={failed}")
                    continue
                try:
                    self._save_png_nx(
                        rendered, out_png,
                        label_mode=label_mode,
                        highlight_kind=highlight_kind,
                        enable_union_rely_styling=enable_union_rely_styling,
                        viz_meta=viz_meta if viz_meta else None,
                        style_opts=style_opts,
                    )
                    done += 1
                except Exception:
                    failed += 1
                total += 1
                if show_progress and progress_every and ((idx + 1) % progress_every == 0):
                    print(f"[viz:{b}] {idx+1}/{planned} done={done} skipped={skipped} failed={failed}")

            stats[b] = {"total": total, "done": done, "skipped": skipped, "failed": failed, "out_dir": str(out_dir)}
            if show_progress:
                print(f"[viz:{b}] finished: total={planned} done={done} skipped={skipped} failed={failed}")
        return stats

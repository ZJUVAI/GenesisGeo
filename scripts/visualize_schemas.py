#!/usr/bin/env python3
"""
基于挖掘/筛选结果中的 rendered 信息可视化 schema（每个 schema 输出一张 PNG）。

用法：python scripts/visualize_schemas.py
所有行为通过文件头部常量控制；无需命令行参数。

说明：本脚本现在推荐使用 Graphviz 的 dot 引擎以获得清晰的层次布局。
请确保已安装 Graphviz 和 pydot 库。
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------------- 配置区（按需修改） ----------------
# 仓库根目录：本脚本位于 repo_root/scripts/ 下
REPO_ROOT = Path(__file__).resolve().parent.parent

# 默认输入：筛选后的最终 JSON（包含 patterns 数组，每项含 rendered/nodes/edges）
INPUT_PATH = REPO_ROOT / "src/newclid/data_discovery/data/filt_out/branched_mining.json"
# 输出目录：仅产出 PNG 文件
OUTPUT_DIR = REPO_ROOT / "src/newclid/data_discovery/data/schema_fig"
# 最多导出多少个 schema（避免一次性生成过多图片）；设为 0 表示导出全部
MAX_PATTERNS = 0
# 已存在同名文件是否覆盖
OVERWRITE = True
# 是否高亮前提/结论（仅 fact 节点）：前提=入度0；结论=出度0且唯一
HIGHLIGHT_KIND = True
# 节点标签显示："full" 显示 rendered.nodes[*].label 原文；"short" 仅显示谓词名
LABEL_VERBOSITY = "full"  # "full" | "short"

# MODIFIED: Increased base figure size again to provide more canvas space
FIG_BASE_SIZE = (12, 9) # Base size for the figure canvas (increased again)
FIG_DPI = 200 # Higher resolution output

# NEW: Define colors for better premise/conclusion highlighting
COLOR_PREMISE = "#e0ffe0"  # Light Green
COLOR_CONCLUSION = "#e0f2ff"  # Light Blue
COLOR_INTERMEDIATE_FACT = "#FFFFFF"  # White
COLOR_RULE = "#f0f0f0"      # Very Light Grey
COLOR_BORDER = "#333333"
COLOR_CONCLUSION_BORDER = "#007bff" # Blue border for conclusion


# ---------------- 工具方法 ----------------
def _escape_label(text: str) -> str:
    """转义用于 DOT 的标签文本。"""
    # 使用双引号包裹，内部转义双引号与反斜杆
    return (
        text.replace("\\", "\\\\").replace("\"", "\\\"")
    )


def _predicate_name(label: str) -> str:
    """从完整原子文本提取谓词名，如 eqangle(X1,...) -> eqangle；失败则原样返回。"""
    try:
        p = label.split("(", 1)[0].strip()
        return p if p else label
    except Exception:
        return label


def _save_png_nx(rendered: Dict, out_png: Path, label_verbosity: str = "full", highlight_kind: bool = False) -> None:
    """使用 networkx + matplotlib 渲染 PNG，依赖系统 graphviz 以获得层次布局。"""
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")  # 无需显示环境
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "缺少依赖：需要 networkx 与 matplotlib。请执行\n"
            "  pip install --user networkx matplotlib\n"
            f"导入错误：{type(e).__name__}: {e}"
        )

    # Check for pydot
    try:
        import pydot
    except ImportError as e:
         raise SystemExit(
            "缺少 pydot 库，无法使用 Graphviz 布局。请执行\n"
            "  pip install --user pydot\n"
            "并确保系统已安装 Graphviz (e.g., 'brew install graphviz' or 'sudo apt-get install graphviz')\n"
            f"导入错误：{type(e).__name__}: {e}"
        )

    nodes = rendered.get("nodes") or []
    edges = rendered.get("edges") or []

    # 统计入度/出度
    indeg: Dict[int, int] = {n.get("idx"): 0 for n in nodes}
    outdeg: Dict[int, int] = {n.get("idx"): 0 for n in nodes}
    for u, v in edges:
        outdeg[u] = outdeg.get(u, 0) + 1
        indeg[v] = indeg.get(v, 0) + 1

    fact_sinks = [n for n in nodes if n.get("type") == "fact" and outdeg.get(n.get("idx", -1), 0) == 0]
    unique_concl_idx = fact_sinks[0]["idx"] if len(fact_sinks) == 1 else None

    G = nx.DiGraph()
    # Set graph attributes for Graphviz layout
    G.graph['graph'] = {'rankdir': 'TB', 'ranksep': '0.7'}
    G.graph['node'] = {'shape': 'circle'}
    G.graph['edges'] = {'arrowsize': '0.8'}

    # 添加节点，记录属性
    for n in nodes:
        idx = n.get("idx")
        ntype = n.get("type")
        raw_label = str(n.get("label", ""))
        label = _predicate_name(raw_label) if label_verbosity == "short" else raw_label
        G.add_node(idx, label=label, ntype=ntype)
    for u, v in edges:
        G.add_edge(u, v)

    # Use graphviz_layout for a hierarchical structure
    try:
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    except Exception as e:
        print(f"Graphviz 'dot' layout failed: {e}. Falling back to spring_layout.")
        pos = nx.spring_layout(G, seed=42)


    # Enhanced node coloring and styling logic
    node_colors, node_edge_colors, node_linewidths, node_sizes, node_shapes = {}, {}, {}, {}, {}
    font_size_labels = 8

    for n_idx, n_data in G.nodes(data=True):
        ntype = n_data.get("ntype")
        if ntype == "rule":
            node_shapes[n_idx] = 's'
            node_colors[n_idx] = COLOR_RULE
            node_edge_colors[n_idx] = COLOR_BORDER
            node_linewidths[n_idx] = 1.0
            node_sizes[n_idx] = 1300
        else: # fact
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
        if shape not in shape_map: shape_map[shape] = []
        shape_map[shape].append(idx)

    # 绘图
    fig, ax = plt.subplots(figsize=FIG_BASE_SIZE)
    ax.axis("off")

    # 边
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
            ax=ax
        )

    # 标签
    labels = {n_idx: n_data["label"] for n_idx, n_data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size_labels, ax=ax)
    
    # NEW: Add margins around the plot to prevent clipping
    # This adds a 10% margin on all sides, effectively "zooming out".
    ax.margins(0.1)

    fig.tight_layout()
    fig.savefig(out_png, format="png", dpi=FIG_DPI)
    plt.close(fig)


def _load_patterns(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    arr = data.get("patterns")
    if not isinstance(arr, list):
        raise SystemExit("输入 JSON 缺少 patterns 数组：" + str(path))
    return arr


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patterns = _load_patterns(INPUT_PATH)

    total = len(patterns)
    limit = MAX_PATTERNS if MAX_PATTERNS and MAX_PATTERNS > 0 else total
    done = 0
    skipped = 0
    failed = 0

    print("Starting schema visualization...")
    for idx, p in enumerate(patterns[:limit]):
        rendered = p.get("rendered") or {}
        nodes = rendered.get("nodes")
        edges = rendered.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes or not edges:
            # print(f"Skipping schema_{idx:04d} due to missing nodes/edges.")
            skipped += 1
            continue

        # 输出文件名
        out_png = OUTPUT_DIR / f"schema_{idx:04d}.png"
        if out_png.exists() and not OVERWRITE:
            skipped += 1
            continue

        try:
            _save_png_nx(rendered, out_png, label_verbosity=LABEL_VERBOSITY, highlight_kind=HIGHLIGHT_KIND)
            done += 1
        except Exception as e:
            print(f"Failed to generate schema_{idx:04d}.png. Error: {e}")
            failed += 1
    
    print("-" * 30)
    print(f"Input total: {total}, Processing limit: {limit}")
    print(f"Generated successfully: {done}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
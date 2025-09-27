#!/usr/bin/env python3
"""
Prune proof graphs from a results JSON and optionally render PNGs.

Usage:
  python scripts/prune_graphs.py <input_json> [output_dir]

Behavior:
  - Produces a sibling file with suffix "_pruned.json" containing rendered graphs per problem after pruning.
  - Keeps metadata (aux_points, point_lines, points, point_rely_on) when available.
  - Optionally renders PNGs using ProofGraphVisualizer.render_rendered.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Defaults (override by CLI)
INPUT_JSON = "outputs/r07_expanded_problems.results_aux.json"
OUTPUT_DIR = "outputs/proof_graphs"
RENDER_PNG = True
LABEL_MODE = "short"  # short | full
FIGSIZE = (15, 20)
FONT_SIZE = 9
RANKSEP = 2.5
NODESEP = 1.2
OVERWRITE = True
PROGRESS_EVERY = 10

# Add src to sys.path for local runs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from newclid.data_discovery.proof_graph import ProofGraph  # noqa: E402
from newclid.data_discovery.graph_pruner import GraphPruner  # noqa: E402
from newclid.data_discovery.proof_graph_visualizer import ProofGraphVisualizer  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {"results": []}
    return obj


def main() -> int:
    args = sys.argv[1:]
    input_json = args[0] if len(args) >= 1 else INPUT_JSON
    output_dir = args[1] if len(args) >= 2 else OUTPUT_DIR

    in_path = Path(input_json)
    if not in_path.exists():
        print(f"[prune] input not found: {in_path}")
        return 1

    # 构建 ProofGraph（便于稳定复用解析逻辑）
    print(f"[prune] building ProofGraph from: {in_path}")
    pg = ProofGraph.build_from_results_json(str(in_path), verbose=False)

    # 修剪
    pruner = GraphPruner()
    pruned_map = pruner.prune_proof_graph(pg)

    # 读取原始对象，用于透传 results 中的元数据
    src_obj = load_json(in_path)
    src_results = src_obj.get("results", []) or []
    # 建 index: problem_id -> src result
    idx_src: Dict[str, Any] = {}
    for r in src_results:
        if isinstance(r, dict):
            pid = str(r.get("problem_id")) if r.get("problem_id") is not None else None
            if pid is not None:
                idx_src[pid] = r

    # 生成输出 results：保留 problem_id, rendered(修剪后), 透传元数据（若存在）
    out_results = []
    for pid, rendered in pruned_map.items():
        base = {"problem_id": pid, "rendered": rendered}
        src = idx_src.get(pid, {})
        # 透传有用的元数据（若存在）
        for k in ("aux_points", "point_lines", "points", "point_rely_on"):
            if k in src:
                base[k] = src[k]
        # 若 ProofGraph 侧缓存了 aux_points，也补充一次
        if "aux_points" not in base:
            ap = (pg.aux_points or {}).get(pid)
            if ap:
                base["aux_points"] = ap
        out_results.append(base)

    out_obj = dict(src_obj)
    out_obj["results"] = out_results

    out_json = in_path.with_name(in_path.stem + "_pruned.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"[prune] pruned json written: {out_json}")

    if RENDER_PNG:
        viz = ProofGraphVisualizer()
        # 输出目录：outputs/proof_graphs/<basename>_pruned/
        out_dir = Path(output_dir) / out_json.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[prune] rendering to: {out_dir}")
        count = 0
        ok = 0
        skipped = 0
        failed = 0
        for rec in out_results:
            pid = str(rec.get("problem_id"))
            rendered = rec.get("rendered")
            # 将 aux_points 附着给 rendered 以便着色逻辑使用
            if isinstance(rendered, dict):
                aux_points = rec.get("aux_points") or (pg.aux_points or {}).get(pid) or []
                if isinstance(aux_points, list):
                    rendered = dict(rendered)
                    rendered["aux_points"] = aux_points
            out_png = out_dir / f"proof_{pid}.png"
            if out_png.exists() and not OVERWRITE:
                skipped += 1
                continue
            try:
                viz.render_rendered(
                    rendered,
                    str(out_png),
                    label_mode=LABEL_MODE,
                    highlight=True,
                    figsize=FIGSIZE,
                    font_size=FONT_SIZE,
                    show_direction_legend=True,
                    layout_ranksep=RANKSEP,
                    layout_nodesep=NODESEP,
                )
                ok += 1
            except Exception as e:
                print(f"[prune] render failed for pid={pid}: {e}")
                failed += 1
            count += 1
            if PROGRESS_EVERY and count % PROGRESS_EVERY == 0:
                print(f"[prune] {count}/{len(out_results)} done={ok} skipped={skipped} failed={failed}")
        print(f"[prune] finished: total={len(out_results)} done={ok} skipped={skipped} failed={failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

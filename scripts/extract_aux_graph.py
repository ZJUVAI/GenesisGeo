#!/usr/bin/env python3
"""
Extract and render problems with non-empty aux_points from a results JSON.

Usage:
  python scripts/extract_aux_graph.py <input_json> [output_dir]

Behavior:
  - Produces a sibling file with suffix "_aux.json" containing only results with aux_points.
  - Renders proof graphs from the filtered JSON, mirroring plot_proof_graphs.py behavior.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

# Defaults (overridable by CLI)
INPUT_JSON = "outputs/r07_problems.results.json"
# INPUT_JSON = "outputs/r07_problems.results.json"
OUTPUT_DIR = "outputs/proof_graphs"
LABEL_MODE = "legend"  # legend | full | short (legend maps to short)
OVERWRITE = True
PROGRESS_EVERY = 10

# Add src to sys.path for local runs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from newclid.data_discovery.aux_extractor import AuxExtractor  # noqa: E402
from newclid.data_discovery.proof_graph import ProofGraph  # noqa: E402
from newclid.data_discovery.proof_graph_visualizer import ProofGraphVisualizer  # noqa: E402


def _map_label_mode(mode: str) -> str:
    if mode == "legend":
        return "short"
    if mode in {"short", "full"}:
        return mode
    return "short"


def main():
    args = sys.argv[1:]
    input_json = args[0] if len(args) >= 1 else INPUT_JSON
    output_dir = args[1] if len(args) >= 2 else OUTPUT_DIR

    in_path = Path(input_json)
    if not in_path.exists():
        print(f"[aux] input not found: {in_path}")
        sys.exit(1)

    # Step 1: Filter into *_aux.json next to input
    aux_json = in_path.with_name(in_path.stem + "_aux.json")
    stats = AuxExtractor().filter_results_with_aux(str(in_path), str(aux_json))
    if stats.get("kept", 0) <= 0:
        print("[aux] no items with aux_points; stop after JSON output.")
        return 0

    # Step 2: Render proof graphs using ProofGraphVisualizer
    base_name = aux_json.stem
    out_dir = Path(output_dir) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[aux] building ProofGraph from: {aux_json}")
    pg = ProofGraph.build_from_results_json(str(aux_json), verbose=False)
    viz = ProofGraphVisualizer()
    mapped_label_mode = _map_label_mode(LABEL_MODE)
    if LABEL_MODE == "legend" and mapped_label_mode != LABEL_MODE:
        print("[aux] label_mode 'legend' 不再支持，已映射为 'short'。")

    # Collect problem ids
    pids = sorted({str(nd.get("problem_id")) for nd in pg.nodes.values() if nd.get("problem_id") is not None})
    total = len(pids)
    done = 0
    skipped = 0
    failed = 0
    print(f"[aux] rendering to: {out_dir}  (total={total})")

    for idx, pid in enumerate(pids):
        out_png = out_dir / f"proof_{pid}.png"
        if out_png.exists() and not OVERWRITE:
            skipped += 1
            continue
        try:
            viz.render_problem(pg, pid, str(out_png), label_mode=mapped_label_mode)
            done += 1
        except Exception as e:
            print(f"Failed to generate proof_{pid}.png. Error: {e}")
            failed += 1
        if PROGRESS_EVERY and ((idx + 1) % PROGRESS_EVERY == 0):
            print(f"[aux] {idx+1}/{total} done={done} skipped={skipped} failed={failed}")

    print("-" * 30)
    print(f"Input total: {total}")
    print(f"Generated successfully: {done}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

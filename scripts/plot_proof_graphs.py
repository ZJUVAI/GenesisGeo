#!/usr/bin/env python3
"""
Plot all proofs from a results JSON into PNGs using ProofGraph internals.

Usage:
  python scripts/plot_proof_graphs.py <input_json> [output_dir]

If arguments are omitted, the script uses the constants defined below.
"""
import os
import sys
import json
from pathlib import Path

# Defaults (will be overridden by CLI args if provided)
INPUT_JSON = "outputs/r07_expanded_problems_lil.results.json"  # change as needed
OUTPUT_DIR = "outputs/proof_graphs"
LABEL_MODE = "legend"  # legend | full | short
# Whether to overwrite existing PNGs
OVERWRITE = True
# Print a progress line every N items (0 to disable periodic logs)
PROGRESS_EVERY = 10

# Add src to sys.path for local runs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from newclid.data_discovery.proof_graph import ProofGraph  # noqa: E402
from newclid.data_discovery.proof_graph_visualizer import ProofGraphVisualizer  # noqa: E402
import inspect  # noqa: E402


def _short_label(raw: str) -> str:
    try:
        if "(" in raw and raw.endswith(")"):
            return raw.split("(", 1)[0].strip()
        return raw.split()[0]
    except Exception:
        return raw


def _fact_points_from_node(nd: dict) -> list:
    """Extract point names from a fact-node dict (newclid ProofGraph node schema)."""
    if not isinstance(nd, dict):
        return []
    args = list(nd.get("args", []) or [])
    pred = str(nd.get("label", ""))
    # handle aconst/rconst: drop trailing constant arg
    if pred in {"aconst", "rconst"} and len(args) >= 1:
        return [a for a in args[:-1]]
    return args


def _map_label_mode_for_visualizer(mode: str) -> str:
    """Map legacy label modes to ProofGraphVisualizer's modes."""
    # ProofGraphVisualizer supports: "short" | "full"
    if mode == "legend":
        return "short"
    if mode in {"short", "full"}:
        return mode
    return "short"


def main():
    args = sys.argv[1:]
    input_json = None
    output_dir = None
    if len(args) >= 1:
        input_json = args[0]
    if len(args) >= 2:
        output_dir = args[1]

    input_json = input_json or INPUT_JSON
    output_dir = output_dir or OUTPUT_DIR

    in_path = Path(input_json)
    if not in_path.exists():
        print(f"[plot] input not found: {in_path}")
        sys.exit(1)

    # Compose output under outputs/proof_graphs/<json_basename>
    base_name = in_path.stem
    out_dir = Path(output_dir) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot] building ProofGraph from: {in_path}")
    pg = ProofGraph.build_from_results_json(str(in_path), verbose=False)
    # Show where ProofGraph is imported from (to diagnose version mismatch)
    try:
        src_path = inspect.getsourcefile(ProofGraph) or str(ProofGraph)
        print(f"[plot] using ProofGraph from: {src_path}")
    except Exception:
        pass

    print(f"[plot] rendering to: {out_dir}")
    viz = ProofGraphVisualizer()
    mapped_label_mode = _map_label_mode_for_visualizer(LABEL_MODE)
    if LABEL_MODE == "legend" and mapped_label_mode != LABEL_MODE:
        print("[plot] label_mode 'legend' 不再支持，已映射为 'short'。")

    # Collect problem ids
    pids = sorted({str(nd.get("problem_id")) for nd in pg.nodes.values() if nd.get("problem_id") is not None})
    total = len(pids)
    done = 0
    skipped = 0
    failed = 0

    print("Starting proof graph rendering...")
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
            print(f"[plot] {idx+1}/{total} done={done} skipped={skipped} failed={failed}")

    print("-" * 30)
    print(f"Input total: {total}")
    print(f"Generated successfully: {done}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()

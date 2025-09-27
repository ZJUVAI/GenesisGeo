#!/usr/bin/env python3
"""
Generate DSL test files from branched_mining.json patterns.

Inputs (defaults are repository-relative):
  - --input:  src/newclid/data_discovery/data/branched_mining.json
  - --outdir: src/newclid/data_discovery/data/schema_tests
  - --start:  starting index (default 0)
  - --limit:  max number of patterns to emit (default None = all)
  - --kind:   which sides to emit: both|schema|before (default both)

Output files (overwrites if exist):
  - schema_{id:04d}.txt
  - schema_before_{id:04d}.txt

Each file contains:
  - All point lines ("point <name> <x> <y>") first, from point_lines if present, else from points.
  - Followed by premises as lines:  "assume <pred> <args...>"
  - Followed by conclusions as lines: "prove <pred> <args...>" (one per conclusion if multiple)

This script uses only Python standard library and does not depend on project modules.
Configuration is defined at the top of file; run with `python scripts/make_schema_tests.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PRED_CLAUSE_RE = re.compile(r"^\s*(?P<pred>\w+)\s*\((?P<args>.*)\)\s*$")


def repo_root_from_here() -> Path:
    # scripts/ is directly under repo root
    return Path(__file__).resolve().parent.parent


def default_paths() -> Tuple[Path, Path]:
    root = repo_root_from_here()
    input_path = root / "src/newclid/data_discovery/data/filt_out/branched_mining.json"
    outdir = root / "src/newclid/data_discovery/data/schema_tests"
    return input_path, outdir

# ---------------- Configuration (edit here) ----------------
REPO_ROOT = repo_root_from_here()
INPUT, OUTDIR = default_paths()
# If you want to use filtered outputs, change to:
# INPUT = REPO_ROOT / "src/newclid/data_discovery/data/filt_out/branched_mining.json"
START = 0
LIMIT = None  # None for all
KIND = "both"  # "both" | "schema" | "before"
# Output variable mode: "name" keeps original point names (current behavior),
# "xi" maps names to Xi variables (e.g., X1). Change this to switch output.
VAR_MODE = "xi"  # "name" | "xi"


def parse_schema(schema: str) -> Tuple[List[Tuple[str, List[str]]], List[Tuple[str, List[str]]]]:
    """Parse a schema string like:
    "p1(...) ∧ p2(...) => c1(...) ∧ c2(...)"
    into (premises, conclusions) where each item is (pred, [args]).
    """
    # Split on => (allow surrounding spaces)
    parts = re.split(r"\s*=>\s*", schema.strip(), maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Schema missing '=>': {schema}")
    left, right = parts
    premises = _split_conjuncts(left)
    conclusions = _split_conjuncts(right)
    return premises, conclusions


def _split_conjuncts(expr: str) -> List[Tuple[str, List[str]]]:
    # Split by the specific conjunction symbol '∧' (U+2227). Also tolerate 'and' or '&' as fallback.
    # We split on '∧' first; if not present, treat entire expr as one clause.
    if '∧' in expr:
        clauses = [c.strip() for c in expr.split('∧') if c.strip()]
    else:
        # Try splitting on ' and ' or ' & ' but only if multiple clauses present
        clauses = [c.strip() for c in re.split(r"\s+(?:and|&)\s+", expr) if c.strip()] if (' and ' in expr or ' & ' in expr) else [expr.strip()]
    parsed: List[Tuple[str, List[str]]] = []
    for clause in clauses:
        m = PRED_CLAUSE_RE.match(clause)
        if not m:
            raise ValueError(f"Bad clause: {clause}")
        pred = m.group('pred')
        args = [a.strip() for a in m.group('args').split(',') if a.strip()]
        parsed.append((pred, args))
    return parsed


def invert_schema_vars(schema_vars: Dict[str, str]) -> Dict[str, str]:
    """Given mapping like {'b': 'X1', 'i': 'X2'}, return {'X1': 'b', 'X2': 'i'}.
    If duplicates occur, last one wins (unlikely under current data)."""
    inv: Dict[str, str] = {}
    for name, x in schema_vars.items():
        inv[x] = name
    return inv


def _is_numeric_token(token: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?", token))


def _is_xi(token: str) -> bool:
    return bool(re.fullmatch(r"X\d+", token))


def map_args(args: Iterable[str], mode: str, xi_to_name: Dict[str, str], name_to_xi: Dict[str, str]) -> List[str]:
    """Map argument tokens according to mode.
    - mode == "name": Xi -> original name (default behavior)
    - mode == "xi": original name -> Xi; Xi stays Xi
    Numbers or fractions remain unchanged.
    """
    mapped: List[str] = []
    for a in args:
        if _is_numeric_token(a):
            mapped.append(a)
            continue
        if mode == "xi":
            # Already Xi? keep; else map name->Xi when possible
            mapped.append(a if _is_xi(a) else name_to_xi.get(a, a))
        else:
            # Default: map Xi->name when possible; names pass through
            mapped.append(xi_to_name.get(a, a))
    return mapped


def _remap_point_lines_for_mode(point_lines: List[str], mode: str, name_to_xi: Dict[str, str]) -> List[str]:
    if mode != "xi":
        return point_lines
    remapped: List[str] = []
    for pl in point_lines:
        parts = pl.strip().split()
        if len(parts) >= 4 and parts[0] == "point":
            name = parts[1]
            parts[1] = name_to_xi.get(name, name)
            remapped.append(" ".join(parts))
        else:
            remapped.append(pl)
    return remapped


def render_problem(point_lines: List[str], premises: List[Tuple[str, List[str]]], conclusions: List[Tuple[str, List[str]]], mode: str, xi_to_name: Dict[str, str], name_to_xi: Dict[str, str]) -> str:
    lines: List[str] = []
    # Points first
    lines.extend(_remap_point_lines_for_mode(point_lines, mode, name_to_xi))
    # Premises
    for pred, args in premises:
        margs = map_args(args, mode, xi_to_name, name_to_xi)
        lines.append(f"assume {pred} {' '.join(margs)}")
    # Conclusions
    for pred, args in conclusions:
        margs = map_args(args, mode, xi_to_name, name_to_xi)
        lines.append(f"prove {pred} {' '.join(margs)}")
    return "\n".join(lines) + "\n"


def _collect_used_names(premises: List[Tuple[str, List[str]]], conclusions: List[Tuple[str, List[str]]], xi_to_name: Dict[str, str]) -> List[str]:
    names: List[str] = []
    def add_from(args: List[str]) -> None:
        for a in args:
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?", a):
                continue
            name = xi_to_name.get(a, a)
            if name not in names:
                names.append(name)
    for _, args in premises:
        add_from(args)
    for _, args in conclusions:
        add_from(args)
    return names


def ensure_point_lines(item: dict, used_names: Iterable[str]) -> List[str]:
    """Return only point lines whose names are referenced in used_names.
    Preserve original order as appears in source data.
    """
    used = set(used_names)
    if item.get("point_lines"):
        res: List[str] = []
        for pl in item["point_lines"]:
            if not pl or not pl.strip():
                continue
            # extract name after 'point '
            m = re.match(r"^\s*point\s+(?P<name>\S+)\s+", pl)
            if not m:
                continue
            name = m.group("name")
            if name in used:
                res.append(pl.strip())
        return res
    # else use structured points
    pts = item.get("points") or []
    res = []
    for p in pts:
        name = p.get("name")
        if name not in used:
            continue
        x = p.get("x")
        y = p.get("y")
        if name is None or x is None or y is None:
            continue
        res.append(f"point {name} {x} {y}")
    return res


def main() -> None:
    input_path = INPUT
    outdir = OUTDIR
    start = START
    limit = LIMIT
    kind = KIND

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = data.get("patterns")
    if not isinstance(patterns, list):
        raise SystemExit("Invalid input: missing 'patterns' array")

    outdir.mkdir(parents=True, exist_ok=True)

    count = 0
    start_idx = start
    end_idx = start_idx + (limit if limit is not None else len(patterns))

    for idx, item in enumerate(patterns):
        if idx < start_idx:
            continue
        if idx >= end_idx:
            break

        schema = item.get("schema")
        schema_before = item.get("schema_before_dependency")
        schema_vars = item.get("rendered", {}).get("schema_vars") or item.get("schema_vars") or {}
        name_to_xi: Dict[str, str] = schema_vars if isinstance(schema_vars, dict) else {}
        xi_to_name = invert_schema_vars(schema_vars) if isinstance(schema_vars, dict) else {}
        # Prepare both sides parsed, to compute used names first
        prem_s, concl_s = ([], [])
        prem_b, concl_b = ([], [])
        if schema:
            prem_s, concl_s = parse_schema(schema)
        if schema_before:
            prem_b, concl_b = parse_schema(schema_before)

        used_names = _collect_used_names(prem_s, concl_s, xi_to_name) + _collect_used_names(prem_b, concl_b, xi_to_name)
        # deduplicate while preserving order
        seen = set()
        used_ordered = [n for n in used_names if not (n in seen or seen.add(n))]

        point_lines = ensure_point_lines(item, used_ordered)

        file_id = f"{idx:04d}"

        if kind in ("both", "schema") and schema:
            prem, concl = prem_s, concl_s
            content = render_problem(point_lines, prem, concl, VAR_MODE, xi_to_name, name_to_xi)
            (outdir / f"schema_{file_id}.txt").write_text(content, encoding="utf-8")

        if kind in ("both", "before") and schema_before:
            content_b = render_problem(point_lines, prem_b, concl_b, VAR_MODE, xi_to_name, name_to_xi)
            (outdir / f"schema_before_{file_id}.txt").write_text(content_b, encoding="utf-8")

        count += 1

    print(f"Emitted {count} pattern(s) starting at {start_idx} into: {outdir}")


if __name__ == "__main__":
    main()

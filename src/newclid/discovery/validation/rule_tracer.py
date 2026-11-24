"""Rule tracer: recover auxiliary construction info and multi-seed coordinates.

Given a normalized rule (A,B,C,...), trace back through the data pipeline to:
  1. Recover the rename_map (A→original_point_name)
  2. Extract auxiliary point constructions from the original fl_problem
  3. Find multiple seed-coordinate sets for cross-validation

Data flow:
  normalized_rules.jsonl  ─── rule_id → {rule_text, rename_map, points, seed, ...}
  occurrences_all.json    ─── rule_text → {seed: count}
  source dataset (JSONL)  ─── seed → {fl_problem, llm_input_renamed, ...}
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

# Parse fl_problem clauses: "point_name@x_y" and "= construction_type args"
_COORD_RE = re.compile(
    r"([a-zA-Z][a-zA-Z0-9]*)@(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)_(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)"
)


class RuleTracer:
    """Trace normalized rules back to original data sources."""

    def __init__(
        self,
        normalized_rules_path: str,
        occurrences_path: str,
        source_dataset_path: str,
    ):
        self.normalized_path = normalized_rules_path
        self.occurrences_path = occurrences_path
        self.source_path = source_dataset_path

        # Indexes (built lazily)
        self._norm_index: dict[str, dict] = {}  # rule_id → norm record
        self._norm_by_seed: dict[int, list[dict]] = defaultdict(
            list
        )  # seed → norm records
        self._occurrences: dict[str, dict[str, int]] = {}  # rule_text → {seed: count}
        self._source_line_offsets: dict[int, list[int]] = defaultdict(
            list
        )  # seed → [byte offsets]

        self._built = False

    # ------------------------------------------------------------------
    # Build indexes
    # ------------------------------------------------------------------

    def build(self) -> "RuleTracer":
        """Build all lookup indexes. Call once before tracing."""
        self._build_norm_index()
        self._build_source_index()
        self._load_occurrences()
        self._built = True
        return self

    def _build_norm_index(self) -> None:
        """Load normalized_rules.jsonl → rule_id and seed indexes."""
        print(f"[tracer] Loading normalized rules: {self.normalized_path}")
        with open(self.normalized_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec["rule_id"]
                self._norm_index[rid] = rec
                seed = rec.get("seed")
                if seed is not None:
                    self._norm_by_seed[seed].append(rec)
        print(f"[tracer]   {len(self._norm_index)} rules indexed")

    def _load_occurrences(self) -> None:
        """Load rule_seed_occurrences_all.json → rule_text → {seed: count}."""
        if not os.path.exists(self.occurrences_path):
            print(f"[tracer] occurrences file not found: {self.occurrences_path}")
            return
        print(f"[tracer] Loading occurrences: {self.occurrences_path}")
        with open(self.occurrences_path, "r", encoding="utf-8") as f:
            self._occurrences = json.load(f)
        print(f"[tracer]   {len(self._occurrences)} unique rule texts")

    def _build_source_index(self) -> None:
        """Build seed → byte-offset index for the source JSONL."""
        if not os.path.exists(self.source_path):
            print(f"[tracer] source dataset not found: {self.source_path}")
            return
        print(f"[tracer] Indexing source dataset: {self.source_path}")
        count = 0
        with open(self.source_path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                # Quick parse: just extract the seed
                try:
                    seed_str = line.split('"seed":')[1].split(",")[0].strip()
                    seed = int(seed_str)
                except (IndexError, ValueError):
                    continue
                self._source_line_offsets[seed].append(offset)
                count += 1
                if count % 500000 == 0:
                    print(f"[tracer]   indexed {count} records...")
        print(
            f"[tracer]   {count} records, {len(self._source_line_offsets)} unique seeds"
        )

    # ------------------------------------------------------------------
    # Source record access
    # ------------------------------------------------------------------

    def get_source_record(self, seed: int, index_in_seed: int = 0) -> dict | None:
        """Read a specific source record by seed and index."""
        offsets = self._source_line_offsets.get(seed, [])
        if index_in_seed >= len(offsets):
            return None

        with open(self.source_path, "r", encoding="utf-8") as f:
            f.seek(offsets[index_in_seed])
            line = f.readline().strip()
            if not line:
                return None
            return json.loads(line)

    def get_source_records(self, seed: int) -> list[dict]:
        """Get all source records for a seed."""
        offsets = self._source_line_offsets.get(seed, [])
        records = []
        with open(self.source_path, "r", encoding="utf-8") as f:
            for off in offsets:
                f.seek(off)
                line = f.readline().strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------
    # Core tracing
    # ------------------------------------------------------------------

    def trace(self, rule_id: str) -> dict[str, Any] | None:
        """Trace a single rule back to its origins.

        Returns a dict with:
          - rule_id, rule_text, rename_map, reverse_map
          - primary_seed: the seed that produced this rule instance
          - aux_constructions: [(aux_point_names, construction_clause), ...]
          - all_seeds: {seed: count} from occurrences data
          - multi_coords: [{seed, index_in_seed, points, source_record}, ...]
        """
        if not self._built:
            self.build()

        # Step 1: get normalize record
        norm_rec = self._norm_index.get(rule_id)
        if norm_rec is None:
            return None

        rule_text = norm_rec["rule_text"]
        rename_map: dict[str, str] = norm_rec.get("rename_map", {})
        reverse_map: dict[str, str] = {v: k for k, v in rename_map.items()}

        primary_seed = norm_rec.get("seed")
        primary_idx = norm_rec.get("index_in_seed", 0)

        # Step 2: get source record and extract aux constructions
        aux_constructions: list[dict] = []
        aux_facts: list[dict] = []  # aux facts from <aux> section
        source_record: dict | None = None
        if primary_seed is not None:
            source_record = self.get_source_record(primary_seed, primary_idx)
            if source_record:
                aux_constructions = _extract_aux_constructions(
                    source_record.get("fl_problem", ""), rename_map
                )
                aux_facts = _extract_aux_facts(
                    source_record.get("llm_output_renamed", ""), rename_map
                )

        # Step 3: get multi-seed coordinates
        seed_counts = self._occurrences.get(rule_text, {})
        multi_coords = []
        for seed_str, count in seed_counts.items():
            seed = int(seed_str)
            # Find a norm record for this seed to get the rename_map + points
            seed_norms = self._norm_by_seed.get(seed, [])
            if not seed_norms:
                continue

            # Use the first norm record for this seed
            # (any of them gives the same rule_text and point mapping)
            sn = seed_norms[0]
            if sn["rule_text"] != rule_text:
                # Different rule_text → different rule, skip
                # (same seed can produce multiple rules)
                continue

            coords = {p["name"]: (p["x"], p["y"]) for p in sn.get("points", [])}

            # Also get source record for this seed
            src = self.get_source_record(seed, sn.get("index_in_seed", 0))

            multi_coords.append(
                {
                    "seed": seed,
                    "index_in_seed": sn.get("index_in_seed", 0),
                    "count": count,
                    "points": coords,
                    "rename_map": sn.get("rename_map", {}),
                    "source_record": src,
                }
            )

        return {
            "rule_id": rule_id,
            "rule_text": rule_text,
            "rename_map": rename_map,
            "reverse_map": reverse_map,
            "primary_seed": primary_seed,
            "primary_index": primary_idx,
            "aux_constructions": aux_constructions,
            "aux_facts": aux_facts,
            "source_record": source_record,
            "all_seeds": seed_counts,
            "multi_coords": multi_coords,
            "norm_record": norm_rec,
        }

    def get_norm_record(self, rule_id: str) -> dict | None:
        """Get normalized record by rule_id."""
        if not self._built:
            self.build()
        return self._norm_index.get(rule_id)


# ------------------------------------------------------------------
# Auxiliary construction extraction
# ------------------------------------------------------------------


def _extract_aux_constructions(
    fl_problem: str,
    rename_map: dict[str, str],
) -> list[dict]:
    """Extract auxiliary point construction clauses from fl_problem.

    A point is "auxiliary" if it appears in fl_problem but NOT in rename_map keys.
    (rename_map maps original_point → normalized_point)

    Returns list of:
      {aux_points: [str], construction: str, clause_text: str}
    """
    if not fl_problem or "?" not in fl_problem:
        return []

    # Split off the goal (after ?)
    construction_part = fl_problem.split("?")[0]

    # Split into semicolon-separated clauses
    clauses = [c.strip() for c in construction_part.split(";") if c.strip()]

    # Parse each clause: "point_names = construction"
    aux_list = []
    for clause in clauses:
        if "=" not in clause:
            continue

        left, right = clause.split("=", 1)
        left = left.strip()
        right = right.strip()

        # Extract point names from left side (before @)
        point_names = []
        for m in _COORD_RE.finditer(left):
            point_names.append(m.group(1))

        # Also handle points without @ (construction args on left side)
        # e.g., "a b c = triangle a b c" → point names are a, b, c
        # But these already appear on the right side, skip

        # Check if any point is auxiliary (not in rename_map)
        aux_points = [p for p in point_names if p not in rename_map]

        if aux_points:
            # Extract construction type from right side
            # "square b a c d" → construction = "square", args = [b, a, c, d]
            right_tokens = right.split()
            if right_tokens:
                aux_list.append(
                    {
                        "aux_points": aux_points,
                        "all_points_in_clause": point_names,
                        "construction_type": right_tokens[0],
                        "construction_args": right_tokens[1:]
                        if len(right_tokens) > 1
                        else [],
                        "clause_text": clause,
                    }
                )

    return aux_list


def _extract_all_points(fl_problem: str) -> dict[str, tuple[float, float]]:
    """Extract all point coordinates from fl_problem."""
    points = {}
    for m in _COORD_RE.finditer(fl_problem or ""):
        name = m.group(1)
        points[name] = (float(m.group(2)), float(m.group(3)))
    return points


# ------------------------------------------------------------------
# Auxiliary construction NDG rules
# ------------------------------------------------------------------

# Known non-degeneracy conditions for each construction type.
# Format: construction_type → list of (condition_name, checker_function_name)
# These are the "require" fields from defs.txt, translated to predicates.
CONSTRUCTION_NDGS: dict[str, list[tuple[str, str]]] = {
    "reflect": [("ncoll", "A B C")],  # A, B, C non-collinear (line BC well-defined)
    "on_circum": [("ncoll", "B C D")],  # 3 defining points non-collinear
    "angle_bisector": [("ncoll", "A B C")],  # angle well-defined
    "midpoint": [("diff", "A B")],  # A ≠ B
    "foot": [("ncoll", "B C D")],  # foot to line, line well-defined
    "on_tline": [("ncoll", "B C D")],
    "on_bline": [("ncoll", "B C D")],
    "on_pline": [("ncoll", "B C D")],
    "on_circle": [("diff", "B C")],  # center ≠ point on circle
    "eqdistance": [],
    "on_line": [("diff", "B C")],
    "on_aline": [("ncoll", "B C D")],
    "on_dia": [("diff", "B C")],
    "incenter": [("ncoll", "A B C")],
    "excenter": [("ncoll", "A B C")],
    "centroid": [("ncoll", "A B C")],
    "orthocenter": [("ncoll", "A B C")],
    "cc_tangent": [("diff", "A B"), ("diff", "C D")],
    "intersection_ll": [("npara", "A B C D")],
    "intersection_lt": [("npara", "A B C D")],
    "intersection_tt": [("diff", "A B"), ("diff", "C D")],
    "intersection_lc": [],
    "intersection_cc": [],
}


def get_construction_ndgs(
    aux_constructions: list[dict],
) -> list[dict]:
    """For each auxiliary construction, compute the required NDG predicates.

    Args are in ORIGINAL point names. Returns list of NDG dicts.
    """
    all_ndgs = []
    for aux_info in aux_constructions:
        ctype = aux_info["construction_type"]
        cargs = aux_info["construction_args"]
        ndg_rules = CONSTRUCTION_NDGS.get(ctype, [])

        for ndg_pred, ndg_pattern in ndg_rules:
            # Map placeholder args (A,B,C...) to actual construction args
            pattern_args = ndg_pattern.split()
            # Map A→cargs[0], B→cargs[1], C→cargs[2], D→cargs[3]
            placeholder_to_real = {}
            for i, pa in enumerate(pattern_args):
                if pa.isalpha() and len(pa) == 1 and pa.isupper():
                    idx = ord(pa) - ord("A")
                    if idx < len(cargs):
                        placeholder_to_real[pa] = cargs[idx]

            real_args = [placeholder_to_real.get(pa, pa) for pa in pattern_args]

            all_ndgs.append(
                {
                    "construction_type": ctype,
                    "aux_points": aux_info["aux_points"],
                    "clause": aux_info["clause_text"],
                    "ndg_predicate": ndg_pred,
                    "ndg_args": real_args,
                }
            )

    return all_ndgs


# ------------------------------------------------------------------
# Aux fact extraction from <aux> section in llm_output_renamed
# ------------------------------------------------------------------

# Regex for facts in <aux>: "pred arg1 arg2 ... [id]"
_AUX_FACT_RE = re.compile(r"(\w+)\s+([A-Za-z0-9_ ]+?)\s*\[(\d+)\]")


def _extract_aux_facts(
    llm_output_renamed: str,
    rename_map: dict[str, str],
) -> list[dict]:
    """Extract auxiliary point facts from the <aux> section of llm_output_renamed.

    The <aux> section has format:
      x00 f : coll a b f [004] cong a f b f [005] ; x01 g : ...

    Where x00/x01 are construction codes, f/g are aux point names, and after ':'
    are the geometric facts generated by the construction.

    Returns list of:
      {aux_points: [str], facts: [{pred, args, id}], clause_text: str}
    """
    if not llm_output_renamed:
        return []

    # Extract <aux>...</aux> content
    aux_match = re.search(r"<aux>(.*?)</aux>", llm_output_renamed, re.DOTALL)
    if not aux_match:
        return []

    aux_content = aux_match.group(1).strip()

    # Split into semicolon-separated clauses
    clauses = [c.strip() for c in aux_content.split(";") if c.strip()]

    aux_list = []
    for clause in clauses:
        if ":" not in clause:
            continue

        left, right = clause.split(":", 1)
        left = left.strip()
        right = right.strip()

        # Parse left side: "x00 f" or "x00 f g" (construction code + point names)
        left_tokens = left.split()
        if not left_tokens:
            continue

        # The construction code starts with 'x' (like x00, x01)
        aux_point_names = []
        for tok in left_tokens:
            if not tok.lower().startswith("x"):
                aux_point_names.append(tok)

        if not aux_point_names:
            continue

        # Parse right side facts: "coll a b f [004] cong a f b f [005]"
        facts = []
        for m in _AUX_FACT_RE.finditer(right):
            facts.append(
                {
                    "predicate": m.group(1),
                    "args": tuple(m.group(2).split()),
                    "id": m.group(3),
                }
            )

        # Separate: which aux points are present
        aux_points_in_clause = [p for p in aux_point_names if p not in rename_map]

        # Map facts: replace aux point names with "[AUX]" marker,
        # given point names with normalized names where possible
        mapped_facts = []
        for fact in facts:
            mapped_args = []
            for a in fact["args"]:
                if a in rename_map:
                    mapped_args.append(rename_map[a])
                elif a in aux_point_names:
                    mapped_args.append(f"[AUX:{a}]")
                else:
                    mapped_args.append(a)
            mapped_facts.append(
                {
                    "predicate": fact["predicate"],
                    "args": mapped_args,
                    "id": fact["id"],
                }
            )

        aux_list.append(
            {
                "aux_points": aux_points_in_clause,
                "all_aux_names": aux_point_names,
                "facts": facts,
                "mapped_facts": mapped_facts,
                "clause_text": clause,
            }
        )

    return aux_list

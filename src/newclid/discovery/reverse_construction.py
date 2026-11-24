"""
Reverse Construction for NDG Counterexample Discovery.

Given a geometry rule (premises -> conclusion), this module generates alternative
construction sequences where the constrained point becomes the output of an
intersection-like construction.  This naturally produces multiple geometric
branches, some of which may serve as counterexamples.

Key insight:
  The original construction enforces hidden NDG conditions that prevent
  degeneracy.  By choosing a different construction order (making the
  constrained point the OUTPUT rather than an INPUT), we sidestep those
  hidden constraints and allow degenerate configurations to emerge
  naturally from intersection constructions.

Usage (standalone test):
  python -m newclid.discovery.reverse_construction \
      --rules_file <path> --normalized_rules <path> \
      --occurrences <path> --source_dataset <path> \
      --limit 10 --output_dir /tmp/rev_test
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from newclid.numerical.geometries import PointNum as _AlgPointNum

# ---------------------------------------------------------------------------
# Predicate symmetry groups (permutations that preserve truth value)
# ---------------------------------------------------------------------------

# Each entry: set of frozensets of argument index pairs that can be swapped
# together.  For example, cong(a,b,c,d) = cong(b,a,c,d) means indices
# (0,1) can be swapped independently.

PRED_SYMMETRY = {
    "cong": [
        {"swap": [(0, 1)]},  # AB = BA
        {"swap": [(2, 3)]},  # CD = DC
        {"swap": [(0, 2), (1, 3)]},  # AB=CD -> CD=AB
    ],
    "para": [
        {"swap": [(0, 1)]},  # AB ∥ CD -> BA ∥ CD
        {"swap": [(2, 3)]},  # AB ∥ CD -> AB ∥ DC
        {"swap": [(0, 2), (1, 3)]},  # AB ∥ CD -> CD ∥ AB
    ],
    "perp": [
        {"swap": [(0, 1)]},
        {"swap": [(2, 3)]},
        {"swap": [(0, 2), (1, 3)]},
    ],
    "npara": [
        {"swap": [(0, 1)]},
        {"swap": [(2, 3)]},
        {"swap": [(0, 2), (1, 3)]},
    ],
    "nperp": [
        {"swap": [(0, 1)]},
        {"swap": [(2, 3)]},
        {"swap": [(0, 2), (1, 3)]},
    ],
    "cyclic": [
        # 8 valid orderings: 4 forward cyclic + 4 reverse cyclic
        {"rotate": [1, 2, 3, 0]},  # ABCD -> BCDA
        {"swap": [(0, 3), (1, 2)]},  # reverse (DCBA from ABCD)
    ],
    "coll": [
        # All 6 permutations of 3 collinear points via adjacent swaps
        {"swap": [(0, 1)]},
        {"swap": [(1, 2)]},
    ],
    "ncoll": [
        {"swap": [(0, 1)]},
        {"swap": [(1, 2)]},
    ],
    "eqangle": [
        # Reverse each ray independently (valid because cross & dot both flip sign)
        {"swap": [(0, 1)]},  # reverse first ray of first angle
        {"swap": [(2, 3)]},  # reverse second ray of first angle
        {"swap": [(4, 5)]},  # reverse first ray of second angle
        {"swap": [(6, 7)]},  # reverse second ray of second angle
        # Swap the two angles: eqangle(a,b,c,d,e,f,g,h) = eqangle(e,f,g,h,a,b,c,d)
        {"swap": [(0, 4), (1, 5), (2, 6), (3, 7)]},
    ],
    "eqratio": [
        {"swap": [(0, 1)]},  # AB = BA
        {"swap": [(2, 3)]},  # CD = DC
        {"swap": [(4, 5)]},  # EF = FE
        {"swap": [(6, 7)]},  # GH = HG
        {"swap": [(0, 4), (1, 5), (2, 6), (3, 7)]},  # swap ratios
    ],
    "midp": [
        # midp(m,a,b): m is midpoint, but m is special
    ],
    "sameclock": [
        {"swap": [(0, 1), (3, 4)]},  # swap triangles
    ],
    "simtri": [
        # Rotate first triangle: ABC->BCA, keep DEF unchanged
        {"rotate": [1, 2, 0, 3, 4, 5]},
        # Rotate second triangle: DEF->EFD, keep ABC unchanged
        {"rotate": [0, 1, 2, 4, 5, 3]},
        {"swap": [(0, 3), (1, 4), (2, 5)]},
    ],
    "contri": [
        {"rotate": [1, 2, 0, 3, 4, 5]},
        {"rotate": [0, 1, 2, 4, 5, 3]},
        {"swap": [(0, 3), (1, 4), (2, 5)]},
    ],
}


def _apply_symmetry(args: list[str], sym: dict) -> list[str]:
    """Apply one symmetry operation to argument list, return new list."""
    result = list(args)
    if "swap" in sym:
        for a, b in sym["swap"]:
            if a < len(result) and b < len(result):
                result[a], result[b] = result[b], result[a]
    if "rotate" in sym:
        order = sym["rotate"]
        if len(order) == len(result):
            result = [result[i] for i in order]
    if "permute" in sym:
        order = sym["permute"]
        if len(order) == len(result):
            result = [result[i] for i in order]
    return result


def predicate_variants(pred_name: str, args: list[str]) -> list[list[str]]:
    """Generate all argument-permuted variants of a predicate.

    Returns list of arg-lists that are semantically equivalent.
    Uses a worklist algorithm to compute the full closure under
    all symmetry operations.
    """
    symmetries = PRED_SYMMETRY.get(pred_name, [])
    if not symmetries:
        return [list(args)]

    variants: set[tuple[str, ...]] = set()
    queue: list[tuple[str, ...]] = [tuple(args)]
    variants.add(tuple(args))

    while queue:
        current = queue.pop()
        for sym in symmetries:
            new_args = tuple(_apply_symmetry(list(current), sym))
            if new_args not in variants:
                variants.add(new_args)
                queue.append(new_args)

    return [list(v) for v in variants]


# ---------------------------------------------------------------------------
# defs.txt parser
# ---------------------------------------------------------------------------


@dataclass
class ConstructionDef:
    """One construction definition from defs.txt."""

    name: str  # e.g. "angle_bisector"
    new_points: list[str]  # e.g. ["x"]
    existing_points: list[str]  # e.g. ["a", "b", "c"]
    all_var_names: list[str]  # e.g. ["x", "a", "b", "c"]
    requires: list[tuple[str, list[str]]]  # e.g. [("ncoll", ["a","b","c"])]
    basics: list[
        tuple[str, list[str]]
    ]  # e.g. [("eqangle", ["b","a","b","x","b","x","b","c"])]

    @property
    def is_multi_point(self) -> bool:
        return len(self.new_points) > 1

    @property
    def is_base(self) -> bool:
        """Base constructions just declare points without constraints."""
        return len(self.basics) == 0 and len(self.requires) == 0

    @property
    def is_intersection(self) -> bool:
        """Intersection constructions naturally produce multiple branches."""
        return self.name.startswith("intersection_")


def _parse_clause_points_preds(
    clause: str,
) -> tuple[list[str], list[str], list[tuple[str, list[str]]]]:
    """Parse a clause like 'x : coll x a b, perp i x b c' or 'a b = diff a b'.

    Returns (left_points, right_points, predicates).
    """
    predicates: list[tuple[str, list[str]]] = []
    left_points: list[str] = []
    right_points: list[str] = []

    if "=" in clause:
        # require format: "a b = diff a b, ncoll a b c"
        left_str, right_str = clause.split("=", 1)
        left_points = left_str.strip().split()
    elif ":" in clause:
        # declare or basics format: "x : coll x a b"
        left_str, right_str = clause.split(":", 1)
        left_points = left_str.strip().split()
    else:
        return [], [], []

    right_str = right_str.strip()
    if not right_str:
        return left_points, right_points, []

    # Split by comma, but NOT within [...] brackets (not used in defs.txt)
    pred_strs = [p.strip() for p in right_str.split(",") if p.strip()]
    for ps in pred_strs:
        parts = ps.split()
        if parts:
            predicates.append((parts[0], parts[1:]))

    return left_points, right_points, predicates


def parse_defs(defs_path: str) -> dict[str, ConstructionDef]:
    """Parse defs.txt into a dict of ConstructionDef objects.

    defs.txt format (5 logical lines per construction):
      Line 1: construction_name new_pts... existing_pts...
      Line 2: declare clause(s)  (new_pt : referenced_pts)  -- may be EMPTY
      Line 3: require clause(s)  (existing_pts = predicates)
      Line 4: basics clause(s)   (new_pt : produced_predicates)
      Line 5: numerics (label only)

    Blocks are separated by blank lines, BUT some constructions have an
    internal blank line (e.g., line 2 is empty for triangle, segment, etc.).
    We detect block boundaries by looking for header lines that match the
    construction definition pattern.
    """
    with open(defs_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # Pattern for a construction header: name followed by space-separated
    # single-letter variable names.  e.g. "angle_bisector x a b c"
    # We distinguish from declare/require/basics lines which contain ':' or '='.
    HEADER_RE = re.compile(r"^([a-z_][a-z0-9_]*)\s+((?:[a-z]\s*)+)$")

    # Find all header line indices.
    # A valid header is a line matching the construction-name pattern that is
    # followed by declare/require/basics content (lines with ':' or '='),
    # NOT just a blank line or another header.
    header_indices = []
    for i, line in enumerate(all_lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that look like declare/require/basics/numerics-only
        if ":" in stripped or "=" in stripped:
            continue
        m = HEADER_RE.match(stripped)
        if not m:
            continue
        name = m.group(1)
        # Construction names are at least 3 chars and contain underscore
        # or are known base types (free, segment, triangle, etc.)
        if len(name) < 3:
            continue
        # Verify: at least one of the next few non-empty lines contains
        # ':' or '=' (declare/require/basics pattern).  Numerics lines
        # like "bisect a b c" are followed by blank lines, not by ':'/'='.
        has_content = False
        for j in range(i + 1, min(i + 6, len(all_lines))):
            nxt = all_lines[j].strip()
            if not nxt:
                continue
            if ":" in nxt or "=" in nxt:
                has_content = True
                break
            # If next non-empty line looks like another header, this isn't one
            if HEADER_RE.match(nxt):
                break
        if has_content:
            header_indices.append(i)

    constructions: dict[str, ConstructionDef] = {}

    for idx, start_line in enumerate(header_indices):
        # Determine end of this block: next header, or EOF
        end_line = (
            header_indices[idx + 1] if idx + 1 < len(header_indices) else len(all_lines)
        )

        # Collect non-empty lines from this block
        block_lines = []
        for j in range(start_line, end_line):
            stripped = all_lines[j].strip()
            if stripped:
                block_lines.append(stripped)

        if len(block_lines) < 4:
            continue

        # Line 1 (block_lines[0]): name + all variable names
        parts = block_lines[0].split()
        name = parts[0]
        var_names = parts[1:]

        # Find which line is declare (contains ':'), require (contains '='),
        # basics (contains ':' after declare), and numerics (last line)
        declare_str = ""
        require_str = ""
        basics_str = ""
        block_lines[-1]

        # Line 1 is header, remaining lines can be in various orders.
        # Heuristic: the line with '=' is require, the first line with ':'
        # after it is basics, and the last line is numerics.
        saw_require = False
        for bl in block_lines[1:-1]:  # exclude header and numerics
            if "=" in bl and not saw_require:
                require_str = bl
                saw_require = True
            elif ":" in bl:
                if not saw_require:
                    declare_str = bl
                else:
                    basics_str = bl

        # Parse declare
        declare_clauses = [c.strip() for c in declare_str.split(";") if c.strip()]
        new_points: list[str] = []
        for clause in declare_clauses:
            left_pts, _, _ = _parse_clause_points_preds(clause)
            for pt in left_pts:
                if pt and pt not in new_points:
                    new_points.append(pt)

        # If no explicit declare, infer from var_names: first vars are new
        if not new_points:
            # For single-new-point: first var is new, rest are existing
            # For multi-new-point (triangle, segment): all vars might be new
            # Check basics: points on left of ':' are new
            basics_clauses_init = [
                c.strip() for c in basics_str.split(";") if c.strip()
            ]
            for clause in basics_clauses_init:
                left_pts, _, _ = _parse_clause_points_preds(clause)
                for pt in left_pts:
                    if pt and pt not in new_points:
                        new_points.append(pt)
            # If still empty, check require
            if not new_points:
                require_clauses_init = [
                    c.strip() for c in require_str.split(";") if c.strip()
                ]
                for clause in require_clauses_init:
                    if "=" in clause:
                        left_str = clause.split("=")[0].strip()
                        for pt in left_str.split():
                            if pt and pt not in new_points:
                                new_points.append(pt)
            # Fallback: first var is new (for simple constructions)
            if not new_points and var_names:
                new_points = [var_names[0]]

        existing_points = [v for v in var_names if v not in new_points]

        # Parse require
        require_clauses = [c.strip() for c in require_str.split(";") if c.strip()]
        requires: list[tuple[str, list[str]]] = []
        for clause in require_clauses:
            _, _, preds = _parse_clause_points_preds(clause)
            requires.extend(preds)

        # Parse basics
        basics_clauses = [c.strip() for c in basics_str.split(";") if c.strip()]
        basics: list[tuple[str, list[str]]] = []
        for clause in basics_clauses:
            _, _, preds = _parse_clause_points_preds(clause)
            basics.extend(preds)

        constructions[name] = ConstructionDef(
            name=name,
            new_points=new_points,
            existing_points=existing_points,
            all_var_names=var_names,
            requires=requires,
            basics=basics,
        )

    return constructions


# ---------------------------------------------------------------------------
# Reverse index: predicate -> constructions that produce it
# ---------------------------------------------------------------------------


@dataclass
class ReverseMatch:
    """A record linking a premise predicate back to a construction."""

    construction: ConstructionDef
    new_point_var: str  # which construction var is the "output"
    basic_pred: tuple[str, list[str]]  # the basics predicate matched
    arg_map: dict[str, str]  # construction var -> premise point name
    require: list[tuple[str, list[str]]]  # require predicates (with construction vars)


def build_reverse_index(
    constructions: dict[str, ConstructionDef],
) -> dict[tuple[str, int], list[ConstructionDef]]:
    """Index constructions by (predicate_name, num_args).

    Returns: {(pred_name, arg_count): [ConstructionDef, ...]}
    """
    index: dict[tuple[str, int], list[ConstructionDef]] = defaultdict(list)
    for cdef in constructions.values():
        seen = set()
        for pred_name, args in cdef.basics:
            key = (pred_name, len(args))
            if key not in seen:
                index[key].append(cdef)
                seen.add(key)
    return index


def match_premise_to_constructions(
    prem_name: str,
    prem_args: list[str],
    reverse_index: dict[tuple[str, int], list[ConstructionDef]],
) -> list[ReverseMatch]:
    """Find all constructions whose basics can match this premise.

    Handles predicate symmetries (e.g., cong(a,b,c,d) = cong(b,a,c,d)).
    """
    key = (prem_name, len(prem_args))
    candidates = reverse_index.get(key, [])
    matches: list[ReverseMatch] = []

    # Generate all equivalent forms of the premise
    prem_variants = predicate_variants(prem_name, prem_args)

    for cdef in candidates:
        for basic_pred in cdef.basics:
            if basic_pred[0] != prem_name:
                continue
            if len(basic_pred[1]) != len(prem_args):
                continue

            basic_variants = predicate_variants(basic_pred[0], list(basic_pred[1]))

            for prem_v in prem_variants:
                for basic_v in basic_variants:
                    arg_map = _try_match_args(basic_v, prem_v, cdef)
                    if arg_map is not None:
                        # Identify which construction var maps to the
                        # "constrained" point (the new_point).
                        # For single-new-point constructions, exactly one
                        # new_point exists.
                        matches.append(
                            ReverseMatch(
                                construction=cdef,
                                new_point_var=cdef.new_points[0]
                                if cdef.new_points
                                else "",
                                basic_pred=(basic_pred[0], list(basic_pred[1])),
                                arg_map=arg_map,
                                require=list(cdef.requires),
                            )
                        )
                        break  # Found a match for this basic_pred

    # Deduplicate by (construction.name, frozenset of arg_map items)
    seen: set[tuple[str, frozenset]] = set()
    unique: list[ReverseMatch] = []
    for m in matches:
        key = (m.construction.name, frozenset(m.arg_map.items()))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def _try_match_args(
    basic_args: list[str],
    prem_args: list[str],
    cdef: ConstructionDef,
) -> dict[str, str] | None:
    """Try to match construction var names to premise point names.

    basic_args uses construction variable names (a, b, x, ...)
    prem_args uses premise point names (A, B, D, ...)

    Returns mapping from construction var -> premise point, or None.
    """
    arg_map: dict[str, str] = {}
    for b_arg, p_arg in zip(basic_args, prem_args):
        if b_arg in arg_map:
            if arg_map[b_arg] != p_arg:
                return None
        else:
            # Only allow mapping construction vars to premise points
            # (not the other way around — a construction var must map
            #  to exactly one premise point, but multiple construction
            #  vars COULD map to the same premise point for degenerate cases)
            arg_map[b_arg] = p_arg

    # Validate: all existing_points must be mapped.
    existing_mapped = set()
    for ep in cdef.existing_points:
        if ep not in arg_map:
            return None
        existing_mapped.add(arg_map[ep])

    # The new point must NOT also be an existing point (no self-construction)
    for new_point in cdef.new_points:
        if new_point in arg_map and arg_map[new_point] in existing_mapped:
            return None

    # At least one new_point must be mapped
    if not any(new_point in arg_map for new_point in cdef.new_points):
        return None

    return arg_map


# ---------------------------------------------------------------------------
# Construction sequence generator
# ---------------------------------------------------------------------------


@dataclass
class ConstructionSequence:
    """One valid construction ordering."""

    steps: list["ConstructionStep"]  # in order
    base_points: list[str]  # points constructed via free/triangle/segment
    score: int = 0  # higher = more likely to produce branches


@dataclass
class ConstructionStep:
    point: str  # point being constructed
    construction_name: str  # e.g. "angle_mirror"
    construction_def: ConstructionDef
    arg_map: dict[str, str]  # construction vars -> actual point names
    premise_index: int | None  # which premise this covers (None for base)


def generate_construction_sequences(
    premises: list[tuple[str, list[str]]],
    conclusion: tuple[str, list[str]],
    premise_matches: dict[int, list[ReverseMatch]],
    constructions: dict[str, ConstructionDef],
    max_sequences: int = 50,
) -> list[ConstructionSequence]:
    """Generate all valid construction sequences for a rule.

    A sequence is valid if:
    - Every point in premises appears in at least one step
    - No step references a point before it's constructed (topological order)
    - All premise predicates are covered by construction basics
    - At least one step uses an intersection-like construction (preferred)
    """
    # Collect all points
    all_points: set[str] = set()
    for _pn, args in premises:
        all_points.update(args)
    for _cn, cargs in [conclusion]:
        all_points.update(cargs)

    point_list = sorted(all_points)

    # Identify which points can be "outputs" of a matched construction
    constrained_points: dict[str, list[tuple[int, ReverseMatch]]] = defaultdict(list)
    for prem_idx, matches in premise_matches.items():
        for match in matches:
            new_pt_var = match.new_point_var
            if new_pt_var in match.arg_map:
                pt = match.arg_map[new_pt_var]
                constrained_points[pt].append((prem_idx, match))

    # "Free" points: never appear as output of any premise match
    free_points = [p for p in point_list if p not in constrained_points]

    # Build construction steps for each point
    point_steps: dict[str, list[ConstructionStep]] = defaultdict(list)

    # Free construction for all points (as fallback)
    free_defs = [
        c
        for c in constructions.values()
        if c.is_base and not c.is_multi_point and c.name == "free"
    ]
    free_def = free_defs[0] if free_defs else None

    # For "free-only" points, only free construction is available
    for pt in free_points:
        if free_def:
            point_steps[pt].append(
                ConstructionStep(
                    point=pt,
                    construction_name="free",
                    construction_def=free_def,
                    arg_map={free_def.new_points[0]: pt},
                    premise_index=None,
                )
            )

    # For constrained points, add matched constructions.
    # ONLY use constructions where the new point is the FIRST variable
    # (the solver requires this for fl_problem syntax).
    for pt, match_list in constrained_points.items():
        if free_def:
            point_steps[pt].append(
                ConstructionStep(
                    point=pt,
                    construction_name="free",
                    construction_def=free_def,
                    arg_map={free_def.new_points[0]: pt},
                    premise_index=None,
                )
            )
        for prem_idx, match in match_list:
            cdef = match.construction
            # Filter 1: blacklist (constructions we never want to use)
            if cdef.name in ("between_bound",):
                continue
            # Filter 2: new point must be first in all_var_names
            new_var = cdef.new_points[0] if cdef.new_points else None
            if new_var and new_var in cdef.all_var_names:
                if cdef.all_var_names.index(new_var) != 0:
                    continue  # skip: solver can't handle it in fl_problem
            # Filter 3: ALL basics must be covered by some premise.
            # e.g. mirror's basics are [coll, cong]; if only coll matches a premise
            # but the extra cong doesn't, mirror is excluded.
            all_covered = True
            for basic_name, basic_args in cdef.basics:
                mapped = [match.arg_map.get(a, a) for a in basic_args]
                # Check if this mapped basic matches ANY premise
                covered = False
                for pn, pa in premises:
                    if pn != basic_name or len(pa) != len(mapped):
                        continue
                    # Try all variant matchings
                    for pv in predicate_variants(pn, pa):
                        if all(m == p for m, p in zip(mapped, pv)):
                            covered = True
                            break
                    if covered:
                        break
                if not covered:
                    all_covered = False
                    break
            if not all_covered:
                continue  # extra basics not covered by any premise
            step = ConstructionStep(
                point=pt,
                construction_name=match.construction.name,
                construction_def=match.construction,
                arg_map=dict(match.arg_map),
                premise_index=prem_idx,
            )
            # Avoid duplicates
            existing = {
                (s.construction_name, frozenset(s.arg_map.items()))
                for s in point_steps[pt]
            }
            key = (step.construction_name, frozenset(step.arg_map.items()))
            if key not in existing:
                point_steps[pt].append(step)

    # Backtracking: try all valid orderings of point constructions.
    # A point can have MULTIPLE constructions (joined by comma in fl_problem),
    # allowing combined constraints like "angle_mirror + on_circle".
    sequences: list[ConstructionSequence] = []
    constructed: set[str] = set()
    covered_premises: set[int] = set()
    current_steps: list[ConstructionStep] = []

    def _inputs_available(steps: list[ConstructionStep]) -> bool:
        """Check if all input points for ALL given steps are constructed."""
        needed: set[str] = set()
        for step in steps:
            for ev in step.construction_def.existing_points:
                if ev in step.arg_map:
                    ep = step.arg_map[ev]
                    if ep != step.point:
                        needed.add(ep)
        return needed.issubset(constructed)

    def _try_subsets(pt: str, candidates: list[ConstructionStep], depth: int):
        """Try subsets of candidate constructions for a point (max 2).

        Only constructions where the new point is the FIRST variable can be
        combined via comma (the solver matches args positionally).
        Others can only be used as single constructions.
        """
        nonlocal sequences
        if len(sequences) >= max_sequences:
            return

        # Split: comma-compatible vs single-only
        comma_ok = []
        single_only = []
        seen_keys = set()
        for c in candidates:
            key = (c.construction_name, c.premise_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cdef = c.construction_def
            # Construction is comma-compatible iff the new point is the FIRST var
            new_var = cdef.new_points[0] if cdef.new_points else None
            new_pos = (
                cdef.all_var_names.index(new_var)
                if new_var and new_var in cdef.all_var_names
                else -1
            )
            if new_pos == 0:
                comma_ok.append(c)
            else:
                single_only.append(c)

        # Combine: all comma_ok candidates can be paired, single_only can only be alone
        all_candidates = comma_ok + single_only

        subset_scores = []
        # Multi-construction: pairs from comma_ok, but ONLY from DIFFERENT premises.
        # Combining two constructions that cover the same premise (e.g.,
        # on_bline + iso_triangle_vertex both matching cong) just adds
        # redundant constraints and causes overconstrained failures.
        if len(comma_ok) >= 2:
            for indices in combinations(range(len(comma_ok)), 2):
                s0, s1 = comma_ok[indices[0]], comma_ok[indices[1]]
                # Skip if both cover the same premise
                if (
                    s0.premise_index is not None
                    and s1.premise_index is not None
                    and s0.premise_index == s1.premise_index
                ):
                    continue
                subset = [s0, s1]
                if not _inputs_available(subset):
                    continue
                prem_set = set(
                    s.premise_index for s in subset if s.premise_index is not None
                )
                # Penalty for extra basics beyond matched premises
                extra = sum(len(s.construction_def.basics) - 1 for s in subset)
                score = len(prem_set) * 100 + 2 - extra * 5
                subset_scores.append((score, subset))

        # Single constructions: all candidates (both comma_ok and single_only)
        for c in all_candidates:
            if _inputs_available([c]):
                prem_set = {c.premise_index} if c.premise_index is not None else set()
                # Strong penalty for unmatched basics: each extra constraint not
                # covered by any premise adds significant geometric burden
                extra_basics = (
                    len(c.construction_def.basics) - 1
                )  # one matches current premise
                score = len(prem_set) * 100 + 1 - extra_basics * 5
                subset_scores.append((score, [c]))

        subset_scores.sort(key=lambda x: -x[0])
        for _score, subset in subset_scores:
            if len(sequences) >= max_sequences:
                return

            constructed.add(pt)
            for step in subset:
                current_steps.append(step)
            new_covered: set[int] = set()
            for step in subset:
                if step.premise_index is not None:
                    if step.premise_index not in covered_premises:
                        covered_premises.add(step.premise_index)
                        new_covered.add(step.premise_index)

            _backtrack(depth + 1)

            for pi in new_covered:
                covered_premises.discard(pi)
            for _ in subset:
                current_steps.pop()
            constructed.discard(pt)

    def _backtrack(depth: int = 0):
        nonlocal sequences
        if len(sequences) >= max_sequences:
            return

        all_pts_done = all(p in constructed for p in point_list)
        if covered_premises == set(premise_matches.keys()) and all_pts_done:
            n_non_base = len(
                [s for s in current_steps if not s.construction_def.is_base]
            )
            n_points_non_base = len(
                set(s.point for s in current_steps if not s.construction_def.is_base)
            )
            score = sum(
                10
                if s.construction_def.is_intersection
                else 5
                if not s.construction_def.is_base
                else 0
                for s in current_steps
            )
            if n_non_base > n_points_non_base:
                score += 15
            sequences.append(
                ConstructionSequence(
                    steps=list(current_steps),
                    base_points=sorted(free_points),
                    score=score,
                )
            )
            return

        # Try to construct points with available non-free candidates first
        for pt in point_list:
            if pt in constructed:
                continue
            candidates = [
                s
                for s in point_steps[pt]
                if not s.construction_def.is_base and _inputs_available([s])
            ]
            if candidates:
                _try_subsets(pt, candidates, depth)
                return

        # No non-free candidates available: construct next free point
        for pt in point_list:
            if pt in constructed:
                continue
            free_steps = [s for s in point_steps[pt] if s.construction_def.is_base]
            if free_steps:
                step = free_steps[0]
                constructed.add(pt)
                current_steps.append(step)
                _backtrack(depth + 1)
                current_steps.pop()
                constructed.discard(pt)
                return

    _backtrack()

    # Sort by score (prefer intersection constructions)
    sequences.sort(key=lambda s: -s.score)
    return sequences


# ---------------------------------------------------------------------------
# fl_problem builder
# ---------------------------------------------------------------------------


def _construction_to_fl_clause(
    steps: list[ConstructionStep],
) -> str:
    """Convert one or more ConstructionSteps for the SAME point to an fl_problem clause.

    When multiple constructions constrain the same point, they are joined
    by commas:  "e = angle_mirror e b a d, on_circle e d b"

    IMPORTANT: args must stay in the ORIGINAL order from defs.txt line 1.
    The solver matches by POSITION (not by name), so reordering breaks
    constructions where the new point is not the first argument.
    The left side of '=' declares which point is new.
    """
    if not steps:
        return ""

    # The point being declared (left side of =)
    first = steps[0]
    declared = [first.arg_map.get(v, v) for v in first.construction_def.new_points]
    left = " ".join(declared)

    rights = []
    for step in steps:
        cdef = step.construction_def
        arg_map = step.arg_map
        # Keep ORIGINAL order from defs.txt (all_var_names)
        actual_args = [arg_map.get(v, v) for v in cdef.all_var_names]
        rights.append(f"{cdef.name} {' '.join(actual_args)}")

    return f"{left} = {', '.join(rights)}"


def build_fl_problem(
    seq: ConstructionSequence,
    rename_map: dict[str, str],
) -> str:
    """Build an fl_problem string from a construction sequence.

    Groups steps by point: multiple constructions for the same point are
    joined by commas (the point must satisfy ALL of them).
    """
    # Remap all steps through rename_map
    remapped_steps = []
    for step in seq.steps:
        remapped_steps.append(
            ConstructionStep(
                point=rename_map.get(step.point, step.point),
                construction_name=step.construction_name,
                construction_def=step.construction_def,
                arg_map={k: rename_map.get(v, v) for k, v in step.arg_map.items()},
                premise_index=step.premise_index,
            )
        )

    # Group by point (preserving order of first appearance)
    point_order = []
    point_steps: dict[str, list[ConstructionStep]] = {}
    for step in remapped_steps:
        if step.point not in point_steps:
            point_order.append(step.point)
            point_steps[step.point] = []
        point_steps[step.point].append(step)

    clauses = [_construction_to_fl_clause(point_steps[pt]) for pt in point_order]
    return "; ".join(clauses)


def build_all_fl_problems(
    seq: ConstructionSequence,
    rename_map: dict[str, str],
    reverse_map: dict[str, str],
) -> list[str]:
    """Build multiple fl_problem variants with different point orderings."""
    # For now, just return one variant
    fl = build_fl_problem(seq, rename_map)
    return [fl]


# ---------------------------------------------------------------------------
# Algebraic branch enumerator — solves constructions directly, no randomness
# ---------------------------------------------------------------------------


def _line_circle_intersections(line_pt, line_dir, center, radius):
    """Return 0, 1, or 2 intersection points of a line and circle.

    Returns [] (no solution) rather than raising when `line_dir` is a
    degenerate (near-zero) vector — this happens legitimately when a
    numeric optimizer probing for a degenerate branch (see
    refine_degenerate_witness) walks two of the points defining that
    direction to (nearly) the same location; the caller treats "no
    intersection here" the same as any other unreachable branch.
    """
    PN = _AlgPointNum
    d = line_dir
    a = d.x * d.x + d.y * d.y
    if a < 1e-20:
        return []
    f = line_pt - center
    b = 2 * (f.x * d.x + f.y * d.y)
    c = f.x * f.x + f.y * f.y - radius * radius
    disc = b * b - 4 * a * c
    if disc < -1e-10:
        return []
    disc = max(disc, 0.0)
    sd = math.sqrt(disc)
    t1 = (-b + sd) / (2 * a)
    t2 = (-b - sd) / (2 * a)
    pts = []
    for t in (t1, t2):
        pts.append(PN(line_pt.x + t * d.x, line_pt.y + t * d.y))
    return pts


def _circle_circle_intersections(c1, r1, c2, r2):
    """Return 0, 1, or 2 intersection points of two circles."""
    PN = _AlgPointNum
    d = math.sqrt(max(c1.distance2(c2), 1e-16))
    if d > r1 + r2 + 1e-8 or d < abs(r1 - r2) - 1e-8:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h_sq = r1 * r1 - a * a
    h = math.sqrt(max(h_sq, 0.0))
    mid = PN(c1.x + a * (c2.x - c1.x) / d, c1.y + a * (c2.y - c1.y) / d)
    rx = h * (c2.y - c1.y) / d
    ry = h * (c2.x - c1.x) / d
    return [
        PN(mid.x + rx, mid.y - ry),
        PN(mid.x - rx, mid.y + ry),
    ]


def _solve_single_construction(name, args, pts, rng):
    """Solve a single construction for the new point.

    Deterministic cases return the exact solution(s).
    Underdetermined cases return a random sample from the constraint set.
    """
    PN = _AlgPointNum
    if name == "free":
        return [pts[args[0]]]
    elif name == "on_line":
        a_pt, b_pt = pts[args[1]], pts[args[2]]
        t = rng.uniform(-1, 1)
        return [PN(a_pt.x + t * (b_pt.x - a_pt.x), a_pt.y + t * (b_pt.y - a_pt.y))]
    elif name == "on_circle":
        o_pt, a_pt = pts[args[1]], pts[args[2]]
        r = math.sqrt(max(o_pt.distance2(a_pt), 1e-16))
        angle = rng.uniform(0, 2 * math.pi)
        return [PN(o_pt.x + r * math.cos(angle), o_pt.y + r * math.sin(angle))]
    elif name == "on_bline":
        a_pt, b_pt = pts[args[1]], pts[args[2]]
        mid = PN((a_pt.x + b_pt.x) / 2, (a_pt.y + b_pt.y) / 2)
        ab = b_pt - a_pt
        perp = PN(-ab.y, ab.x)
        t = rng.uniform(-1, 1)
        return [PN(mid.x + t * perp.x, mid.y + t * perp.y)]
    elif name == "angle_mirror":
        # angle_mirror x a b c: eqangle(b,a, b,c, b,c, b,x)
        # b=args[2] is vertex; a=args[1] is point to reflect; c=args[3] is on axis
        # x = reflection of B across line AD (vertex=b=A, axis=c=D, reflect=a=B)
        vertex = pts[args[2]]
        to_reflect = pts[args[1]]
        axis_pt = pts[args[3]]
        axis = axis_pt - vertex
        v = to_reflect - vertex
        axis_len2 = max(axis.x * axis.x + axis.y * axis.y, 1e-16)
        proj = (v.x * axis.x + v.y * axis.y) / axis_len2
        proj_pt = _AlgPointNum(vertex.x + proj * axis.x, vertex.y + proj * axis.y)
        return [
            _AlgPointNum(2 * proj_pt.x - to_reflect.x, 2 * proj_pt.y - to_reflect.y)
        ]
    elif name == "angle_bisector":
        # x on angle bisector: infinite on a line
        return None
    elif name == "on_dia":
        # perp(x,a,x,b) → x on circle with diameter ab
        return None  # infinite on circle
    elif name == "on_circum":
        # x on circumcircle of a,b,c
        return None  # infinite on circle
    elif name == "on_pline":
        # x on line through a parallel to bc
        return None  # infinite on line
    elif name == "on_tline":
        # x on line through a perpendicular to bc
        return None
    elif name == "on_aline":
        # eqangle(a,x,a,b,d,c,d,e) → defines a line
        return None
    elif name == "iso_triangle_vertex":
        # x on perpendicular bisector of ab
        return None
    elif name == "eqdistance":
        # x on circle centered at a radius bc
        return None
    elif name == "foot":
        # perp(x,a,b,c) and coll(x,b,c) → x is foot of perpendicular from a to bc
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        bc = c_pt - b_pt
        bc_len2 = max(bc.x * bc.x + bc.y * bc.y, 1e-16)
        t = ((a_pt.x - b_pt.x) * bc.x + (a_pt.y - b_pt.y) * bc.y) / bc_len2
        return [_AlgPointNum(b_pt.x + t * bc.x, b_pt.y + t * bc.y)]
    elif name == "orthocenter":
        # perp(x,a,b,c) and perp(x,b,c,a) → intersection of two altitudes
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        # Altitude from A: line through A perpendicular to BC
        bc = c_pt - b_pt
        # Direction perpendicular to BC
        alt_dir = _AlgPointNum(-bc.y, bc.x)
        # Altitude from B: line through B perpendicular to AC
        ac = c_pt - a_pt
        alt_dir2 = _AlgPointNum(-ac.y, ac.x)
        # Intersection of two lines
        return _line_line_intersection(a_pt, alt_dir, b_pt, alt_dir2)
    elif name == "midpoint":
        a_pt, b_pt = pts[args[1]], pts[args[2]]
        return [_AlgPointNum((a_pt.x + b_pt.x) / 2, (a_pt.y + b_pt.y) / 2)]
    elif name == "on_pline0":
        return None  # on line (infinite)
    elif name == "lc_tangent":
        return None  # on line (infinite)
    elif name == "external_bisector":
        return None
    elif name == "eq_triangle":
        return None  # infinite (two possible eq triangles)
    elif name == "mirror":
        # coll(x,a,b) and cong(b,a,b,x) → x is reflection of a across b
        a_pt, b_pt = pts[args[1]], pts[args[2]]
        return [_AlgPointNum(2 * b_pt.x - a_pt.x, 2 * b_pt.y - a_pt.y)]
    else:
        return None  # unknown or underdetermined


def _line_line_degeneracy(d1, d2) -> float:
    """Normalized |sin(angle between d1, d2)| in [0, 1].

    0 means the two lines are exactly parallel/coincident — the point they're
    meant to jointly determine is undetermined (a whole line of solutions)
    rather than unique.  Values close to 0 (but not zero) mean the
    intersection is numerically ill-conditioned: a tiny perturbation of the
    inputs swings the intersection point by a large amount, so a rule whose
    witness configuration lands near this region is only "accidentally" true
    there and may fail on other configurations satisfying the same premises.
    """
    cross = d1.x * d2.y - d1.y * d2.x
    n1 = math.hypot(d1.x, d1.y)
    n2 = math.hypot(d2.x, d2.y)
    denom = n1 * n2
    if denom < 1e-16:
        return 0.0
    return abs(cross) / denom


def _line_line_intersection(p1, d1, p2, d2):
    """Intersection of two lines (p1 + t*d1) and (p2 + s*d2). Returns [point] or []."""
    cross = d1.x * d2.y - d1.y * d2.x
    if abs(cross) < 1e-12:
        return []  # parallel
    dp = p2 - p1
    t = (dp.x * d2.y - dp.y * d2.x) / cross
    return [_AlgPointNum(p1.x + t * d1.x, p1.y + t * d1.y)]


def _get_line_from_construction(name, args, pts):
    """If this construction defines a line constraint, return (point_on_line, direction)."""
    if name == "on_line":
        return (pts[args[1]], pts[args[2]] - pts[args[1]])
    elif name == "on_bline":
        # Perpendicular bisector of ab: line through midpoint, perpendicular to ab
        a_pt, b_pt = pts[args[1]], pts[args[2]]
        mid = _AlgPointNum((a_pt.x + b_pt.x) / 2, (a_pt.y + b_pt.y) / 2)
        ab = b_pt - a_pt
        return (mid, _AlgPointNum(-ab.y, ab.x))
    elif name == "on_tline":
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        bc = c_pt - b_pt
        return (a_pt, _AlgPointNum(-bc.y, bc.x))
    elif name == "on_pline" or name == "on_pline0":
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        return (a_pt, c_pt - b_pt)
    elif name == "iso_triangle_vertex":
        a_pt, b_pt = pts[args[1]], pts[args[2]]
        mid = _AlgPointNum((a_pt.x + b_pt.x) / 2, (a_pt.y + b_pt.y) / 2)
        ab = b_pt - a_pt
        return (mid, _AlgPointNum(-ab.y, ab.x))
    elif name == "angle_bisector":
        # angle_bisector x a b c: x (new=args[0]) on bisector of angle abc
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        ba = a_pt - b_pt
        bc_dir = c_pt - b_pt
        ba_len = math.sqrt(max(ba.x * ba.x + ba.y * ba.y, 1e-16))
        bc_len = math.sqrt(max(bc_dir.x * bc_dir.x + bc_dir.y * bc_dir.y, 1e-16))
        d = _AlgPointNum(
            ba.x / ba_len + bc_dir.x / bc_len, ba.y / ba_len + bc_dir.y / bc_len
        )
        return (b_pt, d)
    elif name == "external_bisector":
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        ba = a_pt - b_pt
        bc_dir = c_pt - b_pt
        ba_len = math.sqrt(max(ba.x * ba.x + ba.y * ba.y, 1e-16))
        bc_len = math.sqrt(max(bc_dir.x * bc_dir.x + bc_dir.y * bc_dir.y, 1e-16))
        d = _AlgPointNum(
            -ba.x / ba_len + bc_dir.x / bc_len, -ba.y / ba_len + bc_dir.y / bc_len
        )
        return (b_pt, d)
    elif name == "on_aline":
        # on_aline x a b c d e: eqangle a x a b d c d e.
        # Forward sketch (numerical/sketch.py:sketch_aline) places x on the
        # line through 'a' (here the anchor/args[1]) with direction angle
        # ang(b-a) + ang(c-d) - ang(e-d).
        a_pt, b_pt, c_pt, d_pt, e_pt = (
            pts[args[1]],
            pts[args[2]],
            pts[args[3]],
            pts[args[4]],
            pts[args[5]],
        )
        ang = (
            math.atan2(b_pt.y - a_pt.y, b_pt.x - a_pt.x)
            + math.atan2(c_pt.y - d_pt.y, c_pt.x - d_pt.x)
            - math.atan2(e_pt.y - d_pt.y, e_pt.x - d_pt.x)
        )
        return (a_pt, _AlgPointNum(math.cos(ang), math.sin(ang)))
    elif name == "on_aline0":
        # on_aline0 x a b c d e f g: eqangle a b c d e f g x.
        # Forward sketch (numerical/sketch.py:sketch_aline0) places x on the
        # line through 'g' (the anchor/args[7]) with direction angle
        # ang(e-f) + ang(c-d) - ang(a-b).
        a_pt, b_pt, c_pt, d_pt, e_pt, f_pt, g_pt = (
            pts[args[1]],
            pts[args[2]],
            pts[args[3]],
            pts[args[4]],
            pts[args[5]],
            pts[args[6]],
            pts[args[7]],
        )
        ang = (
            math.atan2(e_pt.y - f_pt.y, e_pt.x - f_pt.x)
            + math.atan2(c_pt.y - d_pt.y, c_pt.x - d_pt.x)
            - math.atan2(b_pt.y - a_pt.y, b_pt.x - a_pt.x)
        )
        return (g_pt, _AlgPointNum(math.cos(ang), math.sin(ang)))
    elif name == "lc_tangent":
        a_pt, o_pt = pts[args[1]], pts[args[2]]
        oa = a_pt - o_pt
        return (a_pt, _AlgPointNum(-oa.y, oa.x))
    elif name == "angle_mirror":
        # angle_mirror x a b c: eqangle(b,a, b,c, b,c, b,x)
        # b=args[2] is the vertex (A); a=args[1] is point to reflect (B);
        # c=args[3] is on mirror axis (D); x=args[0] is output E
        # E is reflection of B across line AD
        vertex = pts[args[2]]  # b = A
        to_reflect = pts[args[1]]  # a = B
        axis_pt = pts[args[3]]  # c = D
        axis = axis_pt - vertex  # AD
        v = to_reflect - vertex  # AB
        axis_len2 = max(axis.x * axis.x + axis.y * axis.y, 1e-16)
        proj = (v.x * axis.x + v.y * axis.y) / axis_len2
        refl = _AlgPointNum(2 * proj * axis.x - v.x, 2 * proj * axis.y - v.y)
        return (vertex, refl)
    return None


def _get_circle_from_construction(name, args, pts):
    """If this construction defines a circle constraint, return (center, radius)."""
    if name == "on_circle":
        o_pt, a_pt = pts[args[1]], pts[args[2]]
        r = math.sqrt(max(o_pt.distance2(a_pt), 1e-16))
        return (o_pt, r)
    elif name == "eqdistance":
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        r = math.sqrt(max(b_pt.distance2(c_pt), 1e-16))
        return (a_pt, r)
    elif name == "on_dia":
        a_pt, b_pt = pts[args[1]], pts[args[2]]
        center = _AlgPointNum((a_pt.x + b_pt.x) / 2, (a_pt.y + b_pt.y) / 2)
        r = math.sqrt(max(a_pt.distance2(b_pt), 1e-16)) / 2
        return (center, r)
    elif name == "on_circum":
        a_pt, b_pt, c_pt = pts[args[1]], pts[args[2]], pts[args[3]]
        # Circumcenter + radius
        d = 2 * (
            a_pt.x * (b_pt.y - c_pt.y)
            + b_pt.x * (c_pt.y - a_pt.y)
            + c_pt.x * (a_pt.y - b_pt.y)
        )
        if abs(d) < 1e-12:
            return None
        a2 = a_pt.x * a_pt.x + a_pt.y * a_pt.y
        b2 = b_pt.x * b_pt.x + b_pt.y * b_pt.y
        c2 = c_pt.x * c_pt.x + c_pt.y * c_pt.y
        ux = (
            a2 * (b_pt.y - c_pt.y) + b2 * (c_pt.y - a_pt.y) + c2 * (a_pt.y - b_pt.y)
        ) / d
        uy = (
            a2 * (c_pt.x - b_pt.x) + b2 * (a_pt.x - c_pt.x) + c2 * (b_pt.x - a_pt.x)
        ) / d
        center = _AlgPointNum(ux, uy)
        r = math.sqrt(max(center.distance2(a_pt), 1e-16))
        return (center, r)
    return None


def solve_sequence_algebraic(
    seq, rename_map, free_points_samples=5, degeneracy_log=None
):
    """Algebraically enumerate all branches of a construction sequence.

    For free points, tries a grid of sample coordinates.
    For combined (comma) constructions, computes intersection solutions.
    Returns list of {point_name: PointNum} dicts.

    If `degeneracy_log` is a list, every two-line intersection used to place
    a point records its normalized degeneracy (see `_line_line_degeneracy`)
    there, tagged with the point name: (point, degeneracy). A value near 0
    flags that this construction step's two defining lines are nearly
    parallel for this branch — the point is only weakly determined, so a
    rule relying on this step may be numerically unsafe even where it
    nominally checks out.
    """
    # Collect steps by point
    pt_steps = {}
    pt_order = []
    for step in seq.steps:
        pt = step.point
        if pt not in pt_steps:
            pt_order.append(pt)
            pt_steps[pt] = []
        pt_steps[pt].append(step)

    # Generate sample coordinates for free points
    rng = np.random.RandomState(42)
    free_samples = []
    for _ in range(free_points_samples):
        sample = {}
        for step in seq.steps:
            if step.construction_def.is_base:
                pt = step.point
                sample[pt] = _AlgPointNum(rng.uniform(-1, 1), rng.uniform(-1, 1))
        free_samples.append(sample)

    all_solutions = []

    for free_pts in free_samples:
        solutions = [dict(free_pts)]

        for pt in pt_order:
            steps = pt_steps[pt]
            if all(s.construction_def.is_base for s in steps):
                continue  # already placed

            new_solutions = []
            for sol in solutions:
                # Compute possible positions for this point
                # Collect line and circle constraints from all constructions for this point
                lines = []
                circles = []
                for step in steps:
                    cname = step.construction_name
                    cargs = [
                        step.arg_map.get(v, v)
                        for v in step.construction_def.all_var_names
                    ]
                    line = _get_line_from_construction(cname, cargs, sol)
                    if line:
                        # Tag whether this line is a plain "coll" constraint
                        # (on_line: anchored at two *named* points a,b) so the
                        # single-line branch below can split it into the 3
                        # qualitatively distinct regions on ab.
                        lines.append((line[0], line[1], cname == "on_line"))
                    circle = _get_circle_from_construction(cname, cargs, sol)
                    if circle:
                        circles.append(circle)

                # Single underdetermined construction with no line/circle:
                # use random sample from _solve_single_construction
                if len(lines) == 0 and len(circles) == 0 and len(steps) == 1:
                    step = steps[0]
                    det = _solve_single_construction(
                        step.construction_name,
                        [
                            step.arg_map.get(v, v)
                            for v in step.construction_def.all_var_names
                        ],
                        sol,
                        rng,
                    )
                    if det is not None:
                        for r in det:
                            s2 = dict(sol)
                            s2[pt] = r
                            new_solutions.append(s2)
                    continue

                # Use line/circle intersection or random sample
                candidates = []
                if len(lines) == 2:
                    lp0, ld0 = lines[0][0], lines[0][1]
                    lp1, ld1 = lines[1][0], lines[1][1]
                    if degeneracy_log is not None:
                        deg = _line_line_degeneracy(ld0, ld1)
                        degeneracy_log.append((pt, deg))
                    candidates = _line_line_intersection(lp0, ld0, lp1, ld1)
                elif len(lines) == 1 and len(circles) == 1:
                    candidates = _line_circle_intersections(
                        lines[0][0], lines[0][1], circles[0][0], circles[0][1]
                    )
                elif len(circles) == 2:
                    candidates = _circle_circle_intersections(
                        circles[0][0], circles[0][1], circles[1][0], circles[1][1]
                    )
                elif len(lines) == 1 and len(circles) == 0:
                    lp, ld, is_coll = lines[0]
                    if is_coll:
                        # coll x a b (on_line): line lp=a, ld=b-a. Sample once
                        # in each of the 3 qualitatively distinct regions the
                        # two reference points split the line into: before a
                        # (t<0), between a and b (0<t<1), beyond b (t>1).
                        candidates = [
                            _AlgPointNum(lp.x + t * ld.x, lp.y + t * ld.y)
                            for t in (
                                rng.uniform(-1, 0),
                                rng.uniform(0, 1),
                                rng.uniform(1, 2),
                            )
                        ]
                    else:
                        # Single anchor point on the line: either a named
                        # dependency point (on_tline/on_pline/angle_bisector/
                        # external_bisector/lc_tangent/angle_mirror -> a/b, the
                        # vertex) or an implicit midpoint already computed by
                        # _get_line_from_construction (on_bline/
                        # iso_triangle_vertex -> mid of a,b). Either way lp is
                        # that anchor; sample once on each side of it.
                        candidates = [
                            _AlgPointNum(lp.x + t * ld.x, lp.y + t * ld.y)
                            for t in (rng.uniform(-1, 0), rng.uniform(0, 1))
                        ]
                elif len(circles) == 1 and len(lines) == 0:
                    # Single circle: pick a random point on it
                    ctr, r = circles[0]
                    angle = rng.uniform(0, 2 * math.pi)
                    candidates = [
                        _AlgPointNum(
                            ctr.x + r * math.cos(angle), ctr.y + r * math.sin(angle)
                        )
                    ]
                elif len(lines) == 0 and len(circles) == 0:
                    # No constraint at all — shouldn't happen, skip
                    continue

                for c in candidates:
                    s2 = dict(sol)
                    s2[pt] = c
                    new_solutions.append(s2)

            solutions = new_solutions
            if not solutions:
                break

        all_solutions.extend(solutions)

    return all_solutions


# ---------------------------------------------------------------------------
# Locate-and-refine degenerate branch witness
# ---------------------------------------------------------------------------
#
# solve_sequence_algebraic's `degeneracy_log` flags *that* some two-line
# step nearly went parallel somewhere in the random samples, but random
# sampling can only land near a degenerate branch (it's a measure-zero
# boundary), never exactly on it — so a witness point built from that
# sample doesn't cleanly satisfy the premises (see the "fixed sin threshold"
# approach this replaces: too tight and it never fires, too loose and it
# fires on branches that aren't actually parallel, producing invalid
# witnesses).  Instead, treat "how parallel are these two lines" as a
# smooth, differentiable function of the free points feeding the sequence
# up to that step, and directly minimize it with a numeric optimizer —
# this converges to the true boundary to machine precision (~1e-14 in
# practice), at which point sampling a point on the now-truly-shared line
# gives a witness with genuinely near-zero premise violation.


def _build_up_to_step(
    seq: "ConstructionSequence",
    free_values: dict[str, tuple[float, float]],
    stop_at_point: str,
) -> tuple[dict[str, "_AlgPointNum"] | None, tuple | None, tuple | None]:
    """Deterministically place every point up to (not including) `stop_at_point`,
    then return the two lines that would jointly determine it.

    `free_values`: {point_name: (x, y)} for every base/free point in the
    sequence — the parameters an optimizer varies.  Non-free points are
    placed via the same _get_line_from_construction /
    _get_circle_from_construction logic as solve_sequence_algebraic, but
    deterministically: single-line/single-circle underdetermined steps pick
    a fixed nominal point (t=1 along the line, or angle=0 on the circle)
    rather than sampling randomly, since here we only care about whether
    *this specific* two-line step is degenerate, not about enumerating
    branches.

    Returns (points_so_far, line1, line2) — line{1,2} are None if
    `stop_at_point`'s step isn't a two-line intersection (nothing to
    refine) or if placement fails before reaching it.
    """
    pt_steps: dict[str, list] = {}
    pt_order: list[str] = []
    for step in seq.steps:
        pt = step.point
        if pt not in pt_steps:
            pt_order.append(pt)
            pt_steps[pt] = []
        pt_steps[pt].append(step)

    sol: dict[str, _AlgPointNum] = {}
    for pt in pt_order:
        steps = pt_steps[pt]
        if all(s.construction_def.is_base for s in steps):
            if pt not in free_values:
                return None, None, None
            x, y = free_values[pt]
            sol[pt] = _AlgPointNum(x, y)
            continue

        lines = []
        circles = []
        for step in steps:
            cname = step.construction_name
            cargs = [
                step.arg_map.get(v, v) for v in step.construction_def.all_var_names
            ]
            line = _get_line_from_construction(cname, cargs, sol)
            if line:
                lines.append((line[0], line[1]))
            circle = _get_circle_from_construction(cname, cargs, sol)
            if circle:
                circles.append(circle)

        if pt == stop_at_point and len(lines) == 2:
            return sol, lines[0], lines[1]

        # Deterministic placement (nominal point, not a random sample) —
        # only needs to be *some* valid point satisfying this step's
        # constraints, since we're just building context for a later step.
        if len(lines) == 2:
            cand = _line_line_intersection(
                lines[0][0], lines[0][1], lines[1][0], lines[1][1]
            )
        elif len(lines) == 1 and len(circles) == 1:
            cand = _line_circle_intersections(
                lines[0][0], lines[0][1], circles[0][0], circles[0][1]
            )
        elif len(circles) == 2:
            cand = _circle_circle_intersections(
                circles[0][0], circles[0][1], circles[1][0], circles[1][1]
            )
        elif len(lines) == 1 and len(circles) == 0:
            lp, ld = lines[0]
            cand = [_AlgPointNum(lp.x + 1.0 * ld.x, lp.y + 1.0 * ld.y)]
        elif len(circles) == 1 and len(lines) == 0:
            ctr, r = circles[0]
            cand = [_AlgPointNum(ctr.x + r, ctr.y)]
        else:
            return None, None, None

        if not cand:
            return None, None, None
        sol[pt] = cand[0]

    return sol, None, None


def refine_degenerate_witness(
    seq: "ConstructionSequence",
    point: str,
    free_point_names: list[str],
    max_iter: int = 2000,
) -> dict[str, _AlgPointNum] | None:
    """Numerically drive a two-line construction step to exact degeneracy.

    Optimizes the free points feeding into `point`'s construction so that
    the two lines meant to jointly determine it become parallel to machine
    precision, then returns a placement where `point` sits on that now-
    shared line (any point on it satisfies both defining constraints).

    Returns None if `point`'s step isn't a two-line intersection, or if the
    optimizer fails to converge (this branch may simply not have a nearby
    degenerate configuration).
    """
    from scipy.optimize import minimize

    def objective(params: np.ndarray) -> float:
        free_values = {
            name: (params[2 * i], params[2 * i + 1])
            for i, name in enumerate(free_point_names)
        }
        # Nelder-Mead explores freely and can walk two of the free points to
        # (nearly) the same coordinate, which degenerates some intermediate
        # construction step even when `point`'s own step is fine — e.g. a
        # zero-length direction vector feeding a line-circle intersection.
        # Treat any such failure as "not degenerate here" (cost 1.0) rather
        # than letting the exception propagate and abort the whole rule.
        try:
            _sol, line1, line2 = _build_up_to_step(seq, free_values, point)
        except (ZeroDivisionError, ValueError):
            return 1.0
        if line1 is None or line2 is None:
            return 1.0
        return _line_line_degeneracy(line1[1], line2[1]) ** 2

    rng = np.random.RandomState(0)
    best = None
    for attempt in range(4):
        x0 = rng.uniform(-2, 2, size=2 * len(free_point_names))
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-16, "maxiter": max_iter},
        )
        if best is None or res.fun < best.fun:
            best = res
        if best.fun < 1e-20:
            break

    if best is None or best.fun**0.5 > 1e-6:
        return None  # didn't find a nearby degenerate configuration

    free_values = {
        name: (best.x[2 * i], best.x[2 * i + 1])
        for i, name in enumerate(free_point_names)
    }
    try:
        sol, line1, line2 = _build_up_to_step(seq, free_values, point)
    except (ZeroDivisionError, ValueError):
        return None
    if sol is None or line1 is None:
        return None
    lp, ld = line1
    # Any t != 0 places `point` on the (now shared) line without landing
    # exactly on the anchor — t=1.7 is an arbitrary non-degenerate choice.
    sol[point] = _AlgPointNum(lp.x + 1.7 * ld.x, lp.y + 1.7 * ld.y)
    return sol


# ---------------------------------------------------------------------------
# DDAR-based counterexample tester
# ---------------------------------------------------------------------------


def test_sequence(
    seq: ConstructionSequence,
    premises: list[tuple[str, list[str]]],
    conclusions: list[tuple[str, list[str]]],
    rename_map: dict[str, str],
    n_seeds: int = 8,
    max_attempts: int = 500,
) -> dict[str, Any]:
    """Test a construction sequence for counterexamples.

    Builds the fl_problem WITHOUT goal (?), runs DDAR with multiple seeds.
    Checks if any branch gives premises=True, conclusion=False.

    Returns:
        {status: "ce_found"|"all_good"|"no_valid_configs"|"error",
         n_good: int, n_bad: int, bad_configs: [...], ...}
    """
    from newclid.api import GeometricSolverBuilder
    from newclid.discovery.validation.counterexample_search import (
        evaluate_predicate,
        _DDAR_REL_TOL,
    )
    from newclid.numerical.geometries import PointNum

    # Build fl_problem without goal
    fl = build_fl_problem(seq, rename_map)

    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072][
        :n_seeds
    ]
    good_configs = []
    bad_configs = []

    _CE_MARGIN = _DDAR_REL_TOL * 5.0
    _PREM_MARGIN = _DDAR_REL_TOL * 0.1

    for seed in seeds:
        try:
            solver = (
                GeometricSolverBuilder(seed=seed)
                .load_problem_from_txt(fl)
                .build(max_attempts=max_attempts)
            )
            sg = solver.proof.symbols_graph

            # Extract point coordinates
            pts = {}
            # Build pts dict with ORIGINAL point names as keys
            # (rename_map: original -> normalized, sg uses normalized names)
            for orig_name, norm_name in rename_map.items():
                if norm_name in sg.name2node:
                    p = sg.name2node[norm_name].num
                    pts[orig_name] = PointNum(p.x, p.y)

            if len(pts) < len(rename_map):
                continue

            # Check premises
            prem_ok = True
            for pname, pargs in premises:
                ok, viol = evaluate_predicate(pname, pargs, pts)
                if not ok or viol > _PREM_MARGIN:
                    prem_ok = False
                    break

            if not prem_ok:
                continue

            # Check conclusions
            concl_ok = True
            for cname, cargs in conclusions:
                ok, viol = evaluate_predicate(cname, cargs, pts)
                if not ok or viol > _CE_MARGIN:
                    concl_ok = False
                    break

            if concl_ok:
                good_configs.append(
                    {"seed": seed, "points": {n: (p.x, p.y) for n, p in pts.items()}}
                )
            else:
                bad_configs.append(
                    {"seed": seed, "points": {n: (p.x, p.y) for n, p in pts.items()}}
                )

        except Exception:
            continue

    if bad_configs:
        return {
            "status": "ce_found",
            "n_good": len(good_configs),
            "n_bad": len(bad_configs),
            "n_seeds_tried": len(seeds),
            "good_configs": good_configs[:3],
            "bad_configs": bad_configs[:3],
            "fl_problem": fl,
        }
    elif good_configs:
        return {
            "status": "all_good",
            "n_good": len(good_configs),
            "n_bad": 0,
            "n_seeds_tried": len(seeds),
            "good_configs": good_configs[:3],
            "fl_problem": fl,
        }
    else:
        return {
            "status": "no_valid_configs",
            "n_seeds_tried": len(seeds),
            "fl_problem": fl,
        }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def discover_counterexamples(
    rule_id: str,
    rule_text: str,
    rename_map: dict[str, str],
    fl_problem: str,
    constructions: dict[str, ConstructionDef],
    reverse_index: dict[tuple[str, int], list[ConstructionDef]],
    n_seeds: int = 8,
    max_sequences: int = 30,
    verbose: bool = False,
) -> dict[str, Any]:
    """Main entry point: find counterexamples via reverse construction.

    Returns:
        {status: "ce_found"|"all_good"|"no_construction_match"|...,
         rule_id: str,
         n_sequences: int,
         n_ce_found: int,
         results: [...],
         ...}
    """
    from newclid.discovery.utils.rule_parser import parse_predicate, split_rule_text

    start_time = time.time()

    prem_strs, concl_str = split_rule_text(rule_text)
    premises = [
        (n, list(a)) for n, a in (parse_predicate(p) for p in prem_strs if p.strip())
    ]
    conclusions = [
        (n, list(a))
        for c in concl_str.split(",")
        if c.strip()
        for n, a in [parse_predicate(c.strip())]
    ]

    # Step 1: match premises to constructions
    premise_matches: dict[int, list[ReverseMatch]] = {}
    for i, (pname, pargs) in enumerate(premises):
        matches = match_premise_to_constructions(pname, pargs, reverse_index)
        if matches:
            premise_matches[i] = matches
        else:
            if verbose:
                print(f"  [reverse_construction] premise {i} ({pname}) has no matches")

    if not premise_matches:
        return {
            "status": "no_construction_match",
            "rule_id": rule_id,
            "rule_text": rule_text,
            "runtime": time.time() - start_time,
        }

    # Step 2: generate construction sequences
    sequences = generate_construction_sequences(
        premises,
        conclusions[0] if conclusions else ("", []),
        premise_matches,
        constructions,
        max_sequences=max_sequences,
    )

    if verbose:
        print(f"  [reverse_construction] {len(sequences)} sequences generated")

    if not sequences:
        return {
            "status": "no_valid_sequence",
            "rule_id": rule_id,
            "rule_text": rule_text,
            "n_premise_matches": sum(len(v) for v in premise_matches.values()),
            "runtime": time.time() - start_time,
        }

    # Step 3: test each sequence
    all_results = []
    n_ce_found = 0
    for i, seq in enumerate(sequences):
        result = test_sequence(seq, premises, conclusions, rename_map, n_seeds=n_seeds)
        result["seq_index"] = i
        result["seq_steps"] = [f"{s.construction_name}({s.point})" for s in seq.steps]
        all_results.append(result)

        if result["status"] == "ce_found":
            n_ce_found += 1
            if verbose:
                print(
                    f"  [reverse_construction] seq {i}: CE FOUND! "
                    f"good={result['n_good']}, bad={result['n_bad']}"
                )

    runtime = time.time() - start_time
    return {
        "status": "ce_found" if n_ce_found > 0 else "all_good",
        "rule_id": rule_id,
        "rule_text": rule_text,
        "n_sequences": len(sequences),
        "n_tested": len(all_results),
        "n_ce_found": n_ce_found,
        "runtime": runtime,
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# Batch driver (compatible with Part 3 pipeline interface)
# ---------------------------------------------------------------------------


def run_reverse_construction_batch(
    rules_file: str,
    normalized_rules_path: str,
    occurrences_path: str,
    source_dataset_path: str,
    defs_path: str | None = None,
    n_seeds: int = 8,
    max_sequences: int = 30,
    n_workers: int = 1,
    limit: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Run reverse construction on all rules in rules_file.

    Compatible with the Part 3 pipeline interface (similar to discover_all).
    """
    from newclid.discovery.validation.rule_tracer import RuleTracer

    if defs_path is None:
        from newclid.configs import default_defs_path

        defs_path = str(default_defs_path())

    # Parse defs and build index
    print(f"[reverse_construction] Parsing defs from {defs_path}")
    constructions = parse_defs(defs_path)
    print(f"[reverse_construction]   {len(constructions)} constructions parsed")

    reverse_index = build_reverse_index(constructions)
    index_size = sum(len(v) for v in reverse_index.values())
    print(
        f"[reverse_construction]   reverse index: {len(reverse_index)} keys, "
        f"{index_size} entries"
    )

    # Build tracer
    print("[reverse_construction] Building tracer indexes...")
    tracer = RuleTracer(
        normalized_rules_path=normalized_rules_path,
        occurrences_path=occurrences_path,
        source_dataset_path=source_dataset_path,
    )
    tracer.build()

    # Load rules
    rules = []
    with open(rules_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rules.append(json.loads(line))
    if limit:
        rules = rules[:limit]

    print(f"[reverse_construction] Processing {len(rules)} rules...")

    results = []
    t0 = time.time()
    for i, rule in enumerate(rules):
        rid = rule["rule_id"]
        rtext = rule["rule_text"]

        # Get rename_map from normalized record
        norm_rec = tracer.get_norm_record(rid)
        if not norm_rec:
            results.append(
                {"status": "error", "rule_id": rid, "message": "norm record not found"}
            )
            continue

        rename_map = norm_rec.get("rename_map", {})
        src = tracer.get_source_record(
            norm_rec.get("seed"), norm_rec.get("index_in_seed", 0)
        )
        if not src:
            results.append(
                {
                    "status": "error",
                    "rule_id": rid,
                    "message": "source record not found",
                }
            )
            continue

        fl_problem = src.get("fl_problem", "")

        result = discover_counterexamples(
            rid,
            rtext,
            rename_map,
            fl_problem,
            constructions,
            reverse_index,
            n_seeds=n_seeds,
            max_sequences=max_sequences,
            verbose=verbose,
        )

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(rules) - i - 1) if i > 0 else 0
        n_seqs = result.get("n_sequences", 0)
        n_ce = result.get("n_ce_found", 0)
        status = result.get("status", "?")
        rt = result.get("runtime", 0)
        print(
            f"[reverse_construction] [{i + 1}/{len(rules)}] {rid}: {status}, "
            f"seqs={n_seqs}, ce={n_ce}, t={rt:.1f}s "
            f"({elapsed:.0f}s, ETA {eta:.0f}s)"
        )

        results.append(result)

    # Summary
    n_ce = sum(1 for r in results if r.get("n_ce_found", 0) > 0)
    n_match = sum(
        1 for r in results if r.get("status") not in ("no_construction_match", "error")
    )
    total_runtime = time.time() - t0
    print("\n[reverse_construction] Summary:")
    print(f"  Rules with counterexamples: {n_ce}/{len(results)}")
    print(f"  Rules with construction matches: {n_match}/{len(results)}")
    print(f"  Total runtime: {total_runtime:.1f}s")
    print(f"  Avg per rule: {total_runtime / max(len(results), 1):.1f}s")

    return results


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Reverse Construction NDG Discovery")
    p.add_argument(
        "--rules_file",
        required=True,
        help="JSONL rules file (e.g. premise_group_reduced.jsonl)",
    )
    p.add_argument(
        "--normalized_rules", required=True, help="normalized_rules.jsonl path"
    )
    p.add_argument(
        "--occurrences", required=True, help="rule_seed_occurrences_all.json path"
    )
    p.add_argument(
        "--source_dataset", required=True, help="Original dataset JSONL path"
    )
    p.add_argument("--defs", default=None, help="defs.txt path (default: from configs)")
    p.add_argument("--limit", type=int, default=10, help="Max rules to process")
    p.add_argument("--n_seeds", type=int, default=8, help="Seeds per sequence")
    p.add_argument(
        "--max_sequences", type=int, default=30, help="Max sequences per rule"
    )
    p.add_argument(
        "--output_dir",
        default="/tmp/rev_construction_test",
        help="Output directory for results",
    )

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = run_reverse_construction_batch(
        rules_file=args.rules_file,
        normalized_rules_path=args.normalized_rules,
        occurrences_path=args.occurrences,
        source_dataset_path=args.source_dataset,
        defs_path=args.defs,
        n_seeds=args.n_seeds,
        max_sequences=args.max_sequences,
        n_workers=1,
        limit=args.limit,
        verbose=True,
    )

    # Save results
    out_path = os.path.join(args.output_dir, "reverse_construction_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            # Make serializable
            serializable = {k: v for k, v in r.items() if k not in ("results",)}
            serializable["n_results"] = len(r.get("results", []))
            f.write(json.dumps(serializable, ensure_ascii=False, default=str) + "\n")

    print(f"\nResults saved to {out_path}")

    # Also save detailed results for rules with CEs
    detail_path = os.path.join(args.output_dir, "reverse_construction_details.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in results:
            if r.get("n_ce_found", 0) > 0:
                detail = {
                    "rule_id": r["rule_id"],
                    "rule_text": r["rule_text"],
                    "n_ce_found": r["n_ce_found"],
                    "fl_problems": [
                        res.get("fl_problem")
                        for res in r.get("results", [])
                        if res.get("status") == "ce_found"
                    ],
                    "seq_steps": [
                        res.get("seq_steps")
                        for res in r.get("results", [])
                        if res.get("status") == "ce_found"
                    ],
                }
                f.write(json.dumps(detail, ensure_ascii=False, default=str) + "\n")

    print(f"Details saved to {detail_path}")


# ---------------------------------------------------------------------------
# Lightweight interface for ndg_discovery.discover()
# ---------------------------------------------------------------------------

# Module-level cache: parse defs + build index once.
_rev_constructions: dict | None = None
_rev_index: dict | None = None
_rev_constructible: set | None = None


def _init_rev():
    global _rev_constructions, _rev_index, _rev_constructible
    if _rev_constructions is None:
        from newclid.configs import default_defs_path

        _rev_constructions = parse_defs(str(default_defs_path()))
        _rev_index = build_reverse_index(_rev_constructions)
        _rev_constructible = set(pn for (pn, _) in _rev_index.keys())


def try_reverse_construction(
    premises: list[tuple[str, list[str]]],
    conclusions: list[tuple[str, list[str]]],
    rename_map: dict[str, str],
    n_seeds: int = 12,
) -> dict | None:
    """Try to find counterexamples via reverse construction.

    Returns:
        None if translation fails (no sequence covering all premises).
        dict with 'bad_configs', 'good_configs' if build succeeds.
            bad_configs: list of {seed, points: {name: (x,y)}}
            good_configs: list of {seed, points: {name: (x,y)}}
        'min_degeneracy': smallest normalized line-intersection degeneracy
            (see `_line_line_degeneracy`) seen across every sample/sequence.
            Near 0 means some construction step's two defining lines are
            nearly parallel somewhere in the sampled solution space — the
            rule is only weakly/accidentally determined there, a strong
            signal that it may be numerically unsafe even when every sampled
            branch nominally satisfies the conclusion.
    """
    _init_rev()

    # Match constructible premises
    premise_matches = {}
    for i, (pn, pa) in enumerate(premises):
        if pn in _rev_constructible:
            m = match_premise_to_constructions(pn, pa, _rev_index)
            if m:
                premise_matches[i] = m

    if not premise_matches:
        return None

    # Generate sequences (must cover ALL constructible premises)
    seqs = generate_construction_sequences(
        premises,
        conclusions[0] if conclusions else ("", []),
        premise_matches,
        _rev_constructions,
        max_sequences=50,
    )
    if not seqs:
        return None

    bad_configs = []
    good_configs = []
    degeneracy_log: list[tuple[str, float]] = []

    # Algebraic solver: enumerate all branches deterministically.
    # Free points are sampled randomly, but combined constructions
    # (comma-separated) are solved via line/circle intersection.
    from newclid.discovery.validation.counterexample_search import (
        evaluate_predicate,
        _DDAR_REL_TOL,
    )

    _PREM_MARGIN = _DDAR_REL_TOL * 0.1

    # Points whose two-line step showed up as suspiciously close to
    # degenerate in at least one random sample, per sequence — candidates
    # to refine below.  Threshold is deliberately loose (this only decides
    # whether it's worth *trying* to refine, not whether the rule is
    # actually degenerate — that's decided by whether refinement converges
    # and whether the resulting witness violates the conclusion).
    _PROBE_THRESHOLD = 0.05
    suspects: dict[int, set[str]] = defaultdict(set)

    for seq_idx, seq in enumerate(seqs):
        seq_degeneracy_log: list[tuple[str, float]] = []
        sols = solve_sequence_algebraic(
            seq,
            rename_map,
            free_points_samples=max(n_seeds, 20),
            degeneracy_log=seq_degeneracy_log,
        )
        degeneracy_log.extend(seq_degeneracy_log)
        for pt, deg in seq_degeneracy_log:
            if deg < _PROBE_THRESHOLD:
                suspects[seq_idx].add(pt)

        for sol in sols:
            # Check all premises
            prem_ok = True
            for pn, pa in premises:
                ok, viol = evaluate_predicate(pn, pa, sol)
                if not ok or viol > _PREM_MARGIN:
                    prem_ok = False
                    break
            if not prem_ok:
                continue

            # Store as config dict with (x,y) tuples
            pts_dict = {n: (p.x, p.y) for n, p in sol.items()}
            config = {"seed": 0, "points": pts_dict}

            # Check conclusion
            concl_ok = True
            for cn, ca in conclusions:
                ok, viol = evaluate_predicate(cn, ca, sol)
                if not ok:
                    concl_ok = False
                    break

            if concl_ok:
                good_configs.append(config)
            else:
                bad_configs.append(config)

    min_degeneracy = min((d for _pt, d in degeneracy_log), default=None)

    # Locate-and-refine: for any point whose defining step came close to
    # degenerate, numerically drive it to exact degeneracy and sample the
    # now-truly-shared line for a clean witness — see
    # `refine_degenerate_witness`.  This is what actually catches rules
    # like custom_00045: the measure-zero degenerate boundary is (by
    # definition) essentially never hit by random sampling above, so
    # `bad_configs` from the loop alone stays empty even for an unsound
    # rule; this step targets the boundary directly instead of hoping to
    # stumble onto it.
    for seq_idx, points in suspects.items():
        seq = seqs[seq_idx]
        free_point_names = list(
            dict.fromkeys(s.point for s in seq.steps if s.construction_def.is_base)
        )
        if not free_point_names:
            continue
        for pt in points:
            witness = refine_degenerate_witness(seq, pt, free_point_names)
            if witness is None:
                continue
            prem_ok = True
            for pn, pa in premises:
                ok, viol = evaluate_predicate(pn, pa, witness)
                if not ok or viol > _PREM_MARGIN:
                    prem_ok = False
                    break
            if not prem_ok:
                continue  # refinement converged but witness isn't clean here
            concl_ok = all(
                evaluate_predicate(cn, ca, witness)[0] for cn, ca in conclusions
            )
            if not concl_ok:
                pts_dict = {n: (p.x, p.y) for n, p in witness.items()}
                bad_configs.append(
                    {
                        "seed": 0,
                        "points": pts_dict,
                        "source": f"degenerate_witness({pt})",
                    }
                )

    if bad_configs or good_configs:
        return {
            "bad_configs": bad_configs,
            "good_configs": good_configs,
            "min_degeneracy": min_degeneracy,
        }
    return None

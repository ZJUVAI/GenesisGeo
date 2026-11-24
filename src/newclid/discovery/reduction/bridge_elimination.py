"""
Bridge-segment elimination for discovered rules (Part 3 post-processing).

When a rule contains ``coll A B C``, the three segments AB, BC, AC all represent
the **same line**.  In line-based predicates (perp, para, coll, eqangle), any
occurrence of segment XY involving a collinear point can be replaced by an
equivalent segment that uses different points on the same line.  If after such
replacements a point entirely disappears from the conclusions, the ``coll``
premise that introduced it becomes unnecessary and can be removed.

Example
-------
Before::
    coll A B C, ncoll B D C, perp B D C E, perp B E C D => perp A B D E, perp A C D E

Segment AB → CB (same line, both pass through B and C):  ``perp C B D E``
Segment AC → BC (same line, both pass through B and C):  ``perp B C D E``

After (point ``A`` no longer appears in conclusions, ``coll A B C`` removed)::
    ncoll B D C, perp B D C E, perp B E C D => perp C B D E, perp B C D E

Architecture
------------
- Purely structural — no DDAR calls; O(N * P) per rule.
- Segment-based substitution for perp, para, eqangle (pairs of points).
- Point-based substitution for coll, ncoll (3-point relations).
- Metric predicates (cong, eqratio, midp, cyclic, ...) are NOT touched — segment
  length is not preserved under collinearity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from newclid.discovery.utils.rule_parser import split_rule_text


# Predicates where collinear segment substitution is semantically valid.
# These operate on *lines* or *angles* (which depend only on line direction).
# Segment-based predicates: each consecutive pair of arguments forms a segment
# (e.g. perp A B C D → segment AB, segment CD).  coll/ncoll are NOT segment-based
# (they are 3-point relations) and are handled separately.
LINE_PREDICATES = {"perp", "nperp", "para", "npara", "eqangle"}


def _parse_preds(text: str) -> list[list[str]]:
    """Parse comma-separated predicates into [[pred, arg, ...], ...]."""
    result: list[list[str]] = []
    for part in text.split(","):
        tokens = part.strip().split()
        if tokens:
            result.append(tokens)
    return result


def _format_pred(tokens: list[str]) -> str:
    """Inverse of _parse_preds for a single predicate."""
    return " ".join(tokens)


def _build_coll_map(premises: list[list[str]]) -> dict[str, set[str]]:
    """Build a point → same-line-partners map from coll and para premises.

    - ``coll A B C``: A, B, C on the same line.
    - ``para A B A C`` or ``para A B B C``: segments share a point (A or B)
      and are parallel → points A, B, C are collinear.
      (``para A B C D`` with four distinct points does NOT imply collinearity.)

    Transitive closure is not needed because each source premise already
    defines a full clique of same-line relationships.
    """
    coll_map: dict[str, set[str]] = defaultdict(set)

    def _add_coll(a: str, b: str, c: str) -> None:
        coll_map[a].update([b, c])
        coll_map[b].update([a, c])
        coll_map[c].update([a, b])

    for prem in premises:
        if prem[0] == "coll" and len(prem) >= 4:
            _add_coll(prem[1], prem[2], prem[3])
        elif prem[0] == "para" and len(prem) >= 5:
            # para A B C D → segments AB and CD
            p11, p12, p21, p22 = prem[1], prem[2], prem[3], prem[4]
            # Only extract collinearity when the two segments share a point
            # (otherwise para just means two distinct parallel lines).
            if p11 == p21 and p12 != p22:
                _add_coll(p11, p12, p22)  # para A B A C → coll A B C
            elif p11 == p22 and p12 != p21:
                _add_coll(p11, p12, p21)  # para A B C A → coll A B C
            elif p12 == p21 and p11 != p22:
                _add_coll(p12, p11, p22)  # para A B B C → coll B A C
            elif p12 == p22 and p11 != p21:
                _add_coll(p12, p11, p21)  # para A B C B → coll B A C

    return dict(coll_map)


def _try_replace_segment(
    args: list[str],
    pos1: int,
    pos2: int,
    coll_map: dict[str, set[str]],
) -> list[tuple[int, int, str, str]] | None:
    """Find all possible segment replacements for args[pos1:pos2].

    A segment (X, Y) at positions (pos1, pos2) in args can be replaced by
    (X', Y') if X' is a collinear partner of X (or X itself) AND Y' is a
    collinear partner of Y (or Y itself), and (X',Y') ≠ (X,Y).

    Returns a list of (pos1, pos2, new_X, new_Y) candidates, or None if no
    replacement is possible.
    """
    x, y = args[pos1], args[pos2]
    x_partners = coll_map.get(x, set()) | {x}
    y_partners = coll_map.get(y, set()) | {y}

    candidates = []
    for nx in x_partners:
        for ny in y_partners:
            if (nx, ny) != (x, y):
                candidates.append((pos1, pos2, nx, ny))
    return candidates if candidates else None


def bridge_point_eliminate(rule_text: str) -> str | None:
    """Attempt to eliminate bridge points from a single rule via segment substitution.

    Returns simplified rule_text or None if no simplification is possible.
    """
    if "=>" not in rule_text:
        return None

    lhs, rhs = rule_text.split("=>", 1)
    premises = _parse_preds(lhs)
    conclusions = _parse_preds(rhs)
    if not premises or not conclusions:
        return None

    # ---- 1. Build collinearity map from coll premises --------------------
    coll_map = _build_coll_map(premises)
    if not coll_map:
        return None

    # ---- 2. Identify which points are "bridge candidates" ----------------
    # A point is a bridge candidate if:
    #   - It appears in exactly one premise (count distinct premises, not occurrences)
    #   - That premise is a connective (coll, or para that implies collinearity)
    #   - It appears in the conclusion(s)
    point_prem_count: dict[str, int] = defaultdict(int)
    point_connective_prem: dict[str, int] = {}  # point → premise index
    for i, prem in enumerate(premises):
        is_connective = False
        if prem[0] == "coll":
            is_connective = True
        elif prem[0] == "para" and len(prem) >= 5:
            # Same logic as _build_coll_map: para implies collinearity only
            # when the two segments share a point.
            p11, p12, p21, p22 = prem[1], prem[2], prem[3], prem[4]
            if p11 == p21 or p11 == p22 or p12 == p21 or p12 == p22:
                is_connective = True
        if is_connective:
            for arg in set(prem[1:]):  # unique points in this premise
                point_connective_prem[arg] = i
        for arg in set(prem[1:]):  # count premises, not occurrences
            point_prem_count[arg] += 1

    concl_points: set[str] = set()
    for concl in conclusions:
        concl_points.update(concl[1:])

    bridge_candidates: list[tuple[str, int]] = []  # (point, coll_prem_idx)
    for pt in concl_points:
        if point_prem_count.get(pt, 0) == 1 and pt in point_connective_prem:
            bridge_candidates.append((pt, point_connective_prem[pt]))

    if not bridge_candidates:
        return None

    # ---- 3. Iteratively eliminate bridge points one by one ---------------
    # Process bridges one at a time: for each bridge point, try to replace
    # it in ALL conclusions via segment substitution.  If the point is
    # eliminated from all conclusions, remove its coll premise and restart
    # (because eliminating one bridge may expose another).
    set(range(len(premises)))
    remaining_conclusions: list[list[str]] = [list(concl) for concl in conclusions]
    bridge_queue: list[tuple[str, int]] = list(bridge_candidates)
    prem_indices_to_remove: set[int] = set()
    any_eliminated = False

    while bridge_queue:
        bp, coll_prem_idx = bridge_queue.pop(0)

        # Skip if already removed or point no longer in any conclusion
        if coll_prem_idx in prem_indices_to_remove:
            continue
        if not any(bp in concl[1:] for concl in remaining_conclusions):
            # Point already gone — premise can be removed
            prem_indices_to_remove.add(coll_prem_idx)
            any_eliminated = True
            continue

        # Try to eliminate this bridge point from ALL conclusions
        new_concls: list[list[str]] = []
        bp_eliminated = True

        for concl in remaining_conclusions:
            pred = concl[0]
            args = list(concl[1:])

            if bp in args:
                if pred in LINE_PREDICATES:
                    # ---- segment-based predicates (perp, para, eqangle) ----
                    # Each consecutive pair of args forms a segment; replace
                    # the segment containing bp with a collinear-partner segment.
                    seg_positions = [(i, i + 1) for i in range(0, len(args), 2)]
                    replaced = False
                    for seg_start, seg_end in seg_positions:
                        if seg_end >= len(args):
                            break
                        if args[seg_start] != bp and args[seg_end] != bp:
                            continue

                        # Try to replace just this bridge point, keeping the
                        # other point in the segment unchanged if possible.
                        other_pos = seg_end if args[seg_start] == bp else seg_start
                        other_pt = args[other_pos]
                        bp_partners = coll_map.get(bp, set())

                        # Best: keep other_pt, replace bp with a collinear
                        # partner that is not a bridge point itself.
                        best_new = None
                        for partner in bp_partners:
                            if partner == bp:
                                continue
                            # Prefer partner ≠ other_pt (avoid degenerate segment)
                            if partner != other_pt:
                                best_new = partner
                                break
                        if best_new is None and bp_partners:
                            # Fallback: any partner (may create degenerate — skip)
                            best_new = next(iter(bp_partners - {bp}), None)

                        if best_new is not None and best_new != other_pt:
                            args[seg_start if args[seg_start] == bp else seg_end] = (
                                best_new
                            )
                            replaced = True
                            # Continue checking remaining segments — bridge
                            # point may appear in multiple segments within one
                            # conclusion.

                    if not replaced or bp in args:
                        bp_eliminated = False

                elif pred in {"coll", "ncoll"}:
                    # ---- 3-point relations (coll, ncoll) ----
                    # Replace bridge point with any collinear partner not
                    # already in args (avoids degenerate "coll X X Y").
                    bp_partners = coll_map.get(bp, set())
                    best_new = None
                    for partner in bp_partners:
                        if partner != bp and partner not in args:
                            best_new = partner
                            break
                    if best_new is None:
                        best_new = next((p for p in bp_partners if p != bp), None)
                    if best_new is not None:
                        for j in range(len(args)):
                            if args[j] == bp:
                                args[j] = best_new
                    if bp in args:
                        bp_eliminated = False

                else:
                    # Metric / other predicates (cong, cyclic, midp, eqratio,
                    # simtri, contri, ...) — segment substitution is NOT valid
                    # for these.
                    bp_eliminated = False

            new_concls.append([pred] + args)

        if bp_eliminated:
            prem_indices_to_remove.add(coll_prem_idx)
            remaining_conclusions = new_concls
            any_eliminated = True
            # Restart: rebuild bridge candidates (some may now be redundant)
            # Re-check which bridges remain valid
            bridge_queue = [
                (bp2, pidx2)
                for bp2, pidx2 in bridge_candidates
                if pidx2 not in prem_indices_to_remove
                and any(bp2 in c[1:] for c in remaining_conclusions)
            ]

    if not any_eliminated:
        return None

    # ---- 4. Build simplified rule ----------------------------------------
    new_premises = [
        _format_pred(prem)
        for i, prem in enumerate(premises)
        if i not in prem_indices_to_remove
    ]

    # Deduplicate conclusions
    new_concl_strs = []
    seen: set[str] = set()
    for concl in remaining_conclusions:
        c_str = _format_pred(concl)
        if c_str not in seen:
            seen.add(c_str)
            new_concl_strs.append(c_str)

    if not new_concl_strs:
        return None

    new_rule = ", ".join(new_premises) + " => " + ", ".join(new_concl_strs)

    if len(new_premises) >= len(premises):
        return None

    return new_rule


def eliminate_bridge_points(
    rules: list[Any],
) -> tuple[list[Any], dict[str, dict]]:
    """Apply bridge-segment elimination to a list of ``RuleItem`` objects.

    Parameters
    ----------
    rules : list[RuleItem]
        Input rules (typically after reduction, before NDG).

    Returns
    -------
    (simplified_rules, audit)
    """
    audit: dict[str, dict] = {}
    result: list[Any] = []

    for r in rules:
        simplified = bridge_point_eliminate(r.rule_text)
        if simplified and simplified != r.rule_text:
            audit[r.rule_id] = {
                "rule_text": r.rule_text,
                "status": "bridge_eliminated",
                "stage": "bridge_elimination",
                "simplified_rule_text": simplified,
            }
            r.rule_text = simplified
            # Re-parse premises
            from newclid.discovery.utils.rule_parser import parse_predicate

            prem_strs, _concl_str = split_rule_text(simplified)
            r.premises = [
                (name, list(args))
                for name, args in (parse_predicate(p) for p in prem_strs if p.strip())
            ]
            r.premise_count = len(r.premises)
        result.append(r)

    return result, audit

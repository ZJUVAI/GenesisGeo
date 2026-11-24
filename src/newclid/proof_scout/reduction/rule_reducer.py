#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RuleReducer - Main orchestrator for rule reduction.

Implements a greedy algorithm to eliminate redundant rules:
1. Pre-filter by max_premises
2. Score rules by generality (fewer premises = more general)
3. Sort rules by generality (most general first)
4. For each rule, test if it subsumes any other rule
5. Eliminate subsumed rules
6. Return the minimal basis set
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import re
import threading
from pathlib import Path

from newclid.proof_scout.reduction.generality_scorer import GeneralityScorer
from newclid.proof_scout.reduction.subsumption_tester import SubsumptionTester


@dataclass
class RuleWithSource:
    """Rule with its source problem data."""

    rule_id: str  # e.g., "r000042"
    rule_text: str  # "premise1, premise2 => conclusion"
    points: List[Tuple[str, float, float]]  # [(name, x, y), ...]
    premises: List[Tuple[str, List[str]]]  # [(predicate, [args]), ...]
    goal: Tuple[str, List[str]]  # (predicate, [args])
    llm_output_renamed: str = ""  # Proof steps from original problem
    seed: Optional[int] = None  # Construction seed for group reduction

    # Parsed structure (computed lazily)
    _premises_parsed: Optional[List[Tuple[str, ...]]] = None
    conclusions: Optional[List[Tuple[str, ...]]] = None
    generality_score: Optional[Tuple[int, int]] = None  # (-n_premises, n_conclusions)

    def parse(self):
        """Parse rule text into premises and conclusions."""
        if self._premises_parsed is not None:
            return

        if "=>" not in self.rule_text:
            self._premises_parsed = []
            self.conclusions = []
            return

        premise_part, conclusion_part = self.rule_text.split("=>", 1)

        # Parse premises
        self._premises_parsed = []
        for condition in premise_part.split(","):
            parts = re.findall(r"\w+", condition)
            if parts:
                self._premises_parsed.append(tuple(parts))

        # Parse conclusions
        self.conclusions = []
        for condition in conclusion_part.split(","):
            parts = re.findall(r"\w+", condition)
            if parts:
                self.conclusions.append(tuple(parts))


class RuleReducer:
    """Main orchestrator for rule reduction."""

    def __init__(
        self,
        timeout: int = 60,
        seed: int = 42,
        verbose: bool = True,
        n_workers: int = 1,
        batch_size: int = 10,
        max_premises: Optional[int] = None,
        debug: bool = False,
        debug_output_dir: Optional[Path] = None,
        solver_type: str = "python",
        config_path: str = None,
        use_ray: bool = False,
    ):
        """Initialize RuleReducer.

        Args:
            timeout: Timeout in seconds for each subsumption test
            seed: Random seed for reproducibility
            verbose: Print progress messages
            n_workers: Number of parallel workers for subsumption testing
            batch_size: Progress reporting granularity (print every N rules processed)
            max_premises: Maximum number of premises allowed (default: None = no limit)
            debug: Output proof steps when a subsumption test succeeds
            debug_output_dir: Directory to write subsumption_proofs.txt; if None
                and debug=True, proof steps are printed to stdout
            solver_type: "python" for DirectSolver (DDARN), "csolver" for CSolver (C++ DDAR)
            config_path: Path of config for csolver, only used when solver_type="csolver"
            use_ray: Use Ray for distributed parallelism instead of ProcessPoolExecutor
        """
        self.timeout = timeout
        self.seed = seed
        self.verbose = verbose
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.max_premises = max_premises
        self.debug = debug
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        self.solver_type = solver_type
        self.config_path = config_path
        self.use_ray = use_ray

        proof_output_file = None
        if debug and self.debug_output_dir is not None:
            proof_output_file = self.debug_output_dir / "subsumption_proofs.txt"
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            proof_output_file.write_text("", encoding="utf-8")

        self.scorer = GeneralityScorer()
        if solver_type == "csolver":
            from newclid.proof_scout.reduction.subsumption_tester import (
                SubsumptionTesterCSolver,
            )

            self.tester = SubsumptionTesterCSolver(
                timeout=timeout,
                seed=seed,
                proof_output_file=proof_output_file,
                config_path=config_path,
            )
        else:
            self.tester = SubsumptionTester(
                timeout=timeout,
                seed=seed,
                proof_output_file=proof_output_file,
            )

    def reduce(self, rules: List[RuleWithSource]) -> Dict[str, Any]:
        """Reduce rules to a minimal basis set via greedy subsumption.

        Pipeline: max_premises pre-filter → generality sort → greedy elimination.

        Args:
            rules: List of RuleWithSource objects

        Returns:
            Dict with reduction results:
                - basis_rules: List of independent rules
                - eliminated_rules: List of eliminated rules with reasons
                - skipped_by_premises: List of rules skipped due to premise count
                - stats: Statistics about the reduction
        """
        if self.verbose:
            print(f"Starting rule reduction with {len(rules)} rules")

        original_count = len(rules)

        # Pre-filter: remove rules with too many premises
        skipped_by_premises = []
        if self.max_premises is not None:
            kept = []
            for rule in rules:
                if "=>" in rule.rule_text:
                    n_prem = len(
                        [
                            c
                            for c in rule.rule_text.split("=>")[0].split(",")
                            if c.strip()
                        ]
                    )
                else:
                    n_prem = 0
                if n_prem <= self.max_premises:
                    kept.append(rule)
                else:
                    skipped_by_premises.append(
                        {
                            "rule_id": rule.rule_id,
                            "rule_text": rule.rule_text,
                            "n_premises": n_prem,
                            "reason": f"Exceeds max_premises={self.max_premises} (has {n_prem})",
                        }
                    )
            if self.verbose:
                print(
                    f"  Pre-filter: {len(skipped_by_premises)} rules skipped (premises > {self.max_premises})"
                )
                print(f"  Remaining: {len(kept)} rules")
            rules = kept

        if len(rules) == 0:
            if self.verbose:
                print("\nNo rules remaining after pre-filter!")
            return {
                "basis_rules": [],
                "eliminated_rules": [],
                "skipped_by_premises": skipped_by_premises,
                "stats": {
                    "original_count": original_count,
                    "skipped_by_premises_count": len(skipped_by_premises),
                    "basis_count": 0,
                    "eliminated_count": 0,
                    "reduction_rate": 0,
                    "n_subsumption_tests": 0,
                },
            }

        # Step 1: Compute generality scores and parse rules
        if self.verbose:
            print("Step 1: Computing generality scores...")

        for rule in rules:
            rule.generality_score = self.scorer.score(rule.rule_text)
            rule.parse()

        # Step 2: Sort by generality (most general first)
        # Score is (-n_premises, n_conclusions), so reverse=True puts fewer premises first
        sorted_rules = sorted(rules, key=lambda r: r.generality_score, reverse=True)

        if self.verbose:
            print(f"Step 2: Sorted {len(sorted_rules)} rules by generality")
            print(
                f"  Most general: {sorted_rules[0].rule_text[:60]}... (score: {sorted_rules[0].generality_score})"
            )
            print(
                f"  Least general: {sorted_rules[-1].rule_text[:60]}... (score: {sorted_rules[-1].generality_score})"
            )

        # Step 3: Greedy elimination
        if self.verbose:
            if self.use_ray and self.n_workers > 1:
                mode = "Ray pipeline parallel"
            elif self.n_workers > 1:
                mode = "parallel (serial state)"
            else:
                mode = "sequential"
            print(f"Step 3: Greedy elimination ({mode})...")

        active_flags = [True] * len(sorted_rules)
        eliminated_rules = []
        n_tests = 0

        # Get batch_size from instance or use default
        batch_size = getattr(self, "batch_size", 10)

        if self.use_ray and self.n_workers > 1:
            # Ray pipeline parallel: submit multiple rules concurrently
            # Use ray.wait() to process results as they complete without blocking
            import ray
            from newclid.proof_scout.reduction.ray_workers import (
                test_subsumption_batch_worker_ray,
                serialize_rule,
            )

            if self.verbose:
                print(
                    f"  Using Ray with {self.n_workers} workers (pipeline parallel, progress every {batch_size} rules)"
                )

            # Pipeline control: limit in-flight tasks to n_workers * 2
            max_in_flight = self.n_workers * 2
            pending_futures = []  # List of (rule_i_index, future)
            processed_count = 0

            for i, rule_i in enumerate(sorted_rules):
                if not active_flags[i]:
                    continue

                # Build target rules based on CURRENT active_flags
                target_rules = []
                target_indices = []
                for j in range(len(sorted_rules)):
                    if active_flags[j] and j != i:
                        target_rules.append(sorted_rules[j])
                        target_indices.append(j)

                if not target_rules:
                    continue

                # Serialize rules for Ray transmission
                rule_i_data = serialize_rule(rule_i)
                target_rules_data = [serialize_rule(r) for r in target_rules]

                # Submit task
                future = test_subsumption_batch_worker_ray.remote(
                    rule_i_data,
                    target_rules_data,
                    self.timeout,
                    self.seed,
                    self.solver_type,
                    self.config_path,
                )
                pending_futures.append((i, rule_i, target_indices, future))

                # Pipeline control: if we have too many in-flight tasks, wait for one to complete
                while len(pending_futures) >= max_in_flight:
                    # Wait for the next task to complete
                    ready_futures = [f for _, _, _, f in pending_futures]
                    done, _ = ray.wait(ready_futures, num_returns=1)
                    done_ref = done[0]

                    # Find and process the completed task
                    for idx, (rule_idx, rule_obj, targets, f) in enumerate(
                        pending_futures
                    ):
                        if f == done_ref:
                            pending_futures.pop(idx)
                            eliminated_ids = ray.get(done_ref)
                            n_tests += len(targets)
                            processed_count += 1

                            # Update active_flags
                            for rule_id in eliminated_ids:
                                for j, rule_j in enumerate(sorted_rules):
                                    if rule_j.rule_id == rule_id and active_flags[j]:
                                        active_flags[j] = False
                                        eliminated_rules.append(
                                            {
                                                "rule_id": rule_j.rule_id,
                                                "rule_text": rule_j.rule_text,
                                                "subsumed_by": rule_obj.rule_id,
                                                "reason": f"Subsumed by {rule_obj.rule_id}",
                                            }
                                        )

                                        if self.verbose:
                                            print(
                                                f"    Eliminated {rule_j.rule_id} (subsumed by {rule_obj.rule_id})"
                                            )
                                        break

                            # Progress reporting
                            if self.verbose and processed_count % batch_size == 0:
                                n_active = sum(active_flags)
                                print(
                                    f"  Progress: {processed_count}/{len(sorted_rules)} rules processed, "
                                    f"{n_active} active, {n_tests} tests, {len(pending_futures)} in-flight"
                                )
                            break

            # Wait for remaining tasks
            while pending_futures:
                ready_futures = [f for _, _, _, f in pending_futures]
                done, _ = ray.wait(ready_futures, num_returns=1)
                done_ref = done[0]

                for idx, (rule_idx, rule_obj, targets, f) in enumerate(pending_futures):
                    if f == done_ref:
                        pending_futures.pop(idx)
                        eliminated_ids = ray.get(done_ref)
                        n_tests += len(targets)
                        processed_count += 1

                        for rule_id in eliminated_ids:
                            for j, rule_j in enumerate(sorted_rules):
                                if rule_j.rule_id == rule_id and active_flags[j]:
                                    active_flags[j] = False
                                    eliminated_rules.append(
                                        {
                                            "rule_id": rule_j.rule_id,
                                            "rule_text": rule_j.rule_text,
                                            "subsumed_by": rule_obj.rule_id,
                                            "reason": f"Subsumed by {rule_obj.rule_id}",
                                        }
                                    )

                                    if self.verbose:
                                        print(
                                            f"    Eliminated {rule_j.rule_id} (subsumed by {rule_obj.rule_id})"
                                        )
                                    break

                        if self.verbose and processed_count % batch_size == 0:
                            n_active = sum(active_flags)
                            print(
                                f"  Progress: {processed_count}/{len(sorted_rules)} rules processed, "
                                f"{n_active} active, {n_tests} tests"
                            )
                        break

        elif self.n_workers > 1:
            # Parallel execution with serial state updates to avoid race conditions
            # Each rule_i is processed sequentially so active_flags stays consistent,
            # but the subsumption tests for rule_i vs all targets run in parallel.
            from concurrent.futures import ProcessPoolExecutor

            if self.verbose:
                print(
                    f"  Using {self.n_workers} workers (serial state updates, progress every {batch_size} rules)"
                )

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                for i, rule_i in enumerate(sorted_rules):
                    if not active_flags[i]:
                        continue

                    # Progress reporting
                    if self.verbose and (i + 1) % batch_size == 0:
                        n_active = sum(active_flags)
                        print(
                            f"  Progress: {i + 1}/{len(sorted_rules)} rules processed, {n_active} active, {n_tests} tests"
                        )

                    # Build target rules based on CURRENT active_flags (not a stale snapshot)
                    target_rules = []
                    target_indices = []
                    for j in range(len(sorted_rules)):
                        if active_flags[j] and j != i:
                            target_rules.append(sorted_rules[j])
                            target_indices.append(j)

                    if not target_rules:
                        continue

                    # Submit single rule's subsumption tests to worker pool, wait for result
                    future = executor.submit(
                        _test_subsumption_batch_worker,
                        rule_i,
                        target_rules,
                        self.timeout,
                        self.seed,
                        self.solver_type,
                        self.config_path,
                    )

                    try:
                        eliminated_ids = future.result()
                        n_tests += len(target_indices)

                        # Immediately update active_flags before processing next rule
                        for rule_id in eliminated_ids:
                            for j, rule_j in enumerate(sorted_rules):
                                if rule_j.rule_id == rule_id and active_flags[j]:
                                    active_flags[j] = False
                                    eliminated_rules.append(
                                        {
                                            "rule_id": rule_j.rule_id,
                                            "rule_text": rule_j.rule_text,
                                            "subsumed_by": rule_i.rule_id,
                                            "reason": f"Subsumed by {rule_i.rule_id}",
                                        }
                                    )

                                    if self.verbose:
                                        print(
                                            f"    Eliminated {rule_j.rule_id} (subsumed by {rule_i.rule_id})"
                                        )
                                    break
                    except Exception as e:
                        if self.verbose:
                            print(
                                f"  Warning: Error processing rule {rule_i.rule_id}: {e}"
                            )

        else:
            # Sequential execution (original code)
            for i, rule_i in enumerate(sorted_rules):
                if not active_flags[i]:
                    continue

                if self.verbose and (i + 1) % 10 == 0:
                    n_active = sum(active_flags)
                    print(
                        f"  Progress: {i + 1}/{len(sorted_rules)} rules processed, {n_active} active, {n_tests} tests"
                    )

                # Test subsumption: rule_i vs all other rules
                # IMPORTANT: Check all j (not just j > i), because generality score is heuristic
                for j in range(len(sorted_rules)):
                    if i == j or not active_flags[j]:
                        continue

                    rule_j = sorted_rules[j]

                    # Test if rule_i subsumes rule_j
                    n_tests += 1
                    if self.tester.test_subsumption(rule_i, rule_j, debug=self.debug):
                        active_flags[j] = False
                        eliminated_rules.append(
                            {
                                "rule_id": rule_j.rule_id,
                                "rule_text": rule_j.rule_text,
                                "subsumed_by": rule_i.rule_id,
                                "reason": f"Subsumed by {rule_i.rule_id}",
                            }
                        )

                        if self.verbose:
                            print(
                                f"    Eliminated {rule_j.rule_id} (subsumed by {rule_i.rule_id})"
                            )

        # Collect basis rules
        basis_rules = [r for i, r in enumerate(sorted_rules) if active_flags[i]]

        if self.verbose:
            print("\nReduction complete!")
            print(f"  Original rules: {original_count}")
            print(f"  Skipped (premises): {len(skipped_by_premises)}")
            print(f"  Basis rules: {len(basis_rules)}")
            print(f"  Eliminated rules: {len(eliminated_rules)}")
            rate = len(eliminated_rules) / len(rules) * 100 if rules else 0
            print(f"  Reduction rate: {rate:.1f}%")
            print(f"  Total subsumption tests: {n_tests}")

        return {
            "basis_rules": basis_rules,
            "eliminated_rules": eliminated_rules,
            "skipped_by_premises": skipped_by_premises,
            "stats": {
                "original_count": original_count,
                "skipped_by_premises_count": len(skipped_by_premises),
                "basis_count": len(basis_rules),
                "eliminated_count": len(eliminated_rules),
                "reduction_rate": len(eliminated_rules) / len(rules) if rules else 0,
                "n_subsumption_tests": n_tests,
            },
        }

    def reduce_by_seed(
        self, rules: List[RuleWithSource], use_ray: bool = False
    ) -> Dict[str, Any]:
        """Seed-grouped reduction: reduce within each seed group independently.

        1. Group rules by seed
        2. Within each group: max_premises pre-filter → generality sort → greedy elimination
        3. Collect survivors from all groups (+ no-seed rules passed through)

        Global reduction is NOT performed here — the caller (pipeline) is
        responsible for running chunk_reduction / global_reduction on the
        survivors if desired.

        Args:
            rules: List of RuleWithSource objects
            use_ray: If True, use Ray to parallelize across seed groups.
                Each seed group becomes a Ray task processed concurrently.
                If False, seed groups are processed sequentially (original behavior).

        Returns:
            Dict with basis_rules (group survivors), eliminated_rules, and stats.
        """
        # Group rules by seed
        seed_groups: Dict[Any, List[RuleWithSource]] = defaultdict(list)
        no_seed_rules: List[RuleWithSource] = []
        for r in rules:
            if r.seed is not None:
                seed_groups[r.seed].append(r)
            else:
                no_seed_rules.append(r)

        if self.verbose:
            print(f"\n{'=' * 60}")
            mode_str = "Ray parallel" if use_ray else "sequential"
            print(
                f"[reduce_by_seed] {len(rules)} rules, "
                f"{len(seed_groups)} seed groups, "
                f"{len(no_seed_rules)} rules without seed "
                f"(mode: {mode_str})"
            )
            print(f"{'=' * 60}")

        # Group reduction
        group_survivors: List[RuleWithSource] = []
        all_eliminated: List[Dict[str, Any]] = []
        all_skipped: List[Dict[str, Any]] = []
        group_stats: List[Dict[str, Any]] = []
        total_group_eliminated = 0
        total_group_skipped = 0

        if use_ray:
            # Ray parallel: all seed groups processed concurrently
            import ray
            from newclid.proof_scout.reduction.ray_workers import (
                reduce_seed_group_worker,
            )

            if self.verbose:
                print(f"  Submitting {len(seed_groups)} seed groups as Ray tasks...")

            # Submit all seed groups as Ray tasks
            futures = []
            seed_vals = []
            for seed_val, group_rules in sorted(seed_groups.items()):
                if self.verbose:
                    print(
                        f"  Submitting seed group {seed_val}: {len(group_rules)} rules"
                    )
                future = reduce_seed_group_worker.remote(
                    seed_val,
                    group_rules,
                    self.timeout,
                    self.seed,
                    self.solver_type,
                    self.config_path,
                    self.max_premises,
                    self.verbose,
                )
                futures.append(future)
                seed_vals.append(seed_val)

            if self.verbose:
                print(f"  All {len(futures)} tasks submitted. Waiting for results...")

            # Collect results as they complete using ray.wait for progress tracking
            remaining = list(zip(seed_vals, futures))
            completed = 0
            while remaining:
                # Wait for the next task to complete
                ready_futures = [f for _, f in remaining]
                done, _ = ray.wait(ready_futures, num_returns=1)
                done_ref = done[0]

                # Find which seed_val corresponds to this future
                for idx, (sv, f) in enumerate(remaining):
                    if f == done_ref:
                        seed_val = sv
                        remaining.pop(idx)
                        break

                result = ray.get(done_ref)
                completed += 1

                group_survivors.extend(result["basis_rules"])
                all_eliminated.extend(result["eliminated_rules"])
                all_skipped.extend(result["skipped_by_premises"])
                total_group_eliminated += result["stats"]["eliminated_count"]
                total_group_skipped += result["stats"]["skipped_by_premises_count"]
                group_stats.append(
                    {
                        "seed": seed_val,
                        "input": result["stats"]["original_count"],
                        "basis": result["stats"]["basis_count"],
                        "eliminated": result["stats"]["eliminated_count"],
                        "skipped_premises": result["stats"][
                            "skipped_by_premises_count"
                        ],
                    }
                )

                if self.verbose:
                    print(
                        f"  [{completed}/{len(seed_vals)}] Seed group {seed_val}: "
                        f"{result['stats']['original_count']} → {result['stats']['basis_count']} rules"
                    )

        else:
            # Sequential: original behavior
            for seed_val, group_rules in sorted(seed_groups.items()):
                if self.verbose:
                    print(f"\n--- Seed group {seed_val}: {len(group_rules)} rules ---")

                result = self.reduce(group_rules)
                group_survivors.extend(result["basis_rules"])
                all_eliminated.extend(result["eliminated_rules"])
                all_skipped.extend(result["skipped_by_premises"])
                total_group_eliminated += result["stats"]["eliminated_count"]
                total_group_skipped += result["stats"]["skipped_by_premises_count"]
                group_stats.append(
                    {
                        "seed": seed_val,
                        "input": result["stats"]["original_count"],
                        "basis": result["stats"]["basis_count"],
                        "eliminated": result["stats"]["eliminated_count"],
                        "skipped_premises": result["stats"][
                            "skipped_by_premises_count"
                        ],
                    }
                )

        # Add no-seed rules directly to survivors
        group_survivors.extend(no_seed_rules)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(
                f"[reduce_by_seed] Done: "
                f"{len(rules)} → {len(group_survivors)} survivors "
                f"(eliminated {total_group_eliminated}, "
                f"skipped {total_group_skipped})"
            )
            print(f"{'=' * 60}")

        return {
            "basis_rules": group_survivors,
            "eliminated_rules": all_eliminated,
            "skipped_by_premises": all_skipped,
            "stats": {
                "original_count": len(rules),
                "n_groups": len(seed_groups),
                "n_no_seed": len(no_seed_rules),
                "basis_count": len(group_survivors),
                "eliminated_count": total_group_eliminated,
                "skipped_premises_count": total_group_skipped,
                "group_details": group_stats,
            },
        }


def _parse_llm_input(
    llm_input: str,
) -> Tuple[List[Tuple[str, List[str]]], Tuple[str, List[str]]]:
    """Parse llm_input_renamed format into premises and goal.

    Format: <problem> a : ; b : ; c : perp a b a c [000] ; ... ? goal </problem>

    Returns:
        Tuple of (premises, goal) where:
        - premises: List[(predicate, [args]), ...]
        - goal: (predicate, [args])
    """
    # Remove <problem> tags
    text = llm_input.strip()
    if text.startswith("<problem>"):
        text = text[9:].strip()
    if text.endswith("</problem>"):
        text = text[:-10].strip()

    # Split by '?' to separate premises from goal
    if "?" not in text:
        raise ValueError(f"No goal marker '?' found in llm_input: {text}")

    premise_part, goal_part = text.split("?", 1)

    # Parse goal (single predicate)
    goal_tokens = goal_part.strip().split()
    if not goal_tokens:
        raise ValueError(f"Empty goal in llm_input: {text}")
    goal_pred = goal_tokens[0]
    goal_args = goal_tokens[1:] if len(goal_tokens) > 1 else []
    goal = (goal_pred, goal_args)

    # Parse premises
    premises = []
    clauses = premise_part.split(";")

    for clause in clauses:
        clause = clause.strip()
        if not clause or ":" not in clause:
            continue

        # Split by ':' to get point declaration and predicates
        _, pred_part = clause.split(":", 1)
        pred_part = pred_part.strip()

        if not pred_part:
            continue

        # Remove [NNN] tags
        pred_part = re.sub(r"\[\d+\]", "", pred_part).strip()

        # Split by predicate names (known predicates)
        # Common predicates: cong, para, perp, coll, cyclic, eqangle, eqratio, midp, etc.
        predicate_pattern = r"\b(cong|para|perp|coll|cyclic|eqangle|eqratio|midp|eqpoint|sameside|ncoll|on_line|on_circle|on_pline|on_tline|square|rect|iso_triangle|right_triangle|equilateral|parallelogram|trapezoid)\b"

        # Find all predicates and their positions
        matches = list(re.finditer(predicate_pattern, pred_part))

        for i, match in enumerate(matches):
            pred_name = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(pred_part)
            args_str = pred_part[start:end].strip()
            args = args_str.split() if args_str else []
            premises.append((pred_name, args))

    return premises, goal


def _parse_rule_map(rules_file, max_rules=None):
    """Parse rules.txt into an ordered {rule_id: rule_text} dict."""
    rule_map = {}
    with open(rules_file) as f:
        lines = [line.rstrip("\n") for line in f]
    i = 0
    while i + 1 < len(lines):
        rule_id = lines[i].strip()
        rule_text = lines[i + 1].strip()
        i += 2
        if not rule_id or not rule_text:
            continue
        rule_map[rule_id] = rule_text
        if max_rules and len(rule_map) >= max_rules:
            break
    return rule_map


def _build_line_idx_map(rule_map):
    """Build line_idx -> [rule_ids] reverse map for O(1) lookup during JSONL scan.

    Rule IDs of the form "base:NNNNNN" encode a 0-indexed line number.
    Multiple rules can share the same line index (e.g. multiple sub-graphs
    from the same source problem).
    """
    idx_map = {}
    for rule_id in rule_map:
        parts = rule_id.split(":")
        if len(parts) >= 2:
            try:
                idx = int(parts[-1])
                idx_map.setdefault(idx, []).append(rule_id)
            except (ValueError, IndexError):
                pass
    return idx_map


def _entry_to_rule_with_source(rule_id, rule_text, entry):
    """Build a RuleWithSource from a source data entry dict.

    Only extracts the fields needed (llm_input_renamed, point_coords / fl_problem, seed).
    Returns None if the entry cannot be converted (e.g. empty llm_input).
    """
    import re as _re

    llm_input = entry.get("llm_input_renamed", "")
    llm_output = entry.get("llm_output_renamed", "")
    point_coords = dict(entry.get("point_coords") or {})
    seed = entry.get("seed")

    # Fallback: parse point coords from fl_problem when point_coords missing
    if not point_coords:
        fl_problem = entry.get("fl_problem", "")
        if fl_problem:
            _coord_re = _re.compile(
                r"\b([A-Za-z][A-Za-z0-9]*)@"
                r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
                r"_"
                r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
            )
            for m in _coord_re.finditer(fl_problem):
                point_coords[m.group(1)] = [float(m.group(2)), float(m.group(3))]

    if not llm_input:
        return None

    try:
        premises, goal = _parse_llm_input(llm_input)
        points = [(name, coords[0], coords[1]) for name, coords in point_coords.items()]
        return RuleWithSource(
            rule_id=rule_id,
            rule_text=rule_text,
            points=points,
            premises=premises,
            goal=goal,
            llm_output_renamed=llm_output,
            seed=seed,
        )
    except Exception:
        return None


def _scan_jsonl_for_rules(source_data_file, rule_map):
    """Scan a JSONL source file once and return matching entries.

    Strategy 1: explicit rid/pid field in JSONL entry.
    Strategy 2: line-number matching via pre-built reverse index (O(1) per line,
                vs old O(N_rules) per line).

    Returns:
        entry_map: {rule_id -> slim_entry_dict} keeping only fields needed
                   for RuleWithSource construction.
    """
    import json as _json

    KEEP_FIELDS = {
        "llm_input_renamed",
        "llm_output_renamed",
        "point_coords",
        "fl_problem",
        "seed",
        "rid",
        "pid",
    }

    entry_map = {}
    target_rids = set(rule_map.keys())

    with open(source_data_file) as f:
        first_line = f.readline()
        if not first_line:
            return entry_map

        f.seek(0)
        is_jsonl = False
        try:
            data = _json.loads(first_line)
            if isinstance(data, dict) and "entries" not in data:
                is_jsonl = True
        except _json.JSONDecodeError:
            is_jsonl = False

        if not is_jsonl:
            # Standard JSON with "entries" list
            try:
                source_data = _json.load(f)
                for entry in source_data.get("entries", []):
                    rid = entry.get("rid") or entry.get("pid")
                    if rid and rid in target_rids:
                        slim = {k: entry[k] for k in KEEP_FIELDS if k in entry}
                        slim["rid"] = rid
                        entry_map[rid] = slim
                return entry_map
            except _json.JSONDecodeError as e:
                if "Extra data" not in str(e):
                    raise
                f.seek(0)
                is_jsonl = True

        if is_jsonl:
            # Pre-build line_idx -> [rule_ids] reverse map (O(N_rules), very fast)
            line_idx_map = _build_line_idx_map(rule_map)

            line_idx = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except _json.JSONDecodeError:
                    line_idx += 1
                    continue

                # Strategy 1: explicit rid/pid field
                rid = entry.get("rid") or entry.get("pid")
                if rid and rid in target_rids:
                    slim = {k: entry[k] for k in KEEP_FIELDS if k in entry}
                    slim["rid"] = rid
                    entry_map[rid] = slim
                else:
                    # Strategy 2: line-number lookup (O(1))
                    for rule_id in line_idx_map.get(line_idx, []):
                        if rule_id not in entry_map:
                            slim = {k: entry[k] for k in KEEP_FIELDS if k in entry}
                            slim["rid"] = rule_id
                            entry_map[rule_id] = slim

                line_idx += 1

    return entry_map


def load_rules_from_discovery_output(
    rules_file,
    source_data_file,
    max_rules=None,
):
    """Load rules from discovery pipeline output.

    Args:
        rules_file: Path to *_pruned_rules.txt (format: rule_id\\nrule_text\\n...)
        source_data_file: Path to step6_rules_stats.json or raw JSONL with
                          llm_input_renamed and point_coords / fl_problem.
        max_rules: Maximum number of rules to load (for testing).

    Returns:
        (rules, failures) where failures is [(rule_id, rule_text, reason), ...]
    """
    rules = []
    failures = []

    rule_map = _parse_rule_map(rules_file, max_rules)

    try:
        entry_map = _scan_jsonl_for_rules(source_data_file, rule_map)
    except Exception as e:
        print(f"Error loading source data: {e}")
        return [], failures

    for rule_id, rule_text in rule_map.items():
        entry = entry_map.get(rule_id)
        if not entry:
            failures.append((rule_id, rule_text, "No source data entry found"))
            continue
        rule = _entry_to_rule_with_source(rule_id, rule_text, entry)
        if rule is None:
            failures.append(
                (rule_id, rule_text, "Empty llm_input_renamed or parse error")
            )
            continue
        rules.append(rule)
        if max_rules and len(rules) >= max_rules:
            break

    return rules, failures


def stream_chunked_reduce_from_files(
    rules_file,
    source_data_file,
    group_size,
    reducer_cfg,
    output_dir=None,
    verbose=True,
):
    """Streaming chunk reduction: scan source JSONL once, reduce rules chunk-by-chunk.

    Unlike load_rules_from_discovery_output + ChunkedIterativeReducer, this
    function never loads all rules into memory at once:

      1. Parse rules.txt -> rule_map + line_idx reverse map (small, fits in RAM)
      2. Scan source JSONL exactly once line-by-line
      3. As rules are matched, accumulate into a chunk of `group_size`
      4. When chunk is full (or JSONL exhausted), run reduce() on that chunk
      5. Collect survivors across all chunks
      6. Optionally run a final global reduce on survivors

    Args:
        rules_file: Path to rules.txt
        source_data_file: Path to source JSONL or step6_rules_stats.json
        group_size: Max rules per chunk
        reducer_cfg: Dict with keys: timeout, n_workers, batch_size, solver_type,
                     config_path, debug, debug_output_dir, global_reduction (bool)
        output_dir: If set, save per-chunk survivors and final result here
        verbose: Print progress

    Returns:
        (final_survivors, stats)
    """
    import json as _json
    import time as _time

    KEEP_FIELDS = {
        "llm_input_renamed",
        "llm_output_renamed",
        "point_coords",
        "fl_problem",
        "seed",
        "rid",
        "pid",
    }

    timeout = reducer_cfg.get("timeout", 60)
    n_workers = reducer_cfg.get("n_workers", 1)
    batch_size = reducer_cfg.get("batch_size", 10)
    solver_type = reducer_cfg.get("solver_type", "csolver")
    config_path = reducer_cfg.get("config_path", None)
    debug = reducer_cfg.get("debug", False)
    debug_output_dir = reducer_cfg.get("debug_output_dir")
    do_global = reducer_cfg.get("global_reduction", True)

    def make_reducer():
        return RuleReducer(
            timeout=timeout,
            n_workers=n_workers,
            batch_size=batch_size,
            solver_type=solver_type,
            config_path=config_path,
            debug=debug,
            debug_output_dir=debug_output_dir,
            verbose=verbose,
        )

    t_start = _time.time()
    rules_file = Path(rules_file)
    source_data_file = Path(source_data_file)

    # Step 1: parse rules file (small — just rule_id/rule_text pairs)
    rule_map = _parse_rule_map(rules_file)
    n_total_rules = len(rule_map)
    if verbose:
        print(f"[stream_chunk_reduce] {n_total_rules} rules from {rules_file}")

    # Build reverse maps for matching
    line_idx_map = _build_line_idx_map(rule_map)  # line_idx -> [rule_ids]

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Detect JSON vs JSONL
    with open(source_data_file) as f:
        first_line = f.readline()
    is_jsonl = False
    if first_line:
        try:
            d = _json.loads(first_line)
            if isinstance(d, dict) and "entries" not in d:
                is_jsonl = True
        except _json.JSONDecodeError:
            is_jsonl = False

    # For standard JSON, fall back to full load
    if not is_jsonl:
        if verbose:
            print("[stream_chunk_reduce] source is JSON, loading all entries...")
        rules, failures = load_rules_from_discovery_output(rules_file, source_data_file)
        if verbose:
            print(
                f"[stream_chunk_reduce] loaded {len(rules)} rules, {len(failures)} failures"
            )
        reducer = make_reducer()
        result = reducer.reduce(rules)
        survivors = result["basis_rules"]
        stats = {
            "input_count": len(rules),
            "final_survivors_count": len(survivors),
            "elapsed_seconds": _time.time() - t_start,
            "mode": "json_fullload",
        }
        if output_dir:
            with open(output_dir / "final_basis_rules.txt", "w") as f:
                for r in survivors:
                    f.write(f"{r.rule_id}\n{r.rule_text}\n")
        return survivors, stats

    # Step 2: streaming JSONL scan + chunk reduction
    if verbose:
        print(
            f"[stream_chunk_reduce] streaming JSONL scan, group_size={group_size}, "
            f"n_workers={n_workers}"
        )

    all_survivors = []
    chunk_stats_list = []
    chunk_idx = 0
    current_chunk = []
    n_matched = 0
    n_failures = 0
    matched_rids = set()  # avoid duplicate rules from same line

    def flush_chunk():
        nonlocal chunk_idx
        if not current_chunk:
            return
        ci = chunk_idx
        chunk_idx += 1
        if verbose:
            print(f"\n  [chunk {ci}] {len(current_chunk)} rules -> reducing...")
        t0 = _time.time()
        r = make_reducer()
        result = r.reduce(current_chunk)
        basis = result["basis_rules"]
        elapsed = _time.time() - t0
        all_survivors.extend(basis)
        cs = {
            "chunk_id": ci,
            "input": len(current_chunk),
            "survivors": len(basis),
            "eliminated": result["stats"]["eliminated_count"],
            "n_tests": result["stats"]["n_subsumption_tests"],
            "elapsed_seconds": elapsed,
        }
        chunk_stats_list.append(cs)
        if verbose:
            print(
                f"  [chunk {ci}] {len(current_chunk)} -> {len(basis)} survivors ({elapsed:.1f}s)"
            )
        if output_dir:
            chunk_dir = output_dir / f"chunk_{ci:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            with open(chunk_dir / "survivors.txt", "w") as fh:
                for rule in basis:
                    fh.write(f"{rule.rule_id}\n{rule.rule_text}\n")
            import json as _j

            with open(chunk_dir / "stats.json", "w") as fh:
                _j.dump(cs, fh, indent=2)

    with open(source_data_file) as f:
        line_idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                line_idx += 1
                continue

            # Strategy 1: explicit rid/pid
            entry_rid = entry.get("rid") or entry.get("pid")
            if entry_rid and entry_rid in rule_map and entry_rid not in matched_rids:
                candidate_rids = [entry_rid]
            else:
                # Strategy 2: line-number lookup (O(1))
                candidate_rids = [
                    rid
                    for rid in line_idx_map.get(line_idx, [])
                    if rid not in matched_rids
                ]

            for rule_id in candidate_rids:
                rule_text = rule_map.get(rule_id)
                if not rule_text:
                    continue
                slim = {k: entry[k] for k in KEEP_FIELDS if k in entry}
                rule = _entry_to_rule_with_source(rule_id, rule_text, slim)
                if rule is None:
                    n_failures += 1
                    continue
                current_chunk.append(rule)
                matched_rids.add(rule_id)
                n_matched += 1

                if len(current_chunk) >= group_size:
                    flush_chunk()
                    current_chunk = []

            line_idx += 1

    # Flush remaining
    if current_chunk:
        flush_chunk()
        current_chunk = []

    n_unmatched = n_total_rules - n_matched - n_failures
    if verbose:
        print(
            f"\n[stream_chunk_reduce] scan complete: {n_matched} matched, "
            f"{n_failures} parse errors, {n_unmatched} unmatched in JSONL"
        )
        print(f"  {len(all_survivors)} survivors from {chunk_idx} chunks")

    # Step 3: optional global reduction on survivors
    if do_global and len(all_survivors) > 1:
        if verbose:
            print(
                f"\n[stream_chunk_reduce] global reduction on {len(all_survivors)} survivors..."
            )
        t0 = _time.time()
        r = make_reducer()
        global_result = r.reduce(all_survivors)
        final_survivors = global_result["basis_rules"]
        global_elapsed = _time.time() - t0
        if verbose:
            print(
                f"[stream_chunk_reduce] global: {len(all_survivors)} -> {len(final_survivors)} "
                f"({global_elapsed:.1f}s)"
            )
    else:
        final_survivors = all_survivors

    total_elapsed = _time.time() - t_start

    if output_dir:
        with open(output_dir / "final_basis_rules.txt", "w") as fh:
            for rule in final_survivors:
                fh.write(f"{rule.rule_id}\n{rule.rule_text}\n")
        import json as _j

        overall = {
            "input_count": n_total_rules,
            "matched": n_matched,
            "n_chunks": chunk_idx,
            "survivors_after_chunks": len(all_survivors),
            "final_survivors": len(final_survivors),
            "elapsed_seconds": total_elapsed,
            "chunk_stats": chunk_stats_list,
        }
        with open(output_dir / "stream_chunk_stats.json", "w") as fh:
            _j.dump(overall, fh, indent=2)

    return final_survivors, {
        "input_count": n_total_rules,
        "matched": n_matched,
        "n_chunks": chunk_idx,
        "survivors_after_chunks": len(all_survivors),
        "final_survivors_count": len(final_survivors),
        "elapsed_seconds": total_elapsed,
        "chunk_stats": chunk_stats_list,
        "mode": "streaming_jsonl",
    }


def _test_subsumption_batch_worker(
    rule_strong: RuleWithSource,
    rules_weak: List[RuleWithSource],
    timeout: int,
    seed: int,
    solver_type: str = "python",
    config_path: str = None,
) -> List[str]:
    """Worker function for parallel subsumption testing.

    Tests if rule_strong subsumes any rules in rules_weak.

    Args:
        rule_strong: RuleWithSource that might subsume others
        rules_weak: List of RuleWithSource to test against
        timeout: Timeout in seconds for each test
        seed: Random seed
        solver_type: "python" for DirectSolver, "csolver" for CSolver
        config_path: Path of config for CSolver

    Returns:
        List of rule_ids that are subsumed by rule_strong
    """
    if solver_type == "csolver":
        from newclid.proof_scout.reduction.subsumption_tester import (
            SubsumptionTesterCSolver,
        )

        tester = SubsumptionTesterCSolver(
            timeout=timeout, seed=seed, config_path=config_path
        )
    else:
        from newclid.proof_scout.reduction.subsumption_tester import SubsumptionTester

        tester = SubsumptionTester(timeout=timeout, seed=seed)
    eliminated = []

    for rule_weak in rules_weak:
        try:
            if tester.test_subsumption(rule_strong, rule_weak):
                eliminated.append(rule_weak.rule_id)
        except Exception:
            # If test fails, conservatively assume no subsumption
            pass

    return eliminated


class IncrementalReducer:
    """Background reducer that processes seed groups as they arrive from Stage 1.

    Usage:
        queue = Queue()
        reducer = IncrementalReducer(rule_reducer, queue)
        reducer.start()
        # ... Stage 1 pushes ("seed_done", seed, rules) and ("all_done", no_seed_rules) ...
        result = reducer.join()  # blocks until done
    """

    def __init__(self, reducer: RuleReducer, queue, *, verbose: bool = True):
        self.reducer = reducer
        self.queue = queue
        self.verbose = verbose
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Dict[str, Any]] = None
        self._group_survivors: List[RuleWithSource] = []
        self._group_stats: List[Dict[str, Any]] = []
        self._total_group_eliminated = 0
        self._total_group_skipped = 0

    def start(self):
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Consume queue messages and reduce seed groups incrementally."""
        while True:
            msg = self.queue.get()
            if msg[0] == "seed_done":
                _, seed_val, rule_entries = msg
                self._process_seed_group(seed_val, rule_entries)
            elif msg[0] == "all_done":
                _, no_seed_rules = msg
                self._finalize(no_seed_rules)
                break

    def _process_seed_group(self, seed_val, rule_entries: list):
        """Reduce a single seed group."""
        if not rule_entries:
            return

        # Convert entries to RuleWithSource
        rules = _entries_to_rules_with_source(rule_entries)
        if not rules:
            return

        if self.verbose:
            print(f"\n[IncrementalReducer] Seed group {seed_val}: {len(rules)} rules")

        result = self.reducer.reduce(rules)
        survivors = result.get("basis_rules", [])
        self._group_survivors.extend(survivors)
        self._total_group_eliminated += result["stats"]["eliminated_count"]
        self._total_group_skipped += result["stats"]["skipped_by_premises_count"]
        self._group_stats.append(
            {
                "seed": seed_val,
                "input": result["stats"]["original_count"],
                "basis": result["stats"]["basis_count"],
                "eliminated": result["stats"]["eliminated_count"],
            }
        )

    def _finalize(self, no_seed_entries: list):
        """Run global reduction on all group survivors + no-seed rules."""
        no_seed_rules = _entries_to_rules_with_source(no_seed_entries)
        all_survivors = self._group_survivors + no_seed_rules

        if self.verbose:
            print(
                f"\n[IncrementalReducer] Global phase: {len(all_survivors)} survivors"
            )

        if all_survivors:
            global_result = self.reducer.reduce(all_survivors)
        else:
            global_result = {
                "basis_rules": [],
                "eliminated_rules": [],
                "skipped_by_premises": [],
                "stats": {
                    "original_count": 0,
                    "basis_count": 0,
                    "eliminated_count": 0,
                    "skipped_by_premises_count": 0,
                    "reduction_rate": 0,
                    "n_subsumption_tests": 0,
                },
            }

        self._result = {
            "basis_rules": global_result["basis_rules"],
            "eliminated_rules": global_result["eliminated_rules"],
            "skipped_by_premises": global_result["skipped_by_premises"],
            "stats": {
                "original_count": sum(s["input"] for s in self._group_stats)
                + len(no_seed_rules),
                "group_phase": {
                    "n_groups": len(self._group_stats),
                    "n_no_seed": len(no_seed_rules),
                    "survivors": len(all_survivors),
                    "eliminated": self._total_group_eliminated,
                    "group_details": self._group_stats,
                },
                "global_phase": global_result["stats"],
                "basis_count": global_result["stats"]["basis_count"],
                "total_eliminated": self._total_group_eliminated
                + global_result["stats"]["eliminated_count"],
            },
        }

    def join(self, timeout=None) -> Optional[Dict[str, Any]]:
        """Wait for the reducer thread to finish and return results."""
        if self._thread:
            self._thread.join(timeout=timeout)
        return self._result


def _entries_to_rules_with_source(entries: list) -> List[RuleWithSource]:
    """Convert rule entry dicts (from Stage 1 output) to RuleWithSource objects."""
    rules = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        rule_text = e.get("norm_rule") or e.get("rule", "")
        rule_id = e.get("rid", "")
        llm_input = e.get("llm_input_renamed", "")
        llm_output = e.get("llm_output_renamed", "")
        point_coords = e.get("point_coords", {})
        seed = e.get("seed")

        if not rule_text or not llm_input:
            continue

        try:
            premises, goal = _parse_llm_input(llm_input)
            points = [
                (name, coords[0], coords[1]) for name, coords in point_coords.items()
            ]
            rules.append(
                RuleWithSource(
                    rule_id=rule_id,
                    rule_text=rule_text,
                    points=points,
                    premises=premises,
                    goal=goal,
                    llm_output_renamed=llm_output,
                    seed=seed,
                )
            )
        except Exception:
            continue
    return rules


def load_rules_by_ids(
    rule_ids: List[str],
    source_data_file: Path,
) -> Tuple[List[RuleWithSource], List[Tuple[str, str]]]:
    """Load RuleWithSource objects for specific rule IDs from source data.

    Used for checkpoint resume: read rule_id list from survivors.txt,
    then restore full RuleWithSource objects from the original source data.

    Args:
        rule_ids: List of rule IDs to load (e.g., ["r000042", "r000123"])
        source_data_file: Path to step6_rules_stats.json or JSONL file

    Returns:
        Tuple of (rules, failures) where failures is [(rule_id, reason), ...]
    """
    import json

    # Load source data file (supports both JSON and JSONL)
    entries = []
    target_rids = set(rule_ids)
    try:
        with open(source_data_file) as f:
            # More robust detection: read the first line
            first_line = f.readline()
            if not first_line:
                return [], [(rid, "Empty source data file") for rid in rule_ids]

            f.seek(0)
            is_jsonl = False
            try:
                data = json.loads(first_line)
                # If first line is a valid JSON object AND it doesn't have "entries" key
                if isinstance(data, dict) and "entries" not in data:
                    is_jsonl = True
            except json.JSONDecodeError:
                is_jsonl = False

            if not is_jsonl:
                # Standard JSON
                try:
                    source_data = json.load(f)
                    entries = source_data.get("entries", [])
                except json.JSONDecodeError as e:
                    if "Extra data" in str(e):
                        # Fallback to JSONL if we see extra data
                        f.seek(0)
                        is_jsonl = True
                    else:
                        raise e

            if is_jsonl:
                # Assume JSONL (one JSON object per line)
                # Try two strategies:
                # 1. If entries have "rid"/"pid" field, use that
                # 2. Otherwise, match by line number (extract index from rule_id)
                f.seek(0)
                line_idx = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Strategy 1: Check if entry has rid/pid
                        rid = entry.get("rid") or entry.get("pid")
                        if rid and rid in target_rids:
                            if "rid" not in entry and "pid" in entry:
                                entry["rid"] = entry["pid"]
                            entries.append(entry)
                        else:
                            # Strategy 2: Try to match by line number
                            # Extract numeric index from rule_id (e.g., "geometry_clauses10_samples10M_seeded:000132" -> 132)
                            for rule_id in target_rids:
                                # Try to extract the numeric part
                                parts = rule_id.split(":")
                                if len(parts) >= 2:
                                    try:
                                        idx = int(parts[-1])
                                        if idx == line_idx:
                                            entry["rid"] = rule_id
                                            entries.append(entry)
                                            break
                                    except (ValueError, IndexError):
                                        pass
                        line_idx += 1
                    except json.JSONDecodeError:
                        line_idx += 1
                        continue
    except Exception as e:
        print(f"Error loading source data: {e}")
        return [], [(rid, str(e)) for rid in rule_ids]

    entry_map = {entry["rid"]: entry for entry in entries}

    rules = []
    failures = []

    for rule_id in rule_ids:
        entry = entry_map.get(rule_id)
        if not entry:
            failures.append((rule_id, "No source data entry found"))
            continue

        llm_input = entry.get("llm_input_renamed", "")
        llm_output = entry.get("llm_output_renamed", "")
        point_coords = entry.get("point_coords", {})
        seed = entry.get("seed")
        rule_text = entry.get("norm_rule") or entry.get("rule", "")

        if not llm_input or not rule_text:
            failures.append((rule_id, "Empty llm_input_renamed or rule_text"))
            continue

        try:
            premises, goal = _parse_llm_input(llm_input)
            points = [
                (name, coords[0], coords[1]) for name, coords in point_coords.items()
            ]
            rules.append(
                RuleWithSource(
                    rule_id=rule_id,
                    rule_text=rule_text,
                    points=points,
                    premises=premises,
                    goal=goal,
                    llm_output_renamed=llm_output,
                    seed=seed,
                )
            )
        except Exception as e:
            failures.append((rule_id, str(e)))

    return rules, failures


def _reduce_chunk_worker(
    chunk_rules: List[RuleWithSource],
    timeout: int,
    seed: int,
    max_premises: Optional[int],
    batch_size: int,
    solver_type: str,
    config_path: str,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Worker function for serial chunk reduction with intra-chunk parallelism.

    Creates its own RuleReducer instance and reduces a chunk independently.
    n_workers controls parallelism within the subsumption test (intra-chunk).
    """
    reducer = RuleReducer(
        timeout=timeout,
        seed=seed,
        max_premises=None,  # premises already filtered
        n_workers=n_workers,
        batch_size=batch_size,
        verbose=True,
        solver_type=solver_type,
        config_path=config_path,
    )
    return reducer.reduce(chunk_rules)


class ChunkedIterativeReducer:
    """Chunked iterative rule reduction for large rule sets.

    Splits N rules into groups of group_size, reduces each group independently,
    then merges survivors for the next round. Reduces O(n^2) to ~O(n*G*k).
    """

    def __init__(
        self,
        timeout: int = 60,
        seed: int = 42,
        batch_size: int = 10,
        solver_type: str = "csolver",
        config_path: str = None,
        verbose: bool = True,
    ):
        self.timeout = timeout
        self.seed = seed
        self.batch_size = batch_size
        self.solver_type = solver_type
        self.config_path = config_path
        self.verbose = verbose

    @staticmethod
    def filter_by_premises(
        rules: List[RuleWithSource],
        max_premises: int,
    ) -> Tuple[List[RuleWithSource], List[Dict[str, Any]]]:
        """Filter rules exceeding max_premises.

        Returns:
            (kept, skipped) where skipped is a list of dicts with rule info.
        """
        kept = []
        skipped = []
        for rule in rules:
            if "=>" in rule.rule_text:
                n_prem = len(
                    [c for c in rule.rule_text.split("=>")[0].split(",") if c.strip()]
                )
            else:
                n_prem = 0
            if n_prem <= max_premises:
                kept.append(rule)
            else:
                skipped.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_text": rule.rule_text,
                        "n_premises": n_prem,
                        "reason": f"Exceeds max_premises={max_premises} (has {n_prem})",
                    }
                )
        return kept, skipped

    def reduce_one_round(
        self,
        rules: List[RuleWithSource],
        group_size: int,
        n_workers: int = 1,
    ) -> Tuple[List[RuleWithSource], Dict[str, Any]]:
        """One round of chunked reduction.

        Splits rules into chunks, reduces each sequentially (chunks are processed
        one at a time), with intra-chunk parallelism controlled by n_workers.

        Returns:
            (survivors, round_stats)
        """
        import math
        import time as _time

        n_chunks = math.ceil(len(rules) / group_size)
        chunks = [rules[i * group_size : (i + 1) * group_size] for i in range(n_chunks)]

        if self.verbose:
            print(
                f"  Splitting {len(rules)} rules into {n_chunks} chunks (group_size={group_size}, n_workers={n_workers})"
            )

        round_start = _time.time()
        survivors = []
        chunk_stats = []

        # Serial over chunks; intra-chunk parallelism via n_workers
        for ci, chunk in enumerate(chunks):
            if self.verbose:
                print(f"\n  --- Chunk {ci}/{n_chunks}: {len(chunk)} rules ---")
            result = _reduce_chunk_worker(
                chunk,
                self.timeout,
                self.seed,
                None,
                self.batch_size,
                self.solver_type,
                self.config_path,
                n_workers,
            )
            basis = result.get("basis_rules", [])
            survivors.extend(basis)
            chunk_stats.append(
                {
                    "chunk_id": ci,
                    "input": len(chunk),
                    "survivors": len(basis),
                    "eliminated": result["stats"]["eliminated_count"],
                    "n_tests": result["stats"]["n_subsumption_tests"],
                }
            )

        # Sort chunk_stats by chunk_id
        chunk_stats.sort(key=lambda x: x["chunk_id"])

        elapsed = _time.time() - round_start
        total_eliminated = sum(cs["eliminated"] for cs in chunk_stats)
        total_tests = sum(cs["n_tests"] for cs in chunk_stats)

        round_stats = {
            "input_count": len(rules),
            "survivors_count": len(survivors),
            "eliminated_count": total_eliminated,
            "n_chunks": n_chunks,
            "group_size": group_size,
            "n_workers": n_workers,
            "n_subsumption_tests": total_tests,
            "elapsed_seconds": elapsed,
            "chunk_details": chunk_stats,
        }

        if self.verbose:
            print(
                f"\n  Round result: {len(rules)} → {len(survivors)} survivors "
                f"(eliminated {total_eliminated}, {elapsed:.1f}s)"
            )

        return survivors, round_stats

    def reduce_iterative(
        self,
        rules: List[RuleWithSource],
        group_size: int,
        iterations: int = 1,
        n_workers: int = 1,
        output_dir: Optional[Path] = None,
        resume: bool = True,
    ) -> Tuple[List[RuleWithSource], Dict[str, Any]]:
        """Multi-round iterative chunked reduction.

        Args:
            rules: Input rules
            group_size: Rules per chunk
            iterations: Max number of rounds
            n_workers: Intra-chunk parallel workers for subsumption tests
            output_dir: Directory to save per-round results (enables resume)
            resume: If True, check for completed rounds and resume

        Returns:
            (final_survivors, overall_stats)
        """
        import json as _json
        import time as _time

        overall_start = _time.time()
        current_rules = rules
        start_round = 1
        all_round_stats = []

        # Checkpoint resume
        if resume and output_dir is not None:
            last_completed = 0
            for r in range(1, iterations + 1):
                round_dir = output_dir / f"round_{r:03d}"
                survivors_path = round_dir / "survivors.txt"
                if survivors_path.exists() and survivors_path.stat().st_size > 0:
                    last_completed = r
                else:
                    break

            if last_completed > 0:
                # Load survivors from last completed round
                round_dir = output_dir / f"round_{last_completed:03d}"
                survivors_path = round_dir / "survivors.txt"
                with open(survivors_path) as f:
                    lines = [line.strip() for line in f if line.strip()]

                # survivors.txt format: rule_id\nrule_text\n...
                survivor_ids = [
                    lines[i] for i in range(0, len(lines), 2) if i + 1 < len(lines)
                ]

                # Rebuild RuleWithSource from original rules
                rule_map = {r.rule_id: r for r in rules}
                current_rules = [
                    rule_map[rid] for rid in survivor_ids if rid in rule_map
                ]

                # Load accumulated stats
                for r in range(1, last_completed + 1):
                    stats_path = output_dir / f"round_{r:03d}" / "stats.json"
                    if stats_path.exists():
                        with open(stats_path) as f:
                            all_round_stats.append(_json.load(f))

                start_round = last_completed + 1
                if self.verbose:
                    print(
                        f"[Resume] Found {last_completed} completed round(s), "
                        f"loaded {len(current_rules)} survivors from round_{last_completed:03d}"
                    )

        for round_num in range(start_round, iterations + 1):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Round {round_num}/{iterations}: {len(current_rules)} rules")
                print(f"{'=' * 60}")

            # If rules fit in one chunk, this is the final round
            if len(current_rules) <= group_size:
                if self.verbose:
                    print(
                        f"  Rules ({len(current_rules)}) <= group_size ({group_size}), "
                        f"running final single-chunk reduction"
                    )

            survivors, round_stats = self.reduce_one_round(
                current_rules,
                group_size,
                n_workers,
            )
            round_stats["round"] = round_num
            all_round_stats.append(round_stats)

            # Save round results
            if output_dir is not None:
                round_dir = output_dir / f"round_{round_num:03d}"
                round_dir.mkdir(parents=True, exist_ok=True)

                # Save survivors
                survivors_path = round_dir / "survivors.txt"
                with open(survivors_path, "w", encoding="utf-8") as f:
                    for rule in survivors:
                        f.write(f"{rule.rule_id}\n{rule.rule_text}\n")

                # Save stats
                stats_path = round_dir / "stats.json"
                with open(stats_path, "w", encoding="utf-8") as f:
                    _json.dump(round_stats, f, ensure_ascii=False, indent=2)

            # Early termination: no rules eliminated this round
            if len(survivors) == len(current_rules):
                if self.verbose:
                    print(
                        f"\n  No rules eliminated in round {round_num}, stopping early"
                    )
                break

            # Early termination: fits in one chunk
            if len(survivors) <= group_size:
                if self.verbose:
                    print(
                        f"\n  Survivors ({len(survivors)}) <= group_size ({group_size}), done"
                    )
                break

            current_rules = survivors

        # Save final result
        if output_dir is not None:
            final_path = output_dir / "final_basis_rules.txt"
            with open(final_path, "w", encoding="utf-8") as f:
                for rule in survivors:
                    f.write(f"{rule.rule_id}\n{rule.rule_text}\n")
            if self.verbose:
                print(f"\nFinal basis rules saved to {final_path}")

        overall_elapsed = _time.time() - overall_start
        overall_stats = {
            "input_count": len(rules),
            "final_survivors_count": len(survivors),
            "total_eliminated": len(rules) - len(survivors),
            "total_rounds": len(all_round_stats),
            "group_size": group_size,
            "n_workers": n_workers,
            "elapsed_seconds": overall_elapsed,
            "round_stats": all_round_stats,
        }

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(
                f"Iterative reduction complete: {len(rules)} → {len(survivors)} rules "
                f"in {len(all_round_stats)} round(s), {overall_elapsed:.1f}s"
            )
            print(f"{'=' * 60}")

        return survivors, overall_stats


__all__ = [
    "RuleReducer",
    "RuleWithSource",
    "IncrementalReducer",
    "load_rules_from_discovery_output",
    "ChunkedIterativeReducer",
    "load_rules_by_ids",
]

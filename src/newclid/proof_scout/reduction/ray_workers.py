#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ray worker functions for distributed rule reduction.

This module provides Ray remote functions for parallelizing rule reduction tasks:
- Seed group reduction: parallel processing of seed groups
- Subsumption testing: parallel subsumption tests across multiple rules
"""

from typing import List, Dict, Any, Optional

try:
    import ray
except ImportError:
    ray = None


def _reduce_seed_group_worker(
    seed_val: Any,
    group_rules: List,
    timeout: int,
    seed: int,
    solver_type: str,
    config_path: str,
    max_premises: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    """Worker function for reducing a single seed group.

    This function is designed to be called by Ray as a remote task.
    Each seed group is processed independently in parallel.

    Args:
        seed_val: Seed value for this group
        group_rules: List of RuleWithSource objects in this seed group
        timeout: Timeout in seconds for each subsumption test
        seed: Random seed for reproducibility
        solver_type: "python" or "csolver"
        config_path: Path of config for csolver
        max_premises: Maximum number of premises allowed
        verbose: Print progress messages

    Returns:
        Dict with reduction results for this seed group
    """
    from newclid.proof_scout.reduction.rule_reducer import RuleReducer

    # Create a reducer with n_workers=1 (no nested parallelism within seed group)
    reducer = RuleReducer(
        timeout=timeout,
        seed=seed,
        solver_type=solver_type,
        config_path=config_path,
        max_premises=max_premises,
        verbose=verbose,
        n_workers=1,  # Sequential within each seed group
    )

    if verbose:
        print(
            f"[Ray Worker] Processing seed group {seed_val}: {len(group_rules)} rules"
        )

    result = reducer.reduce(group_rules)

    if verbose:
        print(
            f"[Ray Worker] Seed group {seed_val} done: "
            f"{len(group_rules)} → {result['stats']['basis_count']} rules"
        )

    # Add seed_val to result for tracking
    result["seed_val"] = seed_val

    return result


if ray is not None:
    # Create Ray remote version of the worker
    reduce_seed_group_worker = ray.remote(_reduce_seed_group_worker)
else:
    # Fallback if Ray is not available
    reduce_seed_group_worker = _reduce_seed_group_worker


def _test_subsumption_batch_worker_ray(
    rule_strong_data: Dict[str, Any],
    rules_weak_data: List[Dict[str, Any]],
    timeout: int,
    seed: int,
    solver_type: str,
    config_path: str,
) -> List[str]:
    """Ray worker for parallel subsumption testing.

    Tests if rule_strong subsumes any rules in rules_weak.
    This is designed to be called as a Ray remote task.

    Args:
        rule_strong_data: Serialized RuleWithSource data (dict)
        rules_weak_data: List of serialized RuleWithSource data
        timeout: Timeout in seconds for each test
        seed: Random seed
        solver_type: "python" or "csolver"
        config_path: Path of config for csolver

    Returns:
        List of rule_ids that are subsumed by rule_strong
    """
    from newclid.proof_scout.reduction.rule_reducer import RuleWithSource

    # Reconstruct RuleWithSource objects from serialized data
    rule_strong = RuleWithSource(**rule_strong_data)
    rules_weak = [RuleWithSource(**data) for data in rules_weak_data]

    # Use the existing subsumption tester
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


if ray is not None:
    test_subsumption_batch_worker_ray = ray.remote(_test_subsumption_batch_worker_ray)
else:
    test_subsumption_batch_worker_ray = _test_subsumption_batch_worker_ray


def serialize_rule(rule) -> Dict[str, Any]:
    """Serialize a RuleWithSource object to a dict for Ray transmission.

    Args:
        rule: RuleWithSource object

    Returns:
        Dict with all fields needed to reconstruct the rule
    """
    return {
        "rule_id": rule.rule_id,
        "rule_text": rule.rule_text,
        "points": rule.points,
        "premises": rule.premises,
        "goal": rule.goal,
        "llm_output_renamed": rule.llm_output_renamed,
        "seed": rule.seed,
    }

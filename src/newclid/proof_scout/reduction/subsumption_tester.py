#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SubsumptionTester - Test if one rule subsumes another.

A rule R1 subsumes R2 if:
- R1 can solve R2's source problem (using R1 as an additional rule)

This is tested by:
1. Loading R2's source problem
2. Adding R1 as a custom rule
3. Running DirectSolver (Python DDARN) to see if the problem is solved
"""

from pathlib import Path
from typing import Optional


class SubsumptionTester:
    """Test subsumption relationships between rules."""

    def __init__(
        self,
        timeout: int = 60,
        seed: int = 42,
        proof_output_file: Optional[Path] = None,
    ):
        """Initialize SubsumptionTester.

        Args:
            timeout: Timeout in seconds for each subsumption test
            seed: Random seed for reproducibility
            proof_output_file: If set, append proof steps of successful
                subsumptions to this file instead of printing to stdout
        """
        self.timeout = timeout
        self.seed = seed
        self.proof_output_file = Path(proof_output_file) if proof_output_file else None

    def test_subsumption(self, rule_strong, rule_weak, debug: bool = False) -> bool:
        """Test if rule_strong subsumes rule_weak.

        Uses DirectSolver to load rule_weak's source problem and add
        rule_strong as a custom rule. If the problem is solved, rule_strong
        subsumes rule_weak.

        Args:
            rule_strong: RuleWithSource that might subsume rule_weak
            rule_weak: RuleWithSource to test
            debug: If True, output the proof steps when subsumption succeeds.
                Output goes to proof_output_file if set, otherwise stdout.

        Returns:
            True if rule_strong can solve rule_weak's source problem
        """
        try:
            from newclid.api import DirectSolver

            solver = DirectSolver(
                points=rule_weak.points,
                premises=rule_weak.premises,
                goal=rule_weak.goal,
                seed=self.seed,
                custom_rules=[rule_strong.rule_text],
            )
            result = solver.run(timeout=self.timeout)
            if debug and result:
                proof_text = solver.write_proof_steps()
                sep = "─" * 60
                block = (
                    f"\n{sep}\n"
                    f"[DEBUG] Subsumption proof: {rule_strong.rule_id} ⊇ {rule_weak.rule_id}\n"
                    f"  Strong rule: {rule_strong.rule_text}\n"
                    f"  Weak rule:   {rule_weak.rule_text}\n"
                    f"{proof_text}\n"
                    f"{sep}\n"
                )
                if self.proof_output_file is not None:
                    self.proof_output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.proof_output_file, "a", encoding="utf-8") as fh:
                        fh.write(block)
                else:
                    print(block)
            return result
        except Exception:
            return False


def _to_pipe_format(rule_id: str, rule_text: str) -> str:
    """Convert JGEX DSL rule text to CSolver pipe format.

    Input:  rule_id="r42", rule_text="cong a b c d, perp e f g h => para i j k l"
    Output: "r42|cong a b c d,perp e f g h|para i j k l"
    """
    if "=>" not in rule_text:
        return f"{rule_id}||"
    premise_part, conclusion_part = rule_text.split("=>", 1)
    premises = ",".join(p.strip() for p in premise_part.split(","))
    conclusions = ",".join(c.strip() for c in conclusion_part.split(","))
    return f"{rule_id}|{premises}|{conclusions}"


class SubsumptionTesterCSolver(SubsumptionTester):
    """Test subsumption using CSolver (C++ DDAR engine)."""

    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path

    def test_subsumption(self, rule_strong, rule_weak, debug: bool = False) -> bool:
        """Test if rule_strong subsumes rule_weak using CSolver."""
        try:
            from newclid.api import CSolver

            csolver = CSolver(
                points=rule_weak.points,
                premises=rule_weak.premises,
                goals=[rule_weak.goal],
                config_path=self.config_path,
            )
            custom_rule = _to_pipe_format(rule_strong.rule_id, rule_strong.rule_text)
            result = csolver.run(custom_rules=[custom_rule])
            if debug and result:
                sep = "─" * 60
                block = (
                    f"\n{sep}\n"
                    f"[DEBUG] Subsumption proof (CSolver): {rule_strong.rule_id} ⊇ {rule_weak.rule_id}\n"
                    f"  Strong rule: {rule_strong.rule_text}\n"
                    f"  Weak rule:   {rule_weak.rule_text}\n"
                    f"  Custom rule (pipe): {custom_rule}\n"
                    f"{sep}\n"
                )
                if self.proof_output_file is not None:
                    self.proof_output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.proof_output_file, "a", encoding="utf-8") as fh:
                        fh.write(block)
                else:
                    print(block)
            return result
        except Exception:
            return False


__all__ = ["SubsumptionTester", "SubsumptionTesterCSolver", "_to_pipe_format"]

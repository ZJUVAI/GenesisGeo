"""Classical Breadth-First Search based agents."""

from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING, Any

from newclid.agent.agents_interface import (
    DeductiveAgent,
)
from newclid.proof import ProofState

if TYPE_CHECKING:
    from newclid.formulations.rule import Rule
    from newclid.dependencies.dependency import Dependency


class DDARN(DeductiveAgent):
    """Apply Deductive Derivation to exhaustion by Breadth-First Search.

    DDARN will match and apply all available rules level by level
    until reaching a fixpoint we call exhaustion.

    """

    def __init__(self):
        self.rule_buffer: list[Rule] = []
        self.application_buffer: list[Dependency] = []
        self.any_new_statement_has_been_added = True
  
    def run(self, proof: "ProofState", rules: list[Rule], timeout: int = 3600) -> dict[str, Any]:
        """Run DeductiveAgent until saturation or goal found."""
        infos: dict[str, Any] = {}
        for goal in proof.goals:
            if not goal.check_numerical():
                infos["error"] = f"{goal.pretty()} fails numerical check"
                return infos
        t0 = time.time()
        proof.dep_graph.obtain_numerical_checked_eqangle_and_eqratio()
        step = 0
        running = True
        while running and time.time() - t0 < timeout:
            running = self.step(proof=proof, rules=rules)
            step += 1

        infos["runtime"] = time.time() - t0
        infos["success"] = proof.check_goals()
        infos["steps"] = step
        for goal in proof.goals:
            if goal.check():
                infos[goal.pretty() + " succeeded"] = True
            else:
                infos[goal.pretty() + " succeeded"] = False
        return infos

    def step(self, proof: ProofState, rules: list[Rule]) -> tuple[bool, bool]:
        if proof.check_goals():
            return False
        if self.rule_buffer:
            theorem = self.rule_buffer.pop()
            logging.debug("ddarn matching" + str(theorem))
            deps = proof.match_theorem(theorem)
            logging.debug("ddarn matched " + str(len(deps)))
            self.application_buffer.extend(deps)
        elif self.application_buffer:
            dep = self.application_buffer.pop()
            logging.debug(f"ddarn : apply {dep}")
            if proof.apply_dep(dep):
                self.any_new_statement_has_been_added = True
        else:
            if not self.any_new_statement_has_been_added:
                return False
            self.any_new_statement_has_been_added = False
            self.rule_buffer = list(rules)
            logging.debug("ddarn : reload")
        return True

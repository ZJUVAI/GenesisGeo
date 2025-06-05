from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING, Any

from newclid.formulations.rule import Rule


if TYPE_CHECKING:
    from newclid.agent.agents_interface import DeductiveAgent
    from newclid.proof import ProofState


def run_loop(
    deductive_agent: "DeductiveAgent", proof: "ProofState", rules: list[Rule], timeout: int = 3600
) -> dict[str, Any]:
    """Run DeductiveAgent until saturation or goal found."""
    infos: dict[str, Any] = {}
    for goal in proof.goals:
        if not goal.check_numerical():
            infos["error"] = f"{goal.pretty()} fails numerical check"
            return infos
    t0 = time.time()

    step = 0
    running = True
    while running and time.time() - t0 < timeout:
        running = deductive_agent.step(proof=proof, rules=rules)
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

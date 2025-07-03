"""Helper functions to write proofs in a natural language."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from newclid.dependencies.dependency import IN_PREMISES, NUMERICAL_CHECK, Dependency
from newclid.statement import Statement
from newclid.dependencies.symbols import Point

if TYPE_CHECKING:
    from newclid.proof import ProofState

def get_structured_proof(proof_state: "ProofState", id: dict[Statement, str]) -> tuple[str, str, str]:
    def rediger(dep: Dependency) -> str:
        for statement in (dep.statement,) + dep.why:
            if statement not in id:
                id[statement] = f"{len(id):03d}"
        return f"{' '.join(premise.to_str() + ' [' + id[premise] + ']' for premise in dep.why)} ({dep.reason})=> {dep.statement.to_str()} [{id[dep.statement]}]"
    
    def rediger_new_format(dep: Dependency) -> str:
        """Generate proof step in new format: statement [id] rule_id [required_statement_ids]"""
        for statement in (dep.statement,) + dep.why:
            if statement not in id:
                id[statement] = f"{len(id):03d}"
        
        # Extract rule ID from reason string and handle special cases
        reason = dep.reason
        if "Ratio Chasing" in reason or reason == "Ratio":
            rule_id = "a00"
        elif "Angle Chasing" in reason or reason == "Angle":
            rule_id = "a01"
        elif "Shortcut Derivation" in reason or reason == "Shortcut":
            rule_id = "r99"
        elif reason and ' ' in reason:
            rule_id = reason.split()[0]
        else:
            rule_id = reason if reason else "unknown"
        
        # Generate new format: statement [statement_id] rule_id [premise_ids]
        premise_ids = ' '.join(f"[{id[premise]}]" for premise in dep.why)
        return f"{dep.statement.to_str()} [{id[dep.statement]}] {rule_id} {premise_ids}".strip()
    
    def pure_predicate(dep: Dependency) -> str:
        # return f"{' '.join(premise.to_str() + ' [' + id[premise] + ']' for premise in dep.why)} ({dep.reason})=> {dep.statement.to_str()} [{id[dep.statement]}]"
        return f"{dep.statement.to_str()} [{id[dep.statement]}]"
    goals = [goal for goal in proof_state.goals if goal.check()]
    (
        points,
        premises,
        numercial_checked_premises,
        aux_points,
        aux,
        numercial_checked_aux,
        proof_steps,
    ) = proof_state.dep_graph.get_proof_steps(goals)
    points = sorted([p.pretty_name for p in points])
    aux_points = sorted([p.pretty_name for p in aux_points])

    analysis = "<analysis> "
    for line in premises:
        if line.statement not in id:
            id[line.statement] = f"{len(id):03d}"
    sorted_premises = sorted(premises, key=lambda line: id[line.statement])
    analysis_items = []
    for line in sorted_premises:
        analysis_items.append(pure_predicate(line))
    analysis += " ; ".join(analysis_items) + " ; </analysis>"

    numerical_check = ""
    numerical_check_items = []
    for line in numercial_checked_premises:
        if line.statement not in id:
            id[line.statement] = f"{len(id):03d}"
    sorted_numercial_checked_premises = sorted(numercial_checked_premises, key=lambda line: id[line.statement])
    for line in sorted_numercial_checked_premises:
        numerical_check_items.append(pure_predicate(line))
    for line in numercial_checked_aux:
        if line.statement not in id:
            id[line.statement] = f"{len(id):03d}"
    sorted_numercial_checked_aux = sorted(numercial_checked_aux, key=lambda line: id[line.statement])
    for line in sorted_numercial_checked_aux:
        numerical_check_items.append(pure_predicate(line))
    if len(numerical_check_items) > 0:
        numerical_check = "<numerical_check> " + " ; ".join(numerical_check_items) + " ; </numerical_check>"

    proof = "<proof> "
    proof_steps_formatted = []
    for k, line in enumerate(proof_steps):
        if NUMERICAL_CHECK not in line.reason and IN_PREMISES not in line:
            proof_steps_formatted.append(rediger_new_format(line))
    
    # Join proof steps with semicolons
    proof += " ; ".join(proof_steps_formatted) + " ; </proof>"

    return analysis, numerical_check, proof


def write_proof_steps(proof_state: "ProofState", out_file: Optional[Path] = None, print_output: bool = True) -> None:
    """Output the solution to out_file.

    Args:
      proof: Proof state.
      problem: Containing the problem definition and theorems.
      out_file: file to write to, empty string to skip writing to file.
    """

    id: dict[Statement, str] = {}
    goals = [goal for goal in proof_state.goals if goal.check()]
    for k, goal in enumerate(goals):
        id[goal] = f"g{k}"

    def rediger(dep: Dependency) -> str:
        for statement in (dep.statement,) + dep.why:
            if statement not in id:
                id[statement] = str(len(id) - len(goals))
        return f"{', '.join(premise.pretty() + ' [' + id[premise] + ']' for premise in dep.why)} ({dep.reason})=> {dep.statement.pretty()} [{id[dep.statement]}]"

    # solution = "==========================\n"
    # solution += "* From problem construction:\n"
    # solution += f"Points : {', '.join(p.pretty_name for p in proof_state.symbols_graph.nodes_of_type(Point))}\n"
    # proof_deps = proof_state.dep_graph.proof_deps(goals)
    # premises: list[Dependency] = []
    # numercial_checked: list[Dependency] = []
    # proof_steps: list[Dependency] = []
    # for line in proof_deps:
    #     if IN_PREMISES == line.reason:
    #         premises.append(line)
    #     elif NUMERICAL_CHECK == line.reason:
    #         numercial_checked.append(line)
    #     else:
    #         proof_steps.append(line)
    # for line in premises:
    #     solution += rediger(line) + "\n"
    # for line in numercial_checked:
    #     solution += rediger(line) + "\n"

    (
        points,
        premises,
        numercial_checked_premises,
        aux_points,
        aux,
        numercial_checked_aux,
        proof_steps,
    ) = proof_state.dep_graph.get_proof_steps(goals)
    points = sorted([p.pretty_name for p in points if isinstance(p, Point)])
    aux_points = sorted([p.pretty_name for p in aux_points])

    solution = "==========================\n"
    solution += "* From theorem premises:\n"
    solution += f"Points : {', '.join(points)}\n"
    for line in premises:
        solution += rediger(line) + "\n"
    for line in numercial_checked_premises:
        solution += rediger(line) + "\n"

    solution += "\n* Auxiliary Constructions:\n"
    solution += f"Points : {', '.join(aux_points)}\n"
    for line in aux:
        solution += rediger(line) + "\n"
    for line in numercial_checked_aux:
        solution += rediger(line) + "\n"

    solution += "\n* Proof steps:\n"
    for k, line in enumerate(proof_steps):
        if NUMERICAL_CHECK not in line.reason and IN_PREMISES not in line:
            solution += f"{k:03d}. {rediger(line)}\n"
    solution += "=========================="
    if out_file is None and print_output is True:
        print(solution)
    elif out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(solution)
        logging.info("Solution written to %s.", out_file)
    return solution

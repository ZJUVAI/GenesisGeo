import logging
import re
from newclid.formulations.problem import ProblemJGEX
from newclid.statement import Statement
from newclid.api import GeometricSolver


class GeometryGoalFilter:
    def __init__(self, max_clauses: int = 5, min_proof_steps: int = 5, min_clauses_num: int = 3, filteration_rate: float = 0.6):
        self.max_clauses = max_clauses
        self.min_proof_steps = min_proof_steps
        self.min_clauses_num = min_clauses_num
        self.filteration_rate = filteration_rate

    def clauses_num_check(self, problemJGEX: ProblemJGEX) -> bool:
        if len(problemJGEX.constructions) < self.min_clauses_num:
            logging.debug(f"Too few clauses: {len(problemJGEX.constructions)}")
            return False
        else:
            return True

    def proof_check(self, solver: GeometricSolver, goal: Statement) -> bool:
        try:
            _, _, _, _, _, _, proof_steps, = solver.proof.dep_graph.get_proof_steps([
                                                                                    goal])
            if len(proof_steps) < self.min_proof_steps:
                logging.debug(f"Naive proof: {goal}")
                return False
            else:
                return True
        except Exception as e:
            logging.warning(f"error in get_proof_steps {goal}: {e}. Why?")
            return False

    def goal_valid_check(self, name, args, dep_graph):
        if args[-1] == '':
            args = args[:-1]
        # AG1 do not support aconst and rconst
        if name in ('aconst', 'rconst'):  # rconst AB:AB=1, aconst ∠AB AB=0
            return False
        # case: cong AB = AB,
        if name == 'cong':
            left = {args[0], args[1]}
            right = {args[2], args[3]}
            if left == right:
                return False
        # para AB ∥ AB, AB ∥ AC
        if name == 'para':
            if len({args[0], args[1], args[2], args[3]}) < 4:
                return False
        if name == 'eqratio':
            seg_1, seg_2, seg_3, seg_4 = {args[0], args[1]}, {
                args[2], args[3]}, {args[4], args[5]}, {args[6], args[7]}
            # case: eqratio AB/CD = AB/CD
            # case: eqratio AB/CD = CD/AB => cong AB = CD
            # case: eqratio AB/AB = CD/EF => cong CD = EF
            if seg_1 == seg_2 or seg_1 == seg_3 or seg_3 == seg_4 or seg_2 == seg_4:
                return False
            # case: exist two segments with the same length
            sm1 = Statement.from_tokens(['cong'] + list(args[:4]), dep_graph)
            sm2 = Statement.from_tokens(
                ['cong'] + list(args[0:2])+list(args[4:6]), dep_graph)
            if sm1.check() or sm2.check():
                return False
        if name == 'eqangle':
            seg_1, seg_2, seg_3, seg_4 = {args[0], args[1]}, {
                args[2], args[3]}, {args[4], args[5]}, {args[6], args[7]}
            # case: eqangle ∠(AB,CD) = ∠(AB,CD)
            # case: eqangle ∠(AB,CD) = ∠(CD,AB) => perp AB⊥CD
            # case: eqangle ∠(AB,AB) = ∠(CD,EF) => para CD∥EF
            if seg_1 == seg_2 or seg_1 == seg_3 or seg_3 == seg_4 or seg_2 == seg_4:
                return False
            # case: two parallels or perp
            parallel_sets = [
                [args[0], args[1], args[2], args[3]],
                [args[0], args[1], args[4], args[5]],
                [args[0], args[1], args[6], args[7]],
                [args[4], args[5], args[6], args[7]],
                [args[2], args[3], args[4], args[5]],
                [args[2], args[3], args[6], args[7]]
            ]
            for arg_set in parallel_sets:
                sm = Statement.from_tokens(['para'] + arg_set, dep_graph)
                if sm.check():
                    return False
            sm = Statement.from_tokens(['perp'] + list(args[:4]), dep_graph)
            if sm.check():
                return False
            # case: simtri
            a1_args = list(set(args[:4]))
            a2_args = list(set(args[4:]))
            if len(a1_args) == 3 and len(a2_args) == 3:
                simtri_sets = [
                    [*a1_args, *a2_args],
                    [*a1_args, a2_args[0], a2_args[2], a2_args[1]],
                    [*a1_args, a2_args[1], a2_args[0], a2_args[2]],
                    [*a1_args, a2_args[1], a2_args[2], a2_args[0]],
                    [*a1_args, a2_args[2], a2_args[0], a2_args[1]],
                    [*a1_args, a2_args[2], a2_args[1], a2_args[0]]
                ]
                for simtri_set in simtri_sets:
                    sm = Statement.from_tokens(
                        ['simtri']+simtri_set, dep_graph)
                    if sm.check():
                        return False
                for simtri_set in simtri_sets:
                    sm = Statement.from_tokens(
                        ['simtrir']+simtri_set, dep_graph)
                    if sm.check():
                        return False
        if name in ('simtri', 'simtrir', 'contri', 'contrir'):
            # case: simtri △ABC ≅ △ABC
            tri_1 = {args[0], args[1], args[2]}
            tri_2 = {args[3], args[4], args[5]}
            if tri_1 == tri_2:
                return False
        if name == 'sameclock':
            return False

        return True

    def goal_filter(self, possible_goals, dep_graph):
        """filter the equivalent eq goals"""

        def check_equivalence(p1, p2, token_type):
            args1 = [arg.name for arg in p1.args]
            args2 = [arg.name for arg in p2.args]
            statements = [
                Statement.from_tokens(
                    [token_type, args1[i], args1[i + 1], args2[i], args2[i + 1]], dep_graph)
                for i in range(0, len(args1), 2)
            ]
            return all(sm.check() for sm in statements)

        def remove_duplicates(goals, equivalence_fn, token_type):
            unique_goals = []
            for goal in goals:
                if not any(equivalence_fn(existing_goal, goal, token_type) for existing_goal in unique_goals):
                    unique_goals.append(goal)
            return unique_goals

        eqangle_goals = [
            goal for goal in possible_goals if goal.predicate.NAME == 'eqangle']
        eqratio_goals = [
            goal for goal in possible_goals if goal.predicate.NAME == 'eqratio']
        other_goals = [goal for goal in possible_goals if goal.predicate.NAME not in (
            'eqangle', 'eqratio')]

        eqangle_goals = remove_duplicates(
            eqangle_goals, check_equivalence, 'para')
        eqratio_goals = remove_duplicates(
            eqratio_goals, check_equivalence, 'cong')

        return other_goals + eqangle_goals + eqratio_goals

    def proof_similarity_check(self, llm_output: str, proofs_of_used_rules: dict[str, list[tuple]]) -> bool:
        """Check if the LLM output is similar to any of the existing proofs."""

        def extract_rules_and_lines(output: str) -> tuple[str, tuple[str]]:
            """Extract rules from the proof string."""
            proof = output.split('<proof>')[1].split('</proof>')[0].strip()
            lines = [line.strip() for line in proof.split(';')][:-1]
            rules = [line.split(']')[1].strip().split()[0] for line in lines]
            return ('_'.join(rules), tuple(lines))

        def compute_similarity(lines1: tuple[str], lines2: tuple[str]) -> float:
            """Calculate the similarity between two proofs based on their lines."""
            set1, set2 = set(lines1), set(lines2)
            intersection = set1 & set2
            union = set1 | set2
            return len(intersection) / len(union) if union else 0.0

        rules, proof_lines = extract_rules_and_lines(llm_output)
        for proofs in proofs_of_used_rules.get(rules, []):
            similarity = compute_similarity(proof_lines, proofs)
            if similarity > self.filteration_rate:
                return True
        proofs_of_used_rules.setdefault(rules, []).append(proof_lines)
        return False

    def aux_predicates_valid_check(self, llm_output: str) -> bool:

        def is_valid(statement: str, valid_predicates: set) -> bool:
            prefix_match = re.match(r"(x00 \w+)\s*:\s*(.*)", content_item)
            if prefix_match:
                # coll a c e [002] coll b d e [003]
                rest = prefix_match.group(2)
                segments = re.split(r"\s*\[\d+\]", rest)
                # 'coll a c e' , 'coll b d e'
                segments = [seg.strip() for seg in segments if seg.strip()]
                for segment in segments:
                    parts = segment.split()
                    if parts and parts[0] not in valid_aux_predicates:
                        logging.debug(
                            f"Invalid auxiliary predicate: {parts[0]}")
                        return False
            return True

        # <aux> x00 c : perp k n n s [024] cong k n n s [025]; x00 h : ; x00 i : ; x00 j : perp h i h j [009] cong h i h j [010] ; </aux> <proof> cong a k c k [002] r19 [000] [001] ; cong b k c k [003] r19 [000] [001] ; cong a k b k [004] a00 [002] [003] ; </proof>
        valid_aux_predicates = {'perp', 'para',
                                'cong', 'coll', 'eqangle', 'cyclic'}
        aux_match = re.match(r"<aux>\s*(.*)\s*</aux>", llm_output)
        # c : perp a c b c [001] ; c : perp a c b c [001] ;
        if aux_match:
            aux_content = aux_match.group(1)
            for content_item in aux_content.split(';'):
                content_item = content_item.strip()
                if content_item:
                    if not is_valid(content_item, valid_aux_predicates):
                        return False
        return True

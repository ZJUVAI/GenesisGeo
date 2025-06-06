"""Classical Breadth-First Search based agents."""

from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING, Any
import re
from collections import defaultdict

from newclid.agent.agents_interface import (
    DeductiveAgent,
)
from newclid.proof import ProofState
from newclid.formulations.clause import Clause
from newclid.statement import Statement
from newclid.formulations.clause import translate_sentence
from newclid.configs import default_defs_path, default_rules_path
from newclid.formulations.rule import Rule

from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from newclid.formulations.rule import Rule
    from newclid.dependencies.dependency import Dependency


class LMAgent(DeductiveAgent):
    def __init__(self):
        self.rule_buffer: list[Rule] = []
        self.application_buffer: list[Dependency] = []
        self.any_new_statement_has_been_added = True
        
        # Set OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        # transformer
        self.model = None
        self.tokenizer = None

    def inference(self, query: str):
        chat_response = self.client.chat.completions.create(
            model="/c23474/home/zhuminfeng/LLaMA-Factory/saves/qwen2.5math1.5b-ag/full/sft",
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=3000,
            temperature=0.6,
            top_p=0.95,
            extra_body={"top_k": 10,},
        )
        return chat_response.choices[0].message.content
    
    def inference2(self, query: str):
        model_path = "/c23474/home/zhuminfeng/LLaMA-Factory/saves/qwen2.5math1.5b-ag/full/sft"
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        aux_dsl = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return aux_dsl

    def run(self, proof: "ProofState", rules: list[Rule], timeout: int = 3600) -> dict[str, Any]:
        """Run DeductiveAgent until saturation or goal found."""
        infos: dict[str, Any] = {}
        for goal in proof.goals:
            if not goal.check_numerical():
                infos["error"] = f"{goal.pretty()} fails numerical check"
                return infos
        t0 = time.time()

        step = 0
        lm_try = 0
        while lm_try < 5:
            proof.dep_graph.obtain_numerical_checked_eqangle_and_eqratio()
            running = True
            while running and time.time() - t0 < timeout:
                running = self.step(proof=proof, rules=rules)
                step += 1
            if proof.check_goals():
                break
            else:
                with torch.no_grad():
                    # problem -> dsl
                    dsl = self.problem_to_dsl(proof_state=proof)
                    # model inference (dsl) -> aux dsl # <aux>e: coll a c e [002] coll b d e [003]</aux>
                    # aux_dsl = self.inference(dsl)
                    aux_dsl = self.inference2(dsl)
                    # aux dsl -> construction
                    constructions = self.try_dsl_to_constructions(aux_dsl)
                    # add construction
                    max_attempts = 100
                    for _ in range(max_attempts):
                        try:
                            # add construction to problem
                            self.problemJGEX.with_more_construction(constructions)
                            # add construction to proof state
                            self.add_construction(proof=proof, s=constructions)
                            break
                        except Exception as e:
                            continue
                    self.any_new_statement_has_been_added = True
                    lm_try += 1

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

    def try_dsl_to_constructions(self, dsl):
        match = re.search(r"<aux>(.*?)</aux>", dsl) # <aux>e: coll a c e [002] coll b d e [003]</aux>
        if match:
            content = match.group(1).strip()  # e: coll a c e [002] coll b d e [003]
            
            prefix_match = re.match(r"(\w+)\s*:\s*(.*)", content)
            if prefix_match:
                prefix = prefix_match.group(1) # e
                rest = prefix_match.group(2) # coll a c e [002] coll b d e [003]
                segments = re.split(r"\s*\[\d+\]", rest)
                segments = [seg.strip() for seg in segments if seg.strip()]  # 'coll a c e' , 'coll b d e'
                # result = [prefix] + segments
                if len(segments) > 2:
                    return None
                result = prefix + ' = '
                result_constructions = []
                for segment in segments:
                    parts = segment.split()
                    predicate_name, args = self.translate_dsl_to_construction(prefix, parts[0], parts[1:])
                    result_constructions.append(f"{predicate_name} {' '.join(args)}")
                result += ', '.join(result_constructions)
                return result
        return None
    
    def translate_dsl_to_construction(self, point: str, predicate: str, args: list[str]
        ) -> tuple[str, list[str]]:
        """ Translate a predicate into construction
        
        Args:
            point: str: name of the new point
            predicate: str: name of the predicates, e.g., perp, para, etc.
            args: list[str]: list of predicate args.
        
        Return:
            (predicate, args): translated to constructive predicate.
        """
        # 直线垂直
        if predicate == 'perp':
            a, b, c, d = args
            if point in [c, d]:
                a, b, c, d = c, d, a, b
            if point == b:
                a, b = b, a
            if point == d:
                c, d = d, c
            if a == c and a == point:
                return 'on_dia', [a, b, d]
            return 'on_tline', [a, b, c, d]

        # 直线平行
        elif predicate == 'para':
            a, b, c, d = args
            if point in [c, d]:
                a, b, c, d = c, d, a, b
            if point == b:
                a, b = b, a
            return 'on_pline', [a, b, c, d]

        # 全等/等距
        elif predicate == 'cong':
            a, b, c, d = args
            if a == c and a == point:
                return 'on_bline', [a, b, d]
            if point in [c, d]:
                a, b, c, d = c, d, a, b
            if point == b:
                a, b = b, a
            if point == d:
                c, d = d, c
            if b in [c, d]:
                if b == d:
                    c, d = d, c
                return 'on_circle', [a, b, d]
            return 'eqdistance', [a, b, c, d]

        # 共线
        elif predicate == 'coll':
            a, b, c = args
            if point == b:
                a, b = b, a
            if point == c:
                a, b, c = c, a, b
            return 'on_line', [a, b, c]

        # 等角
        elif predicate == 'eqangle':
            a, b, c, d, e, f = args
            if point in [d, e, f]:
                a, b, c, d, e, f = d, e, f, a, b, c
            x, b2, y, c2, d2 = b, c, e, d, f
            if point == b2:
                a, b2, c2, d2 = b2, a, d2, c2
            if point == d2 and x == y:
                return 'angle_bisector', [point, b2, x, c2]
            if point == x:
                return 'eqangle3', [x, a, b2, y, c2, d2]
            return 'on_aline', [a, x, b2, c2, y, d2]

        # 四点共圆
        elif predicate == 'cyclic':
            a, b, c = [x for x in args if x != point]
            return 'on_circum', [point, a, b, c]

        # 其它直接返回
        return predicate, [point] + args if point not in args else args
    
    def problem_to_dsl(self, proof_state: "ProofState") -> str:
        """Convert the problem to a DSL string."""
        problem = self.problemJGEX
        dep_idx: dict[Statement, str] = {}
        defs = proof_state.defs
        
        data_tmp = defaultdict(list)
        for construction in problem.constructions:
            group = {}
            p2deps = defaultdict(list)
            for constr_sentence in construction.sentences:
                cdef = defs[constr_sentence[0]]
                if len(constr_sentence) == len(cdef.declare):
                    mapping = dict(zip(cdef.declare[1:], constr_sentence[1:]))
                else:
                    assert len(constr_sentence) + len(construction.points) == len(cdef.declare)
                    mapping = dict(zip(cdef.declare[1:], construction.points + constr_sentence[1:]))
                for points, bs in cdef.basics:
                    points = tuple([mapping[x] for x in points])
                    for p in points:
                        group[p] = points
                    for b in bs:
                        statement = Statement.from_tokens(translate_sentence(mapping, b), proof_state.dep_graph)
                        p2deps[points].append(statement)

            points = construction.points
            while points:
                p = points[0]
                gr = group[p]
                points = [x for x in points if x not in gr]

                deps = []
                for dep in p2deps[gr]:
                    deps.append(dep)
                data_tmp[' '.join(gr)] = deps

        # <problem> </problem>
        data_problem = '<problem> '
        string_premise = []
        for k, v in data_tmp.items():
            tmp_string = k + ': '
            for dep in v:
                if dep not in dep_idx:
                    dep_idx[dep] = f"{len(dep_idx):03d}"
                tmp_string += dep.to_str() + f' [{dep_idx[dep]}] '
            string_premise.append(tmp_string)
        data_problem += '; '.join([s.strip() for s in string_premise]) + ' ? '
        data_problem += ';'.join([
            (goal[0] + ' ' + ' '.join(goal[1:])) 
            for goal in problem.goals
            ])
        data_problem += ' </problem>'
        return data_problem
    
    def add_construction(self, proof: "ProofState", s: str) -> None:
        clauses = Clause.parse_line(s)
        for clause in clauses:
            proof.add_construction(clause)
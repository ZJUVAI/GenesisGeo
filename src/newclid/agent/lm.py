from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import logging
from typing import TYPE_CHECKING, Any
import re
from collections import defaultdict
import copy
from openai import OpenAI
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import heapq
import ray


from newclid.agent.agents_interface import (
    DeductiveAgent,
)
from newclid.formulations.problem import ProblemJGEX
from newclid.proof import ProofState
from newclid.formulations.clause import Clause
from newclid.statement import Statement
from newclid.formulations.clause import translate_sentence
from newclid.predicates.congruence import Cong
from newclid.predicates.parallelism import Para
from newclid.predicates.perpendicularity import Perp
from newclid.predicates.collinearity import Coll
from newclid.predicates.cyclic import Cyclic
from newclid.predicates.equal_angles import EqAngle

if TYPE_CHECKING:
    from newclid.formulations.rule import Rule
    from newclid.dependencies.dependency import Dependency



class LMAgent(DeductiveAgent):
    def __init__(self, model_path: list[Path], decoding_size: int, beam_size: int, search_depth: int):
        self.any_new_statement_has_been_added = True
        self.decoding_size = decoding_size
        self.beam_size = beam_size
        self.search_depth = search_depth
        
        # Set OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        # transformer
        self.model_path = model_path
        self.models = []
        self.tokenizers = []
        

    def inference(self, query: str):
        chat_response = self.client.chat.completions.create(
            model="LLaMA-Factory/saves/qwen2.5math1.5b-ag/full/sft",
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            # temperature=2,
            # top_p=0.9,
            temperature=0.0,
            logprobs=True,
            n=8,
            extra_body={
                'use_beam_search': True,
                'best_of': 8,
                'stop_token_ids': [62],
            },
        )
        #chat_response.choices[0].logprobs.content[0].logprob
        return chat_response.choices[0].message.content
    
    @torch.no_grad()
    def inference2(self, query: str, response_prefix: str = '<aux>'):
        if not self.models or not self.tokenizers:
            # Clear any existing models/tokenizers
            self.models.clear()
            self.tokenizers.clear()
            
            # Load all models and tokenizers
            for path in self.model_path:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype="auto",
                    device_map="auto", #"sequential",
                    attn_implementation="flash_attention_2"  # Sliding Window Attention is enabled but not implemented for others
                )
                tokenizer = AutoTokenizer.from_pretrained(path)
                self.models.append(model)
                self.tokenizers.append(tokenizer)
        
        aux_dsl_dict = {}
        # Process each model/tokenizer pair
        for model, tokenizer in zip(self.models, self.tokenizers):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # text = query
            model_prompt_inputs = tokenizer([text], return_tensors="pt")
            text += response_prefix
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
            bad_words_ids = tokenizer(["<", " <"]).input_ids
            past_key_values = DynamicCache()
            generated_output = model.generate(
                **model_inputs,
                max_new_tokens=100,
                num_beams=self.decoding_size,
                num_return_sequences=self.decoding_size,
                # num_beam_groups=5,
                # diversity_penalty=1.0,
                # do_sample=True, 
                # # temperature=2, 
                # top_k=50, 
                # top_p=0.9, 
                pad_token_id=151643,
                eos_token_id=2587, #' ;' #29
                # bad_words_ids=bad_words_ids,
                return_dict_in_generate=True, 
                output_scores=True,
                # use_cache=True,
                past_key_values=past_key_values,
            )
            scores = generated_output.sequences_scores
            generated_output = generated_output.sequences[:, model_prompt_inputs.input_ids.shape[1]:]
            aux_dsls = tokenizer.batch_decode(generated_output, skip_special_tokens=True)
            for aux_dsl, score in zip(aux_dsls, scores):
                score = score.item()
                if aux_dsl in aux_dsl_dict:
                    if score > aux_dsl_dict[aux_dsl]:
                        aux_dsl_dict[aux_dsl] = score
                else:
                    aux_dsl_dict[aux_dsl] = score
            
        return aux_dsl_dict # key: aux, value: score


    def run(self, proof: "ProofState", rules: list[Rule], timeout: int = 3600
        ) -> dict[str, Any]:
        """Run DeductiveAgent until saturation or goal found."""
        def proof_info(proof: "ProofState"):
            infos["runtime"] = time.time() - t0
            infos["success"] = proof.check_goals()
            infos["steps"] = step
            for goal in proof.goals:
                if goal.check():
                    infos[goal.pretty() + " succeeded"] = True
                else:
                    infos[goal.pretty() + " succeeded"] = False
            return infos

        infos: dict[str, Any] = {}
        for goal in proof.goals:
            if not goal.check_numerical():
                infos["error"] = f"{goal.pretty()} fails numerical check"
                return infos
        t0 = time.time()
        step = 0

        proof = LMAgent.run_ddar(proof, rules, t0, timeout)

        if not proof.check_goals():
            beam_queue = BeamQueue(max_size=self.beam_size)
            beam_queue.add(node=(self.problemJGEX, proof, '<aux>'), val=0)
            p_dsl = self.problem_to_dsl(self.problemJGEX, proof)
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
            
            future_info = dict()
            running_futures = []
            
            for depth in range(self.search_depth):
                new_queue = BeamQueue(max_size=self.beam_size)  # to replace beam_queue.
                for prev_score, (problem, proof, a_dsl) in beam_queue:
                    # seek help from llm
                    if time.time() - t0 > timeout:
                        break

                    # Stragety 1: insert the aux string into problem and predict the next aux
                    p_dsl = self.problem_to_dsl(problem, proof)
                    aux_dsl_dict = self.inference2(p_dsl, '<aux> x00')
                    for aux_dsl, score in aux_dsl_dict.items():
                        try:
                            aux = self.try_dsl_to_constructions(aux_dsl[len('<aux> x00'):])
                            if aux:
                                # create new problem as new task
                                new_problem = problem.with_more_construction(aux)  # will recreate the problem
                                new_proof = copy.deepcopy(proof)
                                self.add_construction(new_proof, aux)
                                # submit multi tasks
                                future = run_ddar_remote.remote(new_proof, rules, t0, timeout)
                                future_info[future] = (new_problem, prev_score, score, a_dsl)
                                running_futures.append(future)
                                # check any done futures
                                done, running_futures = ray.wait(running_futures, timeout=0)
                                for f in done:
                                    new_proof = ray.get(f)
                                    if new_proof.check_goals():
                                        for task in running_futures:
                                            ray.cancel(task, force=True)
                                        ray.shutdown()
                                        return proof_info(new_proof)
                                    else:
                                        # import pdb; pdb.set_trace()
                                        new_problem, prev_score, score, a_dsl = future_info[f]
                                        new_queue.add(node=(new_problem, new_proof, a_dsl), val=prev_score+score)
                                
                        except Exception as e:
                            # logging.warning(f"Error processing aux construction: {e} {aux}")
                            # import traceback
                            # traceback.print_exc()
                            continue
                            
                # check remaining futures/tasks
                done, remaining_futures = ray.wait(running_futures, num_returns=len(running_futures))
                for f in done:
                    new_proof = ray.get(f)
                    if new_proof.check_goals():
                        for task in remaining_futures:
                            ray.cancel(task, force=True)
                        ray.shutdown()
                        return proof_info(new_proof)
                    else:
                        new_problem, prev_score, score, a_dsl = future_info[f]
                        new_queue.add((new_problem, new_proof, a_dsl), val=prev_score + score)
                    
                    
                beam_queue = new_queue
            ray.shutdown()
        return proof_info(proof)

    def step(self, proof: ProofState, rules: list[Rule]) -> bool:
        return
    def try_dsl_to_constructions(self, content):
        points, premises = content.split(';')[0].split(' : ')

        # points
        points = points.strip().split()
        # currently, we only support one point following alphageometry
        if len(points) == 0 or len(points) > 1:
            return
        points = points[0]
    
        # premises
        premises = re.split(r"\s*\[\d+\]", premises) # coll a c e [002] coll b d e [003] 》'coll a c e' , 'coll b d e'
        premises = [seg.strip() for seg in premises if seg.strip()]  # 
        # currently, we only support two premises following alphageometry
        if len(premises) > 2:
            return 
            # segments = segments[:2]
        # TODO: should we support free points?
        if len(premises) == 0:
            return f'{points} = free {points}'
        result_constructions = []
        for premise in premises:
            parts = premise.split()
            construction = self.translate_dsl_to_construction(points, parts[0], parts[1:])
            result_constructions.append(construction)
        return points + ' = ' + ', '.join(result_constructions)


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
            return Perp.to_constructive(point, tuple(args))

        # 直线平行
        elif predicate == 'para':
            return Para.to_constructive(point, tuple(args))

        # 全等/等距
        elif predicate == 'cong':
            return Cong.to_constructive(point, tuple(args))

        # 共线
        elif predicate == 'coll':
            return Coll.to_constructive(point, tuple(args))

        # 等角
        elif predicate == 'eqangle':
            def dzt(args):
                a1, a2, b1, b2, c1, c2, d1, d2 = args
                if point in [c1, c2, d1, d2]:
                    a1, a2, b1, b2, c1, c2, d1, d2 = c1, c2, d1, d2, a1, a2, b1, b2
                if point in [b1, b2]:
                    a1, a2, b1, b2 = b1, b2, a1, a2
                if point in [d1, d2]:
                    c1, c2, d1, d2 = d1, d2, c1, c2
                if point == a2:
                    a1, a2 = a2, a1
                if point == b2:
                    b1, b2 = b2, b1
                if point == c2:
                    c1, c2 = c2, c1
                # 1. angle_bisector
                if point == a1 and point == c1:
                    b = a2
                    a = b1 if b == b2 else b2
                    c = d1 if b == d2 else d2
                    return 'angle_bisector', [point, a, b, c]
                # 2. eqangle3
                if point == a1 and point == b1:
                    a = a2
                    b = b2
                    d = c1 if c1 == d1 or c1 == d2 else c2
                    e = c2 if d == c1 else c1
                    f = d2 if d == d1 else d1
                    return 'eqangle3', [point, a, b, d, e, f]            
                # 3. on_aline
                a = a2
                b = b2 if a == b1 else b1
                d = c1 if c1 == d1 or c1 == d2 else c2
                c = c2 if d == c1 else c1
                e = d2 if d == d1 else d1
                return 'on_aline', [point, a, b, c, d, e]
            def arrange_angle_points(a, b, c, d):
                if a == c:
                    return (b, a, d)
                elif a == d:
                    return (b, a, c)
                elif b == c:
                    return (a, b, d)
                elif b == d:
                    return (a, b, c)
                else:
                    return None

            a, b, c, d, e, f, g, h = args
            res1 = EqAngle.to_constructive(point, arrange_angle_points(a, b, c, d) + arrange_angle_points(e, f, g, h))
            return res1
            
        # 四点共圆
        elif predicate == 'cyclic':
            return Cyclic.to_constructive(point, tuple(args))

        # 其它直接返回
        return f"{predicate} {' '.join(args)}"
    
    def problem_to_dsl(self, problem: "ProblemJGEX", proof_state: "ProofState") -> str:
        """Convert the problem to a DSL string."""
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
                    points = tuple(p.split('@')[0] for p in construction.points)
                    mapping = dict(zip(cdef.declare[1:], points + constr_sentence[1:]))
                for points, bs in cdef.basics:
                    points = tuple([mapping[x] for x in points])
                    for p in points:
                        group[p] = points
                    for b in bs:
                        statement = Statement.from_tokens(translate_sentence(mapping, b), proof_state.dep_graph)
                        p2deps[points].append(statement)

            points = construction.points
            points = [p.split('@')[0] for p in points]
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
            tmp_string = k + ' : '
            for dep in v:
                if dep not in dep_idx:
                    dep_idx[dep] = f"{len(dep_idx):03d}"
                tmp_string += dep.to_str() + f' [{dep_idx[dep]}] '
            string_premise.append(tmp_string)
        data_problem += ' ; '.join([s.strip() for s in string_premise]) + ' ? '
        data_problem += ' ; '.join([
            (goal[0] + ' ' + ' '.join(goal[1:])) 
            for goal in problem.goals
            ])
        data_problem += ' </problem>'
        return data_problem
    
    def add_construction(self, proof: "ProofState", s: str) -> None:
        clauses = Clause.parse_line(s)
        for clause in clauses:
            proof.add_construction(clause)
    @staticmethod
    def run_ddar(proof: "ProofState", rules: list[Rule], start_time: int, timeout: int = 3600):   
        rule_buffer: list[Rule] = []
        application_buffer: list[Dependency] = []
        any_new_statement_has_been_added = True
        proof.dep_graph.obtain_numerical_checked_premises()
        running = True
        while running and time.time() - start_time < timeout:
            if proof.check_goals():
                running = False
            if rule_buffer:
                theorem = rule_buffer.pop()
                logging.debug("ddarn matching" + str(theorem))
                deps = proof.match_theorem(theorem)
                logging.debug("ddarn matched " + str(len(deps)))
                application_buffer.extend(deps)
            elif application_buffer:
                dep = application_buffer.pop()
                logging.debug(f"ddarn : apply {dep}")
                if proof.apply_dep(dep):
                    any_new_statement_has_been_added = True
            else:
                if not any_new_statement_has_been_added:
                    running = False
                any_new_statement_has_been_added = False
                rule_buffer = list(rules)
                logging.debug("ddarn : reload")
            # TODO: add step later..
            # step += 1
        return proof

@ray.remote(num_cpus=1)
def run_ddar_remote(proof, rules, start_time, timeout):
    """Ray remote function to run DDAR."""
    return LMAgent.run_ddar(proof, rules, start_time, timeout)

class BeamQueue:
    """Keep only the top k objects according to their values."""

    def __init__(self, max_size: int = 512):
        self.queue = []
        self.max_size = max_size
        self.counter = 0
        self.entry_finder = {}
        self.REMOVED = object()

    def add(self, node: object, val: float) -> None:
        """Add a new node to this queue."""

        if len(self.queue) < self.max_size:
            entry = [val, self.counter, node]
            self.counter += 1
            heapq.heappush(self.queue, entry)
            self.entry_finder[node] = entry
        else:
            # Find the minimum node:
            min_val, _, min_node = self.queue[0]
            # replace it if the new node has higher value.
            if val > min_val:
                self.remove(min_node)
                entry = [val, self.counter, node]
                self.counter += 1
                heapq.heappush(self.queue, entry)
                self.entry_finder[node] = entry
    
    def remove(self, node: object) -> None:
        """Mark an existing node as REMOVED."""
        entry = self.entry_finder.pop(node, None)
        if entry:
            entry[-1] = self.REMOVED
        self._rebuild_heap()
    
    def _rebuild_heap(self):
        """Rebuild the heap to remove any invalid entries marked as REMOVED."""
        self.queue = [entry for entry in self.queue if entry[-1] is not self.REMOVED]
        heapq.heapify(self.queue)

    def __iter__(self):
        for val, _, node in self.queue:
            if node is not self.REMOVED:
                yield val, node

    def __len__(self) -> int:
        return len(self.queue)
    
    def __repr__(self) -> str:
        # return f'BeamQueue(max_size={self.max_size}, size={len(self.queue)}])'
        items = ',\n  '.join(f'({val:.4f}, {repr(node)})' for val, _, node in self.queue if node is not self.REMOVED)
        return f'BeamQueue(max_size={self.max_size}, size={len(self.queue)}, items=[\n  {items}\n])'
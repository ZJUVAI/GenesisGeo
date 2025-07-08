import json
from newclid.api import GeometricSolverBuilder
from pathlib import Path

with open("/c23474/home/math/dubhe/Newclid/tests/candidate_rule.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

base_rules = "simtri A B C P Q R => eqangle B A B C Q P Q R, eqratio B A B C Q P Q R\n" \
             "simtrir A B C P Q R => eqangle B A B C Q R Q P, eqratio B A B C Q P Q R\n" \
             "PythagoreanPremises a b c => PythagoreanConclusions a b c\n" \
             "cong a b p q, cyclic a b c p q r, sameclock c a b r p q, sameside c a b r p q => eqangle c a c b r p r q\n" \
             "cong a b p q, cyclic a b c p q r, sameclock c b a r p q, nsameside c a b r p q => eqangle c a c b r p r q\n" \
             "eqratio B A B C Q P Q R, eqratio C A C B R P R Q, sameclock A B C P Q R => simtri A B C P Q R\n" \
             "eqratio B A B C Q P Q R, eqratio C A C B R P R Q, sameclock A B C P R Q => simtrir A B C P Q R\n" \
             "eqratio B A B C Q P Q R, eqangle B A B C Q P Q R, sameclock A B C P Q R => simtri A B C P Q R\n" \
             "eqratio B A B C Q P Q R, eqangle B A B C Q R Q P, sameclock A B C P R Q => simtrir A B C P Q R\n" \
             "simtri A B C P Q R, cong A B P Q => contri A B C P Q R\n" \
             "simtrir A B C P Q R, cong A B P Q => contrir A B C P Q R\n"

i = 0
while i < len(data):
    current_item = data[i]
    print("checking ",current_item['rule_num'])

    rules = "\n".join(
            item['rule_text'] for idx, item in enumerate(data) if idx != i
        ) + "\n"
    
    rules = base_rules + rules
    
    problem = current_item['rule_problem']

    solver = (
            GeometricSolverBuilder(seed=123)
            .load_problem_from_txt(problem)
            .load_rules_from_txt(rules)
            .build()
        )
    
    success = solver.run()

    if success:
        del data[i]
        print("delete rule: ",current_item['rule_num'])
        solver.write_proof_steps(Path('/c23474/home/math/dubhe/Newclid/output.txt'))
    else:
        i += 1

print("Remaining rules:")
for item in data:
    print(item['rule_num'])
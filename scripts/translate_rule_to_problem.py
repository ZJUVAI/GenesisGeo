import re
import random
import argparse
from functools import cmp_to_key
from collections import deque, defaultdict

def _split_top_level(expr: str):
    """Split by commas at top-level (not inside parentheses)."""
    parts = []
    buf = []
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            seg = ''.join(buf).strip()
            if seg:
                parts.append(seg)
            buf = []
        else:
            buf.append(ch)
    seg = ''.join(buf).strip()
    if seg:
        parts.append(seg)
    return parts


# 拆分规则
def split_rule(rule):
    # 拆分规则为premise和goal部分
    premise_part, goal_part = rule.split('=>', 1)

    # 拆分前提部分为每个条件
    premise_part = premise_part.replace('∧', ',')
    goal_part = goal_part.replace('∧', ',')
    premise_conditions = _split_top_level(premise_part)
    premise_tuples = []
    for condition in premise_conditions:
        # 匹配每个条件的格式，假设条件由标识符和参数组成
        parts = re.findall(r'\w+', condition)  # 提取所有单词（标识符和参数）
        if parts:
            premise_tuples.append(tuple(parts))

    # 提取目标部分
    goal_conditions = _split_top_level(goal_part)
    goal_tuples = []
    for condition in goal_conditions:
        # 匹配每个条件的格式，假设条件由标识符和参数组成
        parts = re.findall(r'\w+', condition)  # 提取所有单词（标识符和参数）
        if parts:
            goal_tuples.append(tuple(parts))
    
    return premise_tuples, goal_tuples

# point1 依赖于 point2
def update_dep(point1, point2, deps):
    if point1 == point2 or point1 in deps[point2]:
        return
    deps[point2].append(point1)
    for p, dep in deps.items():
        if point2 in dep:
            update_dep(point1, p, deps)
    for p in deps[point1]:
        update_dep(p, point2, deps)

    return

def translate_perp(args, deps, constructions, state):
    a, b, c, d = args
    if a not in deps[d] and b not in deps[d] and c not in deps[d] and state[d]:
        constructions[d].append(('on_tline', d, c, a, b))
        if len(constructions[d]) == 2:
            state[d] = False
        for p in [a, b, c]:
            update_dep(d, p, deps)
        return True
    elif a not in deps[c] and b not in deps[c] and d not in deps[c] and state[c]:
        constructions[c].append(('on_tline', c, d, a, b))
        if len(constructions[c]) == 2:
            state[c] = False
        for p in [a, b, d]:
            update_dep(c, p, deps)
        return True
    elif a not in deps[b] and c not in deps[b] and d not in deps[b] and state[b]:
        constructions[b].append(('on_tline', b, a, c, d))
        if len(constructions[b]) == 2:
            state[b] = False
        for p in [a, c, d]:
            update_dep(b, p, deps)
        return True
    elif b not in deps[a] and c not in deps[a] and d not in deps[a] and state[a]:
        constructions[a].append(('on_tline', a, b, c, d))
        if len(constructions[a]) == 2:
            state[a] = False
        for p in [b, c, d]:
            update_dep(a, p, deps)
        return True
    return False

def translate_cong(args, deps, constructions, state):
    if len(args) == 4:
        a, b, c, d = args
        if a not in deps[d] and b not in deps[d] and c not in deps[d] and len(constructions[d]) < 2:
            constructions[d].append(('eqdistance', d, c, a, b))
            if len(constructions[d]) == 2:
                state[d] = False
            for p in [a, b, c]:
                update_dep(d, p, deps)
            return True
        elif a not in deps[c] and b not in deps[c] and d not in deps[c] and len(constructions[c]) < 2:
            constructions[c].append(('eqdistance', c, d, a, b))
            if len(constructions[c]) == 2:
                state[c] = False
            for p in [a, b, d]:
                update_dep(c, p, deps)
            return True
        elif a not in deps[b] and c not in deps[b] and d not in deps[b] and len(constructions[b]) < 2:
            constructions[b].append(('eqdistance', b, a, c, d))
            if len(constructions[b]) == 2:
                state[b] = False
            for p in [a, c, d]:
                update_dep(b, p, deps)
            return True
        elif b not in deps[a] and c not in deps[a] and d not in deps[a] and len(constructions[a]) < 2:
            constructions[a].append(('eqdistance', a, b, c, d))
            if len(constructions[a]) == 2:
                state[a] = False
            for p in [b, c, d]:
                update_dep(a, p, deps)
            return True
    else:
        # TODO: Not Implemented
        return False

def translate_cyclic(args, deps, constructions, state):
    count = 0
    for point in reversed(args):
        if not state[point]:
            count += 1
            if count > 3:
                return False
            continue
        cycles = []
        for p in args:
            if p != point and p not in deps[point]:
                cycles.append(p)
                if len(cycles) == 3:
                    break
        if len(cycles) != 3:
            count += 1
            if count > 3:
                return False
            continue
        a, b, c = cycles
        constructions[point].append(('on_circum', point, a, b, c))
        if len(constructions[point]) == 2:
            state[point] = False
        for p in [a, b, c]:
            update_dep(point, p, deps)
    return True

def translate_eqangle(args, deps, constructions, state):
    if len(args) != 8:
        return False
    a, b, c, d, e, f, g, h = args
    if a not in deps[e] and b not in deps[e] and c not in deps[e] and d not in deps[e] and f not in deps[e] and g not in deps[e] and h not in deps[e] and state[e]:
        constructions[e].append(('on_aline0', e, a, b, c, d, f, g, h))
        if len(constructions[e]) == 2:
            state[e] = False
        for p in [a, b, c, d, f, g, h]:
            update_dep(e, p, deps)
        return True
    elif a not in deps[f] and b not in deps[f] and c not in deps[f] and d not in deps[f] and e not in deps[f] and g not in deps[f] and h not in deps[f] and state[f]:
        constructions[f].append(('on_aline0', f, a, b, c, d, e, g, h))
        if len(constructions[f]) == 2:
            state[f] = False
        for p in [a, b, c, d, e, g, h]:
            update_dep(f, p, deps)
        return True
    elif a not in deps[g] and b not in deps[g] and c not in deps[g] and d not in deps[g] and e not in deps[g] and f not in deps[g] and h not in deps[g] and state[g]:
        constructions[g].append(('on_aline0', g, c, d, a, b, h, e, f))
        if len(constructions[g]) == 2:
            state[g] = False
        for p in [a, b, c, d, e, f, h]:
            update_dep(g, p, deps)
    elif a not in deps[h] and b not in deps[h] and c not in deps[h] and d not in deps[h] and e not in deps[h] and f not in deps[h] and g not in deps[h] and state[h]:
        constructions[h].append(('on_aline0', h, c, d, a, b, g, e, f))
        if len(constructions[h]) == 2:
            state[h] = False
        for p in [a, b, c, d, e, f, g]:
            update_dep(h, p, deps)
        return True
    elif b not in deps[a] and c not in deps[a] and d not in deps[a] and e not in deps[a] and f not in deps[a] and g not in deps[a] and h not in deps[a] and state[a]:
        constructions[a].append(('on_aline0', a, e, f, g, h, b, c, d))
        if len(constructions[a]) == 2:
            state[a] = False
        for p in [b, c, d, e, f, g, h]:
            update_dep(a, p, deps)
        return True
    elif a not in deps[b] and c not in deps[b] and d not in deps[b] and e not in deps[b] and f not in deps[b] and g not in deps[b] and h not in deps[b] and state[b]:
        constructions[b].append(('on_aline0', b, e, f, g, h, a, c, d))
        if len(constructions[b]) == 2:
            state[b] = False
        for p in [a, c, d, e, f, g, h]:
            update_dep(b, p, deps)
        return True
    elif a not in deps[c] and b not in deps[c] and d not in deps[c] and e not in deps[c] and f not in deps[c] and g not in deps[c] and h not in deps[c] and state[c]:
        constructions[c].append(('on_aline0', c, g, h, e, f, d, a, b))
        if len(constructions[c]) == 2:
            state[c] = False
        for p in [a, b, d, e, f, g, h]:
            update_dep(c, p, deps)
        return True
    elif a not in deps[d] and b not in deps[d] and c not in deps[d] and e not in deps[d] and f not in deps[d] and g not in deps[d] and h not in deps[d] and state[d]:
        constructions[d].append(('on_aline0', d, g, h, e, f, c, a, b))
        if len(constructions[d]) == 2:
            state[d] = False
        for p in [a, b, c, e, f, g, h]:
            update_dep(d, p, deps)
        return True
    return False

def translate_eqratio(args, deps, constructions, state):
    if len(args) != 8:
        return False
    a, b, c, d, e, f, g, h = args
    if a not in deps[e] and b not in deps[e] and c not in deps[e] and d not in deps[e] and f not in deps[e] and g not in deps[e] and h not in deps[e] and state[e]:
        constructions[e].append(('eqratio', e, a, b, c, d, f, g, h))
        if len(constructions[e]) == 2:
            state[e] = False
        for p in [a, b, c, d, f, g, h]:
            update_dep(e, p, deps)
        return True
    elif a not in deps[f] and b not in deps[f] and c not in deps[f] and d not in deps[f] and e not in deps[f] and g not in deps[f] and h not in deps[f] and state[f]:
        constructions[f].append(('eqratio', f, a, b, c, d, e, g, h))
        if len(constructions[f]) == 2:
            state[f] = False
        for p in [a, b, c, d, e, g, h]:
            update_dep(f, p, deps)
        return True
    elif a not in deps[g] and b not in deps[g] and c not in deps[g] and d not in deps[g] and e not in deps[g] and f not in deps[g] and h not in deps[g] and state[g]:
        constructions[g].append(('eqratio', g, c, d, a, b, h, e, f))
        if len(constructions[g]) == 2:
            state[g] = False
        for p in [a, b, c, d, e, f, h]:
            update_dep(g, p, deps)
    elif a not in deps[h] and b not in deps[h] and c not in deps[h] and d not in deps[h] and e not in deps[h] and f not in deps[h] and g not in deps[h] and state[h]:
        constructions[h].append(('eqratio', h, c, d, a, b, g, e, f))
        if len(constructions[h]) == 2:
            state[h] = False
        for p in [a, b, c, d, e, f, g]:
            update_dep(h, p, deps)
        return True
    elif b not in deps[a] and c not in deps[a] and d not in deps[a] and e not in deps[a] and f not in deps[a] and g not in deps[a] and h not in deps[a] and state[a]:
        constructions[a].append(('eqratio', a, e, f, g, h, b, c, d))
        if len(constructions[a]) == 2:
            state[a] = False
        for p in [b, c, d, e, f, g, h]:
            update_dep(a, p, deps)
        return True
    elif a not in deps[b] and c not in deps[b] and d not in deps[b] and e not in deps[b] and f not in deps[b] and g not in deps[b] and h not in deps[b] and state[b]:
        constructions[b].append(('eqratio', b, e, f, g, h, a, c, d))
        if len(constructions[b]) == 2:
            state[b] = False
        for p in [a, c, d, e, f, g, h]:
            update_dep(b, p, deps)
        return True
    elif a not in deps[c] and b not in deps[c] and d not in deps[c] and e not in deps[c] and f not in deps[c] and g not in deps[c] and h not in deps[c] and state[c]:
        constructions[c].append(('eqratio', c, g, h, e, f, d, a, b))
        if len(constructions[c]) == 2:
            state[c] = False
        for p in [a, b, d, e, f, g, h]:
            update_dep(c, p, deps)
        return True
    elif a not in deps[d] and b not in deps[d] and c not in deps[d] and e not in deps[d] and f not in deps[d] and g not in deps[d] and h not in deps[d] and state[d]:
        constructions[d].append(('eqratio', d, g, h, e, f, c, a, b))
        if len(constructions[d]) == 2:
            state[d] = False
        for p in [a, b, c, e, f, g, h]:
            update_dep(d, p, deps)
        return True
    return False

def translate_midp(args, deps, constructions, state):
    a, b, c = args
    if b not in deps[a] and c not in deps[a] and len(constructions[a]) == 0:
        constructions[a].append(('midpoint', a, b, c))
        state[a] = False
        for p in [b, c]:
            update_dep(a, p, deps)
        return True
    elif a not in deps[b] and c not in deps[b] and len(constructions[b]) == 0:
        constructions[b].append(('online', b, a, c))
        constructions[b].append(('eqdistance', b, a, a, c))
        state[b] = False
        for p in [a, c]:
            update_dep(b, p, deps)
        return True
    elif a not in deps[c] and b not in deps[c] and len(constructions[c]) == 0:
        constructions[c].append(('online', c, a, b))
        constructions[c].append(('eqdistance', c, a, a, b))
        state[c] = False
        for p in [a, b]:
            update_dep(c, p, deps)
        return True
    return False

def translate_para(args, deps, constructions, state):
    a, b, c, d = args
    if a not in deps[d] and b not in deps[d] and c not in deps[d] and state[d]:
        constructions[d].append(('on_pline', d, c, a, b))
        if len(constructions[d]) == 2:
            state[d] = False
        for p in [a, b, c]:
            update_dep(d, p, deps)
        return True
    elif a not in deps[c] and b not in deps[c] and d not in deps[c] and state[c]:
        constructions[c].append(('on_pline', c, d, a, b))
        if len(constructions[c]) == 2:
            state[c] = False
        for p in [a, b, d]:
            update_dep(c, p, deps)
        return True
    elif a not in deps[b] and c not in deps[b] and d not in deps[b] and state[b]:
        constructions[b].append(('on_pline', b, a, c, d))
        if len(constructions[b]) == 2:
            state[b] = False
        for p in [a, c, d]:
            update_dep(b, p, deps)
        return True
    elif b not in deps[a] and c not in deps[a] and d not in deps[a] and state[a]:
        constructions[a].append(('on_pline', a, b, c, d))
        if len(constructions[a]) == 2:
            state[a] = False
        for p in [b, c, d]:
            update_dep(a, p, deps)
        return True
    return False

def translate_coll(args, deps, constructions, state):
    a, b, c = args
    if a not in deps[c] and b not in deps[c] and state[c]:
        constructions[c].append(('on_line', c, a, b))
        if len(constructions[c]) == 2:
            state[c] = False
        for p in [a, b]:
            update_dep(c, p, deps)
        return True
    elif a not in deps[b] and c not in deps[b] and state[b]:
        constructions[b].append(('on_line', b, a, c))
        if len(constructions[b]) == 2:
            state[b] = False
        for p in [a, c]:
            update_dep(b, p, deps)
        return True
    elif b not in deps[a] and c not in deps[a] and state[a]:
        constructions[a].append(('on_line', a, b, c))
        if len(constructions[a]) == 2:
            state[a] = False
        for p in [b, c]:
            update_dep(a, p, deps)
        return True
    return False

def translate_circle(args, deps, constructions, state):
    a, b, c, d = args
    if b not in deps[a] and c not in deps[a] and d not in deps[a] and len(constructions[a]) == 0:
        constructions[a].append(('circle', a, b, c, d))
        state[a] = False
        for p in [b, c, d]:
            update_dep(a, p, deps)
        return True 
    elif a not in deps[c] and b not in deps[c] and a not in deps[d] and b not in deps[d] and state[c] and state[d]:
        constructions[c].append(('on_circle', c, a, b))
        constructions[d].append(('on_circle', d, a, b))
        if len(constructions[c]) == 2:
            state[c] = False
        if len(constructions[d]) == 2:
            state[d] = False
        for p in [a, b]:
            update_dep(c, p, deps)
            update_dep(d, p, deps)
        return True
    return False

def translate_premise(premise, deps, constructions, state):
    type = premise[0]
    args = premise[1:]
    if type in ['simtri', 'simtrir', 'PythagoreanPremises']:
        return False
    if type in ['sameclock', 'ncoll', 'npara', 'sameside', 'nsameside']:
        return True
    if type == 'perp':
        return translate_perp(args, deps, constructions, state)
    if type == 'cong':
        return translate_cong(args, deps, constructions, state)
    if type == 'cyclic':
        return translate_cyclic(args, deps, constructions, state)
    if type == "eqangle":
        return translate_eqangle(args, deps, constructions, state)
    if type == "eqratio":
        return translate_eqratio(args, deps, constructions, state)
    if type == "midp":
        return translate_midp(args, deps, constructions, state)
    if type == "para":
        return translate_para(args, deps, constructions, state)
    if type == "coll":
        return translate_coll(args, deps, constructions, state)
    if type == "circle":
        return translate_circle(args, deps, constructions, state)
    print(type)            
    return False

def shuffle_premise(premise):
    type = premise[0]
    args = list(premise[1:])
    if type in ['sameclock', 'ncoll', 'npara', 'sameside', 'nsameside']:
        return premise
    elif type in ['coll', 'cyclic']:
        random.shuffle(args)
        return (type, *args)
    elif type in ['cong', 'para', 'perp']:
        grouped_list = [args[i:i+2] for i in range(0, len(args), 2)]
        for group in grouped_list:
            random.shuffle(group)
        random.shuffle(grouped_list)
        flattened_list = [item for group in grouped_list for item in group]
        return (type, *flattened_list)
    elif type in ['midp', 'circle']:
        points = args[1:]
        random.shuffle(points)
        return (type, args[0], *points)
    elif type in ['eqangle', 'eqratio']:
        return premise
    # For any other predicate types, keep as-is (no shuffling)
    return premise

def topological_sort(elements, deps):
    # 1. 构建入度表和图的邻接表
    in_degree = defaultdict(int)
    graph = defaultdict(list)

    # 初始化图和入度
    for elem in elements:
        if elem not in deps:
            deps[elem] = []
        for dep in deps[elem]:
            graph[dep].append(elem)  # dep -> elem
            in_degree[elem] += 1
    
    # 2. 找到入度为0的元素
    queue = deque([elem for elem in elements if in_degree[elem] == 0])
    result = []

    # 3. 拓扑排序
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # 对所有相邻节点进行处理
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 如果拓扑排序的结果包含了所有元素，说明排序成功
    if len(result) == len(elements):
        return result
    else:
        # 如果排序结果的长度不等于元素数量，说明图中有环
        return None

def dict2str(points, deps, constructions):
    # 得到拓扑顺序
    sorted_points = topological_sort(points, deps)
    if not sorted_points:
        return None
    sorted_points.reverse()
    res = ""
    for p in sorted_points:
        if p in constructions:
            if len(constructions[p]) == 0:
                res += p + " = free " + p + "; "
                continue
            res += p + " = "
            for c in constructions[p]:
                res += " ".join(c) + ", "
            res = res.strip().rstrip(',')
            res += "; "
    return res.lower().strip().rstrip(';')

# 处理规则文件并拆分规则
def process_rules(input_file, output_file, max_attempts = 10):
    # 打开输入文件并读取内容
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 用于存储结果
    rules_dict = {}

    # 遍历文件中的每一对标题和规则
    for i in range(0, len(lines), 2):  # 假设标题和规则交替出现
        title = lines[i].strip()
        rule = lines[i+1].strip()

        fail_reason = ''
        premise_tuples = []
        goal_tuples = []
        points = []
        deps = {}
        constructions = {}
        problems = []
        success = False

        try:
            # 拆分规则
            premise_tuples, goal_tuples = split_rule(rule)

            points = []
            for premise_tuple in premise_tuples:
                for point in premise_tuple[1:]:
                    points.append(point)
            points = sorted(list(set(points)))

            # 归一化前提（展开 simtri 等）
            tuples_new = []
            for premise_tuple in premise_tuples:
                if premise_tuple[0] == 'simtri':
                    a, b, c, p, q, r = premise_tuple[1:]
                    tuples_new.append(('eqangle', b, a, b, c, q, p, q, r))
                    tuples_new.append(('eqangle', c, a, c, b, r, p, r, q))
                elif premise_tuple[0] == 'simtrir':
                    a, b, c, p, q, r = premise_tuple[1:]
                    tuples_new.append(('eqangle', b, a, b, c, q, p, q, r))
                    tuples_new.append(('eqangle', c, a, c, b, r, q, r, p))
                elif premise_tuple[0] == 'PythagoreanPremises':
                    a, b, c = premise_tuple[1:]
                    tuples_new.append(('perp', a, b, a, c))
                else:
                    tuples_new.append(premise_tuple)
            premise_tuples = tuples_new

            last_fail_on = None

            for _ in range(max_attempts):
                # 打乱以增加成功概率
                for j, premise in enumerate(premise_tuples):
                    premise_tuples[j] = shuffle_premise(premise)

                constructions = {p: [] for p in points}
                state = {p: True for p in points}

                # 点之间的依赖，如果点A在deps[B]中则认为A依赖于B
                deps = {p: [] for p in points}

                success = True
                random.shuffle(premise_tuples)
                for premise_tuple in premise_tuples:
                    if not translate_premise(premise_tuple, deps, constructions, state):
                        success = False
                        last_fail_on = premise_tuple
                        break

                if success:
                    break

            if not success and last_fail_on is not None:
                fail_reason = f"cannot_translate:{last_fail_on[0]}"

            premise = dict2str(points, deps, constructions)
            if not premise:
                success = False
                if not fail_reason:
                    fail_reason = 'toposort_failed_or_no_premise'
            else:
                problems = []
                for goal in goal_tuples:
                    problem = premise + " ? " + " ".join(goal)
                    problems.append(problem.lower())

        except Exception as e:  # 捕获任意异常并继续写出失败原因
            success = False
            fail_reason = f"exception:{type(e).__name__}: {e}"

        # 存储拆分结果，按标题查找
        rules_dict[title] = {
            'rule_raw': rule,
            'premise': premise_tuples,
            'goal': goal_tuples,
            'points': points,
            'deps': deps,
            'constructions': constructions,
            'problem': problems,
            'success': success,
            'fail_reason': fail_reason,
        }

    # 写入拆分后的结果到输出文件
    with open(output_file, 'w', encoding='utf-8') as output:
        for title, rule_data in rules_dict.items():
            output.write(f"Title: {title}\n")
            # 输出被翻译的 schema 原文（按输入文件中的原始规则行）
            raw = rule_data.get('rule_raw', '')
            if raw:
                output.write(f"RuleRaw: {raw}\n")
            output.write(f"Premise: {rule_data['premise']}\n")
            output.write(f"Goal: {rule_data['goal']}\n")
            output.write(f"Points: {rule_data['points']}\n")
            if not rule_data['success']:
                # 兼容旧解析：保留原行；同时追加失败原因
                output.write("Translate Fail.\n")
                fr = rule_data.get('fail_reason')
                if fr:
                    output.write(f"Translate Fail Reason: {fr}\n")
                output.write("\n")
                continue
            # for point, dep in rule_data['deps'].items():
            #     output.write(f"{point}: {dep}\n")
            # constructions = rule_data['constructions']
            # for point, construction in constructions.items():
            #     if len(construction) == 0:
            #         output.write(f"{point}: free\n")
            #     else:
            #         output.write(f"{point}: {construction}\n")
            for problem in rule_data['problem']:
                output.write(f"Problem: {problem}\n")
            output.write("\n")

def _parse_args():
    parser = argparse.ArgumentParser(description='Translate rules file (two-line pairs) into Newclid problems list.')
    parser.add_argument('--input', type=str, required=False,
                        default='/c23474/home/math/dzt/Newclid/src/newclid/data_discovery/data/sch_rules.txt')
    parser.add_argument('--output', type=str, required=False,
                        default='/c23474/home/math/dzt/Newclid/src/newclid/data_discovery/data/sch_split_test.txt')
    parser.add_argument('--max-attempts', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    process_rules(args.input, args.output, args.max_attempts)

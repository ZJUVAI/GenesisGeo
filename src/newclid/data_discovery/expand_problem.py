#!/usr/bin/env python3
"""
expand_problem.py

精简版流水：
- Step1: 从 r07.jsonl 提取含 r07 的 <proof> 步骤到 r07_tmp.txt。
- Step2: 依据 r07_problems.txt 抽取点名并生成新增点到 r07_expand_points.txt。
- Step3: 解析四索引，生成辅助构造到 r07_aux_premises.txt（不再产生日志/汇总）。
- Step4: 将辅助构造插入题面末尾，输出 r07_expanded_problems.txt。
"""

import os
import shutil
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict

HERE = Path(__file__).resolve().parent
IN_PATH = HERE / 'r07.jsonl'
OUT_TMP_PATH = HERE / 'r07_tmp.txt'
PROBLEMS_PATH = HERE / 'r07_problems.txt'
OUT_POINTS_PATH = HERE / 'r07_expand_points.txt'
OUT_AUX_PATH = HERE / 'r07_aux_premises.txt'
OUT_EXPANDED_PROBLEMS_PATH = HERE / 'r07_expanded_problems.txt'

R07_PATTERN = re.compile(r"\br07\b", re.IGNORECASE)
POINT_TOKEN_RE = re.compile(r"\b([a-z])(\d*)\b")  # 单字母后接可选数字
R07_INDEXES_RE = re.compile(r"r07\s*\[(\d+)\]\s*\[(\d+)\]\s*\[(\d+)\]\s*\[(\d+)\]", re.IGNORECASE)


def extract_r07_steps_from_proof(llm_output: str) -> list[str]:
    """从 llm_output 文本中截取 <proof>...</proof>，并返回包含 r07 的步骤列表。"""
    if not llm_output:
        return []
    # 定位 <proof> 片段
    start = llm_output.find('<proof>')
    end = llm_output.find('</proof>')
    if start == -1 or end == -1 or end <= start:
        return []
    proof_body = llm_output[start + len('<proof>'):end]
    # 按分号切分步骤
    steps = [s.strip() for s in proof_body.split(';')]
    # 过滤包含 r07 的步骤（完整单词匹配）
    r07_steps = [s for s in steps if s and R07_PATTERN.search(s)]
    # 追加分号以保持原格式风格
    r07_steps = [s + ' ;' for s in r07_steps]
    return r07_steps


def build_filtered_index_map_from_jsonl(jsonl_path: Path) -> List[int]:
    """返回包含 r07 的题目的原始索引列表（1-based），按出现顺序。"""
    mapping: List[int] = []
    idx = 0
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            idx += 1
            r07_steps = extract_r07_steps_from_proof(obj.get('llm_output', ''))
            if r07_steps:
                mapping.append(idx)
    return mapping


# ---------- 名称序列与解析 ----------

def point_name_key(name: str) -> Tuple[int, int]:
    """将点名映射为排序键：(suffix_index, letter_index)。suffix 为 -1 表示无数字后缀。"""
    m = POINT_TOKEN_RE.fullmatch(name)
    if not m:
        return (10**9, 10**9)  # 非法名放到极后
    letter = m.group(1)
    digits = m.group(2)
    letter_idx = ord(letter) - ord('a')  # 0..25
    suffix = -1 if digits == '' else int(digits)
    return (suffix, letter_idx)


def next_point_name(name: str) -> str:
    """根据序列 a..z, a0..z0, a1..z1 ... 生成 name 的下一个。"""
    m = POINT_TOKEN_RE.fullmatch(name)
    if not m:
        # 若无法解析，则从 'a' 开始
        return 'a'
    letter = m.group(1)
    digits = m.group(2)
    letter_idx = ord(letter) - ord('a')
    suffix = -1 if digits == '' else int(digits)
    if suffix == -1:
        # a..y -> 下一字母；z -> a0
        if letter_idx < 25:
            return chr(ord('a') + letter_idx + 1)
        else:
            return 'a0'
    else:
        # a0..y0..z0 -> a1 开始；常规进位
        if letter_idx < 25:
            return chr(ord('a') + letter_idx + 1) + str(suffix)
        else:
            return 'a' + str(suffix + 1)


def generate_next_names_after(last_name: str, k: int) -> List[str]:
    names: List[str] = []
    cur = last_name
    for _ in range(k):
        cur = next_point_name(cur)
        names.append(cur)
    return names


def parse_points_from_problem(problem_text: str) -> List[str]:
    """从 fl_problem 文本中提取点名（单字母+可选数字），并返回按定义顺序排序后的列表。"""
    tokens = [m.group(0) for m in POINT_TOKEN_RE.finditer(problem_text)]
    # 仅保留合法点名（单字母+数字），剔除重复
    uniq = sorted(set(tokens), key=point_name_key)
    return uniq


# ---------- 文件读写帮助 ----------

def read_two_line_file(path: Path) -> List[Tuple[str, str]]:
    """读取两行一题的文件，返回[(id, content), ...]。"""
    items: List[Tuple[str, str]] = []
    if not path.exists():
        return items
    with path.open('r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            items.append((lines[i].strip(), lines[i + 1]))
    return items


def write_two_line_file(path: Path, items: List[Tuple[str, str]]):
    with path.open('w', encoding='utf-8') as f:
        for pid, content in items:
            f.write(f"{pid}\n")
            f.write(content.rstrip('\n') + "\n")


# ---------- JSONL 解析工具 ----------

_JSONL_CACHE: List[Dict] | None = None


def load_jsonl_objects() -> List[Dict]:
    global _JSONL_CACHE
    if _JSONL_CACHE is None:
        _JSONL_CACHE = []
        if IN_PATH.exists():
            with IN_PATH.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _JSONL_CACHE.append(json.loads(line))
                    except Exception:
                        continue
    return _JSONL_CACHE or []


def parse_indexed_predicates(text: str) -> Dict[int, Tuple[str, List[str]]]:
    """将文本中形如 '<pred> <args...> [NNN]' 的子句解析到 {NNN: (pred, [args])}。
    进一步修正：处理多词段头，如 'g h :'，避免把 'g' 误当谓词。

    做法：按分号 ';' 分段；每段若含有 ':'，且其前缀不含 '['，则剔除到第一个 ':' 为止，再在段内提取
    多个 'pred args [IDX]'。谓词允许数字后缀（如 eqratio3）。无参数的如 'a00 [003]' 跳过。
    """
    idx_map: Dict[int, Tuple[str, List[str]]] = {}
    if not text:
        return idx_map

    # 统一空白，分段处理
    segments = [seg.strip() for seg in text.replace('\n', ' ').split(';')]
    # 段内匹配：pred + 空格 + args(不跨越 '[' 或分号) + [NNN]
    pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+([^\[;]*?)\s*\[(\d+)\]")

    for seg in segments:
        if not seg:
            continue
        # 去掉段头 '... :'（可能是 'd :' 或 'g h :'），仅处理第一个冒号；
        # 但若冒号前已出现 '['，则不视为段头。
        if ':' in seg:
            pre, post = seg.split(':', 1)
            if '[' not in pre:
                seg = post
        # 段内提取多个带索引子句
        for m in pattern.finditer(seg):
            pred = m.group(1).lower()
            if pred in {"problem", "numerical_check", "proof"}:
                continue
            args_str = m.group(2).strip()
            idx = int(m.group(3))
            args = [p.strip() for p in re.split(r"[\s,]+", args_str) if p.strip()]
            if not args:
                continue
            idx_map[idx] = (pred, args)
    return idx_map


def build_idx_to_predicates_for_sample(sample: Dict) -> Dict[int, Tuple[str, List[str]]]:
    """合并解析 llm_input 与 llm_output（包括 <problem> / <proof> / <numerical_check>）中的索引子句。"""
    idx_map: Dict[int, Tuple[str, List[str]]] = {}
    for key in ('llm_input', 'llm_output'):
        text = sample.get(key, '') or ''
        if not text:
            continue
        # 直接解析整个块，利用 [NNN] 提取
        parsed = parse_indexed_predicates(text)
        # 后面的覆盖前面的（一般无冲突）
        idx_map.update(parsed)
    return idx_map


# ---------- 步骤实现 ----------

def step1_build_r07_tmp():
    if not IN_PATH.exists():
        print(f"未找到输入文件: {IN_PATH}")
        return 0
    kept = 0
    total = 0
    with IN_PATH.open('r', encoding='utf-8') as fin, OUT_TMP_PATH.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            total += 1
            r07_steps = extract_r07_steps_from_proof(obj.get('llm_output', ''))
            if not r07_steps:
                continue
            kept += 1
            fout.write(f"{kept:04d}\n")
            fout.write(' '.join(r07_steps).strip() + "\n")
    print(f"Step1: 扫描完成：共 {total} 条，包含 r07 的 {kept} 条。输出: {OUT_TMP_PATH}")
    return kept


def step2_build_expand_points():
    # 读取基础问题与 tmp
    problems = read_two_line_file(PROBLEMS_PATH)
    tmp_items = read_two_line_file(OUT_TMP_PATH)
    if not problems:
        print(f"Step2: 未找到或无法读取 {PROBLEMS_PATH}")
        return
    if not tmp_items:
        print(f"Step2: 未找到或无法读取 {OUT_TMP_PATH}")
        return

    # 建立 r07 过滤映射：tmp 的第 k 条对应原始第 mapping[k-1] 条
    mapping = build_filtered_index_map_from_jsonl(IN_PATH)
    if len(mapping) < len(tmp_items):
        print("Step2 警告: 过滤映射长度小于 tmp 数量，可能存在不一致。将按可用长度对齐。")
    max_n = min(len(tmp_items), len(mapping))

    out_items: List[Tuple[str, str]] = []

    for idx in range(max_n):
        tmp_id, tmp_proof_line = tmp_items[idx]
        orig_idx = mapping[idx]  # 1-based
        if orig_idx <= 0 or orig_idx > len(problems):
            continue
        orig_problem_id, fl_problem = problems[orig_idx - 1]

        # 统计 r07 出现次数
        need_k = len(R07_PATTERN.findall(tmp_proof_line))
        if need_k <= 0:
            continue

        # 解析现有点名，找到“最后一个”
        pts = parse_points_from_problem(fl_problem)
        last_name = pts[-1] if pts else 'z'  # 若未解析到点，默认从 a 开始生成（z 的下一个是 a0，这里更稳妥避免与常见 a 冲突）
        new_pts = generate_next_names_after(last_name, need_k)

        out_items.append((tmp_id, ' '.join(new_pts)))

    write_two_line_file(OUT_POINTS_PATH, out_items)
    print(f"Step2: 生成完成：{len(out_items)} 条。输出: {OUT_POINTS_PATH}")


# ---------- Step 3: 生成辅助构造 ----------

def parse_r07_indices_list_from_tmp_line(tmp_line: str) -> List[Tuple[int, int, int, int]]:
    """从 tmp 的第二行文本中，按出现顺序提取每个 r07 的四索引。"""
    indices = []
    # 按分号分割，逐段匹配 r07 四索引
    for seg in tmp_line.split(';'):
        if 'r07' not in seg.lower():
            continue
        m = R07_INDEXES_RE.search(seg)
        if m:
            a, b, c, d = map(int, m.groups())
            indices.append((a, b, c, d))
    return indices


def resolve_mapping_for_r07(idx_tuple: Tuple[int, int, int, int], idx_map: Dict[int, Tuple[str, List[str]]], *, debug: Dict) -> Dict[str, str] | None:
    """基于四索引与索引->(pred,args) 映射，推断 A,B,C,D,O 的具体点名。
    返回 {'A','B','C','D','O'} 或 None。debug 中写入中间信息用于日志。
    """
    i1, i2, i3, i4 = idx_tuple
    preds = []
    for i in (i1, i2, i3, i4):
        if i not in idx_map:
            debug['missing_index'] = i
            return None
        preds.append((i, idx_map[i][0], idx_map[i][1]))  # (idx, pred, args)

    para_args = None
    ncoll_args = None
    coll_args_list: List[List[str]] = []

    for idx, pred, args in preds:
        p = pred.lower()
        if p == 'para' and len(args) >= 4:
            para_args = args[:4]
        elif p == 'ncoll' and len(args) >= 3:
            ncoll_args = args[:3]
        elif p == 'coll' and len(args) >= 3:
            coll_args_list.append(args[:3])

    debug['found'] = {
        'para': para_args,
        'ncoll': ncoll_args,
        'colls': coll_args_list,
    }

    if para_args is None:
        debug['reason'] = 'no_para_in_four_indices'
        return None
    if ncoll_args is None:
        debug['reason'] = 'no_ncoll_in_four_indices'
        return None
    if len(coll_args_list) < 2:
        debug['reason'] = 'need_two_colls_in_four_indices'
        return None

    A, B, C, D = para_args[0], para_args[1], para_args[2], para_args[3]

    # 通过两条 coll 的交集确定 O，若不唯一，则用启发式消歧
    s1, s2 = set(coll_args_list[0]), set(coll_args_list[1])
    inter = s1.intersection(s2)

    def try_o_candidate(o: str) -> Tuple[bool, str | None, str | None]:
        C_from_coll = None
        D_from_coll = None
        for lst in coll_args_list:
            s = set(lst)
            if o in s and A in s:
                third = [t for t in lst if t not in (o, A)]
                if third:
                    C_from_coll = third[0]
            if o in s and B in s:
                third = [t for t in lst if t not in (o, B)]
                if third:
                    D_from_coll = third[0]
        ok = (C_from_coll is not None) and (D_from_coll is not None)
        return ok, C_from_coll, D_from_coll

    O = None
    C_from_coll = None
    D_from_coll = None

    if len(inter) == 1:
        cand = next(iter(inter))
        ok, c1, d1 = try_o_candidate(cand)
        if ok:
            O, C_from_coll, D_from_coll = cand, c1, d1
        else:
            # 即使交集唯一，若不能导出 C/D，仍继续用备用策略
            pass

    if O is None:
        # 候选包括两条 coll 的所有点（优先交集中的点）
        candidates = list(inter) + [t for t in s1.union(s2) if t not in inter]
        # 若存在 ncoll，优先包含在 ncoll 里的候选
        if ncoll_args:
            nset = set(ncoll_args)
            candidates = sorted(candidates, key=lambda t: (t not in nset))
        # 依次尝试
        for cand in candidates:
            ok, c1, d1 = try_o_candidate(cand)
            if ok:
                O, C_from_coll, D_from_coll = cand, c1, d1
                break

    if O is None:
        debug['reason'] = 'cannot_determine_O'
        return None

    # 确定 C、D，优先 coll 推断，否则退回 para 的 C/D
    C_final = C_from_coll or C
    D_final = D_from_coll or D

    debug['mapping'] = {'A': A, 'B': B, 'C': C_final, 'D': D_final, 'O': O}
    return {'A': A, 'B': B, 'C': C_final, 'D': D_final, 'O': O}


def step3_build_aux_premises():
    # 读取必要文件
    tmp_items = read_two_line_file(OUT_TMP_PATH)
    pts_items = read_two_line_file(OUT_POINTS_PATH)
    if not tmp_items:
        print(f"Step3: 未找到或无法读取 {OUT_TMP_PATH}")
        return
    if not pts_items:
        print(f"Step3: 未找到或无法读取 {OUT_POINTS_PATH}")
        return

    # id -> [x1, x2, ...]
    id_to_newpoints: Dict[str, List[str]] = {pid: content.split() if content.strip() else [] for pid, content in pts_items}

    # 建立映射到原始样本索引
    mapping = build_filtered_index_map_from_jsonl(IN_PATH)
    jsonl_objs = load_jsonl_objects()

    out_items: List[Tuple[str, str]] = []

    summary_ok = 0
    summary_total = 0

    for local_idx, (pid, tmp_line) in enumerate(tmp_items, start=1):
        if pid not in id_to_newpoints:
            continue
        new_pts = id_to_newpoints[pid]
        if not new_pts:
            continue
        if local_idx - 1 >= len(mapping):
            continue
        orig_idx = mapping[local_idx - 1]  # 1-based
        if orig_idx <= 0 or orig_idx > len(jsonl_objs):
            continue
        sample = jsonl_objs[orig_idx - 1]
        idx_map = build_idx_to_predicates_for_sample(sample)

        # 从 tmp 提取每个 r07 的四索引
        r07_indices_list = parse_r07_indices_list_from_tmp_line(tmp_line)
        if not r07_indices_list:
            continue

        # 对齐数量（以较小者为准）
        n = min(len(new_pts), len(r07_indices_list))
        clauses: List[str] = []
        for i in range(n):
            xname = new_pts[i]
            idx_tuple = r07_indices_list[i]
            summary_total += 1
            debug: Dict = {}
            mapping_dict = resolve_mapping_for_r07(idx_tuple, idx_map, debug=debug)
            if not mapping_dict:
                continue
            A = mapping_dict['A']
            B = mapping_dict['B']
            C = mapping_dict['C']
            D = mapping_dict['D']
            clause = f"; {xname} = on_line {xname} {C} {D}, on_pline {xname} {B} {A} {C}"
            clauses.append(clause)
            summary_ok += 1
            # 记录成功计数（已不产生日志文件）

        if clauses:
            out_items.append((pid, ' '.join(clauses)))

    # 写出结果
    write_two_line_file(OUT_AUX_PATH, out_items)
    rate = (summary_ok / summary_total) if summary_total else 0.0
    print(f"Step3: 生成完成：{len(out_items)} 条。输出: {OUT_AUX_PATH}（ok/total={summary_ok}/{summary_total}, rate={rate:.2%}）")



##########################
# 附加：过滤工具函数
##########################

def filter_r07_jsonl_remove_multi(jsonl_path: Path = IN_PATH, backup_dir: Path = HERE / 'backups') -> dict:
    """就地过滤 r07.jsonl：删除 <proof> 中包含两个及以上 r07 的样本。
    - 会先将原文件备份到 backups/ 目录（时间戳后缀）。
    - 返回统计信息：{"before":N, "kept":K, "removed":R, "backup":path}。
    """
    stats = {"before": 0, "kept": 0, "removed": 0, "backup": None}
    if not jsonl_path.exists():
        return stats

    # 读取并计算
    lines = [ln.rstrip("\n") for ln in jsonl_path.read_text(encoding='utf-8').splitlines()]
    stats["before"] = len([ln for ln in lines if ln.strip()])
    kept_objs = []
    removed = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            # 无法解析的行，保留原样以免误删
            kept_objs.append(s)
            continue
        r07_steps = extract_r07_steps_from_proof(obj.get('llm_output', ''))
        if len(r07_steps) >= 2:
            removed += 1
            continue
        kept_objs.append(obj)

    stats["removed"] = removed
    stats["kept"] = len(kept_objs)

    # 备份
    try:
        backup_dir.mkdir(exist_ok=True)
    except Exception:
        pass
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_path = backup_dir / f"r07.jsonl.bak-{ts}"
    backup_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    stats["backup"] = str(backup_path)

    # 写回过滤后的文件
    with jsonl_path.open('w', encoding='utf-8') as f:
        for obj in kept_objs:
            if isinstance(obj, str):
                f.write(obj + "\n")
            else:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return stats


def tidy_process_data():
    """整理中间产物：将 r07_tmp.txt / r07_expand_points.txt 及历史日志/汇总移动到 logs/arch-<ts>/。
    不删除核心输出：r07_aux_premises.txt、r07_expanded_problems.txt、r07_problems.txt、r07.jsonl。
    """
    ts_dir = HERE / 'logs' / ('arch-' + __import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S'))
    ts_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        OUT_TMP_PATH,
        OUT_POINTS_PATH,
        HERE / 'r07_aux_premises.log',
        HERE / 'r07_aux_premises_summary.tsv',
    ]
    moved = []
    for p in candidates:
        try:
            if p.exists():
                shutil.move(str(p), str(ts_dir / p.name))
                moved.append(p.name)
        except Exception:
            pass
    # 不保留 r07_aux_premises.txt
    try:
        if OUT_AUX_PATH.exists():
            OUT_AUX_PATH.unlink()
            moved.append('(deleted) ' + OUT_AUX_PATH.name)
    except Exception:
        pass
    print(f"Clean: 移动中间文件到 {ts_dir}: {', '.join(moved) if moved else '无'}")


# ---------- Step 4: 将辅助构造注入题面，生成扩展题库 ----------

def _insert_aux_into_problem_text(problem: str, aux: str) -> str:
    """将辅助构造子句 aux 插入到题面 problem 的末尾：
    - 若存在 '?'（待证部分），则在第一个 '?' 之前插入；
    - 否则直接追加到末尾。
    要求 aux 已以分号开头，如 '; x = on_line ...'。
    """
    if not aux.strip():
        return problem
    aux_str = aux.strip()
    # 确保前后空格
    if '?' in problem:
        pos = problem.find('?')
        left = problem[:pos].rstrip()
        right = problem[pos:].lstrip()
        return f"{left} {aux_str} {right}"
    else:
        return f"{problem.rstrip()} {aux_str}"


def step4_build_expanded_problems():
    """读取 r07_problems.txt 与 r07_aux_premises.txt，按当前 r07.jsonl 的映射将辅助构造
    插入对应题目末尾（'?' 前），输出 r07_expanded_problems.txt。未命中的题保持原样。
    注意：这里使用 build_filtered_index_map_from_jsonl(IN_PATH) 作为 r07 子集到原题索引的映射，
    将 pid(0001..）映射为原题下标 orig_idx。
    """
    problems = read_two_line_file(PROBLEMS_PATH)
    if not problems:
        print(f"Step4: 未找到或无法读取 {PROBLEMS_PATH}")
        return

    aux_items = read_two_line_file(OUT_AUX_PATH)
    aux_by_pid: Dict[str, str] = {pid: content for pid, content in aux_items}

    mapping = build_filtered_index_map_from_jsonl(IN_PATH)  # [orig_idx1, orig_idx2, ...]
    updated = 0

    # 构建便于更新的列表副本
    probs_list = list(problems)

    # 遍历 r07 序列 pid -> 原题索引
    for i, orig_idx in enumerate(mapping, start=1):
        pid = f"{i:04d}"
        if pid not in aux_by_pid:
            continue
        if not (1 <= orig_idx <= len(probs_list)):
            continue
        old_id, old_text = probs_list[orig_idx - 1]
        # 插入辅助构造
        new_text = _insert_aux_into_problem_text(old_text, aux_by_pid[pid])
        probs_list[orig_idx - 1] = (old_id, new_text)
        updated += 1

    write_two_line_file(OUT_EXPANDED_PROBLEMS_PATH, probs_list)
    # 按要求不保留 r07_aux_premises.txt
    try:
        if OUT_AUX_PATH.exists():
            OUT_AUX_PATH.unlink()
    except Exception:
        pass
    print(f"Step4: 生成完成：共 {len(probs_list)} 题，更新 {updated} 题。输出: {OUT_EXPANDED_PROBLEMS_PATH}（已删除 r07_aux_premises.txt）")

def main():
    # 执行 Step1 ~ Step3
    step1_build_r07_tmp()
    step2_build_expand_points()
    step3_build_aux_premises()
    step4_build_expanded_problems()


if __name__ == '__main__':
    main()

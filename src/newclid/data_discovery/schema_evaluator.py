#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SchemaBatchEvaluator: 对一组规则执行 翻译->解析->求解，返回 results 列表。
一个输入（json）对应一个输出（<basename>.<kind>.results.json），并写出派生的中间文件（rules/split）。
"""
import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger('schema_eval')

# 准备路径，保持与 schema_eval 一致的相对定位
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
SCRIPTS_ROOT = os.path.join(REPO_ROOT, 'scripts')
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

try:
    from newclid import GeometricSolverBuilder, proof_writing
except Exception as e:  # pragma: no cover
    logger.warning('Evaluator: newclid solver not available: %s', e)
    GeometricSolverBuilder = None  # type: ignore
    proof_writing = None  # type: ignore


def _load_translator():
    try:
        import importlib.util
        mod_path = os.path.join(REPO_ROOT, 'scripts', 'translate_rule_to_problem.py')
        if not os.path.exists(mod_path):
            logger.warning('Translator script not found: %s', mod_path)
            return None
        spec = importlib.util.spec_from_file_location('translate_rule_to_problem', mod_path)
        if spec is None or spec.loader is None:
            logger.warning('Failed to create import spec for translator')
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return getattr(mod, 'process_rules', None)
    except Exception as e:  # pragma: no cover
        logger.warning('Could not load translator: %s', e)
        return None


def _write_two_line_rules(records: List[Tuple[str, str]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for title, rule in records:
            f.write(title + "\n")
            f.write(rule + "\n")


def _read_split_output(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    title = None
    success = False
    problems: List[str] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('Title: '):
                    if title is not None:
                        out[title] = {'success': success, 'problems': problems}
                    title = line[len('Title: '):]
                    success = True
                    problems = []
                elif line == 'Translate Fail.':
                    success = False
                elif line.startswith('Problem: '):
                    problems.append(line[len('Problem: '):])
            if title is not None:
                out[title] = {'success': success, 'problems': problems}
    except FileNotFoundError:
        logger.error('Split output not found: %s', path)
    return out


def _solve_problem(problem_txt: str, max_attempts: int) -> Tuple[bool, str]:
    if GeometricSolverBuilder is None or proof_writing is None:
        return False, 'solver_unavailable'
    try:
        builder = GeometricSolverBuilder(123)
        builder.load_problem_from_txt(problem_txt)
        solver = builder.build(max_attempts=max_attempts)
        ok = solver.run()
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            proof_writing.write_proof_steps(solver.proof)
        return ok, buf.getvalue()
    except Exception as e:
        return False, f'exception: {e}'


class SchemaBatchEvaluator:
    def __init__(self, out_dir: str, max_attempts: int = 100, topn_print: int = 10) -> None:
        self.out_dir = out_dir
        self.max_attempts = max_attempts
        self.topn_print = topn_print
        self._translate_rules = _load_translator()

    # ---------- helpers moved from script ----------
    @staticmethod
    def _norm_rule_str(s: str) -> str:
        if '=>' not in s:
            return s.strip()
        left, right = s.split('=>', 1)
        left = left.replace('∧', ',')
        right = right.replace('∧', ',')
        left = ' '.join(left.split())
        right = ' '.join(right.split())
        return f"{left.strip()} => {right.strip()}"

    @staticmethod
    def _write_json(path: str, obj: Any) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # 分组/合并输出相关逻辑已移除，保持最简职责

    # ---------- records loading ----------
    def load_records_from_branched(self, json_path: str, kind: str) -> List[Tuple[str, str]]:
        """Load patterns from branched JSON and build (title, rule) records for given kind.
        kind: 'schema' or 'schema_before'.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        patterns: List[Dict[str, Any]] = data.get('patterns', data.get('patterns_summary_topN', []))
        if not patterns:
            patterns = data.get('top_k', [])
        records: List[Tuple[str, str]] = []
        for idx, p in enumerate(patterns):
            pid = p.get('pattern_id', idx)
            s_schema = p.get('schema')
            s_before = p.get('schema_before_dependency')
            if kind == 'schema' and s_schema:
                rule = self._norm_rule_str(s_schema)
                title = f'pat_{pid:04d}_schema'
                records.append((title, rule))
            if kind == 'schema_before' and s_before:
                rule = self._norm_rule_str(s_before)
                title = f'pat_{pid:04d}_schema_before'
                records.append((title, rule))
        return records

    def run(self, records: List[Tuple[str, str]], kind: str, rules_filename: str, split_filename: str) -> List[Dict[str, Any]]:
        os.makedirs(self.out_dir, exist_ok=True)
        out_rules = os.path.join(self.out_dir, rules_filename)
        out_split = os.path.join(self.out_dir, split_filename)

        # 写入两行规则文件
        _write_two_line_rules(records, out_rules)

        # 翻译
        translations: Dict[str, Any] = {}
        if self._translate_rules is None:
            logger.error('process_rules() not available; skipping translation step (%s).', kind)
        else:
            logger.info('Translating %s -> %s', kind, out_split)
            self._translate_rules(out_rules, out_split, self.max_attempts)
            translations = _read_split_output(out_split)

        # 求解
        logger.info('Evaluating problems with solver (max_attempts=%d) for %s ...', self.max_attempts, kind)
        results: List[Dict[str, Any]] = []
        for title, rule in records:
            t: Dict[str, Any] = {'title': title, 'kind': kind, 'rule': rule}
            trans = translations.get(title, {'success': False, 'problems': []})
            t['translate_success'] = bool(trans.get('success'))
            t['problems'] = []
            for prob in trans.get('problems', []):
                ok, proof = _solve_problem(prob, self.max_attempts)
                t['problems'].append({'problem': prob, 'solver_success': ok, 'proof': proof})
            results.append(t)
        return results

    # 高层封装：从 JSON 加载并处理某一类 kind，并将结果写为一个输出文件
    def process_kind(self, json_path: str, kind: str) -> List[Dict[str, Any]]:
                """从给定 json_path 抽取指定 kind 的规则，翻译+求解，并将结果写为一个输出文件。

                命名约定（一个输入对应一个输出）：
                    <basename>.<kind>.results.json  （例如 branched_mining.schema.results.json）

                为便于审计，内部中间文件（两行规则 / 拆分输出）也按输入名派生：
                    <basename>.<kind>.rules.txt, <basename>.<kind>.split.txt
                """
                assert kind in ('schema', 'schema_before')
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                records = self.load_records_from_branched(json_path, kind)
                rules_file = f'{base_name}.{kind}.rules.txt'
                split_file = f'{base_name}.{kind}.split.txt'
                results = self.run(records, kind, rules_filename=rules_file, split_filename=split_file)

                # 一个输入 -> 一个输出：仅写 results
                out_results = os.path.join(self.out_dir, f'{base_name}.{kind}.results.json')
                self._write_json(out_results, {'results': results})
                logger.info('[%s] saved -> %s (records=%d)', kind, out_results, len(results))
                return results



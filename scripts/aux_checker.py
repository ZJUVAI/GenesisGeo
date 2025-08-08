#!/usr/bin/env python3
"""
AuxChecker
功能：
1. validate-translation: 验证lm.py中translate函数是否适配aux.jsonl文件格式
2. validate-aux: 预处理aux数据并测试几何问题求解（无中间文件）
3. pipeline: 完整验证流水线
使用示例:
python scripts/aux_checker.py validate-translation input.jsonl
python scripts/aux_checker.py validate-aux input.jsonl
python scripts/aux_checker.py pipeline input.jsonl
"""
import json
import re
import sys
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
from newclid.configs import default_defs_path
from newclid.formulations.definition import DefinitionJGEX
from newclid import GeometricSolverBuilder, GeometricSolver
from newclid.predicates import NAME_TO_PREDICATE
# 从lm.py导入LMAgent类，确保使用最新版本
from newclid.agent.lm import LMAgent

class AuxChecker:
    """AuxChecker - 综合aux验证工具"""
    def __init__(self, seed: int = 123, max_attempts: int = 100):
        """
        初始化AuxChecker
        Args:
            seed: 随机种子
            max_attempts: 求解器最大尝试次数
        """
        self.seed = seed
        self.max_attempts = max_attempts
        self.defs = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))
        # 创建LMAgent实例来调用translate函数
        # 使用空的模型路径，因为我们只需要调用translate方法
        self.lm_agent = LMAgent(model_path=[], decoding_size=1, beam_size=1, search_depth=1)
    def extract_aux_from_llm_output(self, llm_output: str) -> Optional[str]:
        """从LLM输出中提取aux部分的内容"""
        aux_pattern = r'<aux>\s*(.*?)\s*</aux>'
        match = re.search(aux_pattern, llm_output, re.DOTALL)
        if match:
            aux_content = match.group(1).strip()
            # 移除开头可能的编号，如 "x00 "
            aux_content = re.sub(r'^x\d+\s+', '', aux_content)
            return aux_content
        return None
    def extract_aux_points(self, llm_output: str) -> Set[str]:
        """从llm_output的<aux>标签中提取点名"""
        aux_match = re.search(r'<aux>(.*?)</aux>', llm_output, re.DOTALL)
        if not aux_match:
            return set()
        aux_contents = aux_match.group(1).strip().split(';')
        points = set()
        for aux_content in aux_contents:
            aux_content = aux_content.strip()
            if not aux_content:
                continue
            # 检查是否包含 "x00 点名 :" 的格式
            aux_points_match = re.match(r'x\d+\s+(.*?)\s*:\s*.*', aux_content)
            if aux_points_match:
                aux_points = aux_points_match.group(1).strip().split()
                points.update(aux_points)
        return points
    def remove_clauses_with_aux_points(self, fl_problem: str, aux_points: Set[str]) -> str:
        """从fl_problem中移除包含aux点的子句"""
        if not aux_points:
            return fl_problem
        # 分割为问题部分和查询部分
        if ' ? ' in fl_problem:
            problem_part, query_part = fl_problem.split(' ? ', 1)
        else:
            return fl_problem
        # 分割子句
        clauses = [clause.strip() for clause in problem_part.split(';') if clause.strip()]
        # 过滤掉包含aux点的子句
        filtered_clauses = []
        for clause in clauses:
            # 检查子句中是否包含aux点
            clause_points = set(re.findall(r'\b[a-z]\d*\b', clause))
            if not clause_points.intersection(aux_points):
                filtered_clauses.append(clause)
        # 重新组合
        result = '; '.join(filtered_clauses)
        if query_part:
            result += ' ? ' + query_part
        return result
    def validate_lm_translation(self, input_file: str) -> bool:
        """
        验证lm.py中translate函数对aux.jsonl格式的适配性
        合并了原retranslate和validate-lm功能
        """
        print(f"验证lm.py翻译函数适配性: {input_file}")
        inconsistencies_found = 0
        total_processed = 0
        # 准备输出文件
        output_file = Path(__file__).parent / "translation_validation_log.txt"
        log_entries = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        # 解析JSON行
                        data = json.loads(line)
                        # 获取llm_output_renamed和fl_problem字段
                        llm_output_renamed = data.get('llm_output_renamed', '')
                        fl_problem = data.get('fl_problem', '')
                        llm_output = data.get('llm_output', '')
                        if not llm_output_renamed:
                            continue
                        # 提取aux内容
                        aux_content = self.extract_aux_from_llm_output(llm_output_renamed)
                        if not aux_content:
                            continue
                        total_processed += 1
                        # 初始化变量用于错误日志记录
                        aux_points = None
                        aux_premises = None
                        preparsed_premises = []
                        aux_clauses = None
                        aux_constructions = []
                        retranslated_premises = []
                        try:
                            # 解析aux内容
                            aux_points, aux_premises = aux_content.split(';')[0].split(' : ')
                            aux_points = aux_points.strip().split()
                            # 目前只支持一个点（遵循alphageometry）
                            if len(aux_points) == 0 or len(aux_points) > 1:
                                continue
                            aux_points = aux_points[0]
                            # 解析premises
                            aux_premises = re.split(r"\s*\[\d+\]", aux_premises.strip())
                            preparsed_premises = []
                            for aux_premise in aux_premises:
                                aux_premise = aux_premise.strip()
                                if aux_premise == '':
                                    continue
                                aux_premise_parts = aux_premise.split(" ")
                                if aux_premise_parts[0] in NAME_TO_PREDICATE:
                                    pred = NAME_TO_PREDICATE[aux_premise_parts[0]]
                                    preparsed = pred.preparse(aux_premise_parts[1:])
                                    preparsed_aux_premise = aux_premise_parts[0] + ' ' + ' '.join(preparsed)
                                    preparsed_premises.append(preparsed_aux_premise)
                            # 使用lm.py的函数进行翻译
                            aux_clauses = self.lm_agent.try_dsl_to_constructions(aux_content)
                            if aux_clauses is None:
                                continue
                            aux_constructions = aux_clauses.split(" = ")[1].split(", ")
                            retranslated_premises = []
                            for con in aux_constructions:
                                con_items = con.split(' ')
                                if con_items[0] in self.defs:
                                    cdef = self.defs[con_items[0]]
                                    if len(con_items) == len(cdef.declare):
                                        mapping = dict(zip(cdef.declare[1:], con_items[1:]))
                                        for aux_points_def, bs in cdef.basics:
                                            for b in bs:
                                                premise = b[0] + ' ' + ' '.join([mapping[x] for x in b[1:]])
                                                if b[0] in NAME_TO_PREDICATE:
                                                    pred = NAME_TO_PREDICATE[b[0]]
                                                    preparsed = pred.preparse([mapping[x] for x in b[1:]])
                                                    preparsed_premise = b[0] + ' ' + ' '.join(preparsed)
                                                    retranslated_premises.append(preparsed_premise)
                            # 检查一致性：只有当原始predicates中有不被重译predicates包含的情况才认为是不一致
                            preparsed_set = set(preparsed_premises)
                            retranslated_set = set(retranslated_premises)
                            missing_in_retranslated = preparsed_set - retranslated_set
                            if missing_in_retranslated:
                                inconsistencies_found += 1
                                log_entry = f"\n--- 第{line_num}行发现翻译不一致 ---\n"
                                log_entry += f"原始: {', '.join(preparsed_premises)}\n"
                                log_entry += f"重译: {', '.join(retranslated_premises)}\n"
                                log_entry += f"构造: {aux_constructions}\n"
                                log_entry += f"缺失的predicates: {', '.join(missing_in_retranslated)}\n"
                                log_entries.append(log_entry)
                                print(log_entry)
                        except Exception as e:
                            inconsistencies_found += 1
                            # 构建详细的错误日志条目
                            error_entry = f"\n--- 第{line_num}行lm.py函数处理失败 ---\n"
                            error_entry += f"错误信息: {e}\n"
                            error_entry += f"fl_problem: {fl_problem}\n"
                            error_entry += f"llm_output: {llm_output}\n"
                            error_entry += f"原始aux内容: {aux_content}\n"
                            error_entry += f"解析得到的aux点: {aux_points}\n"
                            error_entry += f"原始premises: {', '.join(preparsed_premises)}\n"
                            error_entry += f"lm.py返回的aux_clauses: {aux_clauses}\n"
                            error_entry += f"解析得到的构造: {aux_constructions}\n"
                            error_entry += f"重译premises: {', '.join(retranslated_premises)}\n"
                            error_entry += f"原始llm_output_renamed:\n{llm_output_renamed}\n"
                            error_entry += "-" * 50 + "\n"
                            log_entries.append(error_entry)
                            print(f"第{line_num}行lm.py函数处理失败: {e}")
                    except json.JSONDecodeError as e:
                        print(f"第{line_num}行JSON解析错误: {e}")
                        continue
                    except Exception as e:
                        print(f"第{line_num}行处理错误: {e}")
                        continue
        except FileNotFoundError:
            print(f"错误: 找不到文件 '{input_file}'")
            return False
        except Exception as e:
            print(f"读取文件错误: {e}")
            return False
        # 写入日志文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"翻译验证日志 - {input_file}\n")
                f.write("=" * 60 + "\n\n")
                if log_entries:
                    f.writelines(log_entries)
                else:
                    f.write("未发现翻译不一致的情况\n")
                f.write(f"\n\n总结:\n")
                f.write(f"处理总数: {total_processed}\n")
                f.write(f"不一致数: {inconsistencies_found}\n")
                if total_processed > 0:
                    f.write(f"成功率: {(total_processed - inconsistencies_found) / total_processed:.2%}\n")
            print(f"详细日志已保存到: {output_file}")
        except Exception as e:
            print(f"保存日志文件失败: {e}")
        if total_processed > 0:
            success_rate = (total_processed - inconsistencies_found) / total_processed
            print(f"\n翻译验证完成:")
            print(f"  处理总数: {total_processed}")
            print(f"  不一致数: {inconsistencies_found}")
            print(f"  成功率: {success_rate:.2%}")
        else:
            print("未找到可处理的aux数据")
        return inconsistencies_found == 0
    def preprocess_jsonl_in_memory(self, input_file: str) -> List[str]:
        """
        在内存中预处理JSONL文件，返回处理后的问题列表
        不写入中间文件
        """
        processed_problems = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        fl_problem = data.get('fl_problem', '')
                        llm_output_renamed = data.get('llm_output_renamed', '')
                        if not fl_problem:
                            continue
                        # 提取aux点
                        aux_points = self.extract_aux_points(llm_output_renamed)
                        # 移除包含aux点的子句
                        filtered_problem = self.remove_clauses_with_aux_points(fl_problem, aux_points)
                        if filtered_problem.strip():
                            processed_problems.append(filtered_problem)
                    except json.JSONDecodeError as e:
                        print(f"第{line_num}行JSON解析错误: {e}")
                        continue
                    except Exception as e:
                        print(f"第{line_num}行预处理错误: {e}")
                        continue
        except FileNotFoundError:
            print(f"错误: 找不到文件 '{input_file}'")
            return []
        except Exception as e:
            print(f"读取文件错误: {e}")
            return []
        print(f"预处理完成: 从{input_file}中提取{len(processed_problems)}个问题")
        return processed_problems
    def solve_geometry_problem(self, problem_text: str) -> Tuple[bool, Optional[GeometricSolver]]:
        """解决单个几何问题"""
        try:
            solver_builder = GeometricSolverBuilder(seed=self.seed)
            solver_builder.load_problem_from_txt(problem_text)
            solver: GeometricSolver = solver_builder.build(max_attempts=self.max_attempts)
            success = solver.run()
            return success, solver if success else None
        except Exception as e:
            return False, None
    def test_problems_from_list(self, problems: List[str]) -> Dict[str, int]:
        """
        从问题列表测试求解，而不是从文件读取
        返回统计结果字典
        """
        print(f"开始测试{len(problems)}个问题...")
        # 输出文件路径
        solved_file = Path(__file__).parent / "aux_solved.jsonl"
        unsolved_file = Path(__file__).parent / "aux_unsolved.jsonl"
        unbuild_file = Path(__file__).parent / "aux_unbuild.jsonl"
        solved_problems = 0
        build_problems = 0
        processed_problems = 0
        solved_first_write = True
        unsolved_first_write = True
        unbuild_first_write = True
        for i, problem in enumerate(problems, 1):
            if random.random() < 0.95:
                continue
            processed_problems += 1
            try:
                solver_builder = GeometricSolverBuilder(self.seed)
                solver_builder.load_problem_from_txt(str(problem))
                solver: GeometricSolver = solver_builder.build(max_attempts=self.max_attempts)
                build_problems += 1
                success = solver.run()
                if success:
                    solved_problems += 1
                    writemode = 'w' if solved_first_write else 'a'
                    solved_first_write = False
                    with open(solved_file, writemode, encoding='utf-8') as f:
                        f.write(problem + '\n')
                else:
                    writemode = 'w' if unsolved_first_write else 'a'
                    unsolved_first_write = False
                    with open(unsolved_file, writemode, encoding='utf-8') as f:
                        f.write(problem + '\n')
            except Exception as e:
                writemode = 'w' if unbuild_first_write else 'a'
                unbuild_first_write = False
                with open(unbuild_file, writemode, encoding='utf-8') as f:
                    f.write(problem + '\n')
            # 每100个问题输出一次进度
            if processed_problems % 100 == 0 or i == len(problems):
                print(f"进度 {processed_problems}/{len(problems)}: 已求解={solved_problems}, 已构建={build_problems}, 已处理={processed_problems}")
        results = {
            'total': processed_problems,
            'solved': solved_problems,
            'build': build_problems,
            'unbuild': processed_problems - build_problems
        }
        print(f"\n测试完成:")
        print(f"  处理总数: {results['total']}")
        print(f"  成功求解: {results['solved']}")
        print(f"  成功构建: {results['build']}")
        print(f"  构建失败: {results['unbuild']}")
        if results['total'] > 0:
            print(f"  求解率: {results['solved']/results['total']:.2%}")
            print(f"  构建率: {results['build']/results['total']:.2%}")
        return results
    def validate_aux_problems(self, input_file: str) -> Dict[str, int]:
        """
        预处理aux数据并测试几何问题求解（无中间文件）
        """
        print(f"开始aux问题验证: {input_file}")
        # 步骤1: 在内存中预处理
        processed_problems = self.preprocess_jsonl_in_memory(input_file)
        if not processed_problems:
            print("预处理后没有可用问题")
            return {'total': 0, 'solved': 0, 'build': 0, 'unbuild': 0}
        # 步骤2: 直接测试预处理后的问题
        results = self.test_problems_from_list(processed_problems)
        return results
    def run_full_pipeline(self, input_file: str) -> Dict[str, any]:
        """
        完整验证流水线：
        1. 验证lm.py翻译函数适配性
        2. 验证aux问题求解
        """
        print("="*60)
        print("开始完整验证流水线")
        print("="*60)
        pipeline_results = {}
        # 步骤1: 验证翻译适配性
        print("\n步骤1: 验证lm.py翻译函数适配性")
        print("-" * 40)
        translation_valid = self.validate_lm_translation(input_file)
        pipeline_results['translation_valid'] = translation_valid
        # 步骤2: 验证aux问题求解
        print("\n步骤2: 验证aux问题求解")
        print("-" * 40)
        solve_results = self.validate_aux_problems(input_file)
        pipeline_results['solve_results'] = solve_results
        # 总结
        print("\n" + "="*60)
        print("完整验证流水线结果总结")
        print("="*60)
        print(f"翻译函数适配性: {'✓ 通过' if translation_valid else '✗ 失败'}")
        if solve_results['total'] > 0:
            print(f"问题求解统计:")
            print(f"  总问题数: {solve_results['total']}")
            print(f"  求解成功: {solve_results['solved']} ({solve_results['solved']/solve_results['total']:.1%})")
            print(f"  构建成功: {solve_results['build']} ({solve_results['build']/solve_results['total']:.1%})")
        return pipeline_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AuxChecker - 综合aux验证工具')
    parser.add_argument('command', choices=[
        'validate-translation',  # 验证lm.py翻译函数适配性
        'validate-aux',         # 预处理+测试aux问题
        'pipeline'              # 完整流水线
    ], help='要执行的命令')
    parser.add_argument('input_file', help='输入JSONL文件路径')
    parser.add_argument('--seed', type=int, default=123, 
                       help='随机种子 (默认: 123)')
    parser.add_argument('--max-attempts', type=int, default=100, 
                       help='求解器最大尝试次数 (默认: 100)')
    args = parser.parse_args()
    # 检查输入文件是否存在
    if not Path(args.input_file).exists():
        print(f"错误: 输入文件 '{args.input_file}' 不存在")
        sys.exit(1)
    # 创建AuxChecker实例
    checker = AuxChecker(seed=args.seed, max_attempts=args.max_attempts)
    # 执行相应命令
    try:
        if args.command == 'validate-translation':
            success = checker.validate_lm_translation(args.input_file)
            sys.exit(0 if success else 1)
        elif args.command == 'validate-aux':
            results = checker.validate_aux_problems(args.input_file)
            # 根据结果设置退出码
            sys.exit(0 if results['total'] > 0 else 1)
        elif args.command == 'pipeline':
            results = checker.run_full_pipeline(args.input_file)
            # 综合判断是否成功
            success = (results.get('translation_valid', False) and 
                      results.get('solve_results', {}).get('total', 0) > 0)
            sys.exit(0 if success else 1)
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

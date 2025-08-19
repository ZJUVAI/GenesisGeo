"""
主管线脚本 - 迭代规则更新管线
实现完整的迭代规则发现和性能评估流程
"""

import os
import shutil
from datetime import datetime
from .solver_utils import solve_problems_batch
from .data_processor import load_discovery_data
from .rule_extractor import create_rule_extractor


class IterativeRulesPipeline:
    """
    迭代规则更新管线
    功能：管理整个迭代规则发现流程，包括性能评估、规则提取和更新
    """
    
    def __init__(self, config: dict):
        """
        初始化管线配置
        功能：设置管线运行所需的所有配置参数
        参数：config (dict) - 包含文件路径、求解器配置等的配置字典
        返回值：无
        被调用：check_discovery.py 中的 run_pipeline_with_config() 调用
        """
        self.config = config
        self.rules_file = config['rules_file']
        self.discovery_data_file = config['discovery_data_file']
        self.dev_jgex_file = config['dev_jgex_file']
        self.default_rules_file = config['default_rules_file']
        self.max_iterations = config['max_iterations']
        self.backup_dir = config['backup_dir']
        self.log_dir = config['log_dir']
        self.rule_extractor_config = config['rule_extractor_config']
        
        # 创建必要的目录
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def run(self) -> dict:
        """
        运行完整的迭代管线
        功能：执行多轮迭代，每轮包括性能评估、规则提取和更新
        参数：无
        返回值：dict - 包含最终性能统计的字典
        被调用：check_discovery.py 中的 run_pipeline_with_config() 调用
        """
        print("=== 开始迭代规则发现管线 ===")
        
        # 初始化规则文件
        self.initialize_rules()
        
        all_results = []
        
        initial_result = {
            'iteration': 0,
            'performance': self.evaluate_current_performance(),
            'new_rules_count': 0,
            'timestamp': datetime.now().isoformat()
        }
        all_results.append(initial_result)
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- 第 {iteration} 轮迭代 ---")
            
            iteration_result = self.single_iteration(iteration)

            # 如果没有提取到新规则，提前结束
            if iteration_result['new_rules_count'] == 0:
                print("未提取到新规则，结束迭代")
                break

            all_results.append(iteration_result)
            
        
        # 在run的最后打印最终报告
        print("\n=== 最终性能报告 ===")
        if all_results == []:
            print(f"程序错误，未进行迭代")
            return []

        if len(all_results) == 1:
            final_performance = all_results[-1]['performance']

            print(f"未提取到新规则，仅迭代一次")
            print(f"求解成功率: {final_performance['solve_rate']:.2%}")
            print(f"总迭代次数: {len(all_results)}")

        else:
            initial_performance = all_results[0]['performance']
            final_performance = all_results[-1]['performance']
            
            print(f"初始求解成功率: {initial_performance['solve_rate']:.2%}")
            print(f"最终求解成功率: {final_performance['solve_rate']:.2%}")
            print(f"总体提升: {final_performance['solved'] - initial_performance['solved']} 题")
            print(f"总迭代次数: {len(all_results)}")
        
        return all_results
    
    def single_iteration(self, iteration: int) -> dict:
        """
        执行单次迭代
        功能：执行一轮完整的性能评估、规则提取和更新流程
        参数：iteration (int) - 当前迭代次数
        返回值：dict - 包含本次迭代结果的字典
        被调用：run() 方法调用
        """
        # 提取并添加新规则
        new_rules_count = self.extract_and_add_rules()
        print(f"提取到新规则: {new_rules_count} 条")
        
        # 如果有新规则，重新评估性能
        if new_rules_count > 0:
            performance = self.evaluate_current_performance()
            print(f"更新后求解成功率: {performance['solve_rate']:.2%} ({performance['solved']}/{performance['total']})")
            result = {
                'iteration': iteration,
                'performance_after': performance,
                'timestamp': datetime.now().isoformat()
            }
        else:
            result = {
                'new_rules_count': new_rules_count,
                'timestamp': datetime.now().isoformat()
            }
        
        self.log_iteration_results(iteration, result)
        return result
    
    def initialize_rules(self):
        """
        初始化规则文件（从默认规则复制）
        功能：如果规则文件不存在，从默认规则文件复制初始规则
        参数：无
        返回值：无
        被调用：run() 方法调用
        """
        if not os.path.exists(self.rules_file):
            print(f"初始化规则文件: {self.rules_file}")
            shutil.copy2(self.default_rules_file, self.rules_file)
        else:
            print(f"使用现有规则文件: {self.rules_file}")
    
    def evaluate_current_performance(self) -> dict:
        """
        评估当前规则集的求解性能
        功能：使用当前规则集对dev_jgex问题进行批量求解并统计性能
        参数：无
        返回值：dict - 包含求解统计信息的字典 {"total": int, "solved": int, "solve_rate": float}
        被调用：single_iteration() 方法调用
        """
        print("正在评估当前规则集性能...")
        results = solve_problems_batch(
            self.dev_jgex_file, 
            self.rules_file, 
            self.config['max_attempts']
        )
        return results
    
    def extract_and_add_rules(self) -> int:
        """
        提取新规则并添加到规则集
        功能：从discovery数据中提取新规则并追加到现有规则文件
        参数：无
        返回值：int - 成功添加的新规则数量
        被调用：single_iteration() 方法调用
        """
        # 加载discovery数据
        discovery_data = load_discovery_data(self.discovery_data_file)
        
        # 创建规则提取器
        extractor = create_rule_extractor(self.rule_extractor_config)
        
        # 提取规则
        new_rules = extractor.extract_rules(discovery_data)
        
        if new_rules:
            # 备份当前规则文件
            backup_path = os.path.join(
                self.backup_dir, 
                f"rules_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            shutil.copy2(self.rules_file, backup_path)
            
            # 追加新规则到文件
            with open(self.rules_file, 'a', encoding='utf-8') as f:
                f.write('\n')  # 确保换行
                for rule in new_rules:
                    f.write(f"{rule}\n")
        
        return len(new_rules)
    
    def log_iteration_results(self, iteration: int, results: dict):
        """
        记录迭代结果
        功能：将迭代结果写入日志文件
        参数：iteration (int) - 迭代次数, results (dict) - 迭代结果字典
        返回值：无
        被调用：single_iteration() 方法调用
        """
        log_file = os.path.join(self.log_dir, f"iteration_{iteration:03d}.log")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"迭代 {iteration} 结果\n")
            f.write(f"时间: {results['timestamp']}\n")
            f.write(f"迭代版本性能: {results['performance']}\n")
            f.write(f"新规则数量: {results['new_rules_count']}\n")

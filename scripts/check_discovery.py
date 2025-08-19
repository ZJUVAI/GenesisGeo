#!/usr/bin/env python3
"""
配置管理和主管线调用接口
用于设置配置参数并调用迭代规则更新管线
"""

import sys
import os
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.newclid.data_discovery.iterative_rules_pipeline import IterativeRulesPipeline

# 配置参数（写死在代码中）
RULES_FILE = "src/newclid/data_discovery/rules_with_discovery.txt"
DISCOVERY_DATA_FILE = "src/newclid/data_discovery/discovery_aux_data.jsonl"
DEV_JGEX_FILE = "problems_datasets/test_jgex.txt"
DEFAULT_RULES_FILE = "src/newclid/default_configs/rules.txt"

# 求解器配置
MAX_ATTEMPTS = 100
SOLVER_TIMEOUT = 30

# 管线配置
MAX_ITERATIONS = 10
BACKUP_DIR = "src/newclid/data_discovery/backups"
LOG_DIR = "src/newclid/data_discovery/logs"

# 规则提取器配置（简化版超参数）
RULE_EXTRACTOR_CONFIG = {
    "method": "basic",           # 提取方法: basic, advanced, ml_based
    "example_param": 0.8,        # 示例参数
}


def main():
    """
    主函数：解析命令行参数并调用管线
    功能：程序入口点，设置配置并启动迭代规则更新管线
    参数：无（从命令行解析）
    返回值：无
    被调用：程序启动时调用
    """
    parser = argparse.ArgumentParser(description='迭代规则发现管线')
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS,
                       help=f'最大迭代次数 (默认: {MAX_ITERATIONS})')
    parser.add_argument('--method', choices=['basic', 'advanced', 'ml_based'], 
                       default='basic', help='规则提取方法 (默认: basic)')
    
    args = parser.parse_args()
    
    # 更新配置
    config = {
        'rules_file': RULES_FILE,
        'discovery_data_file': DISCOVERY_DATA_FILE,
        'dev_jgex_file': DEV_JGEX_FILE,
        'default_rules_file': DEFAULT_RULES_FILE,
        'max_attempts': MAX_ATTEMPTS,
        'solver_timeout': SOLVER_TIMEOUT,
        'max_iterations': args.max_iterations,
        'backup_dir': BACKUP_DIR,
        'log_dir': LOG_DIR,
        'rule_extractor_config': {
            'method': args.method,
            'example_param': RULE_EXTRACTOR_CONFIG['example_param']
        }
    }
    
    run_pipeline_with_config(config)


def run_pipeline_with_config(config: dict):
    """
    使用指定配置运行管线
    功能：创建并运行迭代规则更新管线
    参数：config (dict) - 包含所有配置参数的字典
    返回值：无
    被调用：main() 函数调用
    """
    pipeline = IterativeRulesPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()

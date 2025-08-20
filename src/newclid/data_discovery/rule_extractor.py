"""
规则提取器框架（可配置多种实现）
提供从discovery数据中提取几何规则的功能
这是最重要的模块，支持多种提取方法
"""

from typing import List, Dict
from .data_processor import parse_llm_input, parse_llm_output


class RuleExtractor:
    """
    规则提取器基类
    功能：定义规则提取的通用接口
    """
    
    def __init__(self, config: dict):
        """
        初始化提取器配置
        功能：设置规则提取器的配置参数
        参数：config (dict) - 提取器配置字典，包含method、example_param等
        返回值：无
        被调用：create_rule_extractor() 函数调用
        """
        self.config = config
        self.method = config.get('method', 'basic')
        self.example_param = config.get('example_param', 0.8)
    
    def extract_rules(self, discovery_data: List[Dict]) -> List[str]:
        """
        从discovery数据中提取规则（基类方法，需要子类实现）
        功能：解析discovery数据并提取可用的几何规则
        参数：discovery_data (List[Dict]) - 包含llm_input_renamed和llm_output_renamed的数据列表
        返回值：List[str] - 提取的规则字符串列表
        被调用：iterative_rules_pipeline.py 中的 extract_and_add_rules() 调用
        """
        raise NotImplementedError("子类必须实现 extract_rules 方法")


class BasicRuleExtractor(RuleExtractor):
    """
    基础规则提取器
    功能：实现基本的规则提取逻辑（框架实现）
    """
    
    def extract_rules(self, discovery_data: List[Dict]) -> List[str]:
        """
        基础规则提取逻辑（框架实现）
        功能：使用基础方法从证明步骤中提取几何规则
        参数：discovery_data (List[Dict]) - discovery数据列表
        返回值：List[str] - 提取的规则字符串列表
        被调用：父类的 extract_rules() 接口调用
        """
        extracted_rules = []
        
        print(f"开始基础规则提取，处理 {len(discovery_data)} 条数据...")
        
        for i, data in enumerate(discovery_data):
            if (i + 1) % 100 == 0:
                print(f"处理进度: {i + 1}/{len(discovery_data)}")
            
            # 获取LLM输入和输出
            llm_input = data.get('llm_input_renamed', '')
            llm_output = data.get('llm_output_renamed', '')
            
            if not llm_input or not llm_output:
                continue
            
            # 解析LLM输出中的证明步骤
            parsed_output = parse_llm_output(llm_output)
            proof_steps = parsed_output.get('proof_steps', [])
            
            # 从每个证明步骤提取规则
            for step in proof_steps:
                rules_from_step = self._extract_from_proof_step(step)
                for rule in rules_from_step:
                    if self._validate_rule_format(rule):
                        extracted_rules.append(rule)
        
        # 去重
        unique_rules = list(set(extracted_rules))
        print(f"基础提取完成: {len(extracted_rules)} -> {len(unique_rules)} 条唯一规则")
        
        return unique_rules
    
    def _extract_from_proof_step(self, proof_step: str) -> List[str]:
        """
        从单个证明步骤提取规则
        功能：分析单个证明步骤，提取其中蕴含的几何规则
        参数：proof_step (str) - 单个证明步骤的文本
        返回值：List[str] - 从该步骤提取的规则列表
        被调用：extract_rules() 方法内部调用
        """
        # TODO: 实现具体的证明步骤解析和规则提取逻辑
        # 这里是您需要重点实现的核心算法
        
        # 框架示例：简单的模式匹配
        rules = []
        
        # 示例：如果步骤包含特定模式，提取为规则
        # 实际实现需要根据几何推理的具体格式来设计
        if "eqangle" in proof_step.lower():
            # 提取角度相等相关的规则
            pass
        elif "cong" in proof_step.lower():
            # 提取线段相等相关的规则
            pass
        elif "simtri" in proof_step.lower():
            # 提取相似三角形相关的规则
            pass
        
        return rules
    
    def _validate_rule_format(self, rule: str) -> bool:
        """
        验证规则格式
        功能：检查提取的规则是否符合预期的格式规范
        参数：rule (str) - 待验证的规则字符串
        返回值：bool - 规则格式是否有效
        被调用：extract_rules() 方法内部调用
        """
        # TODO: 实现规则格式验证逻辑
        # 检查规则是否符合几何推理系统的语法要求
        
        if not rule or not rule.strip():
            return False
        
        # 基本格式检查
        # 实际验证逻辑需要根据具体的规则格式定义
        return True


class AdvancedRuleExtractor(RuleExtractor):
    """
    高级规则提取器（预留接口）
    功能：实现更复杂的规则提取算法
    """
    
    def extract_rules(self, discovery_data: List[Dict]) -> List[str]:
        """
        高级规则提取逻辑（待实现）
        功能：使用高级算法（如模式挖掘、频率分析等）提取规则
        参数：discovery_data (List[Dict]) - discovery数据列表
        返回值：List[str] - 提取的规则字符串列表
        被调用：父类的 extract_rules() 接口调用
        """
        print("高级规则提取器暂未实现")
        return []


class MLBasedRuleExtractor(RuleExtractor):
    """
    基于机器学习的规则提取器（预留接口）
    功能：使用机器学习方法自动发现和提取规则
    """
    
    def extract_rules(self, discovery_data: List[Dict]) -> List[str]:
        """
        ML规则提取逻辑（待实现）
        功能：使用机器学习模型从大量证明数据中学习和提取规则
        参数：discovery_data (List[Dict]) - discovery数据列表
        返回值：List[str] - 提取的规则字符串列表
        被调用：父类的 extract_rules() 接口调用
        """
        print("ML规则提取器暂未实现")
        return []


def create_rule_extractor(config: dict) -> RuleExtractor:
    """
    根据配置创建规则提取器
    功能：工厂函数，根据配置参数创建相应的规则提取器实例
    参数：config (dict) - 提取器配置，必须包含'method'字段
    返回值：RuleExtractor - 对应的规则提取器实例
    被调用：iterative_rules_pipeline.py 中的 extract_and_add_rules() 调用
    """
    method = config.get('method', 'basic')
    
    if method == 'basic':
        return BasicRuleExtractor(config)
    elif method == 'advanced':
        return AdvancedRuleExtractor(config)
    elif method == 'ml_based':
        return MLBasedRuleExtractor(config)
    else:
        print(f"警告: 未知的提取方法 '{method}'，使用默认的基础提取器")
        return BasicRuleExtractor(config)

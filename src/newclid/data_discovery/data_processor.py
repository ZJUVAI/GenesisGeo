"""
数据处理和解析
提供discovery数据的加载功能
"""

import json
import os
from typing import List, Dict


def load_discovery_data(file_path: str) -> List[Dict]:
    """
    加载discovery数据
    功能：从jsonl文件中读取discovery数据，每行一个JSON对象
    参数：file_path (str) - discovery_aux_data.jsonl文件路径
    返回值：List[Dict] - [{"llm_input_renamed": str, "llm_output_renamed": str, ...}, ...]
    被调用：iterative_rules_pipeline.py 中的 extract_and_add_rules() 调用
    """
    data = []
    
    if not os.path.exists(file_path):
        print(f"警告: Discovery数据文件 {file_path} 不存在")
        return data
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
        
        print(f"成功加载 {len(data)} 条discovery数据")
        
    except Exception as e:
        print(f"错误: 读取discovery数据文件失败: {e}")
    
    return data


def parse_llm_input(llm_input: str) -> Dict:
    """
    解析llm_input_renamed内容
    功能：解析LLM输入文本，提取问题的前提和目标
    参数：llm_input (str) - llm_input_renamed字段的内容
    返回值：Dict - {"problem": str, "premises": list, "goal": str}
    被调用：rule_extractor.py 中的规则提取函数调用
    """
    # TODO: 实现LLM输入解析逻辑
    # 这里先返回空的结构，具体解析逻辑后续实现
    return {
        "problem": "",
        "premises": [],
        "goal": ""
    }


def parse_llm_output(llm_output: str) -> Dict:
    """
    解析llm_output_renamed内容
    功能：解析LLM输出文本，提取数值检查和证明步骤
    参数：llm_output (str) - llm_output_renamed字段的内容
    返回值：Dict - {"numerical_check": list, "proof_steps": list}
    被调用：rule_extractor.py 中的规则提取函数调用
    """
    # TODO: 实现LLM输出解析逻辑
    # 这里先返回空的结构，具体解析逻辑后续实现
    return {
        "numerical_check": [],
        "proof_steps": []
    }

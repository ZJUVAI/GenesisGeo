# 迭代规则发现管线 - 工作总结与待办事项

## 项目概述

本项目实现了一个迭代式几何规则发现管线，用于从discovery数据中提取新的几何推理规则，并通过这些规则提升DDAR引擎在dev_jgex问题集上的求解性能。

## 代码结构

```
scripts/
└── check_discovery.py              # 配置和调用接口

src/newclid/data_discovery/
├── __init__.py                     # 模块初始化文件
├── iterative_rules_pipeline.py     # 主管线脚本
├── solver_utils.py                 # 求解器工具
├── data_processor.py               # 数据处理工具
├── rules_manager.py                # 规则管理工具（空框架）
├── rule_extractor.py               # 规则提取器（核心模块）
└── summary_and_todo.md             # 本文件
```

## 程序运行流程

### 启动入口
```bash
python scripts/check_discovery.py [--max-iterations N] [--method METHOD]
```

### 执行流程

1. **配置初始化** (`check_discovery.py`)
   - 设置固定的文件路径配置
   - 解析命令行参数
   - 调用主管线

2. **主管线执行** (`iterative_rules_pipeline.py`)
   ```
   IterativeRulesPipeline.run()
   ├── initialize_rules()                    # 初始化规则文件
   └── for 每次迭代:
       ├── single_iteration()
       │   ├── evaluate_current_performance()  # 评估当前性能
       │   ├── extract_and_add_rules()        # 提取并添加新规则
       │   └── log_iteration_results()        # 记录结果
       └── 打印最终报告
   ```

3. **性能评估** (`solver_utils.py`)
   ```
   evaluate_current_performance()
   └── solve_problems_batch()
       ├── _load_dev_jgex_problems()         # 加载问题集
       ├── solve_single_problem()            # 逐个求解
       └── 返回统计结果 {total, solved, solve_rate}
   ```

4. **规则提取** (`rule_extractor.py`)
   ```
   extract_and_add_rules()
   ├── load_discovery_data()                # 加载discovery数据
   ├── create_rule_extractor()              # 创建提取器
   └── extractor.extract_rules()
       ├── parse_llm_output()               # 解析证明步骤
       ├── _extract_from_proof_step()       # 从步骤提取规则
       └── _validate_rule_format()          # 验证规则格式
   ```

## 已实现功能

### ✅ 完整实现的模块

1. **check_discovery.py** - 配置管理和程序入口
   - 命令行参数解析
   - 配置参数设置
   - 主管线调用

2. **iterative_rules_pipeline.py** - 主管线逻辑
   - 迭代循环控制
   - 性能评估调度
   - 规则提取调度
   - 日志记录
   - 最终报告生成

3. **solver_utils.py** - 求解器封装
   - 单问题求解接口
   - 批量问题求解（集成加载和统计）
   - dev_jgex问题文件解析

4. **data_processor.py** - 数据处理
   - discovery数据加载（jsonl格式）
   - 基本的LLM输入输出解析框架

### 🔧 框架实现的模块

1. **rules_manager.py** - 规则文件管理（空框架）
   - `read_rules_file()` - TODO
   - `write_rules_file()` - TODO  
   - `append_rules_to_file()` - TODO
   - `backup_rules_file()` - TODO
   - `deduplicate_rules()` - TODO
   - `validate_rules_syntax()` - TODO
   - `copy_default_rules()` - TODO

2. **rule_extractor.py** - 规则提取器（部分框架）
   - `BasicRuleExtractor` 类结构完整
   - `_extract_from_proof_step()` - TODO（核心算法）
   - `_validate_rule_format()` - TODO
   - `parse_llm_input()` - TODO
   - `parse_llm_output()` - TODO

## 待实现功能清单

### 🚨 高优先级（核心功能）

1. **solver_utils.py**
   - [ ] 确认并实现自定义规则文件的设置方法
   - [ ] 完善GeometricSolverBuilder的规则文件配置

2. **rule_extractor.py**
   - [ ] `_extract_from_proof_step()` - 从证明步骤提取规则的核心算法
   - [ ] `_validate_rule_format()` - 规则格式验证逻辑
   
3. **data_processor.py**
   - [ ] `parse_llm_input()` - 解析LLM输入格式
   - [ ] `parse_llm_output()` - 解析LLM输出中的证明步骤

### 🔧 中优先级（支撑功能）

1. **rules_manager.py** - 规则文件操作
   - [ ] `read_rules_file()` - 读取规则文件
   - [ ] `write_rules_file()` - 写入规则文件
   - [ ] `append_rules_to_file()` - 追加规则
   - [ ] `backup_rules_file()` - 备份规则文件
   - [ ] `deduplicate_rules()` - 去重规则

### 🎯 低优先级（增强功能）

1. **rule_extractor.py** - 高级提取器
   - [ ] `AdvancedRuleExtractor` 具体实现
   - [ ] `MLBasedRuleExtractor` 具体实现

2. **验证和优化**
   - [ ] 规则语法验证
   - [ ] 性能优化
   - [ ] 错误处理增强

## 数据文件准备

### ✅ 已准备
- `src/newclid/data_discovery/discovery_aux_data.jsonl` - 过滤后的discovery数据

### 📝 需要准备
- `src/newclid/data_discovery/rules_with_discovery.txt` - 将从default_configs/rules.txt复制
- `src/newclid/data_discovery/backups/` - 规则备份目录
- `src/newclid/data_discovery/logs/` - 迭代日志目录

## 下一步工作重点

1. **立即任务**: 实现solver_utils.py中的自定义规则文件设置
2. **核心任务**: 实现rule_extractor.py中的规则提取核心算法
3. **支撑任务**: 实现data_processor.py中的解析功能
4. **测试任务**: 运行完整管线并验证功能

## 技术依赖

- **Newclid库**: GeometricSolverBuilder, GeometricSolver
- **数据格式**: dev_jgex.txt, discovery_aux_data.jsonl
- **Python模块**: json, os, shutil, datetime, typing

## 配置参数

```python
# 文件路径（固定）
RULES_FILE = "src/newclid/data_discovery/rules_with_discovery.txt"
DISCOVERY_DATA_FILE = "src/newclid/data_discovery/discovery_aux_data.jsonl" 
DEV_JGEX_FILE = "problems_datasets/dev_jgex.txt"
DEFAULT_RULES_FILE = "src/newclid/default_configs/rules.txt"

# 管线参数（可调整）
MAX_ITERATIONS = 10
MAX_ATTEMPTS = 100

# 提取器配置（可扩展）
RULE_EXTRACTOR_CONFIG = {
    "method": "basic",
    "example_param": 0.8
}
```

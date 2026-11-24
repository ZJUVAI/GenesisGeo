"""共享数据模型（对应伪代码 §6 关键数据结构）。

Pipeline 中跨模块传递的数据一律使用本文件定义的 dataclass，避免散乱的 dict。

设计约束：所有 dataclass 尽量 frozen=True；不依赖业务模块，只被它们引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 基础几何数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateInstance:
    """单个谓词实例，如 cong(a, b, c, d)。"""

    predicate: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Point:
    """带坐标的点。"""

    name: str
    x: float
    y: float


# ---------------------------------------------------------------------------
# 完整证明图节点（Part 1 第一子步：构图产物）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactNode:
    """证明图中的事实节点（几何命题）。

    id 为原始数据中方括号内的局部编号（如 "000"）。
    produced_by 为生成该 fact 的 rule 节点 id；给定前提（题设 / numerical_check）为 None。
    """

    id: str
    predicate: str
    args: tuple[str, ...]
    produced_by: str | None = None


@dataclass(frozen=True)
class RuleNode:
    """证明图中的规则节点（一次推理步骤）。

    id 形如 "R{step}"；code 为规则代码（如 "r111"、"AR"）。
    premises / conclusion 均以 FactNode.id 引用。
    """

    id: str
    step: int
    code: str
    premises: tuple[str, ...]
    conclusion: str


# ---------------------------------------------------------------------------
# Step 2 产物
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposition:
    """Step 2 提取出的"前提集 + 结论"对。"""

    premises: tuple[PredicateInstance, ...]
    conclusion: PredicateInstance


@dataclass(frozen=True)
class PropositionRecord:
    """从一张证明图提取的命题：无辅助点前提 → 结论，并保留坐标与来源序号。

    - premises：图中所有不含辅助点的给定前提。
    - conclusion：图的最终结论（goal）。
    - points：命题涉及点的坐标（前提+结论中出现的点），便于后续数值处理。
    - proposition_id / seed / index_in_seed：溯源信息（子图 id 形如 "59:0#017"）。
    """

    proposition_id: str
    seed: int | None
    index_in_seed: int
    premises: tuple[PredicateInstance, ...]
    conclusion: PredicateInstance
    points: tuple[Point, ...]


@dataclass(frozen=True)
class NormalizedRule:
    """规范化 + 去重后的规则。

    - rule_text：规范化规则文本（点名重命名为 A,B,C…）。
    - rename_map：原始点名 → 重命名后点名。
    - points：重命名后点的坐标。
    - rule_id / seed / index_in_seed：所保留的代表命题（同规范形中序号最小者）。
    """

    rule_id: str
    seed: int | None
    index_in_seed: int
    rule_text: str
    rename_map: dict[str, str]
    points: tuple[Point, ...]


# ---------------------------------------------------------------------------
# 贯穿 Part 1 输出 / Part 2 输入的核心结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleWithSource:
    """一条规范化规则及其来源题目信息。

    Part 1 输出的 JSONL 每行对应一个 RuleWithSource；
    Part 2 加载 JSONL 后也构建此对象。
    """

    rule_id: str
    rule_text: str
    points: tuple[Point, ...]
    premises: tuple[PredicateInstance, ...]
    goal: PredicateInstance
    llm_output_renamed: str
    seed: int | None
    pid: str
    subgraph_id: int


# ---------------------------------------------------------------------------
# Step 1 子图渲染结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphNode:
    """证明图中的节点。"""

    idx: int
    type: str  # "fact" | "rule"
    label: str
    points: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedSubgraph:
    """修剪后的子图序列化结构。"""

    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[int, int], ...]


# ---------------------------------------------------------------------------
# 中间结果结构
# ---------------------------------------------------------------------------


@dataclass
class PrunedItem:
    """Step 1 → Step 2 的传递单元。"""

    pid: str
    subgraph_id: int
    rendered: RenderedSubgraph


@dataclass
class ExtractionRecord:
    """Step 2-5 之间流转的中间记录。

    在各 step 中逐步填充字段，未填充字段保持 None。
    """

    pid: str
    subgraph_id: int
    rule_text: str
    rename_map: dict[str, str] | None = None
    proposition: Proposition | None = None
    rendered: RenderedSubgraph | None = None
    eqpoint_info: list[tuple[str, str]] | None = None
    # Step 4 填充
    problem_statement: str | None = None
    points: tuple[Point, ...] | None = None
    premises: tuple[PredicateInstance, ...] | None = None
    goal: PredicateInstance | None = None
    seed: int | None = None
    llm_output_renamed: str | None = None
    # Step 5 填充
    signature: str | None = None
    normalized_rule_text: str | None = None


# ---------------------------------------------------------------------------
# Part 1 / Part 2 引擎返回结构
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Part 1 (RuleExtractionEngine) 的返回值。"""

    rules_file: str
    input_count: int
    output_count: int
    skipped_rules: int
    step_timing: dict[str, float]
    output_files: dict[str, str]


@dataclass
class SeedReductionStats:
    """Seed 规约阶段统计。"""

    input_size: int
    output_size: int
    output_file: str
    n_groups: int
    avg_reduction_rate: float
    group_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DivideConquerStats:
    """分治规约阶段统计。"""

    input_size: int
    output_size: int
    n_subsumption_tests: int
    reduction_rate: float


@dataclass
class ReductionResult:
    """Part 2 规约引擎的返回值。"""

    output_path: str
    input_count: int
    output_count: int
    stage_timing: dict[str, float]
    seed_reduction: SeedReductionStats | None = None
    divide_conquer_reduction: DivideConquerStats | None = None


# ---------------------------------------------------------------------------
# Pipeline 汇总
# ---------------------------------------------------------------------------


@dataclass
class PipelineSummary:
    """写入 pipeline_summary.json 的顶层汇总结构。"""

    output_dir: str
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_part(self, part_name: str, enabled: bool, stats: dict[str, Any]) -> None:
        """记录一个 Part 的执行结果。"""
        self.parts[part_name] = {"enabled": enabled, **stats}

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化的字典。"""
        return {"output_dir": self.output_dir, "parts": self.parts}

"""原始合成数据的文本解析工具（Part 1 第一子步）。

从一条原始 JSONL 记录的三个字段中抽取构图所需的结构化信息：
- ``fl_problem``：``point@x_y`` 格式的点坐标（重命名后点名）。
- ``llm_input_renamed`` 的 ``<problem>``：题设前提 fact，以及 ``?`` 之后的 goal。
- ``llm_output_renamed`` 的 ``<aux>`` / ``<numerical_check>`` / ``<trivial>`` / ``<proof>``：
  辅助 fact、数值检查 fact、以及推理步骤。

本模块只做纯文本解析，不构图；构图逻辑见 graph_manager.py。
"""

from __future__ import annotations

import re

from newclid.discovery.data_models import PredicateInstance, Point


# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------

# 剥离 <analysis> / <proof> 等标签
_TAGS_RE = re.compile(r"</?\w+>")

# 一个事实片段：pred arg arg ... [id]
_FACT_RE = re.compile(r"(\w+)\s+([A-Za-z0-9_ ]+?)\s*\[(\d+)\]")

# 方括号内的局部 id，如 [004]
_BRACKET_ID_RE = re.compile(r"\[(\d+)\]")

# fl_problem 中的 point@x_y（坐标可为负数 / 科学计数法）
_COORD_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9]*)"  # 点名
    r"@"
    r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"  # x
    r"_"
    r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"  # y
)


# ---------------------------------------------------------------------------
# 标签处理
# ---------------------------------------------------------------------------


def strip_tags(text: str) -> str:
    """去掉形如 <tag> / </tag> 的标签。"""
    return _TAGS_RE.sub("", text or "")


def extract_tag_content(text: str, tag: str) -> str:
    """提取 <tag>...</tag> 之间的内容；缺失返回空串。"""
    if not text:
        return ""
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# fact / proof 解析
# ---------------------------------------------------------------------------


def parse_fact_segments(section: str) -> list[tuple[str, tuple[str, ...], str]]:
    """从一段文本中解析所有 ``pred args [id]`` 事实片段。

    Returns
    -------
    list[(predicate, args, id)]
        按出现顺序排列；args 为点参数元组。
    """
    content = strip_tags(section).replace("\n", " ")
    out: list[tuple[str, tuple[str, ...], str]] = []
    for m in _FACT_RE.finditer(content):
        pred = m.group(1)
        args = tuple(tok for tok in m.group(2).split() if tok)
        id_local = m.group(3)
        out.append((pred, args, id_local))
    return out


def parse_proof_step(
    line: str,
) -> tuple[str, tuple[str, ...], str, str, tuple[str, ...]] | None:
    """解析一条 proof 语句，如 ``eqangle a b a c a c b c [008] r111 [001]``。

    结构：``<concl_pred> <concl_args...> [concl_id] <rule_code> [prem_id]...``

    Returns
    -------
    (concl_pred, concl_args, concl_id, rule_code, premise_ids) 或 None（无法解析）。
    """
    s = line.strip()
    if not s:
        return None

    # 第一个 [NNN] 作为结论 id 锚点
    first = _BRACKET_ID_RE.search(s)
    if not first:
        return None
    concl_id = first.group(1)
    left = s[: first.start()].strip()
    right = s[first.end() :].strip()
    if not left or not right:
        return None

    left_tokens = left.split()
    concl_pred = left_tokens[0]
    concl_args = tuple(left_tokens[1:])

    right_tokens = right.split()
    rule_code = right_tokens[0]
    premise_ids = tuple(_BRACKET_ID_RE.findall(" ".join(right_tokens[1:])))

    return concl_pred, concl_args, concl_id, rule_code, premise_ids


# ---------------------------------------------------------------------------
# 坐标 / goal / aux 点
# ---------------------------------------------------------------------------


def extract_points(fl_problem: str) -> tuple[Point, ...]:
    """从 fl_problem 解析点坐标（重命名后点名）。按出现顺序去重。"""
    seen: set[str] = set()
    points: list[Point] = []
    for m in _COORD_RE.finditer(fl_problem or ""):
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)
        points.append(Point(name=name, x=float(m.group(2)), y=float(m.group(3))))
    return tuple(points)


def extract_goal(problem_section: str) -> PredicateInstance | None:
    """从 <problem> 内容中提取 ``?`` 之后的 goal 谓词。"""
    content = strip_tags(problem_section)
    if "?" not in content:
        return None
    goal_text = content.split("?", 1)[1].strip()
    tokens = goal_text.split()
    if not tokens:
        return None
    return PredicateInstance(predicate=tokens[0], args=tuple(tokens[1:]))


def extract_aux_points(aux_section: str) -> tuple[str, ...]:
    """从 <aux> 内容中提取辅助构造点名。

    aux 片段形如 ``x00 m : ...``，以 ``x`` 开头的 token 后紧跟点名。
    """
    content = strip_tags(aux_section).replace("\n", " ")
    aux_points = []

    for part in content.split(";"):
        part = part.strip()
        if not part:
            continue

        left = part.split(":", 1)[0].strip()
        tokens = left.split()

        if tokens and tokens[0].lower().startswith("x"):
            aux_points.extend(tokens[1:])

    return tuple(aux_points)

"""配置加载与默认值填充（对应伪代码 §2）。

读取 JSON 配置文件，递归合并到 DEFAULT_CONFIG 上，确保 global / part1_extract /
part2_reduction 三段都有完整的字段；缺失的可选字段由本文件填默认值。
"""

from __future__ import annotations

import json
import os
from typing import Any


# ---------------------------------------------------------------------------
# 默认配置（与伪代码 §2 的字段表格对齐）
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "global": {
        "output_dir": None,  # 必填，无默认值
        "n_workers": 30,
        "save_intermediates": True,
    },
    "part1_extract": {
        "enabled": False,
        "input": None,
        "output": None,  # 默认 {output_dir}/part1/extracted_rules.jsonl
        "rule_skip_predicates": ["aconst", "rconst"],
    },
    "part2_reduction": {
        "enabled": False,
        "input": None,
        "output": None,  # 默认 {output_dir}/part2/extracted_rules.txt
        "max_premises": None,
        "max_perp_premises": None,  # 阶段0: perp 前提数 > 该值则丢弃; null = 不过滤
        "extra_rules_path": None,  # 额外 sources 规则(如上一轮已验证规则库), 不参与本轮规约/NDG
        "engine": "full",
        "timeout": 3600,
        "seed_reduction": {
            "enabled": False,
        },
        "divide_conquer_reduction": {
            "enabled": False,
            "min_chunk_size": 30,
        },
        # 阶段0.5: NDG 发现 + 应用（原独立的 part3_ndg，已并入 Part 2，且提前
        # 到任何规约判定之前——规约阶段的 subsumption 判定会把 sources 当作
        # 无条件成立的定理，若 source 本身需要 guard 才成立，用它淘汰别的
        # 规则这个决定就可能是错的，且规约丢弃不可逆）。
        "ndg": {
            "enabled": False,
            "n_seeds": 10,
            "n_ce_trials": 3,
            "n_workers": 8,
            "degeneracy_threshold": 0.001,
            "rule_timeout_seconds": 120.0,
            "min_good_ratio": 0.0,  # good/(good+bad) 采样占比低于该值直接丢弃; 0 = 不检查
            "normalized_rules_path": None,  # 默认 {output_dir}/part1/normalized_rules.jsonl
            "occurrences_path": None,  # 默认 {output_dir}/part1/rule_seed_occurrences_all.json
            "source_dataset_path": None,  # 默认取 part1_extract.input
        },
    },
}


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict[str, Any]:
    """加载 JSON 配置文件并填充默认值。

    Parameters
    ----------
    config_path : str
        配置文件路径（JSON 格式）。

    Returns
    -------
    dict
        合并默认值后的完整配置字典。

    Raises
    ------
    FileNotFoundError
        配置文件不存在。
    ValueError
        配置文件解析失败或必填字段缺失。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 允许 // 整行注释（标准 JSON 不支持）
    lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("//")]
    try:
        user_cfg = json.loads("\n".join(lines))
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件解析失败: {exc}") from exc

    user_cfg = _strip_comments(user_cfg)
    cfg = _merge_configs(DEFAULT_CONFIG, user_cfg)
    _validate_config(cfg)
    return cfg


def _strip_comments(obj: Any) -> Any:
    """递归删除以 _comment 开头的键。"""
    if isinstance(obj, dict):
        return {
            k: _strip_comments(v)
            for k, v in obj.items()
            if not str(k).startswith("_comment")
        }
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _merge_configs(default: dict, user: dict) -> dict:
    """递归合并 user 配置到 default 上（user 优先）。"""
    merged = dict(default)
    for key, value in user.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_config(cfg: dict[str, Any]) -> None:
    """校验必填字段。

    - global.output_dir 不能为 None
    - part1_extract.enabled=True 时 input 不能为 None
    """
    if not cfg.get("global", {}).get("output_dir"):
        raise ValueError("global.output_dir 为必填项")

    part1 = cfg.get("part1_extract", {})
    if part1.get("enabled") and not part1.get("input"):
        raise ValueError("part1_extract.enabled=True 时必须指定 part1_extract.input")


def resolve_output(
    explicit: str | None,
    output_dir: str,
    sub_dir: str,
    default_filename: str,
) -> str:
    """解析输出路径：若 explicit 非 None 则直接用，否则拼 output_dir/sub_dir/default_filename。"""
    if explicit:
        return explicit
    return os.path.join(output_dir, sub_dir, default_filename)

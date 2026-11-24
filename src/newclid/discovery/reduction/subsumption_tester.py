"""规约的原子操作：可推导性判定（对应伪代码 §5.3）。

核心：给定目标规则 R=(P → g) 与一组"其它规则" sources，判断在 base DDAR 规则 +
sources 作为自定义规则的前提下，能否从 P 推出 g。可推出 ⇒ R 冗余。

判定引擎用 CSolver（C++ DDAR），仅在本文件引用。sources 可为多条 → 天然支持
"多条规则合力推出第三条"（非两两 subsumption）。

规则的坐标来自 Part 1 存下的重命名后 points（单一实现）。多坐标实现共识以降低
浮点巧合误判，留作后续增强（见 is_derivable 的 TODO）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tqdm import tqdm

from newclid.discovery.utils.rule_parser import to_pipe_format


@dataclass
class RuleItem:
    """规约过程中流转的轻量规则表示。"""

    rule_id: str
    seed: int | None
    index_in_seed: int
    rule_text: str
    points: list[tuple[str, float, float]]  # [(name, x, y)]
    premises: list[tuple[str, list[str]]] = field(default_factory=list)
    goal: tuple[str, list[str]] | None = None
    premise_count: int = 0
    guards: str = ""  # NDG guard predicates (numerical-only)

    @classmethod
    def from_record(cls, rec: dict) -> "RuleItem":
        from newclid.discovery.utils.rule_parser import parse_predicate, split_rule_text

        prem_strs, concl_str = split_rule_text(rec["rule_text"])
        premises = [
            (name, list(args))
            for name, args in (parse_predicate(p) for p in prem_strs if p.strip())
        ]
        cname, cargs = parse_predicate(concl_str)
        goal = (cname, list(cargs))
        points = [(p["name"], p["x"], p["y"]) for p in rec.get("points", [])]
        return cls(
            rule_id=rec["rule_id"],
            seed=rec.get("seed"),
            index_in_seed=rec.get("index_in_seed", 0),
            rule_text=rec["rule_text"],
            points=points,
            premises=premises,
            goal=goal,
            premise_count=len(premises),
            guards=rec.get("guards", ""),
        )


def load_rules(jsonl_path: str) -> list[RuleItem]:
    """从 normalized_rules.jsonl 加载为 RuleItem 列表。"""
    rules: list[RuleItem] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="[part2] 加载规则", unit="行"):
            line = line.strip()
            if line:
                rules.append(RuleItem.from_record(json.loads(line)))
    return rules


class SubsumptionTester:
    """封装 CSolver 的可推导性判定。"""

    def __init__(self, seed: int = 42, config_path: str | None = None) -> None:
        self.seed = seed
        self.config_path = config_path

    def is_derivable(self, target: RuleItem, sources: list[RuleItem]) -> bool:
        """在 base + sources 规则下，能否从 target 的前提推出 target 的结论。

        Returns True ⇒ target 冗余（可被 sources 推出）。
        任何异常（构造失败/超时等）视为不可推出，保守保留该规则。
        """
        if not sources or target.goal is None:
            return False
        try:
            from newclid.api import CSolver

            csolver = CSolver(
                points=target.points,
                premises=target.premises,
                goals=[target.goal],
                config_path=self.config_path,
                seed=self.seed,
            )
            custom = [to_pipe_format(s.rule_id, s.rule_text, s.guards) for s in sources]
            # TODO(鲁棒性): 多坐标实现取共识，只有多次都可推才判冗余，降低浮点巧合误判。
            return bool(csolver.run(custom_rules=custom))
        except Exception:
            return False

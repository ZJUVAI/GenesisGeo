"""规则规范化 + 等价去重（规范化 Step 2 / Step 3）。

作用于 part1 提取的命题（PropositionRecord）：
1. Step 1 谓词转化（predicate_rewrite）。
2. Step 2 规范化：逐谓词按对称性规范化参数（utils.symmetry）→ 前提按（谓词, 参数）排序
   → 点名按出现顺序重命名为 A,B,C…（超过 26 个不同点则丢弃整条规则）→ 重命名后再规范
   化+排序定稿。
3. Step 3 去重：规范形完全相同的规则只保留序号最小者（seed, index_in_seed, 根节点号）。
"""

from __future__ import annotations

from tqdm import tqdm

from newclid.discovery.data_models import (
    NormalizedRule,
    Point,
    PredicateInstance,
    PropositionRecord,
)
from newclid.discovery.extraction.predicate_rewrite import rewrite_predicate
from newclid.discovery.utils.rule_parser import build_rule_text
from newclid.discovery.utils.symmetry import normalize_predicate

Pred = tuple[str, tuple[str, ...]]


def _norm_args(pred: Pred) -> Pred:
    """按对称性规范化单个谓词的参数；未知谓词原样保留（不做规范化）。"""
    name, args = pred
    try:
        return name, tuple(
            normalize_predicate(PredicateInstance(predicate=name, args=args))
        )
    except ValueError:
        return name, tuple(args)


def _collect_points(preds: list[Pred]) -> list[str]:
    """按首次出现顺序收集点名（跳过以数字开头的常量参数）。"""
    seen: dict[str, None] = {}
    for _name, args in preds:
        for a in args:
            if a and not a[0].isdigit() and a not in seen:
                seen[a] = None
    return list(seen)


def _dedup_sorted(preds: list[Pred]) -> list[Pred]:
    """前提去重（保序）后按 (谓词, 参数) 排序。"""
    uniq = list(dict.fromkeys(preds))
    uniq.sort(key=lambda p: (p[0], p[1]))
    return uniq


def normalize_proposition(prop: PropositionRecord) -> NormalizedRule | None:
    """把一条命题规范化为 NormalizedRule；点数超过 26 返回 None（丢弃）。"""
    coord = {p.name: (p.x, p.y) for p in prop.points}

    # Step 1: 谓词转化
    premises = [rewrite_predicate(pi.predicate, pi.args, coord) for pi in prop.premises]
    conclusion = rewrite_predicate(
        prop.conclusion.predicate, prop.conclusion.args, coord
    )

    # Step 2a: 用原始点名先规范化参数 + 排序（得到与原始写法无关的稳定起点）
    premises = _dedup_sorted([_norm_args(p) for p in premises])
    conclusion = _norm_args(conclusion)

    # Step 2b: 按出现顺序重命名为 A,B,C…
    points = _collect_points([*premises, conclusion])
    if len(points) > 26:
        return None
    rename = {p: chr(ord("A") + i) for i, p in enumerate(points)}

    def apply_rename(pred: Pred) -> Pred:
        name, args = pred
        return _norm_args((name, tuple(rename.get(a, a) for a in args)))

    # Step 2c: 重命名后再规范化 + 排序定稿
    premises = _dedup_sorted([apply_rename(p) for p in premises])
    conclusion = apply_rename(conclusion)
    rule_text = build_rule_text(premises, conclusion)

    renamed_points = tuple(
        Point(name=rename[p], x=coord[p][0], y=coord[p][1])
        for p in points
        if p in coord
    )
    return NormalizedRule(
        rule_id=prop.proposition_id,
        seed=prop.seed,
        index_in_seed=prop.index_in_seed,
        rule_text=rule_text,
        rename_map={p: rename[p] for p in points},
        points=renamed_points,
    )


def _id_key(rule: NormalizedRule) -> tuple[int, int, int]:
    """序号排序键：(seed, index_in_seed, 根节点号)。"""
    seed = rule.seed if rule.seed is not None else 0
    root = rule.rule_id.split("#")[-1] if "#" in rule.rule_id else "0"
    try:
        root_n = int(root)
    except ValueError:
        root_n = 0
    return (seed, rule.index_in_seed, root_n)


# rule_text -> seed(str) -> 出现次数。溯源统计：记录去重前每条规范形分别来自
# 多少个不同 seed、每个 seed 里出现几次，供人工反查规则在源数据集中的泛化程度。
SeedOccurrences = dict[str, dict[str, int]]


def _record_occurrence(occurrences: SeedOccurrences, rule: NormalizedRule) -> None:
    seed_key = str(rule.seed) if rule.seed is not None else "null"
    bucket = occurrences.setdefault(rule.rule_text, {})
    bucket[seed_key] = bucket.get(seed_key, 0) + 1


def normalize_and_dedup(
    props: list[PropositionRecord],
) -> tuple[list[NormalizedRule], SeedOccurrences]:
    """规范化一批命题并按规范形去重（同形保留序号最小者）。

    同时统计去重前每条规范形 rule_text 分别来自多少个 seed、各出现几次
    （在合并 / 规约之前，反映原始命题层面的泛化程度）。
    """
    best: dict[str, NormalizedRule] = {}
    occurrences: SeedOccurrences = {}
    for prop in tqdm(props, desc="[part1] 规范化去重", unit="条"):
        rule = normalize_proposition(prop)
        if rule is None:
            continue
        _record_occurrence(occurrences, rule)
        cur = best.get(rule.rule_text)
        if cur is None or _id_key(rule) < _id_key(cur):
            best[rule.rule_text] = rule
    return sorted(best.values(), key=_id_key), occurrences


def _normalize_chunk(
    props: list[PropositionRecord],
) -> tuple[dict[str, NormalizedRule], SeedOccurrences]:
    """单个分块内规范化 + 局部去重（Ray worker 里跑，块内串行）。"""
    best: dict[str, NormalizedRule] = {}
    occurrences: SeedOccurrences = {}
    for prop in props:
        rule = normalize_proposition(prop)
        if rule is None:
            continue
        _record_occurrence(occurrences, rule)
        cur = best.get(rule.rule_text)
        if cur is None or _id_key(rule) < _id_key(cur):
            best[rule.rule_text] = rule
    return best, occurrences


def _merge_best(dicts: list[dict[str, NormalizedRule]]) -> dict[str, NormalizedRule]:
    best: dict[str, NormalizedRule] = {}
    for d in dicts:
        for text, rule in d.items():
            cur = best.get(text)
            if cur is None or _id_key(rule) < _id_key(cur):
                best[text] = rule
    return best


def _merge_occurrences(dicts: list[SeedOccurrences]) -> SeedOccurrences:
    merged: SeedOccurrences = {}
    for d in dicts:
        for rule_text, seed_counts in d.items():
            bucket = merged.setdefault(rule_text, {})
            for seed_key, count in seed_counts.items():
                bucket[seed_key] = bucket.get(seed_key, 0) + count
    return merged


def normalize_and_dedup_parallel(
    props: list[PropositionRecord],
    *,
    n_workers: int = 30,
    chunk_size: int = 5000,
) -> tuple[list[NormalizedRule], SeedOccurrences]:
    """normalize_and_dedup 的 Ray 并行版：分块规范化(各 worker 内先局部去重)，
    再在主进程合并各块结果做最终去重。

    每条命题的规范化互相独立，天然可分块并行；去重是 reduce 步骤，块内局部去重
    只是减少跨进程传输量，最终仍需在主进程按 _id_key 合并一次全局最小值。
    seed 出现次数统计同理按块局部汇总，再在主进程合并求和。
    """
    import ray

    from newclid.discovery.reduction.parallel import ensure_ray, run_bounded

    chunks = [props[i : i + chunk_size] for i in range(0, len(props), chunk_size)]
    if len(chunks) <= 1:
        return normalize_and_dedup(props)

    ensure_ray(n_workers)
    remote = ray.remote(_normalize_chunk)
    args = [(c,) for c in chunks]

    partials: list[dict[str, NormalizedRule]] = []
    occurrence_partials: list[SeedOccurrences] = []
    for chunk_best, chunk_occurrences in tqdm(
        run_bounded(remote, args, inflight=n_workers),
        total=len(chunks),
        desc="[part1] 规范化去重(并行)",
        unit="块",
    ):
        partials.append(chunk_best)
        occurrence_partials.append(chunk_occurrences)

    best = _merge_best(partials)
    occurrences = _merge_occurrences(occurrence_partials)
    return sorted(best.values(), key=_id_key), occurrences


def rule_to_output(rule: NormalizedRule) -> dict:
    """NormalizedRule 的落盘结构：规范文本 + 重命名映射 + 重命名后坐标 + 序号。"""
    return {
        "rule_id": rule.rule_id,
        "seed": rule.seed,
        "index_in_seed": rule.index_in_seed,
        "rule_text": rule.rule_text,
        "rename_map": rule.rename_map,
        "points": [{"name": p.name, "x": p.x, "y": p.y} for p in rule.points],
    }

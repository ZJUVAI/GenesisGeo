"""Seed 分组规约 + 前提合并（规约策略阶段 1、2）。

阶段1：按 seed 把规则分组，每组内做贪心累积规约；组间用 Ray 并行（每组一个任务）。
阶段2：合并前提完全一致的规则（p→g1, p→g2 ⇒ p→g1,g2）。

贪心规约 greedy_reduce 是组内/分块共用的核心：前提少者优先，逐条测"能否被已保留集
推出"，不能才保留 —— 得到一个基（basis）。
"""

from __future__ import annotations

from collections import defaultdict

from tqdm import tqdm

from newclid.discovery.reduction.parallel import ensure_ray, run_bounded
from newclid.discovery.reduction.subsumption_tester import RuleItem, SubsumptionTester


# ---------------------------------------------------------------------------
# 贪心累积规约（核心）
# ---------------------------------------------------------------------------


def greedy_reduce(
    rules: list[RuleItem],
    tester: SubsumptionTester,
    *,
    stage: str = "",
    audit: dict[str, dict] | None = None,
    extra_sources: list[RuleItem] | None = None,
) -> list[RuleItem]:
    """贪心累积求基：保留"无法被已保留规则集推出"的规则。

    顺序：前提数升序（更一般优先保留），再按 rule_id。互相可推的两条只留先处理的那条。

    第一条规则在被加入 basis 时，尚无其它规则可供测试（basis 为空），因此它永远
    不会被判定为冗余，即便后续加入的规则组合起来本可以推出它（例如它本不需要
    辅助点，只是恰好排在最前面）。收尾时用最终 basis 中的其它规则反过来测一次
    第一条规则，弥补这个不对称。

    extra_sources: 额外的、始终可用作 sources 但不参与本轮规约（不会被丢弃/
    合并、不计入返回值）的规则集合，通常是已验证过的历史规则库。每次
    is_derivable 判定时都附加在 basis 之后一起喂给 CSolver。

    audit: 若非 None，记录每条被丢弃规则的 {stage, dropped_by}（rule_id -> record）。
    """
    extra = extra_sources or []
    ordered = sorted(rules, key=lambda r: (r.premise_count, r.rule_id))
    basis: list[RuleItem] = []
    for r in ordered:
        if tester.is_derivable(r, basis + extra):
            if audit is not None:
                audit[r.rule_id] = {
                    "rule_text": r.rule_text,
                    "status": "dropped",
                    "stage": stage,
                    "dropped_by": [s.rule_id for s in basis + extra],
                }
            continue  # 可由已保留规则推出 → 冗余
        basis.append(r)

    if len(basis) > 1 and tester.is_derivable(basis[0], basis[1:] + extra):
        if audit is not None:
            audit[basis[0].rule_id] = {
                "rule_text": basis[0].rule_text,
                "status": "dropped",
                "stage": f"{stage}#tailcheck",
                "dropped_by": [s.rule_id for s in basis[1:] + extra],
            }
        basis.pop(0)

    return basis


def leave_one_out_reduce(
    rules: list[RuleItem],
    tester: SubsumptionTester,
    *,
    stage: str = "",
    audit: dict[str, dict] | None = None,
    extra_sources: list[RuleItem] | None = None,
) -> list[RuleItem]:
    """留一法规约：每条规则用"其余全部规则"（而非仅已保留的 basis）测试可推导性。

    与 greedy_reduce 的区别：greedy_reduce 只拿"已经进入 basis 的规则"作为 sources，
    一条规则一旦先被保留就再也不会被后加入的规则组合反推掉（例如规则3、4合力才能
    推出规则2，但2先于3、4被处理入 basis 的情况）。这里改为对每条规则用当前存活集合
    中除它自己外的全部规则去测，能推出就立即删除（而非批量删除，避免"A、B 互相可
    由含对方的集合推出"导致两条同时被误删）。

    只扫描一轮：is_derivable 在理想情形下单调（sources 越多越容易判定冗余），
    一轮扫描中规则被测试时看到的 sources 只会比最终存活集合更大，故不会出现
    "这轮没删、下一轮才被删"的情况——除非 CSolver 因 sources 过多而超时，保守
    返回不可推导（见 is_derivable 的 except 分支），那属于工程限制，不再用外层
    循环兜底。

    代价随存活集合大小增长（每次测试的 sources 数 ≈ 存活规则数），只适合较小的块
    （如分治规约的 chunk，而非未分块的整个 seed 组）。

    extra_sources: 同 greedy_reduce，始终附加在 sources 里、不参与本轮规约的
    额外规则集合。
    """
    extra = extra_sources or []
    current = sorted(rules, key=lambda r: (r.premise_count, r.rule_id))
    i = 0
    while i < len(current):
        target = current[i]
        others = current[:i] + current[i + 1 :]
        if tester.is_derivable(target, others + extra):
            if audit is not None:
                audit[target.rule_id] = {
                    "rule_text": target.rule_text,
                    "status": "dropped",
                    "stage": stage,
                    "dropped_by": [s.rule_id for s in others + extra],
                }
            current.pop(i)
            continue
        i += 1
    return current


# ---------------------------------------------------------------------------
# 阶段 2a：合并前提一致的规则前，先组内规约一次
# ---------------------------------------------------------------------------


def _premise_key(r: RuleItem) -> tuple:
    """前提集合的规范键（无序）。"""
    return tuple(sorted(f"{n} {' '.join(a)}" for n, a in r.premises))


def _reduce_premise_group(
    rules: list[RuleItem],
    seed: int,
    config_path: str | None,
    extra_sources: list[RuleItem] | None = None,
) -> tuple[list[RuleItem], dict[str, dict]]:
    """单个"前提一致"组内规约（组内规模小，用留一法在 worker 内串行执行）。"""
    if len(rules) <= 1:
        return rules, {}
    tester = SubsumptionTester(seed=seed, config_path=config_path)
    audit: dict[str, dict] = {}
    survivors = leave_one_out_reduce(
        rules,
        tester,
        stage="premise_group_reduce",
        audit=audit,
        extra_sources=extra_sources,
    )
    return survivors, audit


def reduce_by_premise_group(
    rules: list[RuleItem],
    *,
    n_workers: int = 30,
    use_ray: bool = True,
    config_path: str | None = None,
    seed: int = 42,
    extra_sources: list[RuleItem] | None = None,
) -> tuple[list[RuleItem], dict[str, dict]]:
    """阶段 2a：按前提集合分组，组内规约一次；组间并行。

    分治规约（divide_conquer_reducer）打乱分块，同前提的规则未必被分到同一块
    互相比较过；这里按 _premise_key 精确分组，专门补一次组内 subsumption 判定，
    使随后 merge_same_premise 合并时组内已不存在冗余结论。仅做组内互相判定，
    不改变规则的前提/结论文本，因此不影响其可追溯性（rule_id/seed 不变）。
    """
    groups: dict[tuple, list[RuleItem]] = defaultdict(list)
    for r in rules:
        groups[_premise_key(r)].append(r)
    multi_groups = {k: g for k, g in groups.items() if len(g) > 1}
    print(
        f"[premise_group_reduce] {len(rules)} 条规则 -> {len(groups)} 个前提组"
        f"（其中 {len(multi_groups)} 组需要组内规约）"
    )

    if not multi_groups:
        return rules, {}

    audit: dict[str, dict] = {}
    survivors: list[RuleItem] = []
    single_groups = [g[0] for k, g in groups.items() if k not in multi_groups]

    if use_ray and len(multi_groups) > 1:
        import ray

        ensure_ray(n_workers)
        remote = ray.remote(_reduce_premise_group)
        args = [(g, seed, config_path, extra_sources) for g in multi_groups.values()]
        for group_survivors, group_audit in tqdm(
            run_bounded(remote, args, inflight=n_workers),
            total=len(multi_groups),
            desc="[part2] 前提组内规约",
            unit="组",
        ):
            survivors.extend(group_survivors)
            audit.update(group_audit)
    else:
        for g in tqdm(multi_groups.values(), desc="[part2] 前提组内规约", unit="组"):
            group_survivors, group_audit = _reduce_premise_group(
                g,
                seed,
                config_path,
                extra_sources=extra_sources,
            )
            survivors.extend(group_survivors)
            audit.update(group_audit)

    survivors.extend(single_groups)
    print(f"[premise_group_reduce] 组内规约后剩 {len(survivors)} 条")
    return survivors, audit


# ---------------------------------------------------------------------------
# 阶段 2：合并前提一致的规则
# ---------------------------------------------------------------------------


def _conclusion_symmetry_key(concl_text: str) -> tuple:
    """结论文本 -> 按谓词对称性规范化后的 (name, args) 键，供去重使用。

    "perp B C D E" 和 "perp C B D E" 描述的是同一个几何关系（线段方向不影响
    垂直/平行/相等关系），字符串比较区分不出来，会导致 merge_same_premise
    把它们当成两个不同结论保留在同一条多结论规则里——见 custom_00006
    (ncoll B C D, perp B D C E, perp B E C D => perp C B D E, perp B C D E)。
    用 utils.symmetry.normalize_predicate 规范化后再比较即可识别出重复。
    未知谓词（normalize_predicate 会抛 ValueError）原样返回，不做规范化。
    """
    from newclid.discovery.data_models import PredicateInstance
    from newclid.discovery.utils.rule_parser import parse_predicate
    from newclid.discovery.utils.symmetry import normalize_predicate

    name, args = parse_predicate(concl_text)
    try:
        norm_args = normalize_predicate(PredicateInstance(predicate=name, args=args))
    except ValueError:
        norm_args = args
    return (name, tuple(norm_args))


def merge_same_premise(rules: list[RuleItem]) -> tuple[list[RuleItem], dict[str, dict]]:
    """把前提完全一致的规则合并为一条多结论规则（p→g1, p→g2 ⇒ p→g1,g2）。

    注意：应在规约**之后**做——若在规约前合并，会把多个结论耦合，导致无法单独
    丢弃其中可被推出的结论，削弱规约能力。
    """
    groups: dict[tuple, list[RuleItem]] = defaultdict(list)
    for r in rules:
        groups[_premise_key(r)].append(r)

    audit: dict[str, dict] = {}
    merged: list[RuleItem] = []
    for _key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        rep = min(group, key=lambda r: (r.seed or 0, r.index_in_seed, r.rule_id))
        lhs = rep.rule_text.split("=>", 1)[0].strip()
        # 收集去重后的所有结论 —— 按对称性规范化后的键去重，而不是原始
        # 字符串，避免 "perp B C D E" / "perp C B D E" 这类语义相同、
        # 参数顺序不同的写法都被当成独立结论保留。
        concls: list[str] = []
        seen_keys: set[tuple] = set()
        for r in group:
            c = r.rule_text.split("=>", 1)[1].strip()
            key = _conclusion_symmetry_key(c)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            concls.append(c)
        rep.rule_text = f"{lhs} => {', '.join(concls)}"
        merged.append(rep)
        for r in group:
            if r.rule_id != rep.rule_id:
                audit[r.rule_id] = {
                    "rule_text": r.rule_text,
                    "status": "merged",
                    "stage": "merge_same_premise",
                    "merged_into": rep.rule_id,
                }
    return merged, audit


# ---------------------------------------------------------------------------
# 阶段 1：按 seed 分组规约（Ray 并行）
# ---------------------------------------------------------------------------


def _reduce_group(
    rules: list[RuleItem],
    seed: int,
    config_path: str | None,
    extra_sources: list[RuleItem] | None = None,
) -> tuple[list[RuleItem], dict[str, dict]]:
    """单个 seed 组的规约（在 worker 内串行执行 greedy_reduce）。"""
    tester = SubsumptionTester(seed=seed, config_path=config_path)
    audit: dict[str, dict] = {}
    survivors = greedy_reduce(
        rules,
        tester,
        stage="seed_reduce",
        audit=audit,
        extra_sources=extra_sources,
    )
    return survivors, audit


def reduce_by_seed(
    rules: list[RuleItem],
    *,
    n_workers: int = 30,
    use_ray: bool = True,
    config_path: str | None = None,
    extra_sources: list[RuleItem] | None = None,
) -> tuple[list[RuleItem], dict[str, dict]]:
    """阶段 1：按 seed 分组，组内规约；组间并行。返回所有组的幸存者。

    extra_sources: 额外的、始终可用作 sources 但不参与规约的规则集合
    （见 greedy_reduce），每个 seed 组都会附加同一份 extra_sources。
    """
    groups: dict[object, list[RuleItem]] = defaultdict(list)
    for r in rules:
        groups[r.seed].append(r)
    print(f"[seed_reduce] {len(rules)} 条规则 -> {len(groups)} 个 seed 组")

    audit: dict[str, dict] = {}
    survivors: list[RuleItem] = []
    if use_ray and len(groups) > 1:
        import ray

        ensure_ray(n_workers)
        remote = ray.remote(_reduce_group)
        args = [
            (g, (seed if isinstance(seed, int) else 0), config_path, extra_sources)
            for seed, g in groups.items()
        ]
        for group_survivors, group_audit in tqdm(
            run_bounded(remote, args, inflight=n_workers),
            total=len(groups),
            desc="[part2] seed 分组规约",
            unit="组",
        ):
            survivors.extend(group_survivors)
            audit.update(group_audit)
    else:
        for seed, g in tqdm(groups.items(), desc="[part2] seed 分组规约", unit="组"):
            group_survivors, group_audit = _reduce_group(
                g,
                seed if isinstance(seed, int) else 0,
                config_path,
                extra_sources=extra_sources,
            )
            survivors.extend(group_survivors)
            audit.update(group_audit)

    print(f"[seed_reduce] 组内规约后剩 {len(survivors)} 条")
    return survivors, audit

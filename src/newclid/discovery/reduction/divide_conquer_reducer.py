"""分治规约（规约策略阶段 3）。

对 seed 组规约后剩下的全部规则做跨组规约：分块 → 每块内 greedy 规约（Ray 并行）→
汇总幸存者 → 若数量仍显著下降则重新打乱分块迭代，直到稳定（不动点）。

分块打散了 seed 边界，能消除"不同 seed 却互推"的冗余。块内串行 greedy，块间并行。
（更激进的两两 pipelined merge 见 git 历史的老实现；此处先用简明的迭代分块版。）
"""

from __future__ import annotations

import random

from tqdm import tqdm

from newclid.discovery.reduction.parallel import ensure_ray, run_bounded
from newclid.discovery.reduction.seed_reducer import greedy_reduce, leave_one_out_reduce
from newclid.discovery.reduction.subsumption_tester import RuleItem, SubsumptionTester


def _reduce_chunk(
    rules: list[RuleItem],
    seed: int,
    config_path: str | None,
    stage: str = "divide_conquer",
    extra_sources: list[RuleItem] | None = None,
) -> tuple[list[RuleItem], dict[str, dict]]:
    tester = SubsumptionTester(seed=seed, config_path=config_path)
    audit: dict[str, dict] = {}
    basis = greedy_reduce(
        rules, tester, stage=stage, audit=audit, extra_sources=extra_sources
    )
    survivors = leave_one_out_reduce(
        basis,
        tester,
        stage=stage,
        audit=audit,
        extra_sources=extra_sources,
    )
    return survivors, audit


def _split(rules: list[RuleItem], chunk_size: int) -> list[list[RuleItem]]:
    return [rules[i : i + chunk_size] for i in range(0, len(rules), chunk_size)]


def reduce(
    rules: list[RuleItem],
    *,
    chunk_size: int = 200,
    n_workers: int = 30,
    use_ray: bool = True,
    max_rounds: int = 20,
    config_path: str | None = None,
    seed: int = 42,
    extra_sources: list[RuleItem] | None = None,
) -> tuple[list[RuleItem], dict[str, dict]]:
    """分块迭代规约至稳定。

    每轮：打乱顺序 → 分块 → 块内 greedy（并行）→ 汇总。若幸存/上轮 > shrink_ratio
    （几乎不再减少）或到 max_rounds 则停止。规则数 <= chunk_size 时单块一次收敛。

    extra_sources: 额外的、始终可用作 sources 但不参与规约的规则集合
    （见 seed_reducer.greedy_reduce），每个 chunk 都会附加同一份 extra_sources。

    audit 按轮次记录 stage="divide_conquer_round_{rnd}"；同一条规则若跨轮多次
    被判冗余（不应发生，规约后规模只减不增），以最后一次记录为准。
    """
    current = list(rules)
    audit: dict[str, dict] = {}
    rng = random.Random(seed)
    for rnd in range(max_rounds):
        before = len(current)
        stage = f"divide_conquer_round_{rnd}"
        if before <= chunk_size:
            current, round_audit = _reduce_chunk(
                current,
                seed,
                config_path,
                stage=stage,
                extra_sources=extra_sources,
            )
            audit.update(round_audit)
            print(f"[divide_conquer] 轮{rnd}: 单块 {before} -> {len(current)}")
            break

        rng.shuffle(current)
        chunks = _split(current, chunk_size)
        survivors: list[RuleItem] = []
        if use_ray and len(chunks) > 1:
            import ray

            ensure_ray(n_workers)
            remote = ray.remote(_reduce_chunk)
            args = [(c, seed, config_path, stage, extra_sources) for c in chunks]
            for chunk_survivors, chunk_audit in tqdm(
                run_bounded(remote, args, inflight=n_workers),
                total=len(chunks),
                desc=f"[part2] 分治规约 轮{rnd}",
                unit="块",
            ):
                survivors.extend(chunk_survivors)
                audit.update(chunk_audit)
        else:
            for c in tqdm(chunks, desc=f"[part2] 分治规约 轮{rnd}", unit="块"):
                chunk_survivors, chunk_audit = _reduce_chunk(
                    c,
                    seed,
                    config_path,
                    stage=stage,
                    extra_sources=extra_sources,
                )
                survivors.extend(chunk_survivors)
                audit.update(chunk_audit)

        current = survivors
        print(
            f"[divide_conquer] 轮{rnd}: {before} -> {len(current)} ({len(chunks)} 块)"
        )

    return current, audit

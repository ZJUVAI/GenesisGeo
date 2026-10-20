"""自动发现区分谓词：多 seed 重建 + 全局反例搜索 + 候选筛选。

思路:
  1. 从 fl_problem 用不同 seed 重建多组坐标
  2. 对每组坐标，做大幅扰动的全局搜索，找"前提成立但结论不成立"的配置
  3. 如找到反例，生成所有候选谓词，筛选一致区分的

供 reduction.orchestrator._run_ndg_stage 直接 import 调用（discover_all /
save_discover_results），不提供独立 CLI；参数（rules_file / n_seeds / n_workers 等）
统一走 discovery_config.json 的 part2_reduction.ndg 段。
"""

from __future__ import annotations

import json
import os
import re
import signal
import time
import zlib
from itertools import combinations, permutations

import numpy as np

from newclid.api import GeometricSolverBuilder
from newclid.discovery.validation.rule_tracer import RuleTracer
from newclid.discovery.validation.counterexample_search import (
    evaluate_predicate as _eval_pred,
    CounterexampleFinder,
    _DDAR_REL_TOL,
)
from newclid.numerical.geometries import PointNum
from newclid.discovery.utils.rule_parser import parse_predicate, split_rule_text


# ============================================================================
# Multi-seed rebuild
# ============================================================================

def rebuild_with_seeds(fl_problem: str, rename_map: dict, seeds: list[int],
                       max_attempts: int = 500, strip_goal: bool = True) -> list[dict]:
    """Rebuild problem with different seeds.

    If strip_goal=True: remove the goal from fl_problem before building.
    This allows the construction to produce BOTH branches — some seeds will
    naturally produce configurations where the conclusion holds, others
    where it doesn't.  Both are valid counterexample sources.
    """
    fl_no_coords = re.sub(r'@-?[0-9.]+(?:_-?[0-9.]+)?', '', fl_problem)

    # Strip goal if requested: "constructions ? goal" → "constructions"
    if strip_goal and "?" in fl_no_coords:
        fl_no_coords = fl_no_coords.split("?")[0].strip()

    coord_sets = []
    for seed in seeds:
        try:
            solver = (GeometricSolverBuilder(seed=seed)
                      .load_problem_from_txt(fl_no_coords)
                      .build(max_attempts=max_attempts))
            sg = solver.proof.symbols_graph
            pts = {}
            for on, nn in rename_map.items():
                if on in sg.name2node:
                    p = sg.name2node[on].num
                    pts[nn] = PointNum(p.x, p.y)
            if len(pts) == len(rename_map):
                coord_sets.append({"seed": seed, "points": pts})
        except Exception:
            continue
    return coord_sets


# ============================================================================
# Aggressive global counterexample search
# ============================================================================

def _find_distant_ce(
    premises: list[tuple[str, list[str]]],
    conclusions: list[tuple[str, list[str]]],
    nominal: dict[str, PointNum],
    n_trials: int = 5,
    base_seed: int = 42,
) -> dict[str, PointNum] | None:
    """Aggressive global search for a DISTANT counterexample.

    Uses very large perturbations (up to 20x coordinate range) to hop
    to entirely different branches of the premise manifold.

    base_seed must be deterministic per call site (e.g. derived from the
    rule_id) — using time.time() here previously made whether a rule
    survived NDG depend on the wall-clock moment discover() happened to run,
    so the same rule could be kept or dropped on different batch runs with
    no code or data change.
    """
    for trial in range(n_trials):
        # Each trial: start from perturbed nominal, optimize
        rng = np.random.RandomState((base_seed + trial * 137) % (2**32))
        # Build perturbation: mix of random directions with large scale
        x0 = np.zeros(2 * len(nominal))
        point_names = sorted(nominal.keys())
        # Large noise
        for i, name in enumerate(point_names):
            nom = nominal[name]
            # Scale proportional to coordinate range
            noise_scale = 2.0 + rng.random() * 8.0  # 2-10x
            x0[2*i] = rng.normal(0, noise_scale)
            x0[2*i+1] = rng.normal(0, noise_scale)

        # Optimize
        finder = CounterexampleFinder(premises, conclusions, nominal)
        # Force global-scale search by overriding perturbation
        try:
            result = finder.search(max_restarts=150, random_seed=rng.randint(0, 2**31))
            if result.get("counterexample_found"):
                ce_raw = result["counterexample_points"]
                ce_pts = {n: PointNum(x, y) for n, (x, y) in ce_raw.items()}
                # Check it's genuinely different (>1% in any coord)
                max_diff = 0.0
                for n in ce_pts:
                    d = abs(ce_pts[n] - nominal[n])
                    max_diff = max(max_diff, d)
                if max_diff > 0.01:
                    return ce_pts
        except Exception:
            continue

    return None


# ============================================================================
# Candidate predicates
# ============================================================================

def generate_all_candidates(point_names: list[str]) -> list[tuple[str, list[str]]]:
    candidates = []
    for a, b, c in permutations(point_names, 3):
        candidates.append(("obtuse", [a, b, c]))
        candidates.append(("acute", [a, b, c]))
    triples = list(combinations(point_names, 3))
    for (a, b, c), (d, e, f) in combinations(triples, 2):
        if len({a, b, c, d, e, f}) >= 4:
            candidates.append(("sameclock", [a, b, c, d, e, f]))
    for a, b, c in combinations(point_names, 3):
        candidates.append(("ncoll", [a, b, c]))
    # sameside A B C X Y Z: A vs. B->C同侧 X vs. Y->Z. 复用sameclock已有的
    # "两组三点各自互不相同、六点里至少4个不同"枚举方式(triple组合成对)，
    # 只是sameside的两个三元组内部有(顶点, 边起点, 边终点)的角色区分——枚举
    # 全部有序三元组(permutations)覆盖所有角色分配，交给counterexample_search
    # 的数值评估和symmetry.py的_normalize_sameside在下游做等价折叠。
    ordered_triples = list(permutations(point_names, 3))
    for (a, b, c), (x, y, z) in combinations(ordered_triples, 2):
        if len({a, b, c, x, y, z}) >= 4:
            candidates.append(("sameside", [a, b, c, x, y, z]))
            candidates.append(("nsameside", [a, b, c, x, y, z]))
    segs = list(combinations(point_names, 2))
    for (a, b), (c, d) in combinations(segs, 2):
        if len({a, b, c, d}) >= 3:
            candidates.append(("perp", [a, b, c, d]))
            candidates.append(("nperp", [a, b, c, d]))
    for (a, b), (c, d) in combinations(segs, 2):
        if len({a, b, c, d}) >= 3:
            candidates.append(("para", [a, b, c, d]))
            candidates.append(("npara", [a, b, c, d]))
    return candidates


# ============================================================================
# Main discovery
# ============================================================================

def discover(rule_id: str, rule_text: str, rename_map: dict,
             fl_problem: str, n_seeds: int = 8,
             n_ce_trials: int = 5, min_good_ratio: float = 0.0) -> dict:
    """Discover distinguishing predicates for one rule.

    min_good_ratio: 在满足前提的采样配置中, 结论成立(good)的占比若低于该阈值
        (即绝大多数采样都是 bad, 只有零星 good), 认为这条规则大概率本身就是
        错的、这几个 good 只是偶然凑巧, 不值得费力找一个 guard 把它们圈出来
        —— 直接判定 status="mostly_bad" 丢弃, 不进入候选谓词搜索。
        0 或负数关闭此检查(恢复原始行为: 只要能找到一致区分谓词就保留)。
    """
    prem_strs, concl_str = split_rule_text(rule_text)
    premises = [(n, list(a)) for n, a in (parse_predicate(p) for p in prem_strs if p.strip())]
    conclusions = [(n, list(a)) for c in concl_str.split(",") if c.strip()
                   for n, a in [parse_predicate(c.strip())]]
    point_names = sorted(rename_map.values())

    # Helper: check that a predicate is SATISFIED WITH MARGIN
    _PREM_MARGIN = _DDAR_REL_TOL * 0.1

    def _prem_ok(pts: dict) -> bool:
        """Premises must be truly satisfied (violation ≈ 0)."""
        return all(
            _eval_pred(n, a, pts)[0] and _eval_pred(n, a, pts)[1] <= _PREM_MARGIN
            for n, a in premises
        )

    _CE_MARGIN = _DDAR_REL_TOL * 5.0

    def _concl_clearly_bad(pts: dict) -> bool:
        """Conclusion must be clearly violated (violation ≥ 5× DDAR tolerance)."""
        results = [_eval_pred(n, a, pts) for n, a in conclusions]
        return any(not ok and viol >= _CE_MARGIN for ok, viol in results)

    bad_configs = []  # list of (points_dict, source_description)
    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768][:n_seeds]

    # =====================================================================
    # Step 1: Reverse construction (prioritized) — numerically build the
    # premises directly from the rule text, no dependence on the original
    # fl_problem/proof.  Handles both random sampling and (via
    # try_reverse_construction's internal degenerate-witness refinement)
    # measure-zero degenerate branches that random sampling would almost
    # never hit on its own.
    # =====================================================================
    from newclid.discovery.reverse_construction import try_reverse_construction

    # rename_map from RuleTracer maps FL-names → rule-names (e.g., 'a'→'A').
    # try_reverse_construction needs rule-names → FL-names.  Invert it.
    rev_rename = {v: k for k, v in rename_map.items()}
    # Use a larger sample count than n_seeds for the degeneracy signal itself:
    # min_degeneracy only reliably approaches its true (possibly near-zero)
    # value with enough samples — too few can mask a real degenerate branch
    # by chance (see reverse_construction._line_line_degeneracy).  150 is
    # empirically motivated: on "para A B C D, para A C B D => cong A B C D"
    # (missing an ncoll A B C guard — see reverse_construction module
    # docstring), the min degeneracy across free-point samples only crosses
    # try_reverse_construction's refine-probe threshold (0.05) somewhere
    # between 60 and 100 samples (0.076 at 60, 0.029 at 100); 60 alone
    # silently missed this rule's degenerate branch and let it through.
    rev_result = try_reverse_construction(premises, conclusions, rev_rename,
                                          n_seeds=max(n_seeds, 150))
    min_degeneracy = rev_result.get("min_degeneracy") if rev_result else None

    good_sets = []
    if rev_result is not None:
        for bc in rev_result.get("bad_configs", []):
            pts = {n: PointNum(x, y) for n, (x, y) in bc["points"].items()}
            if _prem_ok(pts) and _concl_clearly_bad(pts):
                bad_configs.append((pts, f"reverse_construction(seed={bc['seed']})"))
        for gc in rev_result.get("good_configs", []):
            pts = {n: PointNum(x, y) for n, (x, y) in gc["points"].items()}
            if _prem_ok(pts):
                good_sets.append({"seed": gc["seed"], "points": pts})

    if rev_result is not None and not bad_configs:
        # Translation succeeded but sampling found no counterexample.  This
        # branch is decided purely by min_degeneracy — no further fallback
        # strategies are run (measured empirically to have zero unique
        # contribution here: on the 250k_new2 rule set, Strategy A/B/C never
        # found a counterexample once reverse construction had already
        # translated the rule and sampled it without finding one — the
        # earlier ~minutes-per-rule cost of running them anyway, e.g. on
        # custom_00045's structural neighbors, bought nothing).  Prefer
        # dropping a possibly-good rule over risking keeping a bad one:
        # ndg_apply.py's degeneracy_threshold makes the actual accept/reject
        # call from min_degeneracy.
        return {"status": "pass", "rule_id": rule_id, "rule_text": rule_text,
                "n_good": len(good_sets), "n_bad": 0,
                "min_degeneracy": min_degeneracy}

    if rev_result is None:
        # =================================================================
        # Step 2: translation failed — reverse construction couldn't even
        # build a numeric sequence for these premises (no min_degeneracy
        # signal available).  Fall back to rebuilding the rule's own
        # original fl_problem with fresh seeds.
        # =================================================================
        # Strategy B: build WITHOUT the goal — lets different seeds land on
        # both sides of the conclusion naturally.
        bad_seeds = [s + 10000 for s in seeds]
        bad_sets_raw = rebuild_with_seeds(fl_problem, rename_map, bad_seeds, strip_goal=True)
        for gs in bad_sets_raw:
            pts = gs["points"]
            if _prem_ok(pts) and _concl_clearly_bad(pts):
                bad_configs.append((pts, f"no_goal_build(seed={gs['seed']})"))

        if len(good_sets) < 2:
            # strip_goal=False: a "good" configuration must actually satisfy the
            # conclusion.  Building WITHOUT the goal (the old inherited default)
            # let premise-satisfying counterexamples into good_sets, which makes
            # the pair test below unsatisfiable by construction — surviving
            # requires a candidate to differ on every (good, bad) pair AND agree
            # across all goods, and a good that violates the conclusion behaves
            # like just another bad.  That silently produced n_surviving=0 /
            # no_valid_candidate for every rule reaching this branch.
            # Step 1's try_reverse_construction classifies good vs bad with an
            # explicit conclusion check (reverse_construction.py), so this
            # restores the intended, consistent semantics.
            good_sets = rebuild_with_seeds(fl_problem, rename_map, seeds,
                                           strip_goal=False)

        # Strategy C (cheap): only when Strategy B also found nothing, one
        # last global-search attempt on a SINGLE good config (not looped
        # over every good_set, unlike the original) — the original cost
        # tens of minutes on rules like 366815:8#041 for zero payoff on
        # most rules, but it did have 2/51 unique hits in the same test, so
        # it's worth one attempt rather than dropping straight to
        # "insufficient".  n_ce_trials still controls how many random
        # restarts that single attempt gets internally.
        if not bad_configs and good_sets:
            # Deterministic per-rule seed: same rule_id always probes the same
            # perturbations, so a rule's survival can't flip between batch
            # runs depending on wall-clock time (see _find_distant_ce docstring).
            ce_base_seed = zlib.crc32(rule_id.encode("utf-8")) % (2**31)
            ce_pts = _find_distant_ce(premises, conclusions, good_sets[0]["points"],
                                      n_trials=n_ce_trials, base_seed=ce_base_seed)
            if ce_pts is not None and _prem_ok(ce_pts) and _concl_clearly_bad(ce_pts):
                bad_configs.append((ce_pts, f"cheap_global_search(seed={good_sets[0]['seed']})"))

        if len(good_sets) < 2:
            return {"status": "insufficient_good_sets", "rule_id": rule_id,
                    "rule_text": rule_text, "n_good": len(good_sets),
                    "min_degeneracy": min_degeneracy}

        if not bad_configs:
            # Translation failed AND every fallback found nothing — unlike
            # the "translation succeeded, no counterexample" branch above,
            # there's no min_degeneracy signal here to distinguish "this
            # really does look degenerate" from "genuinely holds
            # everywhere".  With zero positive evidence either way, default
            # to dropping rather than treating it as "pass" (which
            # ndg_apply.py would otherwise keep unconditionally when
            # min_degeneracy is None) — same bias towards not letting a bad
            # theorem slip through applied consistently.
            return {"status": "no_counterexample_no_degeneracy_signal",
                    "rule_id": rule_id, "rule_text": rule_text,
                    "n_good": len(good_sets), "n_bad": 0}

    # 绝大多数满足前提的采样都是 bad、只有零星 good：这不是"需要一个 guard
    # 圈出适用范围"的情形，而是规则本身大概率就是错的，这几个 good 只是偶然
    # 凑巧撞上的特例。此时不再进入候选谓词搜索（搜索出的谓词也只是把这几个
    # 罕见的 good 硬圈出来，guard 本身没有几何意义），直接判定丢弃。
    # ndg_apply.py 对非 pass/success 的状态一律 reason=status 丢弃，此状态
    # 无需额外改动即可被正确处理。
    n_good_total = len(good_sets)
    n_bad_total = len(bad_configs)
    if min_good_ratio > 0 and n_good_total + n_bad_total > 0:
        good_ratio = n_good_total / (n_good_total + n_bad_total)
        if good_ratio < min_good_ratio:
            return {"status": "mostly_bad", "rule_id": rule_id,
                    "rule_text": rule_text,
                    "n_good": n_good_total, "n_bad": n_bad_total,
                    "good_ratio": good_ratio,
                    "min_degeneracy": min_degeneracy}

    # Normalize all point sets to unit scale before pair testing.
    # Shifts centroid to origin and scales average distance to 1.0,
    # making the DDAR relative tolerance (0.001) meaningful regardless
    # of the original coordinate magnitude.
    def _normalize_pts(pts: dict) -> dict:
        cx = sum(p.x for p in pts.values()) / max(len(pts), 1)
        cy = sum(p.y for p in pts.values()) / max(len(pts), 1)
        shifted = {k: PointNum(v.x - cx, v.y - cy) for k, v in pts.items()}
        avg_r = sum((p.x**2 + p.y**2)**0.5 for p in shifted.values()) / max(len(shifted), 1)
        if avg_r < 1e-12:
            return pts
        return {k: PointNum(v.x / avg_r, v.y / avg_r) for k, v in shifted.items()}

    for gs in good_sets:
        gs["points"] = _normalize_pts(gs["points"])
    new_bad = []
    for pts, desc in bad_configs:
        new_bad.append((_normalize_pts(pts), desc))
    bad_configs = new_bad

    all_pairs = []
    for gs in good_sets:
        for bc in bad_configs:
            all_pairs.append((gs["points"], bc[0], gs["seed"]))

    if len(all_pairs) < 2:
        return {"status": "insufficient_pairs", "rule_id": rule_id,
                "rule_text": rule_text,
                "n_good": len(good_sets), "n_bad": len(bad_configs),
                "n_pairs": len(all_pairs),
                "min_degeneracy": min_degeneracy}

    # Step 3: generate + filter candidates
    candidates = generate_all_candidates(point_names)
    surviving = []
    for pred_name, args in candidates:
        consistent = True
        all_good_true = None
        for good_pts, bad_pts, _seed in all_pairs:
            good_ok, _ = _eval_pred(pred_name, args, good_pts)
            bad_ok, _ = _eval_pred(pred_name, args, bad_pts)
            if good_ok == bad_ok:
                consistent = False
                break
            if all_good_true is None:
                all_good_true = good_ok
            elif all_good_true != good_ok:
                consistent = False
                break
        if consistent:
            surviving.append({
                "predicate": f"{pred_name} {' '.join(args)}",
                "true_at_good": all_good_true,
                "n_pairs_tested": len(all_pairs),
            })

    return {
        "status": "success",
        "rule_id": rule_id,
        "rule_text": rule_text,
        "n_good_sets": len(good_sets),
        "n_pairs": len(all_pairs),
        "n_candidates": len(candidates),
        "n_surviving": len(surviving),
        "surviving_predicates": surviving,
        "min_degeneracy": min_degeneracy,
    }


class _RuleTimeout(Exception):
    """Raised inside the SIGALRM handler when a single rule's discover() call
    runs past the per-rule time budget."""


def _discover_with_timeout(timeout_seconds: float | None, *args, **kwargs) -> dict:
    """Run discover(*args, **kwargs), aborting and returning a timeout status
    if it takes longer than timeout_seconds.

    Uses SIGALRM rather than a thread/process timeout because discover()'s
    slow paths (scipy.optimize inside _find_distant_ce /
    refine_degenerate_witness, GeometricSolverBuilder rebuilds) are plain
    blocking calls with no cooperative cancellation point — SIGALRM is the
    only mechanism that can interrupt them without restructuring discover()
    itself.  Only safe from the main thread of the main interpreter (true
    for both discover_all's serial loop and each Ray worker process, since
    each worker is its own single-threaded process).  timeout_seconds=None
    or <=0 disables the timeout (matches historical unbounded behavior).
    """
    if not timeout_seconds or timeout_seconds <= 0:
        return discover(*args, **kwargs)

    def _handler(signum, frame):
        raise _RuleTimeout()

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return discover(*args, **kwargs)
    except _RuleTimeout:
        rule_id = args[0] if args else kwargs.get("rule_id", "?")
        rule_text = args[1] if len(args) > 1 else kwargs.get("rule_text", "")
        return {"status": "timeout", "rule_id": rule_id, "rule_text": rule_text,
                "timeout_seconds": timeout_seconds}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


# ============================================================================
# Driver: 对整份规则文件跑 discover()（供 pipeline 直接 import 调用）
# ============================================================================

def discover_all(
    rules_file: str,
    normalized_rules_path: str,
    occurrences_path: str,
    source_dataset_path: str,
    n_seeds: int = 10,
    n_ce_trials: int = 3,
    n_workers: int = 1,
    limit: int | None = None,
    rule_id: str | None = None,
    rule_timeout_seconds: float = 120.0,
    min_good_ratio: float = 0.0,
) -> list[dict]:
    """对 rules_file 中的每条规则跑 discover()，返回结果列表。

    n_workers > 1 时用 Ray 并行（每个 worker 内部重建 RuleTracer，因其持有
    文件句柄，不能跨进程序列化传递）。

    rule_timeout_seconds: 单条规则处理超过该时长（默认 120s）就放弃这条规则，
        返回 status="timeout" 而不是无限期等下去 —— 少数规则（尤其翻译失败、
        走 Strategy B/廉价 Strategy C fallback 的那一支）在某些前提组合下可能
        触发很慢的 CSolver 重建或 scipy 全局搜索。ndg_apply.py 把非 pass/
        success 状态一律丢弃，所以超时的规则会被丢弃而不是卡住整条流水线。
        设为 0 或 None 关闭超时（恢复无限期等待）。

    min_good_ratio: 见 discover() 同名参数——good/(good+bad) 采样占比低于此值
        视为规则本身大概率错误，直接丢弃(status="mostly_bad")而不再
        搜索候选 guard。0 关闭此检查。
    """
    print("[ndg_discovery] Building tracer indexes...")
    tracer = RuleTracer(
        normalized_rules_path=normalized_rules_path,
        occurrences_path=occurrences_path,
        source_dataset_path=source_dataset_path,
    )
    tracer.build()

    def process_rule(rid, rule_text):
        norm_rec = tracer.get_norm_record(rid)
        if not norm_rec:
            return {"status": "error", "rule_id": rid, "message": "norm record not found"}
        rename_map = norm_rec.get("rename_map", {})
        src = tracer.get_source_record(norm_rec.get("seed"), norm_rec.get("index_in_seed", 0))
        if not src:
            return {"status": "error", "rule_id": rid, "message": "source record not found"}
        return _discover_with_timeout(rule_timeout_seconds, rid, rule_text, rename_map,
                                      src.get("fl_problem", ""),
                                      n_seeds=n_seeds, n_ce_trials=n_ce_trials,
                                      min_good_ratio=min_good_ratio)

    rules = []
    with open(rules_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rules.append(json.loads(line))
    if rule_id:
        rules = [r for r in rules if r["rule_id"] == rule_id]
    if limit:
        rules = rules[:limit]

    if n_workers > 1:
        from newclid.discovery.reduction.parallel import ensure_ray, run_bounded
        ensure_ray(n_workers)
        import ray

        @ray.remote
        def _process_remote(rid, rule_text, seeds, ce_trials, timeout_seconds,
                            norm_path, occ_path, src_path, good_ratio):
            _tracer = RuleTracer(
                normalized_rules_path=norm_path,
                occurrences_path=occ_path,
                source_dataset_path=src_path,
            )
            _tracer.build()
            norm_rec = _tracer.get_norm_record(rid)
            if not norm_rec:
                return {"status": "error", "rule_id": rid, "message": "norm not found"}
            rename_map = norm_rec.get("rename_map", {})
            src = _tracer.get_source_record(norm_rec.get("seed"), norm_rec.get("index_in_seed", 0))
            if not src:
                return {"status": "error", "rule_id": rid, "message": "source not found"}
            # Each Ray worker is its own single-threaded process, so SIGALRM
            # is safe here too (see _discover_with_timeout docstring).
            return _discover_with_timeout(timeout_seconds, rid, rule_text, rename_map,
                                          src.get("fl_problem", ""),
                                          n_seeds=seeds, n_ce_trials=ce_trials,
                                          min_good_ratio=good_ratio)

        tasks = [(r["rule_id"], r["rule_text"], n_seeds, n_ce_trials, rule_timeout_seconds,
                  normalized_rules_path, occurrences_path, source_dataset_path, min_good_ratio)
                 for r in rules]
        results = []
        from tqdm import tqdm
        for res in tqdm(run_bounded(_process_remote, tasks, inflight=n_workers),
                        total=len(rules), desc="[ndg_discovery] Processing rules", unit="rule"):
            results.append(res)
    else:
        results = []
        t0 = time.time()
        for i, r in enumerate(rules):
            result = process_rule(r["rule_id"], r["rule_text"])
            n_surv = result.get("n_surviving", 0)
            n_pairs = result.get("n_pairs", 0)
            status = result.get("status", "?")
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(rules) - i - 1) if i > 0 else 0
            print(f"[ndg_discovery] [{i+1}/{len(rules)}] {r['rule_id']}: {status}, "
                  f"pairs={n_pairs}, surv={n_surv}  ({elapsed:.0f}s, ETA {eta:.0f}s)")
            results.append(result)

    surviving_count = sum(1 for r in results if r.get("n_surviving", 0) > 0)
    print(f"[ndg_discovery] Rules with distinguishing predicates: {surviving_count}/{len(results)}")
    return results


def save_discover_results(results: list[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "distinguishing.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[ndg_discovery] Saved to {out_path}")
    return out_path

"""Build a single geometry problem and export its full DDAR proof closure.

Unlike `generation/pipeline.py` (which targets one specific goal, adds
auxiliary points, filters goals, etc.), this module:

1. Samples a random construction statement (no auxiliary points, no goal).
2. Builds several candidate numeric coordinate assignments for it and keeps
   the best-spread ones.
3. Runs DDAR to full closure (not scoped to any single goal) on the best
   candidate and exports every derived statement together with its
   dependency structure (premises + rule), including statements that follow
   directly from the construction with an empty derivation.
"""

from __future__ import annotations

import hashlib
import logging
import random

from newclid.api import CSolver, GeometricSolverBuilder
from newclid.dependencies.dependency import IN_PREMISES
from newclid.dependencies.symbols import Point
from newclid.generation.sampler import ProblemSampler
from newclid.numerical.geometries import PointNum

logger = logging.getLogger(__name__)


def score_points(point_coords: dict[str, tuple[float, float]]) -> float:
    """Score how evenly spread a set of points is.

    Returns min pairwise distance / max pairwise distance. Close to 1 means
    points are roughly equidistant from each other (neither cramped nor
    stretched); close to 0 means at least one pair is much closer together
    than another pair is far apart.

    This is stricter than the build-time degeneracy checks
    (`check_too_close_numerical` / `check_too_far_numerical`): a build can
    pass those and still score poorly here.
    """
    coords = list(point_coords.values())
    n = len(coords)
    if n < 2:
        return 0.0

    dists = []
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            dists.append(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5)

    max_dist = max(dists)
    if max_dist == 0:
        return 0.0
    return min(dists) / max_dist


def build_solver(fl_statement: str, build_seed: int):
    """Build `fl_statement` into a full `ProofState` with the given seed.

    Raises on failure. `build_problemJGEX` wraps the underlying build errors
    (degenerate configurations, invalid intersections, ...) into a generic
    `Exception` once its single attempt is exhausted, so callers that want
    to tolerate failures should catch broadly.
    """
    return (
        GeometricSolverBuilder(seed=build_seed)
        .load_problem_from_txt(fl_statement)
        .build(max_attempts=1)
    )


def extract_point_coords(solver) -> dict[str, tuple[float, float]]:
    """Extract the numeric coordinates of every point in a built solver."""
    return {
        name: (float(point.num.x), float(point.num.y))
        for name, point in solver.proof.symbols_graph.name2node.items()
        if isinstance(point, Point) and isinstance(point.num, PointNum)
    }


def try_build(fl_statement: str, build_seed: int):
    """Attempt one numeric build. Returns (coords, None) on success,
    (None, error_message) on failure."""
    try:
        solver = build_solver(fl_statement, build_seed)
    except Exception as e:
        return None, repr(e)

    coords = extract_point_coords(solver)
    if not coords:
        return None, "no points in built problem"
    return coords, None


def pick_best_builds(
    fl_statement: str,
    build_seed_base: int,
    n_best: int = 5,
    n_trials: int = 300,
    stop_after_successes: int | None = None,
) -> list[dict]:
    """Try up to `n_trials` numeric builds with distinct seeds and keep the
    `n_best` best-spread successful ones.

    Searching stops early once `stop_after_successes` builds have succeeded
    (default `4 * n_best`), so well-behaved constructions do not pay for the
    full budget; hard constructions use all `n_trials` and may still fall
    short of `n_best` -- the caller decides whether that is acceptable.

    Returns up to `n_best` entries sorted by score descending, each
    `{"build_seed": int, "score": float, "point_coords": dict}`.
    """
    if stop_after_successes is None:
        stop_after_successes = n_best * 4

    candidates = []
    for i in range(n_trials):
        build_seed = build_seed_base + i
        point_coords, err = try_build(fl_statement, build_seed)
        if point_coords is None:
            logger.debug(f"Build seed {build_seed} failed: {err}")
            continue
        candidates.append(
            {
                "build_seed": build_seed,
                "score": score_points(point_coords),
                "point_coords": {
                    name: list(xy) for name, xy in point_coords.items()
                },
            }
        )
        if len(candidates) >= stop_after_successes:
            break

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:n_best]


def run_full_closure(
    fl_statement: str,
    build_seed: int,
    max_level: int = 500,
    custom_rules: list[str] | None = None,
):
    """Build `fl_statement` with `build_seed` and run DDAR to full closure.

    No goal is passed, so the solver explores every derivable statement
    instead of stopping once a target is proved. `light=False` is required
    so that CSolver uses this already-built ProofState (and merges the C++
    closure into `solver.proof.dep_graph.hyper_graph`) rather than silently
    rebuilding its own numeric input via `build_ddar_input`.

    Returns the built `GeometricSolver` with the full closure merged in.
    """
    solver = build_solver(fl_statement, build_seed)
    csolver = CSolver(fl_statement, seed=build_seed, solver=solver, light=False)
    csolver.run(max_level=max_level, custom_rules=custom_rules)
    return solver


def export_proof_graph(solver) -> list[dict]:
    """Export the full DDAR closure as a list of proof-graph nodes.

    Each node is `{"id": str, "statement": str, "rule": str,
    "premises": list[str]}`.

    - Nodes derived from the construction carry `rule == "premise"` and an
      empty `premises` list (the "derived directly with no dependency" case).
    - Every other node names the rule that produced it and the ids of the
      nodes it was derived from. `rule` is the raw reason string from DDAR
      (a rule id such as `"r35"`, or an algebraic-reasoning label such as
      `"AR with Dist"` / `"Numerical Check"` / `"Trivial"`).

    Nodes are emitted in topological order, so a node's premises always
    appear earlier in the list. `hyper_graph` insertion order is *not*
    usable for this: the C++ `simplify_ar` pass runs after the main search
    and rewrites dependencies onto statements that were inserted later.
    """
    hyper_graph = solver.proof.dep_graph.hyper_graph

    # Assign ids in insertion order, so ids stay stable for a given build.
    stmt_id: dict = {}
    for statement, dep in hyper_graph.items():
        stmt_id.setdefault(statement, f"{len(stmt_id):04d}")
        for premise in dep.why:
            stmt_id.setdefault(premise, f"{len(stmt_id):04d}")

    nodes_by_stmt = {}
    for statement, dep in hyper_graph.items():
        is_premise = dep.reason == IN_PREMISES
        nodes_by_stmt[statement] = {
            "id": stmt_id[statement],
            "statement": statement.to_str(),
            "rule": "premise" if is_premise else dep.reason,
            "premises": [] if is_premise else [stmt_id[p] for p in dep.why],
        }

    # Kahn's algorithm over premise -> conclusion edges, breaking ties by
    # insertion order so the output is deterministic.
    insertion_index = {stmt: i for i, stmt in enumerate(hyper_graph)}
    dependency_count = {stmt: len(dep.why) for stmt, dep in hyper_graph.items()}
    dependents: dict = {stmt: [] for stmt in hyper_graph}
    for statement, dep in hyper_graph.items():
        for premise in dep.why:
            if premise in dependents:
                dependents[premise].append(statement)

    ready = [s for s in hyper_graph if dependency_count[s] == 0]
    ready.sort(key=lambda s: insertion_index[s])
    ordered: list = []
    while ready:
        stmt = ready.pop(0)
        ordered.append(stmt)
        for dependent in dependents[stmt]:
            dependency_count[dependent] -= 1
            if dependency_count[dependent] == 0:
                ready.append(dependent)
        ready.sort(key=lambda s: insertion_index[s])

    # Any statement left over is part of a dependency cycle (unexpected for a
    # DAG closure, but keep it rather than silently dropping derivations).
    if len(ordered) < len(hyper_graph):
        remaining = [s for s in hyper_graph if s not in set(ordered)]
        remaining.sort(key=lambda s: insertion_index[s])
        logger.warning(f"{len(remaining)} statement(s) in a dependency cycle")
        ordered.extend(remaining)

    return [nodes_by_stmt[statement] for statement in ordered]


def pick_clause_count(
    sample_seed: int, min_clauses: int = 3, max_clauses: int = 20
) -> int:
    """Deterministically pick a clause count in [min_clauses, max_clauses]
    from the sample seed, so the same seed always yields the same problem.

    Note the value is the sampler's `length` argument; the resulting
    `fl_statement` may contain a slightly different number of clauses (the
    sampler can emit one extra or one fewer clause per step).
    """
    if min_clauses > max_clauses:
        raise ValueError("min_clauses must be <= max_clauses")
    return random.Random(sample_seed).randint(min_clauses, max_clauses)


def build_one_problem(
    sample_seed: int,
    n_clauses: int | None = None,
    min_clauses: int = 3,
    max_clauses: int = 20,
    n_best: int = 5,
    n_trials: int = 300,
    max_level: int = 500,
    min_proof_nodes: int = 0,
    construction_config: dict | None = None,
    custom_rules: list[str] | None = None,
) -> dict | None:
    """Sample one construction, find its best coordinate builds, run the
    full DDAR closure on the best one, and package everything into a single
    record.

    `n_clauses=None` draws the clause count randomly from the sample seed
    (see `pick_clause_count`); pass an int to pin it.

    Returns None (caller should retry with a different sample_seed) if:
    - fewer than `n_best` successful builds were found within `n_trials`, or
    - the closure produced fewer than `min_proof_nodes` nodes.
    """
    if n_clauses is None:
        n_clauses = pick_clause_count(sample_seed, min_clauses, max_clauses)

    sampler = ProblemSampler(seed=sample_seed, construction_config=construction_config)
    fl_statement = sampler.generate(
        length=n_clauses,
        add_auxiliary=False,
        prune=False,
        with_coords=False,
        rename=True,
    )

    candidates = pick_best_builds(
        fl_statement,
        build_seed_base=sample_seed * 1_000_003,
        n_best=n_best,
        n_trials=n_trials,
    )
    if len(candidates) < n_best:
        logger.debug(
            f"Sample seed {sample_seed}: only {len(candidates)}/{n_best} "
            f"builds found in {n_trials} trials, discarding"
        )
        return None

    best = candidates[0]
    solver = run_full_closure(
        fl_statement,
        build_seed=best["build_seed"],
        max_level=max_level,
        custom_rules=custom_rules,
    )
    proof_graph = export_proof_graph(solver)
    if len(proof_graph) < min_proof_nodes:
        logger.debug(
            f"Sample seed {sample_seed}: closure has {len(proof_graph)} nodes "
            f"(< {min_proof_nodes}), discarding"
        )
        return None

    return {
        "fl_statement": fl_statement,
        # Stable, content-derived id: identical constructions sampled in two
        # different runs get the same hash. `problem_id` (the positional id
        # used for filenames) is assigned by the driver on write.
        "fl_hash": hashlib.sha1(fl_statement.encode("utf-8")).hexdigest()[:12],
        "n_clauses": len(fl_statement.split(";")),
        "sampler_length": n_clauses,
        "sample_seed": sample_seed,
        "candidate_builds": candidates,
        "chosen_build_seed": best["build_seed"],
        "chosen_point_coords": best["point_coords"],
        "proof_graph": proof_graph,
    }

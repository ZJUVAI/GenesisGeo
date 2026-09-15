"""Batch draw test using benchmarks/construction_tests.txt.

For each problem: build solver, call draw_bw() with random highlight modes.
Tests three modes: normal (no highlight), highlight_clauses, highlight_predicates.

Usage:
    conda run -n Genesis python scripts/test_draw_bw.py
    conda run -n Genesis python scripts/test_draw_bw.py --input benchmarks/construction_tests.txt --out-dir out_bw_batch
    conda run -n Genesis python scripts/test_draw_bw.py --random-seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newclid.api import GeometricSolverBuilder
from newclid.numerical.draw_bw import draw_bw


SEED = 998244353

# Template predicates for testing (will be filled with actual point names)
PREDICATE_TEMPLATES = [
    ("cong", 4, "{0} {1} {2} {3}"),
    ("para", 4, "{0} {1} {2} {3}"),
    ("perp", 4, "{0} {1} {2} {3}"),
    ("coll", 3, "{0} {1} {2}"),
    ("midp", 3, "{0} {1} {2}"),
    ("cyclic", 4, "{0} {1} {2} {3}"),
    ("eqangle", 8, "{0} {1} {2} {3} {4} {5} {6} {7}"),
    ("eqratio", 8, "{0} {1} {2} {3} {4} {5} {6} {7}"),
    ("simtri", 6, "{0} {1} {2} {3} {4} {5}"),
    ("contri", 6, "{0} {1} {2} {3} {4} {5}"),
    ("circle", 4, "{0} {1} {2} {3}"),
]


def load_problems(path: Path) -> list[tuple[str, str]]:
    """Read (name, problem_text) pairs from a two-line-per-problem file."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(lines) - 1, 2):
        pairs.append((lines[i], lines[i + 1]))
    return pairs


def generate_random_predicates(point_names: list[str], rng: random.Random, count: int = 1) -> list[str]:
    """Generate random predicate strings for testing.

    Cycles through different predicate types to ensure coverage.
    """
    if len(point_names) < 3:
        return []

    predicates = []
    for i in range(count):
        pname, nargs, template = PREDICATE_TEMPLATES[i % len(PREDICATE_TEMPLATES)]

        # Pick random points (with replacement for simplicity)
        selected = rng.choices(point_names, k=nargs)
        pred_str = f"{pname} {template.format(*selected)}"
        predicates.append(pred_str)

    return predicates


def make_safe_filename_suffix(predicates: list[str]) -> str:
    """Convert predicate list to safe filename suffix.

    Example: ["cong a b c d", "cyclic a b c d"] -> "congabcd_cyclicabcd"
    """
    parts = []
    for pred in predicates:
        # Remove spaces: "cong a b c d" -> "congabcd"
        compact = pred.replace(" ", "")
        parts.append(compact)
    return "_".join(parts)[:80]  # Limit length


def process_one(name: str, problem_text: str, out_dir: Path, rng: random.Random) -> tuple[int, int, str]:
    """Build solver and draw with random highlight modes.

    Returns (num_success, num_total, error_message).
    """
    try:
        builder = GeometricSolverBuilder(seed=SEED)
        builder.load_problem_from_txt(problem_text)
        solver = builder.build(max_attempts=100)

        safe_name = name.replace("/", "_").replace("\\", "_")
        if safe_name.endswith(".gex"):
            safe_name = safe_name[:-4]

        # Get all point names
        sg = solver.proof.symbols_graph
        from newclid.dependencies.symbols import Point
        point_names = [pt.name for pt in sg.nodes_of_type(Point)]

        # Get number of clauses
        num_clauses = len(builder.problemJGEX.constructions)

        # Randomly choose how many variants to generate (1-3)
        num_variants = rng.randint(1, 3)

        # Decide modes for each variant
        modes = rng.choices(["normal", "clauses", "predicates"], k=num_variants)

        success_count = 0

        for i, mode in enumerate(modes):
            if mode == "normal":
                suffix = "_normal"
                out_path = out_dir / f"{safe_name}{suffix}.svg"
                draw_bw(solver.proof, builder.problemJGEX, save_to=out_path)

            elif mode == "clauses" and num_clauses > 0:
                # Randomly select 1-3 clause indices
                k = min(rng.randint(1, 3), num_clauses)
                indices = rng.sample(range(num_clauses), k=k)
                # Include indices in filename: "name_c012.svg" for clauses 0,1,2
                indices_str = "".join(map(str, sorted(indices)))
                suffix = f"_c{indices_str}"
                out_path = out_dir / f"{safe_name}{suffix}.svg"
                draw_bw(solver.proof, builder.problemJGEX,
                       save_to=out_path, highlight_clauses=indices)

            elif mode == "predicates" and len(point_names) >= 3:
                # Generate 1-3 random predicates
                k = rng.randint(1, 3)
                preds = generate_random_predicates(point_names, rng, count=k)
                # Include predicate info in filename
                pred_suffix = make_safe_filename_suffix(preds)
                suffix = f"_p{pred_suffix}"
                out_path = out_dir / f"{safe_name}{suffix}.svg"
                draw_bw(solver.proof, builder.problemJGEX,
                       save_to=out_path, highlight_predicates=preds)
            else:
                # Fallback to normal if mode not applicable
                suffix = "_normal"
                out_path = out_dir / f"{safe_name}{suffix}.svg"
                draw_bw(solver.proof, builder.problemJGEX, save_to=out_path)

            success_count += 1

        return success_count, num_variants, ""

    except Exception:
        return 0, 1, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="benchmarks/construction_tests.txt",
        help="Path to problem file (two lines per problem: name, problem_text)",
    )
    parser.add_argument(
        "--out-dir",
        default="out_bw_batch",
        help="Directory for output SVGs",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for variant selection (default: 42)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.random_seed)

    problems = load_problems(input_path)
    print(f"Loaded {len(problems)} problems from {input_path}")
    print(f"Random seed: {args.random_seed}")
    print(f"Testing modes: normal, highlight_clauses, highlight_predicates\n")

    total_success = 0
    total_attempted = 0
    failed = []

    for name, text in tqdm(problems, ncols=80, unit="problem"):
        success, attempted, err = process_one(name, text, out_dir, rng)
        total_success += success
        total_attempted += attempted
        if success < attempted:
            failed.append((name, err))

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"Results: {total_success}/{total_attempted} SVGs generated successfully")
    print(f"{'='*60}")

    if failed:
        print(f"\nFailed ({len(failed)} problems):")
        for name, err in failed:
            print(f"\n  [{name}]")
            last_line = [ln for ln in err.strip().splitlines() if ln.strip()][-1]
            print(f"    {last_line}")

    if total_success > 0:
        print(f"\nSVGs written to: {out_dir}/")
        print(f"\nGenerated variants include:")
        print(f"  • normal: all black")
        print(f"  • clauses: random construction clauses highlighted in red")
        print(f"  • predicates: random predicate instances highlighted in red")


if __name__ == "__main__":
    main()

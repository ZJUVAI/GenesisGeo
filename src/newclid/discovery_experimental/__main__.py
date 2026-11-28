"""Discover DDAR rules from generated geometry proofs."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rule-timeout", type=float, default=6)
    parser.add_argument("--singular-timeout", type=float, default=1)
    parser.add_argument(
        "--candidate-input",
        action="store_true",
        help="Read already extracted candidates for a paired benchmark.",
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if not math.isfinite(args.rule_timeout) or not math.isfinite(args.singular_timeout):
        parser.error("timeouts must be finite")
    if not 0 < args.singular_timeout < args.rule_timeout:
        parser.error("Singular timeout must be positive and below the rule timeout")
    if not args.input.is_file():
        parser.error("input must be a JSONL file")

    from .pipeline import run

    print(
        json.dumps(
            run(
                args.input,
                args.output,
                workers=args.workers,
                limit=args.limit,
                rule_timeout=args.rule_timeout,
                singular_timeout=args.singular_timeout,
                candidate_input=args.candidate_input,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

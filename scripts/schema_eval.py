#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script entry to evaluate schemas. Run:
    python scripts/schema_eval.py
"""
import os
import sys
import logging

# Configure logging early
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('schema_eval')

# Adjust sys.path for in-repo imports when run directly from scripts/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Import evaluator from src/
from newclid.data_discovery.schema_evaluator import SchemaBatchEvaluator


class Args:
    json: str = os.path.join(REPO_ROOT, 'src', 'newclid', 'data_discovery', 'data', 'branched_mining.json')
    out_dir: str = os.path.join(REPO_ROOT, 'src', 'newclid', 'data_discovery', 'data')
    max_attempts: int = 100
    topn_print: int = 10


def main():
    args = Args()
    os.makedirs(args.out_dir, exist_ok=True)

    evaluator = SchemaBatchEvaluator(args.out_dir, max_attempts=args.max_attempts, topn_print=args.topn_print)
    evaluator.process_kind(args.json, 'schema')
    evaluator.process_kind(args.json, 'schema_before')


if __name__ == '__main__':
    main()

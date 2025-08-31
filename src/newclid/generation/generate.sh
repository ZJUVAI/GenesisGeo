#!/bin/bash

# MELIAD_PATH=$(pwd)/../meliad_lib/meliad
# export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH

python generate.py --n_clauses=20 --n_threads=10 --n_samples=100000 --log_level=info --timeout=7200
# python -m cProfile -o dataset/profile.prof -s cumulative -m generate --max_clauses=4 --search_depth=9 --n_threads=1 --n_samples=1 --log_level=info

# python equiv_analyze.py geometry_depth${search_depth}_raw.csv dataset/output.txt
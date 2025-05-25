#!/bin/bash

# MELIAD_PATH=$(pwd)/../meliad_lib/meliad
# export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH

# start_time=$(date +%s)
search_depth=5

python generate.py --max_clauses=20 --search_depth=10 --n_threads=50 --n_samples=300000 --log_level=info --time_limit=7200
# python -m cProfile -o dataset/profile.prof -s cumulative -m generate --max_clauses=4 --search_depth=9 --n_threads=1 --n_samples=1 --log_level=info

python equiv_analyze.py geometry_depth${search_depth}_raw.csv dataset/output.txt

# python analyze.py 
# end_time=$(date +%s)
# execution_time=$((end_time - start_time))
# echo "Execution time: $execution_time seconds"
# echo "Execution time: $execution_time seconds" >> execution_time.log

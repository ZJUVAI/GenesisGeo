#!/bin/bash

# Tranlation tests
# python -m unittest translate_test.py

## SOLVE
INPUT_FILE=input.txt
OUTPUT_DIR=output_dir
SEARCH_DEPTH=5
BEAM_SIZE=3
TIME_LIMIT=600
SEED=998244353
LOG_LEVEL=info

python newclid_solve.py \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --search_depth ${SEARCH_DEPTH} \
    --beam_size ${BEAM_SIZE} \
    --time_limit ${TIME_LIMIT} \
    --seed ${SEED} \
    --log_level ${LOG_LEVEL}
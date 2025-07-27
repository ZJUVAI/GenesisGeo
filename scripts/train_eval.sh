#!/bin/bash
export LOGLEVEL=WARNING

###############################################
# Training

# Model directory - modify this as needed
model_dir="sft3"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen/Qwen2.5-Math-1.5B \
    --train_type full \
    --dataset ./dataset/geometry_clauses15_samples15M.jsonl \
    --columns '{"llm_input": "query", "llm_output": "response"}' \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16 \
    --full_determinism true \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --padding_free true \
    --packing true \
    --use_liger_kernel true \
    --num_train_epochs 1 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 5000 \
    --logging_steps 100 \
    --max_length 2048 \
    --output_dir models/$model_dir \
    --add_version false \
    --save_only_model true


###############################################
# Evaluation

# Dataset options
datasets=("dev_jgex.txt" "imo_ag_30.txt")

# Decoding configurations (decoding_size beam_size)
configs=(
    "8 64"
    # "32 512"
)

# Checkpoint options - modify this list as needed
checkpoints=(
    # "checkpoint-5000"
    "checkpoint-10000"
    # "checkpoint-12140"
    "checkpoint-20000"
    "checkpoint-40000"
    "checkpoint-80000"
    "checkpoint-120000"
    "checkpoint-160000"
    "checkpoint-200000"
)

echo "Starting evaluation tasks..."
echo "Will process ${#checkpoints[@]} checkpoints, ${#datasets[@]} datasets, and ${#configs[@]} configurations"
echo "Total commands to execute: $((${#checkpoints[@]} * ${#datasets[@]} * ${#configs[@]}))"
echo "=================================="

# Loop through all checkpoints
for checkpoint in "${checkpoints[@]}"; do
    echo "Processing checkpoint: $checkpoint"
    echo "=================================="
    
    # Loop through all datasets
    for dataset in "${datasets[@]}"; do
        # Loop through all configurations
        for config in "${configs[@]}"; do
            # Split configuration parameters
            read -r decoding_size beam_size <<< "$config"
            
            # Build complete command
            cmd="python scripts/evaluation.py --problems_path problems_datasets/$dataset --model_path ./models/$model_dir/$checkpoint --max_workers 8 --decoding_size $decoding_size --beam_size $beam_size --search_depth 4"
            
            # Print current command to execute
            echo "Executing command:"
            echo "$cmd"
            echo "----------------------------------"
            
            # Execute command
            eval "$cmd"
            
            # Check command execution status
            if [ $? -eq 0 ]; then
                echo "✓ Command executed successfully"
            else
                echo "✗ Command execution failed"
            fi
            
            echo "=================================="
        done
    done
    
    echo "Completed checkpoint: $checkpoint"
    echo "=================================="
done

echo "All evaluation tasks completed!"
echo "Processed ${#checkpoints[@]} checkpoints total."
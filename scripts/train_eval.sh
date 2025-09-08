#!/bin/bash
export LOGLEVEL=WARNING

###############################################
# Training

# Model directory - modify this as needed
model_dir="sft15"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen/Qwen3-0.6B-Base \
    --dataset datasets/0901/geometry_clauses25-30_all_filtered2.jsonl \
    --columns '{"llm_input_renamed": "query", "llm_output_renamed": "response"}' \
    --system 'You are a helpful assistant.' \
    --max_length 2048 \
    --packing true \
    --padding_free true \
    --dataset_num_proc 16 \
    --dataloader_num_workers 4 \
    --train_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero1 \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --save_steps 10000 \
    --logging_steps 500 \
    --output_dir models/$model_dir \
    --add_version false \
    --save_only_model true
    # --truncation_strategy left \
    # --full_determinism true \
    # --lr_scheduler_type cosine_with_min_lr \
    # --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    # --use_chat_template false \


###############################################
# Evaluation

# Dataset options
datasets=(
    "dev_jgex.txt" 
    "dev_imo.txt"
)

# Decoding configurations (decoding_size beam_size)
configs=(
    # "8 64"
    "32 512"
)

# Checkpoint options - modify this list as needed
checkpoints=(
    # "checkpoint-10000"
    # "checkpoint-20000"
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
            cmd="python scripts/evaluation.py --problems_path problems_datasets/$dataset --model_path ./models/$model_dir/$checkpoint --max_workers 40 --decoding_size $decoding_size --beam_size $beam_size --search_depth 4"
            
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
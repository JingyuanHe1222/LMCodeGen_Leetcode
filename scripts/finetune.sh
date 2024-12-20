#!/bin/bash

DATA_PATH="ArtificialZeng/leetcode_code_generation"

# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B-Instruct"

OUTPUT_DIR="outputs"
OUTPUT_PATH="${OUTPUT_DIR}/qwen_finetuned_1e-6"

bz=8

cd ./finetune

deepspeed finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --lang "python" \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size $bz \
    --per_device_eval_batch_size $bz \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --learning_rate 1e-6 \
    --warmup_steps 1 \
    --logging_steps 5 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --deepspeed configs/ds_config_zero3_new.json \
    --bf16 True

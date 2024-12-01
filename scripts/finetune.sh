#!/bin/bash

DATA_PATH="ArtificialZeng/leetcode_code_generation"
OUTPUT_PATH="outputs/deepseekcoder_fintuned"
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"

cd ./finetune

deepspeed finetune_deepseekcoder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3_new.json \
    --bf16 True

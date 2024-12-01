#!/bin/bash
#SBATCH --job-name=finetune_qwen_1e-6
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --partition=general  
#SBATCH --mem=64G 
#SBATCH --gres=gpu:A6000:2

#SBATCH --exclude=babel-4-[1,17,25,33,37],babel-1-23 

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"


eval "$(conda shell.bash hook)"
conda activate pyserini

DATA_PATH="ArtificialZeng/leetcode_code_generation"

# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B-Instruct"

OUTPUT_DIR="/data/user_data/jingyuah/models" # "outputs"
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

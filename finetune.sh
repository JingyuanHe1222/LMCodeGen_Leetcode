#!/bin/bash
#SBATCH --job-name=qwen_ft
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2

#SBATCH --partition=general  
#SBATCH --mem=16G 

#SBATCH --gres=gpu:A6000

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

eval "$(conda shell.bash hook)"
conda activate sub_idx


echo "FT Job Starts"

model_path="Qwen/Qwen2.5-Coder-1.5B-Instruct"
dataset="ArtificialZeng/leetcode_code_generation"

exp_name="qwen_1.5b_ft_1e-6"
output_dir="/data/user_data/jingyuah/models/${exp_name}"

lang="python"
num_epoch=5

python finetune.py \
  --model_path $model_path \
  --dataset $dataset \
  --lang $lang \
  --exp_name $exp_name \
  --seq_length 2048 \
  --epoch $num_epoch \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.1 \
  --weight_decay 0.1 \
  --log_freq 5 \
  --eval_freq 100 \
  --save_freq 100 \
  --output_dir $output_dir \
  --exp_name $exp_name 


  echo "FT Job Starts"
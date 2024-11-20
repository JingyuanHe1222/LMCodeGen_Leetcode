#!/bin/bash
#SBATCH --job-name=inf_code_gen
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

eval "$(conda shell.bash hook)"
conda activate sub_idx


echo "Inf Job Starts"

model_path="Qwen/Qwen2.5-Coder-1.5B-Instruct"
dataset="ArtificialZeng/leetcode_code_generation"


python inf.py \
    --model_path $model_path \
    --dataset $dataset \
    --lang 'python' \
    --batch_size 2 \
    --gen_max_tokens 512 


echo "Inf Job Ends"


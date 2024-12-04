#!/bin/bash

#model_path="deepseek-ai/deepseek-coder-1.3b-instruct"
model_path="Qwen/Qwen2.5-Coder-1.5B-Instruct"
dataset="ArtificialZeng/leetcode_code_generation"

output_file="outputs/qwen/qwen_java_shots4.txt"
template="verbalizer/prompt_java.txt"
shots=4

echo "Inf Job Starts" > $output_file
# plain inference on model without any prompts or examples
python inf.py \
    --model_path $model_path \
    --dataset $dataset \
    --template $template \
    --lang 'java' \
    --batch_size 8 \
    --shots $shots \
    --gen_max_tokens 512 >> $output_file 2>&1


# in-context learning 
# template="templates/qwen_simple.jsonl"
# shots=2
# python in_context.py \
#     --model_path $model_path \
#     --dataset $dataset \
#     --template $template \
#     --shots $shots \
#     --lang 'python' \
#     --batch_size 8 \
#     --gen_max_tokens 512 

echo "Inf Job Ends" >> $output_file
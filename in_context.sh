


echo "Inf Job Starts"

model_path="Qwen/Qwen2.5-Coder-1.5B-Instruct"
dataset="ArtificialZeng/leetcode_code_generation"


# in-context learning 
template="templates/qwen_simple.jsonl"
shots=0
python in_context.py \
    --model_path $model_path \
    --dataset $dataset \
    --template $template \
    --shots $shots \
    --lang 'python' \
    --batch_size 8 \
    --gen_max_tokens 512 


echo "Inf Job Ends"
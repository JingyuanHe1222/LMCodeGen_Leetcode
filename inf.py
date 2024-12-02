import argparse
import pickle
from tqdm import tqdm 
import random

import torch 
from codebleu import calc_codebleu


from datasets import Features, Value, ClassLabel
from datasets import load_dataset

from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



torch.cuda.manual_seed(42)
torch.manual_seed(42)


def get_args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model_path", type=str, help="model for inference", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, help="name of dataset")
    parser.add_argument("--lang", type=str, default="python", help="Select language from ['python', 'c++', 'java', 'javascript']")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--gen_max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--template", type=str, default="prompt.txt")
    args = parser.parse_args()

    return args


def generate(prompts, model, sampling_params): 
    outputs = model.generate(prompts, sampling_params)
    generated_text = [out.outputs[0].text for out in outputs]
    return generated_text

def eval_batch(prompts, refs, model, sampling_params, lang): 
    code_gen_out = generate(prompts, model, sampling_params)
    '''
    for i in range(len(code_gen_out)):
        temp = code_gen_out[i].split("```python")
        
        python_code = temp[1].split("```")
        code_gen_out[i] = python_code[0]
    '''

    for i in range(len(prompts)):
        print("*****EXAMPLE*****\n\n\n")
        print("generated:")
        print(code_gen_out[i])
        print("ref:")
        print(refs[i])

    scores = calc_codebleu(refs, code_gen_out, lang=lang, weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return scores 

def main(): 
    args = get_args()
    random.seed(args.seed)

    # only load the corresponding lang
    class_names = ["content", args.lang]
    ft = Features({"sequence": Value("string"), "label": ClassLabel(names=class_names)})
    dataset = load_dataset(args.dataset, split="train")

    # split data with seed 
    dataset_split = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_data = dataset_split["train"]
    dataset_split = dataset_split["test"].train_test_split(test_size=0.5, seed=args.seed)
    eval_data = dataset_split["train"]
    test_data = dataset_split["test"]
    # only do inf on test 
    del eval_data

    # batch dataset 
    def collate_fn(features):

        def extract_answer_code(answer): 
            # get the code part only 
            code = answer.split('```')[1]
            # remove the lang prefix from code solution 
            parts = code.split('\n')
            code = '\n'.join(parts[1:])
            return code if code else ""
            
        questions = [sample["content"] for sample in features]
        answers = [extract_answer_code(sample[args.lang]) for sample in features]
        return {
            "questions": questions, 
            "answers": answers, 
        }
    

    dataloader = DataLoader(
        test_data, 
        collate_fn=collate_fn, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # init fast inf model 
    # tokenizer = AutoTokenizer.from_pretrained(args.model_p  ath)
    llm = LLM(args.model_path,trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        repetition_penalty=args.repetition_penalty, 
        max_tokens=args.gen_max_tokens, 
        seed=args.seed
    )

    all_scores = {
        "codebleu": [], 
        "ngram_match_score": [], 
        "weighted_ngram_match_score": [],  
        "syntax_match_score": [], 
        "dataflow_match_score": []
    }
    with open(args.template, 'r') as file:
        verbalizer = file.read()

    if args.shots != 0: 
        example_indices = random.sample(range(len(train_data)), args.shots)
        
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)): 
        # inf
        prompts = batch["questions"]
        for p in range(len(prompts)):
            temp = ""
            for i in range(args.shots): 
                example = train_data[example_indices[i]]
                temp += f"*** Leetcode Example Question {i+1} ***\n"  # 1-indexing 
                temp += f"{example['content']}\n"
                temp += f"**Code solution:** \n {example[args.lang]}\n" # example in target lang; data format **
            prompts[p] = temp + verbalizer.format(question=prompts[p])

        refs = batch["answers"]
        scores = eval_batch(prompts, refs, model=llm, sampling_params=sampling_params, lang=args.lang)
        # process scores 
        for key in scores: 
            all_scores[key].append(scores[key])

    for key in scores: 
        all_scores[key] = sum(all_scores[key])/len(all_scores[key])

    print(f"{args.model_path} on {args.dataset}-{args.lang} achieves scores: \n{all_scores}")



if __name__ == "__main__": 
    main()
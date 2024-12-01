import argparse
import copy 
import json
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
    parser.add_argument("--template_path", type=str, default="templates/qwen_simple.jsonl", help="a template to wrap around the context of the prompt")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--gen_max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    args = parser.parse_args()

    return args


def generate(prompts, model, sampling_params): 
    outputs = model.generate(prompts, sampling_params)
    generated_text = [out.outputs[0].text for out in outputs]
    return generated_text

def eval_batch(prompts, refs, model, sampling_params, lang): 
    code_gen_out = generate(prompts, model, sampling_params)
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

    # init tokenizer to wrap chat template 
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # read template 
    templates = []
    with open(args.template_path) as json_file:
        for line in json_file:
            templates.append(json.loads(line))

    # batch dataset 
    def collate_fn(features):

        def format_context(): 
            # format the n-shot example text 
            if args.shots == 0: 
                return ""
            template = "Improve your coding skill from the following leetcode example questions: \n"
            example_indices = random.sample(range(len(train_data)), args.shots)
            for i in range(args.shots): 
                example = train_data[example_indices[i]]
                template += f"*** Leetcode Example Question {i+1} ***\n"  # 1-indexing 
                template += f"{example['content']}\n"
                template += f"**Code solution:** \n {example[args.lang]}\n" # example in target lang; data format **
            return template 

        def apply_template(context): 
            # wrap against json dict
            messages = copy.deepcopy(templates)
            n_shot_contexts = format_context()
            messages[1]["content"] = n_shot_contexts + messages[1]["content"] + context
            # wrap template 
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        def extract_answer_code(answer): 
            # get the code part only 
            code = answer.split('```')[1]
            # remove the lang prefix from code solution 
            parts = code.split('\n')
            code = '\n'.join(parts[1:])
            return code if code else ""
        
        
        questions = [apply_template(sample["content"]) for sample in features]
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
    llm = LLM(args.model_path)

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        repetition_penalty=args.repetition_penalty, 
        max_tokens=args.gen_max_tokens, 
        seed=args.seed, 
    )

    all_scores = {
        "codebleu": [], 
        "ngram_match_score": [], 
        "weighted_ngram_match_score": [],  
        "syntax_match_score": [], 
        "dataflow_match_score": []
    }

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)): 
        # inf
        prompts = batch["questions"]
        refs = batch["answers"]
        scores = eval_batch(prompts, refs, model=llm, sampling_params=sampling_params, lang=args.lang)
        # process scores 
        for key in scores: 
            all_scores[key].append(scores[key])

    for key in scores: 
        all_scores[key] = sum(all_scores[key])/len(all_scores[key])

    print(f"{args.model_path} on {args.dataset}-{args.lang} with {args.shots} inference achieves scores: \n{all_scores}")



if __name__ == "__main__": 
    main()
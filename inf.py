import argparse
import pickle
from tqdm import tqdm 

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
    del train_data, eval_data

    # batch dataset 
    def collate_fn(features):
        questions = [sample["content"] for sample in features]
        answers = [sample[args.lang] for sample in features]
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(args.model_path)

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        repetition_penalty=args.repetition_penalty, 
        max_tokens=args.gen_max_tokens
    )

    all_scores = {
        "codebleu": [], 
        "ngram_match_score": [], 
        "weighted_ngram_match_score": [],  
        "syntax_match_score": [], 
        "dataflow_match_score": []
    }

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)): 
        breakpoint()
        prompts = batch["questions"]
        refs = batch["answers"]
        scores = eval_batch(prompts, refs, model=llm, sampling_params=sampling_params, lang=args.lang)
        for key in scores: 
            all_scores[key].append(scores[key])

    print(f"{args.model_name} on {args.dataset}-{args.lang} achieves scores: \n{all_scores}")



if __name__ == "__main__": 
    main()
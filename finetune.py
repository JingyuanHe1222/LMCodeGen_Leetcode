import argparse
import os
from tqdm import tqdm

import torch

from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from trl import SFTTrainer
from transformers import TrainingArguments, TrainingArguments, logging, set_seed



class CustomData(Dataset): 
    def __init__(self, data, tokenizer, max_len=2048, lang="python"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang

    def extract_answer_code(self, answer): 
        # get the code part only 
        code = answer.split('```')[1]
        # remove the lang prefix from code solution 
        parts = code.split('\n')
        code = '\n'.join(parts[1:])
        return code if code else ""

    def __getitem__(self, idx): 
        questions = self.data[idx]["content"]
        answers = self.extract_answer_code(self.data[idx][self.lang])

        question_results = self.tokenizer(
            questions,
            truncation=True,
            max_length=self.max_len,  
            padding=False,
            return_tensors="pt",
            add_special_tokens=False
        )

        answer_ids = self.tokenizer(
            answers, 
            truncation=True,
            max_length=self.max_len,  
            padding=False,
            return_tensors="pt",
            add_special_tokens=False, 
            return_attention_mask=False, 
        )

        return {
            "input_ids": question_results['input_ids'], 
            "attention_mask": question_results['attention_mask'], 
            "label": answer_ids['input_ids'], 
        }


    def __len__(self): 
        return len(self.data)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="ArtificialZeng/leetcode_code_generation")
    parser.add_argument("--lang", type=str, default="python", help="language to fine-tune on")
    
    parser.add_argument("--exp_name", type=str)

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=100, type=int)

    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def extract_answer_code(answer): 
    # get the code part only 
    try: 
        code = answer.split('```')[1]
    except: 
        code = answer.split('```')[0] # only one example encounter this...
    # remove the lang prefix from code solution 
    parts = code.split('\n')
    code = '\n'.join(parts[1:])
    return code if code else ""

    
def main(args):

    # load model 
    print("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        device_map={"": Accelerator().process_index},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path) 
    print_trainable_parameters(model)
    
    # process data splits
    dataset = load_dataset(args.dataset, split="train")

    dataset_split = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_data = dataset_split["train"]
    dataset_split = dataset_split["test"].train_test_split(test_size=0.5, seed=args.seed)
    eval_data = dataset_split["train"]

    # train_dataset = CustomData(train_data, tokenizer, args.seq_length, args.lang)
    # eval_dataset = CustomData(eval_data, tokenizer, args.seq_length, args.lang)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_on_start=True, 
        logging_steps=args.log_freq, 
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        load_best_model_at_end=True,
        num_train_epochs=args.epoch,
        metric_for_best_model="eval_loss", 
        save_total_limit=2, 
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=args.exp_name,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    # trainer = Trainer(
    #     model=model, 
    #     args=training_args, 
    #     train_dataset=train_dataset, 
    #     eval_dataset=eval_dataset, 
    # )

    def preprocess_function(example):
        # returning a list of samples 
        question = example["content"]
        answer = extract_answer_code(example[args.lang])
        text = f"### Question:\n{question}\n\n### Answer:\n{answer}"
        return text    

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        formatting_func=preprocess_function,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=True
    )

    print("Training...")
    trainer.train()



if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
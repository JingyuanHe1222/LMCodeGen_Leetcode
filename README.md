# LMCodeGen_Leetcode
LM Code Generation project for CMU 11667 FA24


## CodeBleu

    pip install tree-sitter-python==0.21
    pip install codebleu
    
    
## Inference

    bash scripts/inf.sh 

or in slurm: 

    sbatch scripts/inf_slurm.sh


## In-Content

    bash scripts/in_context.sh 

or in slurm: 

    sbatch scripts/in_context_slurm.sh


## Finetune

    bash scripts/finetune.sh

or in slurm: 

    sbatch scripts/finetune.sh





## Statistics 

<!-- ------------------------------ -->
### Baseline

#### ***Python***

Model                              | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.229710704538744 | 0.01602361731893948 | 0.08882963909258695        | 0.30382401780040463 | 0.5101655439430446
| deepseek-ai/deepseek-coder-1.3b-instruct | 0.1299426628892564 | 0.014579863872695368| 0.0443358317753529        | 0.15995708731386077 | 0.3008978685951167


<!-- ------------------------------ -->
### In-Context

#### ***Python***

Model                              | Template                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/qwen_simple.jsonl | 0         | 0.2426334885735626| 0.023860909555950393| 0.12461768989960881        | 0.3426517943797025 | 0.4794035604589891
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/qwen_simple.jsonl | 1         | 0.2451417490079226| 0.03740582286751913 | 0.13490852147710572        | 0.35501781357354173| 0.45323483811352394
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/qwen_simple.jsonl | 2         | 0.23180871274242767| 0.04479062266627481| 0.1330601867362257         | 0.33139134964085226| 0.41799269192635813


<!-- ------------------------------ -->
### Fine-Tuning

#### ***Python***

Model                              | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.2610019584292215 | 0.029828865479395342| 0.1573694166705269        | 0.3492735190945361 | 0.507536032472428
| deepseek-ai/deepseek-coder-1.3b-instruct | 0.17047070107756407 | 0.013446862387470082| 0.05191944056633911        | 0.21340642824632017 | 0.40311007311012675

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
| Qwen/Qwen2.5-Coder-3B-Instruct | 0.2462009168232043 | 0.02453480689013888 | 0.12786808672495403        | 0.34166027945836736 | 0.4907404942193567
| deepseek-ai/deepseek-coder-1.3b-instruct | 0.1299426628892564 | 0.014579863872695368| 0.0443358317753529        | 0.15995708731386077 | 0.3008978685951167


#### ***java***

Model                              | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.18685390194794613 | 0.011963906570062443 | 0.04714539590092179        | 0.18968858256294532 | 0.49861772275785504
| Qwen/Qwen2.5-Coder-3B-Instruct | 0.17480732712229172 | 0.010215454474163131 | 0.03741425217998402        | 0.16858412172319065 | 0.4830154801118292
| deepseek-ai/deepseek-coder-1.3b-instruct | 0.07220883166644357 | 0.003524907806668263| 0.00561304307545241        | 0.05913988722582831 | 0.22055748855782525

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


#### ***java***

Model                              | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.30977036588567675 | 0.061891130805922824 | 0.23127925034798466        | 0.44569735945712324 | 0.5002137229316763
| deepseek-ai/deepseek-coder-1.3b-instruct | 0.20380132664894324 | 0.04447950183656576 | 0.11096579635849103        | 0.3164572898019216 | 0.34330271859879435

# LMCodeGen_Leetcode
LM Code Generation project for CMU 11667 FA24


## Inference

    bash inf.sh 

or in slurm: 

    sbatch inf.sh


## CodeBleu

    pip install tree-sitter-python==0.21


## Statistics 

<!-- ------------------------------ -->
### Baseline

#### ***Python***

Model                              | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.2272671388797994 | 0.015910792689906885 | 0.08882963909258695        | 0.29162434417580824 | 0.5101655439430446


<!-- ------------------------------ -->
### In-Context

#### ***Python***

Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | verbalizer/prompt.txt | 0         | 0.22736769047157787| 0.015910792689906885| 0.09182186307500868        | 0.29162434417580824 | 0.5101137619455877
| Qwen/Qwen2.5-Coder-1.5B-Instruct | verbalizer/prompt_python.txt | 0         | 0.24052859432032397| 0.01955953337474523 | 0.12454930922399235        | 0.3374881106149699| 0.4805174240675887
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_python2.txt | 0         | 0.24215609516457082| 0.018862656370835625| 0.11955113259797612         | 0.3574958914692374| 0.47271470022023426
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 2         | 0.2427969666283093| 0.02018918811061197| 0.12202217247187458         | 0.3640395589814406| 0.4649369469493102
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 4         | 0.24131754839965838| 0.01903401286898284| 0.11845568018390874         | 0.355754486289066| 0.47202601425667573
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 8         | 0.1584848793180661| 0.015259170049969174| 0.055996858142944986         | 0.18700042753991938| 0.37568306153943076
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 5         | 0.24705894038316303| 0.021329870437724295| 0.1306045237641184         | 0.359456205286188| 0.4768451620446216

Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| deepseek-ai/deepseek-coder-1.3b-instruct | N/A | 0         | 0.13385174472066855| 0.015889369738179265| 0.045207386318725305        | 0.1763489814241062 | 0.2979612414016635
| deepseek-ai/deepseek-coder-1.3b-instruct | verbalizer/prompt.txt | 0         | 0.21504140072301933| 0.029322011697573844 | 0.10316930015703125        | 0.3191568005536727| 0.40851749048379943
| deepseek-ai/deepseek-coder-1.3b-instruct | verbalizer/deepseek_prompt.txt | 0         | 0.22154951054829272| 0.03220860132797948| 0.08711396378436835         | 0.30917778991844674| 0.3720667329389041
| deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 2         | 0.2082731280379006| 0.03620140059051431| 0.09078762541592303         | 0.3129420850784446| 0.3931614010667206
|deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 4         | 0.2029518659470796| 0.03999000268328664| 0.08507234046737103         | 0.3140230111671443| 0.37272210947051637
| deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 8         | 0.07340528316179015| 0.022718187769511435| 0.03601860230128292         | 0.10503934958899934| 0.12984499298736696



#### ***Java***
Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | verbalizer/prompt_java.txt | 0         | 0.31126644488734523| 0.04600979600158717| 0.20646619172882125         | 0.4889758044400094| 0.5036139873789632
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_java.txt | 2         | 0.31523915400567293| 0.05314839370513022| 0.23115396342573552         | 0.4895273382514287| 0.4871269206403976
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_java.txt | 4         | 0.31768086168240145| 0.05198436610105995| 0.2300794808788181         | 0.4905334166646837| 0.4981261830850442

Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| deepseek-ai/deepseek-coder-1.3b-instruct | verbalizer/deepseek_prompt.txt | 0         | 0.29566562005486235| 0.07056999904567654| 0.1814067281749894         | 0.4687450288968518| 0.453466147830746
| deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 2         | 0.311417122249295| 0.1001271801400451| 0.19341789613870214         | 0.4724320370347916| 0.475454087548048
|deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 4         | 0.3188062483128827| 0.12027399722382058| 0.2060335623330785         | 0.4823078614148741| 0.46660957227975847


<!-- ------------------------------ -->
### Scaling Up


#### ***Python***
Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/deepseek_prompt.txt | 0 | 0.2365068348571344 | 0.031476593881604654 | 0.12529971381322924       | 0.3118301027130953 | 0.4519972002070494
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/deepseek_prompt.txt | 2 | 0.2324779013679328 | 0.03543981352965757 | 0.11611923081640614       | 0.3303634128569884 | 0.44798914826867925
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/deepseek_prompt.txt | 4 | 0.23890479009518997 | 0.056271539987321435 | 0.12128848597367742       | 0.34445591166737155 | 0.4336032227523893

#### ***Java***
Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/prompt.txt | 0 | 0.16215023591485625| 0.0035412276241528085| 0.014950838334448819        | 0.1241914970369472 | 0.5059173806638763 
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/deepseek_prompt.txt | 0 | 0.310024100494679 | 0.07314342623149687 | 0.2096812137977044       | 0.47399052703468325 | 0.4663320823724581
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/deepseek_prompt.txt | 2 | 0.3221974874016451 | 0.08750551155965851 | 0.2096812137977044       | 0.47399052703468325 | 0.4663320823724581
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/deepseek_prompt.txt | 4 | 0.3354103364488185 | 0.11434979323787338 | 0.2372369243307168       | 0.5100206437391517 | 0.48003398448753165
| deepseek-ai/deepseek-coder-6.7b-instruct | verbalizer/deepseek_prompt.txt | 8 (too long, can't parse)| 0.25| 0.0| 0.0        | 0.0 | 0.0 




### Deployment

#### ***python***

#### ft + in-context

Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | verbalizer/prompt_shots.txt | 0         | 0.26520419405175875| 0.16177608182722186| 0.16177608182722186         | 0.3882973223766777| 0.47922243628007544
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 2         | 0.22790118952098| 0.07497215723608647| 0.13537410190334428         | 0.3178320530505136| 0.3834264458939756
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 4         | 0.24495386266628055| 0.08422297359148066| 0.15207771294059264         | 0.34330841671887585| 0.40020634741417316



Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| deepseek-ai/deepseek-coder-1.3b-instruct | verbalizer/deepseek_prompt.txt | 0         | 0.21235916789973025| 0.02782308231306178| 0.10055145107162522         | 0.3085396127311261| 0.412522525483108
| deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 2         | 0.2029732050070879| 0.035951873013259666| 0.09929546507228512         | 0.28956589760525414| 0.38707958433755246
|deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 4         | 0.2011936097216442| 0.03760770616265476| 0.088810121561726         | 0.2858364035135594| 0.39252020764863643


#### ***Java***

#### ft + in-context

Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | verbalizer/prompt_shots.txt | 0         | 0.33012411570770883| 0.06135170196677177| 0.24961192243769267         | 0.48560996088211184| 0.5239228775442586
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 2         | 0.3253962297322123| 0.11139963714959084| 0.23115396342573552         | 0.4893625130311877| 0.4611858003163795
| Qwen/Qwen2.5-Coder-1.5B-Instruct | templates/prompt_shots.txt | 4         | 0.3272682124366119| 0.12280796353057581| 0.2483040068174084         | 0.4833954095012772| 0.4545654698971865


Model                              | Verbalizer                    | shots     | codebleu          | ngram_match_score   | weighted_ngram_match_score | syntax_match_score  | dataflow_match_score 
| -------------------------------- | -------------               | --------- | ---------------   | -----------------   | -------------------------- | -----------------   | -----------------    |
| deepseek-ai/deepseek-coder-1.3b-instruct | verbalizer/deepseek_prompt.txt | 0         | 0.28248633449322724| 0.07613729092552933| 0.184881047792029         | 0.4419914759090907| 0.42693552334625984
| deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 2         | 0.27533805911709835| 0.08062596876740984| 0.17135847838085294         | 0.4289855132948561| 0.42038227602527467
|deepseek-ai/deepseek-coder-1.3b-instruct | templates/deepseek_prompt.txt | 4         | 0.27447512686308556| 0.09001045755410408| 0.17667068912951875         | 0.4305689299684055| 0.40065043080031415

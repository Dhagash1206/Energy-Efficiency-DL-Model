# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 6 across 6 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 8 calibration, 16 held out; 2 held-out repeat(s), 16 maximum generated tokens.
All 269 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token |
|---|---:|---:|---:|
| ordinary | 32 | 0.037400 | 2.104184 |
| fixed | 32 | 0.036840 | 2.239229 |
| never_observe | 32 | 0.041857 | 2.406311 |
| selected_observe | 32 | 0.041603 | 2.640725 |
| always_observe | 32 | 0.040927 | 2.719697 |
| bandit | 32 | 0.037053 | 2.256791 |
| adaptive | 32 | 0.033218 | 1.922158 |

Observed feedback cost per request (selected/all): selected_observe 4.687500 J, always_observe 6.515500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and CPU timing noise do not establish a speed or energy advantage, and there is no novelty claim.

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
| ordinary | 32 | 0.035038 | 2.076137 |
| fixed | 32 | 0.035506 | 2.236490 |
| never_observe | 32 | 0.038857 | 2.281088 |
| selected_observe | 32 | 0.039333 | 2.527379 |
| always_observe | 32 | 0.039470 | 2.564031 |
| bandit | 32 | 0.034453 | 2.104178 |
| adaptive | 32 | 0.034977 | 2.116275 |

Observed feedback cost per request (selected/all): selected_observe 3.549500 J, always_observe 4.098000 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and CPU timing noise do not establish a speed or energy advantage, and there is no novelty claim.

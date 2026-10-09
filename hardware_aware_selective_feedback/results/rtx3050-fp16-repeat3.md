# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 6 across 6 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 8 calibration, 16 held out; 3 held-out repeat(s), 16 maximum generated tokens.
All 381 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token |
|---|---:|---:|---:|
| ordinary | 48 | 0.037662 | 1.684565 |
| fixed | 48 | 0.036775 | 1.687952 |
| never_observe | 48 | 0.043695 | 1.951799 |
| selected_observe | 48 | 0.040762 | 1.961363 |
| always_observe | 48 | 0.040802 | 1.965448 |
| bandit | 48 | 0.035608 | 1.631327 |
| adaptive | 48 | 0.036874 | 1.664036 |

Observed feedback cost per request (selected/all): selected_observe 0.539000 J, always_observe 0.662500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and CPU timing noise do not establish a speed or energy advantage, and there is no novelty claim.

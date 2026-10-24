# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp32 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 2 across 2 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 4 calibration, 8 held out; 2 held-out repeat(s), 16 maximum generated tokens.
All 133 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token |
|---|---:|---:|---:|
| ordinary | 16 | 0.112348 | 4.392223 |
| fixed | 16 | 0.086302 | 3.916281 |
| never_observe | 16 | 0.081016 | 3.741449 |
| selected_observe | 16 | 0.089186 | 4.181785 |
| always_observe | 16 | 0.091540 | 4.240984 |
| bandit | 16 | 0.084205 | 3.640500 |
| adaptive | 16 | 0.075542 | 3.482609 |

Observed feedback cost per request (selected/all): selected_observe 7.542000 J, always_observe 8.512500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in rtx3050-fp32-energy-safe.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

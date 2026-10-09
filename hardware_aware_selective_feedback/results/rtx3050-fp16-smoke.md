# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 1 across 1 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 2 calibration, 2 held out; 1 held-out repeat(s), 4 maximum generated tokens.
All 27 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token |
|---|---:|---:|---:|
| ordinary | 2 | 0.039443 | 3.478500 |
| fixed | 2 | 0.035516 | 2.913500 |
| never_observe | 2 | 0.039181 | 2.691750 |
| selected_observe | 2 | 0.043397 | 3.130375 |
| always_observe | 2 | 0.040283 | 4.294625 |
| bandit | 2 | 0.031278 | 3.226375 |
| adaptive | 2 | 0.033848 | 2.953250 |

Observed feedback cost per request (selected/all): selected_observe 1.754500 J, always_observe 6.411500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and CPU timing noise do not establish a speed or energy advantage, and there is no novelty claim.

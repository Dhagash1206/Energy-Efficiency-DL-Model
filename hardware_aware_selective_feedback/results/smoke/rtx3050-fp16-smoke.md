# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 1 across 1 calibration prompts.
Best executed fixed setting varies across prompts: False.
Prompts: 2 calibration, 2 held out; 1 held-out repeat(s), 4 maximum generated tokens.
All 27 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token |
|---|---:|---:|---:|
| ordinary | 2 | 0.023999 | 1.304875 |
| fixed | 2 | 0.023821 | 0.613875 |
| never_observe | 2 | 0.028201 | 1.317250 |
| selected_observe | 2 | 0.028655 | 2.052000 |
| always_observe | 2 | 0.031230 | 1.311625 |
| bandit | 2 | 0.023385 | 1.311125 |
| adaptive | 2 | 0.025200 | 1.238125 |

Observed feedback cost per request (selected/all): selected_observe 2.939000 J, always_observe -0.022500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in rtx3050-fp16-smoke.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

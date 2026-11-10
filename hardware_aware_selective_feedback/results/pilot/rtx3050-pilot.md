# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 2 across 2 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 4 calibration, 4 held out; 2 held-out repeat(s), 8 maximum generated tokens.
All 77 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token | Greedy mismatches |
|---|---:|---:|---:|---:|
| ordinary | 8 | 0.025048 | 1.250391 | 0 |
| fixed | 8 | 0.029826 | 1.337797 | 0 |
| never_observe | 8 | 0.025110 | 1.295891 | 0 |
| selected_observe | 8 | 0.026690 | 1.480922 | 0 |
| always_observe | 8 | 0.027415 | 1.545594 | 0 |
| bandit | 8 | 0.024606 | 1.118750 | 0 |
| adaptive | 8 | 0.024100 | 1.206703 | 0 |

Observed feedback cost per request (selected/all): selected_observe 1.290000 J, always_observe 3.731500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in rtx3050-pilot.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

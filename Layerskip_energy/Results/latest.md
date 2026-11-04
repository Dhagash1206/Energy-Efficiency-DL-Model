# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp32 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 6 across 6 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 8 calibration, 16 held out; 3 held-out repeat(s), 64 maximum generated tokens.
All 381 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token | Greedy mismatches |
|---|---:|---:|---:|---:|
| ordinary | 48 | 0.105173 | 4.129220 | 0 |
| fixed | 48 | 0.064210 | 2.784979 | 0 |
| never_observe | 48 | 0.063219 | 2.764236 | 0 |
| selected_observe | 48 | 0.070298 | 3.112311 | 0 |
| always_observe | 48 | 0.072359 | 3.214446 | 0 |
| bandit | 48 | 0.065241 | 2.791580 | 0 |
| adaptive | 48 | 0.060672 | 2.671720 | 0 |

Observed feedback cost per request (selected/all): selected_observe 21.680000 J, always_observe 28.613000 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

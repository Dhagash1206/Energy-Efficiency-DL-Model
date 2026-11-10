# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 1 across 1 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 2 calibration, 2 held out; 1 held-out repeat(s), 4 maximum generated tokens.
All 27 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token | Greedy mismatches |
|---|---:|---:|---:|---:|
| ordinary | 2 | 0.045174 | 1.956500 | 0 |
| fixed | 2 | 0.044255 | 1.557625 | 0 |
| never_observe | 2 | 0.051316 | 1.956750 | 0 |
| selected_observe | 2 | 0.055799 | 2.470750 | 0 |
| always_observe | 2 | 0.061004 | 2.459250 | 0 |
| bandit | 2 | 0.044795 | 1.940375 | 0 |
| adaptive | 2 | 0.052540 | 1.420250 | 0 |

Observed feedback cost per request (selected/all): selected_observe 2.056000 J, always_observe 2.010000 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in rtx3050-smoke.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

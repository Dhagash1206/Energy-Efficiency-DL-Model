# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 4 across 4 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 16 calibration, 16 held out; 3 held-out repeat(s), 32 maximum generated tokens.
386/389 executed requests matched native greedy token IDs. Excluded from adaptive/bandit selection after a mismatch: depth=4 length=3. Policies with mismatches are not valid exact-output comparisons.

| Policy | Runs | Latency s/token | Energy J/token | Greedy mismatches |
|---|---:|---:|---:|---:|
| ordinary | 48 | 0.026583 | 0.924262 | 0 |
| fixed | 48 | 0.023034 | 0.799397 | 3 |
| never_observe | 48 | 0.025272 | 0.883632 | 0 |
| selected_observe | 48 | 0.027231 | 0.999564 | 0 |
| always_observe | 48 | 0.085561 | 1.084138 | 0 |
| bandit | 48 | 0.024574 | 0.882391 | 0 |
| adaptive | 48 | 0.024306 | 0.842391 | 0 |

Observed feedback cost per request (selected/all): selected_observe 3.850500 J, always_observe 4.886500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in rtx3050-heldout.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

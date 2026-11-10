# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp32 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 3 across 3 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 4 calibration, 4 held out; 2 held-out repeat(s), 32 maximum generated tokens.
All 97 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token | Greedy mismatches |
|---|---:|---:|---:|---:|
| ordinary | 8 | 0.120784 | 4.844008 | 0 |
| fixed | 8 | 0.074070 | 3.612066 | 0 |
| never_observe | 8 | 0.069928 | 3.470680 | 0 |
| selected_observe | 8 | 0.075701 | 3.831516 | 0 |
| always_observe | 8 | 0.077652 | 4.026406 | 0 |
| bandit | 8 | 0.085561 | 3.863668 | 0 |
| adaptive | 8 | 0.076396 | 3.658246 | 0 |
| fixed_depth12 | 8 | 0.070973 | 3.452809 | 0 |
| del | 8 | 0.093979 | 4.714074 | 0 |

Observed feedback cost per request (selected/all): selected_observe 11.381500 J, always_observe 19.247000 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in pilot.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

## DEL comparison (independent greedy paper port)

This is a local implementation comparison, not a reproduction of published DEL speedups.
DEL's shadow projections and decisions are measured. Adaptive request-level decisions/updates
remain outside the generation meter; shared calibration is separate. These are not end-to-end savings.
Positive reduction means adaptive used less cost. No significance claim is made.

| Comparison | Exact-output valid | GPU energy reduction | Latency reduction |
|---|---|---:|---:|
| adaptive vs del | True | 22.40% | 18.71% |
| adaptive vs fixed_depth12 | True | -5.95% | -7.64% |

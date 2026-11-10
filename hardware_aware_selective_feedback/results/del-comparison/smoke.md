# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp32 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 1 across 1 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 2 calibration, 2 held out; 1 held-out repeat(s), 8 maximum generated tokens.
All 31 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token | Greedy mismatches |
|---|---:|---:|---:|---:|
| ordinary | 2 | 0.157594 | 7.395125 | 0 |
| fixed | 2 | 0.162299 | 6.897250 | 0 |
| never_observe | 2 | 0.149459 | 7.405500 | 0 |
| selected_observe | 2 | 0.159161 | 7.866250 | 0 |
| always_observe | 2 | 0.163232 | 9.294812 | 0 |
| bandit | 2 | 0.124870 | 5.961500 | 0 |
| adaptive | 2 | 0.144760 | 6.106063 | 0 |
| fixed_depth12 | 2 | 0.112147 | 5.963313 | 0 |
| del | 2 | 0.189841 | 9.329812 | 0 |

Observed feedback cost per request (selected/all): selected_observe 3.686000 J, always_observe 15.114500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in smoke.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

## DEL comparison (independent greedy paper port)

This is a local implementation comparison, not a reproduction of published DEL speedups.
DEL's shadow projections and decisions are measured. Adaptive request-level decisions/updates
remain outside the generation meter; shared calibration is separate. These are not end-to-end savings.
Positive reduction means adaptive used less cost. No significance claim is made.

| Comparison | Exact-output valid | GPU energy reduction | Latency reduction |
|---|---|---:|---:|
| adaptive vs del | True | 34.55% | 23.75% |
| adaptive vs fixed_depth12 | True | -2.39% | -29.08% |

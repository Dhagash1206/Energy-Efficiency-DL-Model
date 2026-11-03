# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp32 on cpu.
Objective: latency. Energy boundary: cpu_packages.
Counter: Windows EMI CPU package cumulative energy (pWh).
Observer pairs: 1 across 1 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 4 calibration, 4 held out; 1 held-out repeat(s), 8 maximum generated tokens.
All 45 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token |
|---|---:|---:|---:|
| ordinary | 4 | 0.316458 | 11.137473 |
| fixed | 4 | 0.442984 | 16.299304 |
| never_observe | 4 | 0.349404 | 12.364600 |
| selected_observe | 4 | 0.406832 | 14.625141 |
| always_observe | 4 | 0.442991 | 16.220315 |
| bandit | 4 | 0.336674 | 12.162497 |
| adaptive | 4 | 0.320250 | 11.601508 |

Observed feedback cost per request (selected/all): selected_observe 0.476497 s, always_observe 0.735495 s.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and CPU timing noise do not establish a speed or energy advantage, and there is no novelty claim.

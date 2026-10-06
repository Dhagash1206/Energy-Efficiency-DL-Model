# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp32 on cpu.
Objective: energy. Energy boundary: cpu_packages.
Counter: Windows EMI CPU package cumulative energy (pWh).
Observer pairs: 1 across 1 calibration prompts.
Best executed fixed setting varies across prompts: True.
Prompts: 2 calibration, 2 held out; 1 held-out repeat(s), 4 maximum generated tokens.
All 27 executed requests matched native greedy token IDs.

| Policy | Runs | Latency s/token | Energy J/token |
|---|---:|---:|---:|
| ordinary | 2 | 0.752212 | 12.763160 |
| fixed | 2 | 0.796088 | 12.763198 |
| never_observe | 2 | 0.860152 | 13.696231 |
| selected_observe | 2 | 0.808662 | 13.361432 |
| always_observe | 2 | 0.919942 | 14.259261 |
| bandit | 2 | 0.954259 | 16.806651 |
| adaptive | 2 | 0.920174 | 15.818055 |

Observed feedback cost per request (selected/all): selected_observe -1.339194 J, always_observe 2.252120 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and CPU timing noise do not establish a speed or energy advantage, and there is no novelty claim.

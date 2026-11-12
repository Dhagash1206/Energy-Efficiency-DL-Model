# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 3 across 3 calibration prompts.
Best executed fixed setting varies across prompts: False.
Prompts: 4 calibration, 4 held out; 1 held-out repeat(s), 256 maximum generated tokens.
All 57 executed requests matched native greedy token IDs.

Observer projection period: 4 verification cycle(s).

| Policy | Runs | Latency s/token | Total J/token | Static idle J/token | Dynamic excess J/token | Greedy mismatches |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 4 | 0.035091 | 1.477726 | 0.367210 | 1.110515 | 0 |
| fixed | 4 | 0.022238 | 0.948633 | 0.232708 | 0.715925 | 0 |
| fixed_d12_k3 | 4 | 0.027993 | 1.205459 | 0.292932 | 0.912527 | 0 |
| never_observe | 4 | 0.025414 | 1.075541 | 0.265948 | 0.809593 | 0 |
| selected_observe | 4 | 0.026058 | 1.115130 | 0.272688 | 0.842441 | 0 |
| always_observe | 4 | 0.024960 | 1.066759 | 0.261196 | 0.805563 | 0 |
| bandit | 4 | 0.028008 | 1.209877 | 0.293090 | 0.916786 | 0 |
| adaptive | 4 | 0.022571 | 0.964938 | 0.236194 | 0.728744 | 0 |

Observed feedback cost per request (selected/all): selected_observe 4.457000 J, always_observe -2.206000 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in fp16-m4-quality-pilot.json.
Calibration work is reported separately and is not allocated to any single baseline.
This is a correctness and feasibility run. Small samples and hardware measurement variability do not establish a speed or energy advantage, and there is no novelty claim.

Idle subtraction is a resident-model reference estimate, not a physical power decomposition.
Negative dynamic excess values are retained. Selection still optimizes total measured energy.

## Calibration-inclusive energy break-even

Stationary-workload projections, not measured lifetime savings. All shared calibration
and idle sampling is charged to adaptive; fixed baselines have zero calibration charge.
Model loading, greedy reference generation, and decisions outside Meter remain excluded.

| Baseline | Exact-output energy comparison | Break-even requests |
|---|---|---:|
| ordinary | valid | 64 |
| fixed | candidate mean request energy is not lower | no supported crossing |
| never_observe | valid | 295 |
| fixed_d12_k3 | valid | 136 |

## Task quality (one greedy completion per task)

Timing repeats are not independent quality samples. Scores use repeat 0, including greedy mismatches.

| Policy | GSM8K accuracy | GSM8K tasks | HumanEval pass@1 | HumanEval tasks |
|---|---:|---:|---:|---:|
| bandit | 0.0000 | 2 | not executed | 2 |
| adaptive | 0.0000 | 2 | not executed | 2 |
| selected_observe | 0.0000 | 2 | not executed | 2 |
| never_observe | 0.0000 | 2 | not executed | 2 |
| fixed | 0.0000 | 2 | not executed | 2 |
| always_observe | 0.0000 | 2 | not executed | 2 |
| fixed_d12_k3 | 0.0000 | 2 | not executed | 2 |
| ordinary | 0.0000 | 2 | not executed | 2 |

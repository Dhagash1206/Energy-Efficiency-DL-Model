# LayerSkip selective-feedback experiment

Model: facebook/layerskip-llama3.2-1B at 81deb0f88734409cca506bcefcfb5c6bd2667565; fp16 on cuda:0.
Objective: energy. Energy boundary: nvidia_gpu.
Counter: NVML cumulative GPU energy counter (mJ).
Observer pairs: 3 across 3 calibration prompts.
Best executed fixed setting varies across prompts: False.
Prompts: 4 calibration, 4 held out; 1 held-out repeat(s), 32 maximum generated tokens.
All 65 executed requests matched native greedy token IDs.

Observer projection period: 1 verification cycle(s).

| Policy | Runs | Latency s/token | Total J/token | Static idle J/token | Dynamic excess J/token | Greedy mismatches |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 4 | 0.026827 | 1.507695 | not measured | not measured | 0 |
| fixed | 4 | 0.022912 | 1.201328 | not measured | not measured | 0 |
| fixed_d12_k3 | 4 | 0.026652 | 1.458891 | not measured | not measured | 0 |
| never_observe | 4 | 0.028065 | 1.553789 | not measured | not measured | 0 |
| selected_observe | 4 | 0.030711 | 1.821039 | not measured | not measured | 0 |
| always_observe | 4 | 0.029915 | 1.803945 | not measured | not measured | 0 |
| bandit | 4 | 0.026641 | 1.376461 | not measured | not measured | 0 |
| adaptive | 4 | 0.023200 | 1.234594 | not measured | not measured | 0 |
| frozen_context | 4 | 0.026132 | 1.373406 | not measured | not measured | 0 |
| del | 4 | 0.030514 | 1.674875 | not measured | not measured | 0 |

Observed feedback cost per request (selected/all): selected_observe 7.969000 J, always_observe 8.538500 J.
Observer overhead is measured by same-prompt paired executions. Raw runs, order, token IDs, valid feedback, and paired deltas are in p5_live_experiment.json.
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
| ordinary | valid | 120 |
| fixed | candidate mean request energy is not lower | no supported crossing |
| never_observe | valid | 102 |
| fixed_d12_k3 | valid | 145 |

## Task quality (one greedy completion per task)

Timing repeats are not independent quality samples. Scores use repeat 0, including greedy mismatches.

| Policy | GSM8K accuracy | GSM8K tasks | HumanEval pass@1 | HumanEval tasks |
|---|---:|---:|---:|---:|
| frozen_context | 0.0000 | 2 | not executed | 2 |
| fixed | 0.0000 | 2 | not executed | 2 |
| always_observe | 0.0000 | 2 | not executed | 2 |
| bandit | 0.0000 | 2 | not executed | 2 |
| del | 0.0000 | 2 | not executed | 2 |
| adaptive | 0.0000 | 2 | not executed | 2 |
| never_observe | 0.0000 | 2 | not executed | 2 |
| fixed_d12_k3 | 0.0000 | 2 | not executed | 2 |
| selected_observe | 0.0000 | 2 | not executed | 2 |
| ordinary | 0.0000 | 2 | not executed | 2 |

## DEL comparison (independent greedy paper port)

This is a local implementation comparison, not a reproduction of published DEL speedups.
DEL's shadow projections and decisions are measured. Adaptive request-level decisions/updates
remain outside the generation meter; shared calibration is separate. These are not end-to-end savings.
Positive reduction means adaptive used less cost. No significance claim is made.

| Comparison | Exact-output valid | GPU energy reduction | Latency reduction |
|---|---|---:|---:|
| adaptive vs del | True | 26.29% | 23.97% |
| adaptive vs fixed_d12_k3 | True | 15.37% | 12.95% |

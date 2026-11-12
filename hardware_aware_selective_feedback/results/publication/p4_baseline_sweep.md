# Phase P4: Development Fixed Baseline Sweep & Selection

- **Frozen At (UTC):** `2026-11-10T16:32:40.183902+00:00`
- **Selection Objective:** `energy_j_per_token`
- **Winning Action:** `fixed_d4_k1` (depth=4, draft_length=1)
- **Winner Metric:** `1.16394921875`
- **Candidates Evaluated:** 13
- **Total Tuning Energy:** 4620.5840 J
- **Total Tuning Time:** 80.38 s

## Evaluated Candidates

| Candidate | Depth | Draft Length | Runs | Mismatches | Latency (s/tok) | Energy (J/tok) | Valid Winner |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `fixed_d4_k1` | 4 | 1 | 4 | 0 | 0.021722 | 1.163949 | Yes |
| `fixed_d4_k2` | 4 | 2 | 4 | 0 | 0.019741 | 1.166734 | Yes |
| `fixed_d4_k3` | 4 | 3 | 4 | 0 | 0.021045 | 1.239824 | Yes |
| `fixed_d8_k1` | 8 | 1 | 4 | 0 | 0.022252 | 1.302594 | Yes |
| `fixed_d4_k4` | 4 | 4 | 4 | 0 | 0.021792 | 1.313270 | Yes |
| `fixed_d8_k2` | 8 | 2 | 4 | 0 | 0.023070 | 1.315973 | Yes |
| `fixed_d8_k3` | 8 | 3 | 4 | 0 | 0.022529 | 1.325816 | Yes |
| `fixed_d8_k4` | 8 | 4 | 4 | 0 | 0.025437 | 1.457215 | Yes |
| `fixed_d12_k1` | 12 | 1 | 4 | 0 | 0.025965 | 1.476098 | Yes |
| `ordinary` | 0 | 0 | 4 | 0 | 0.026274 | 1.504348 | Yes |
| `fixed_d12_k2` | 12 | 2 | 4 | 0 | 0.027306 | 1.538145 | Yes |
| `fixed_d12_k3` | 12 | 3 | 4 | 0 | 0.028207 | 1.606609 | Yes |
| `fixed_d12_k4` | 12 | 4 | 4 | 0 | 0.028639 | 1.638582 | Yes |

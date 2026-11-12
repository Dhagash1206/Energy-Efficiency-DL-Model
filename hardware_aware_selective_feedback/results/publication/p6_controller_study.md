# Phase P6: Controller Ablation Study & Economic Probe Analysis

- **Adaptive Beats Fixed Baseline:** `False`
- **Deployment Recommendation:** Retain the development-selected fixed baseline as deployment choice; adaptive selector overhead exceeds its benefit.
- **Probe Purchase Justified Economically:** `False`

## Controller Comparisons

| Controller | Runs | Energy (J/token) | Latency (s/token) | Greedy Mismatches |
|---|---:|---:|---:|---:|
| `fixed` | 4 | 1.201328 | 0.022912 | 0 |
| `frozen_context` | 4 | 1.373406 | 0.026132 | 0 |
| `bandit` | 4 | 1.376461 | 0.026641 | 0 |
| `adaptive` | 4 | 1.234594 | 0.023200 | 0 |
| `ordinary` | 4 | 1.507695 | 0.026827 | 0 |
| `never_observe` | 4 | 1.553789 | 0.028065 | 0 |
| `selected_observe` | 4 | 1.821039 | 0.030711 | 0 |
| `always_observe` | 4 | 1.803945 | 0.029915 | 0 |

## Economic Probe Decision Rule

- **Measured Probe Hurdle Cost:** 0.0011 J
- **Expected Future Benefit:** 0.0000 J
- **Net Expected Gain:** -0.0011 J
- **Economic Decision:** `REJECT PROBES`

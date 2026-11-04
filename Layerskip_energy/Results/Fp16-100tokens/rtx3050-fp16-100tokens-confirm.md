# Fixed decoding configuration sweep

8 held-out prompts; 2 repeats; 100 maximum tokens.
Energy boundary: nvidia_gpu. 30/32 requests matched greedy tokens.

| Depth | Draft length | Energy J/token | Latency s/token | Median paired energy delta J/request | Greedy matches |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1.349146 | 0.026584 | -28.025000 | False |
| 0 | 0 | 1.662457 | 0.031798 | +0.000000 | True |

Only comparisons with a correct ordinary baseline and no candidate mismatches are eligible.
Exploratory sweep: confirm any promising setting on fresh prompts and repeats.

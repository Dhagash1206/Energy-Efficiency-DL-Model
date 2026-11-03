# Fixed decoding configuration sweep

8 held-out prompts; 2 repeats; 200 maximum tokens.
Energy boundary: nvidia_gpu. 28/32 requests matched greedy tokens.

| Depth | Draft length | Energy J/token | Latency s/token | Median paired energy delta J/request | Greedy matches |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1.112307 | 0.021220 | -67.910500 | False |
| 0 | 0 | 1.426454 | 0.026188 | +0.000000 | True |

Only comparisons with a correct ordinary baseline and no candidate mismatches are eligible.
Exploratory sweep: confirm any promising setting on fresh prompts and repeats.

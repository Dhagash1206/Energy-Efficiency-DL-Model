# Fixed decoding configuration sweep

8 held-out prompts; 2 repeats; 200 maximum tokens.
Energy boundary: nvidia_gpu. 108/144 requests matched greedy tokens.

| Depth | Draft length | Energy J/token | Latency s/token | Median paired energy delta J/request | Greedy matches |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 0.925238 | 0.017469 | -98.697000 | False |
| 4 | 2 | 0.992830 | 0.018909 | -84.995500 | False |
| 3 | 1 | 1.027213 | 0.019536 | -73.764500 | False |
| 6 | 2 | 1.055404 | 0.019773 | -72.357500 | False |
| 4 | 1 | 1.080328 | 0.020882 | -66.334000 | False |
| 6 | 1 | 1.118036 | 0.021360 | -58.397000 | False |
| 2 | 1 | 1.243311 | 0.023908 | -32.321500 | False |
| 2 | 2 | 1.246358 | 0.023600 | -10.927000 | False |
| 0 | 0 | 1.395264 | 0.026759 | +0.000000 | True |

Only comparisons with a correct ordinary baseline and no candidate mismatches are eligible.
Exploratory sweep: confirm any promising setting on fresh prompts and repeats.

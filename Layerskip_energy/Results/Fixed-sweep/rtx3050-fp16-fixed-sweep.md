# Fixed decoding configuration sweep

8 held-out prompts; 2 repeats; 16 maximum tokens.
Energy boundary: nvidia_gpu. All 208 requests matched greedy tokens.

| Depth | Draft length | Energy J/token | Latency s/token | Median paired energy Δ J/request |
|---:|---:|---:|---:|---:|
| 4 | 2 | 1.535375 | 0.033609 | -2.308500 |
| 4 | 1 | 1.551594 | 0.033074 | -3.030000 |
| 8 | 1 | 1.567621 | 0.033613 | -3.090000 |
| 12 | 1 | 1.632473 | 0.036486 | -1.144500 |
| 8 | 2 | 1.672168 | 0.036040 | -0.252000 |
| 0 | 0 | 1.711527 | 0.036645 | +0.000000 |
| 12 | 2 | 1.721035 | 0.036876 | -0.347000 |
| 4 | 3 | 1.722270 | 0.036939 | -0.711000 |
| 12 | 3 | 1.731141 | 0.037799 | -0.977000 |
| 12 | 4 | 1.771125 | 0.037854 | -0.433500 |
| 8 | 3 | 1.834500 | 0.038372 | -0.894000 |
| 8 | 4 | 1.955082 | 0.043258 | +2.253000 |
| 4 | 4 | 1.991387 | 0.042854 | +3.696500 |

Exploratory sweep: confirm any promising setting on fresh prompts and repeats.

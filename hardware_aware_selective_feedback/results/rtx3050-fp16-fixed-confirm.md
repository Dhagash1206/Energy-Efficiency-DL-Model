# Fixed decoding configuration sweep

8 held-out prompts; 2 repeats; 16 maximum tokens.
Energy boundary: nvidia_gpu. 74/80 requests matched greedy tokens.

| Depth | Draft length | Energy J/token | Latency s/token | Median paired energy delta J/request | Greedy matches |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1.492836 | 0.032761 | -1.474500 | True |
| 8 | 1 | 1.608973 | 0.035667 | -1.808000 | False |
| 4 | 2 | 1.622520 | 0.035759 | -0.876500 | False |
| 0 | 0 | 1.690754 | 0.037696 | +0.000000 | True |
| 8 | 2 | 1.722449 | 0.037660 | +1.230500 | False |

Exploratory sweep: confirm any promising setting on fresh prompts and repeats.

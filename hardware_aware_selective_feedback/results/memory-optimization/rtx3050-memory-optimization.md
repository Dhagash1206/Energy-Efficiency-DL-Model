# LayerSkip memory optimization

Same-session randomized before/after comparison with the current pre-change
engine saved in a temporary snapshot (source SHA-256 hashes are in the JSON).
RTX 3050, FP16, 8 prompts, seed 43, 2 repeats, maximum 32 generated tokens.
Both implementations were warmed up. Energy boundary: NVIDIA GPU.

| Metric, depth 4 / draft length 1 | Before | After |
|---|---:|---:|
| Maximum peak allocated GPU memory | 2428.69 MiB | 2387.37 MiB |
| Maximum working memory above pre-request allocation | 62.44 MiB | 21.12 MiB |
| GPU energy per token | 1.510766 J | 1.515699 J |
| Latency per token | 0.024812 s | 0.024573 s |

Working memory decreased approximately 66%; total peak allocated memory
decreased 41.32 MiB (1.7%). Model weights dominate total GPU allocation.
Energy increased about 0.3%, so this experiment does not show an energy saving
for LayerSkip. Latency decreased about 1%; more repeats are needed to establish
such a small timing difference.

All 64 measured requests (including ordinary decoding controls) matched native
greedy token IDs. Every paired before/after output was identical. The 27 unit
tests passed, including a new check that final-position projection preserves
the KV cache and predictions on the small test model.

Changes: project only the final prompt position into the vocabulary, and release
obsolete forward outputs before later decoding work. GPU utilization percentage
was not measured. PyTorch allocated-memory peaks exclude driver memory and
reserved but unused allocator blocks. Raw data: `rtx3050-memory-optimization.json`.

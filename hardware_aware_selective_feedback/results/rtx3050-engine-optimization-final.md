# Decoder optimization check

Randomized same-session comparison of pinned pre-optimization Git revision
`71b0a23` and the working-tree engine.
FP16 CUDA, 8 prompts (seed 43), 3 repeats, 16 generated-token maximum, warm-up
for both implementations. Measurement boundary: NVIDIA GPU. All 96 measured
requests matched native greedy tokens.

| Path | Baseline seconds/token | Optimized seconds/token | Baseline J/token | Optimized J/token |
|---|---:|---:|---:|---:|
| Ordinary | 0.034797 | 0.029404 | 2.141505 | 1.921201 |
| Depth 4, draft length 1 | 0.030464 | 0.026257 | 1.742505 | 1.742492 |

Ordinary latency decreased 15.5%; speculative latency decreased 13.8%.
Ordinary GPU energy decreased 10.3%; speculative GPU energy was effectively
unchanged. These short measurements do not establish a general energy improvement.
Raw measurements and the baseline Git revision are in
`rtx3050-engine-optimization-final.json`.

Changes reuse rotary embeddings across layers, avoid all-ones padding masks for
unpadded inputs, and reuse device token tensors instead of uploading token IDs
again. Cache format conversions and host synchronization for decisions remain.

The separate `rtx3050-mismatch-regression.json` check reproduces the existing
FP16 mismatch on `gsm8k-test-245` for (4, 2), (8, 1), and (8, 2). Old and new
engines produce identical tokens on those checks. The cause is unresolved.
The preliminary `rtx3050-engine-optimization.json` used a moving Git HEAD and
is superseded by this pinned comparison.

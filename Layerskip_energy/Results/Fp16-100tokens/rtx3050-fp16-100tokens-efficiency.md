# 100-token efficiency evaluation

## Setup

- Model: `facebook/layerskip-llama3.2-1B` at pinned revision `81deb0f88734409cca506bcefcfb5c6bd2667565`
- Hardware: CUDA GPU, FP16; energy measured by NVML at the GPU boundary
- Workload: 8 held-out prompts (4 GSM8K and 4 HumanEval), 2 repeats, up to 100 generated tokens per request
- Comparison: ordinary greedy against LayerSkip depth 4 / draft length 1; all outputs checked token-for-token against greedy
- Run: [`rtx3050-fp16-100tokens-confirm.json`](rtx3050-fp16-100tokens-confirm.json)

## Results

| Decode mode | Requests | Greedy matches | Energy (J/token) | Latency (s/token) | Energy change vs greedy | Latency change vs greedy | Eligible exact-output comparison |
|---|---:|---:|---:|---:|---:|---:|---|
| Ordinary greedy (depth 0) | 16 | 16/16 (100%) | 1.6625 | 0.03180 | baseline | baseline | Yes |
| LayerSkip (depth 4, draft 1) | 16 | 14/16 (87.5%) | 1.3491 | 0.02658 | -18.9% | -16.4% | **No** |

The candidate had two mismatches, both on `gsm8k-test-880` at generated token 21 (both repeats). Its lower measured cost therefore does not qualify as an efficiency improvement while preserving greedy output. Exclude it from exact-output claims.

## Takeaway

This run establishes the ordinary greedy baseline at **1.6625 GPU J/token** and **31.80 ms/token** on this workload. The tested speculative configuration is faster and uses less measured GPU energy, but it is not token-equivalent on one prompt, so the project does not yet demonstrate a valid exact-output efficiency gain for this setting.

Measurements cover the NVIDIA GPU only, not the whole computer. This is a small held-out workload; the report does not score answer quality or execute generated code. Use additional prompts and diagnose the FP16 mismatch before making a broader performance claim.

# Phase P3: Model Quality Development Diagnostics

- **Evaluated Tasks:** 4
- **Precision:** fp16 on cuda
- **Peak GPU Memory:** 2426.52 MiB
- **Wall Time:** 27.29s

## Summary by Dataset

| Dataset | Tasks | Passed | Repetition Loops | Truncation (Hit Budget) | Main Failure Mode |
|---|---:|---:|---:|---:|---|
| gsm8k | 2 | 0/2 | 0 | 2 | truncation |
| humaneval | 2 | 0/2 | 0 | 2 | truncation |

## Per-Task Failure Diagnostics

| Task ID | Dataset | Prompt Tokens | Output Tokens | Finish Reason | Repetitive | Diagnosed Failure | Latency (s) |
|---|---|---:|---:|---|:---:|---|---:|
| `gsm8k-test-3` | gsm8k | 39 | 256 | `stop_string` | No | `truncation` | 7.181 |
| `gsm8k-test-1126` | gsm8k | 56 | 256 | `stop_string` | No | `truncation` | 5.977 |
| `HumanEval/25` | humaneval | 121 | 256 | `length` | No | `truncation` | 6.650 |
| `HumanEval/141` | humaneval | 210 | 256 | `length` | No | `truncation` | 6.174 |

## P3 Decision Gate Assessment

- **Capability vs Decoding:** Full-depth baseline answers the tasks directly; any failure here represents checkpoint capability or formatting, not speculative decoding error.
- **Repetition & Truncation:** Unwanted repetition loops and token limit exhaustions are separated from semantic reasoning failures.

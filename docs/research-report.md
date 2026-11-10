# Energy-Efficiency Research Report

## Executive summary

This repository investigates whether a pretrained LayerSkip language model can
produce the same greedy output with lower measured inference cost on consumer
hardware. The main contribution is an experimental controller around
self-speculative decoding: early layers draft tokens, the full model verifies
them, and a hardware-cost selector chooses decoding configurations. The
repository also contains experiments for selective intermediate-layer feedback,
the DEL adaptive-decoding baseline, decoder-engine optimization, memory
reduction, and FP16/fixed-configuration sweeps.

The strongest measured result is the latest FP32 CUDA run:

- **Adaptive policy:** 2.671720 GPU J/token and 0.060672 s/token.
- **Ordinary greedy baseline:** 4.129220 GPU J/token and 0.105173 s/token.
- Relative to ordinary greedy, this is **35.3% lower measured GPU energy** and
  **42.3% lower measured latency** (1.73x the token throughput).
- **381/381 measured requests matched native greedy token IDs.**

This is a feasibility result, not a proof of general efficiency. It uses one
RTX 3050 Laptop GPU, one pinned LayerSkip checkpoint, a small prompt workload,
and an NVIDIA-GPU energy boundary rather than whole-laptop energy. The adaptive
held-out run bought no observers, so it does **not** establish that selective
feedback caused the improvement. A matched fixed depth-12/draft-3 baseline and
an isolated feedback ablation are still required.

## 1. What the codebase implements

The executable research system is in
[hardware_aware_selective_feedback/](../hardware_aware_selective_feedback/).
[engine.py](../hardware_aware_selective_feedback/engine.py) implements native
greedy decoding and LayerSkip self-speculative decoding with KV-cache rollback.
The early layers draft tokens; the remaining layers verify them, retaining the
accepted prefix and correcting rejected tokens. Native greedy output is the
exact-output reference.

[selector.py](../hardware_aware_selective_feedback/selector.py) represents an
action as `(exit depth, draft length, observed exits)`. It records executed
cost per token and can select ordinary decoding or speculative actions. It also
supports context-aware cost tables and an independent executed-cost bandit.
The selector is a heuristic; it has no formal optimality or value-of-information
guarantee.

[feedback.py](../hardware_aware_selective_feedback/feedback.py) counts only
one-step observations belonging to the committed prefix. Predictions from
rejected or post-stop positions are excluded rather than counted as failures.
[measurement.py](../hardware_aware_selective_feedback/measurement.py) measures
cumulative counters and labels the measurement boundary. On the main runs this
was NVML GPU energy in millijoules converted to joules.

The runner in [run.py](../hardware_aware_selective_feedback/run.py) separates
calibration and held-out prompts, randomizes policy order, records raw
per-request measurements, and checks every output against native greedy tokens.
The reporting conventions are described in
[hardware_aware_selective_feedback/README.md](../hardware_aware_selective_feedback/README.md).

## 2. Measurement conventions

Unless stated otherwise, the CUDA results use:

- `facebook/layerskip-llama3.2-1B`, revision
  `81deb0f88734409cca506bcefcfb5c6bd2667565`
- NVIDIA GeForce RTX 3050 Laptop GPU, `cuda:0`
- NVML cumulative GPU energy counter
- GPU boundary only, not CPU, display, memory, storage, or whole-laptop energy
- GSM8K and HumanEval prompt workloads
- Token-equivalence checks against native greedy decoding

`J/token` is total measured request energy divided by generated tokens.
`s/token` is measured request latency divided by generated tokens. These
measurements include prompt prefill and internal generation work, but exclude
model loading and request-level controller decisions outside the meter.
Repeated executions of the same prompt are not independent prompts.

Exact greedy agreement means the generated token IDs are identical to the
native greedy reference. It does **not** mean that GSM8K answers are correct or
that HumanEval code passes tests; the repository does not score those tasks.

## 3. Latest adaptive selective-feedback experiment

Source: [latest.md](../hardware_aware_selective_feedback/results/latest.md) and
[latest.json](../hardware_aware_selective_feedback/results/latest.json).

Setup: 8 calibration prompts, 16 held-out prompts, 3 held-out repeats, and a
64-token maximum. There were 48 held-out requests per policy and 3,072 held-out
tokens per policy. All 381 measured requests, including calibration and
held-out requests, matched native greedy output.

| Policy | Runs | Latency s/token | GPU energy J/token | Energy vs ordinary | Latency vs ordinary |
|---|---:|---:|---:|---:|---:|
| ordinary greedy | 48 | 0.105173 | 4.129220 | baseline | baseline |
| fixed | 48 | 0.064210 | 2.784979 | -32.6% | -38.9% |
| never observe | 48 | 0.063219 | 2.764236 | -33.1% | -39.9% |
| selected observe | 48 | 0.070298 | 3.112311 | -24.6% | -33.2% |
| always observe | 48 | 0.072359 | 3.214446 | -22.2% | -31.2% |
| bandit | 48 | 0.065241 | 2.791580 | -32.4% | -38.0% |
| **adaptive** | **48** | **0.060672** | **2.671720** | **-35.3%** | **-42.3%** |

The adaptive policy selected depth 12/draft length 3 for 47 held-out requests
and depth 8/draft length 3 for one. It used no paid observations during
held-out adaptive requests. Observer calibration was still performed before
the held-out phase, and its cost was reported separately rather than amortized:
8,991.701 J of calibration energy. The measured observer overhead was 21.680
J/request for selected observation and 28.613 J/request for observing all
configured exits. Therefore, the latest table demonstrates a strong adaptive
decoding result, but not a benefit from purchasing feedback at serving time.

The earlier smaller FP32 run in
[benchmark/rtx3050-fp32-energy-safe.md](../hardware_aware_selective_feedback/results/benchmark/rtx3050-fp32-energy-safe.md)
also had exact agreement (133/133), but is superseded for the main result by
the larger latest run. It measured ordinary at 4.392223 J/token and adaptive
at 3.482609 J/token, a 20.7% reduction.

## 4. Methods and results

### 4.1 Native greedy decoding

Native greedy decoding runs the full model autoregressively and is the
correctness and cost reference. It is not an efficiency method, but every
candidate method is required to reproduce its token IDs before an exact-output
efficiency claim is accepted.

In the latest FP32 run, the reference cost was 4.129220 GPU J/token and
0.105173 s/token. In the latest FP16 16-token repeat, the ordinary reference
was 1.684565 GPU J/token and 0.037662 s/token, with exact agreement.

### 4.2 LayerSkip fixed self-speculation

Fixed LayerSkip uses an early exit at a selected depth and drafts a fixed
number of tokens before full-model verification. It can reduce cost when the
accepted draft prefix repays the extra draft and verification work.

The most recent dedicated confirmation is
[rtx3050-fp16-fixed-confirm.md](../hardware_aware_selective_feedback/results/fixed-sweep/rtx3050-fp16-fixed-confirm.md):

| Depth | Draft length | Energy J/token | Latency s/token | Greedy result |
|---:|---:|---:|---:|---|
| 4 | 1 | 1.492836 | 0.032761 | valid |
| 8 | 1 | 1.608973 | 0.035667 | invalid |
| 4 | 2 | 1.622520 | 0.035759 | invalid |
| 0 | 0 | 1.690754 | 0.037696 | valid baseline |
| 8 | 2 | 1.722449 | 0.037660 | invalid |

The depth-4/draft-1 candidate was about 11.7% lower energy and 13.1% lower
latency than the ordinary baseline in this 16-token run, while preserving
greedy output. The other tested candidates cannot support exact-output claims.
The earlier exploratory sweep had a more favorable depth-4/draft-1 number, but
the confirmation artifact is the appropriate result to cite.

Longer-output tests were less reliable. In the latest 100-token confirmation,
[rtx3050-fp16-100tokens-confirm.md](../hardware_aware_selective_feedback/results/fp16-100tokens/rtx3050-fp16-100tokens-confirm.md),
depth 4/draft 1 measured 1.349146 J/token versus the 1.662457 J/token ordinary
baseline and 0.026584 versus 0.031798 s/token. However, it mismatched greedy
on both repeats of `gsm8k-test-880` at token 21, so the apparent 18.9% energy
and 16.4% latency gains are **not valid exact-output improvements**.

The 200-token sweep had even lower candidate costs (best 0.925238 J/token),
but only 28/32 requests matched greedy and every candidate row was invalid.
Those values are diagnostic evidence of the speed/correctness tradeoff, not
deployable savings.

### 4.3 Selective intermediate-exit feedback

The system can observe intermediate-layer predictions during speculative
decoding. `selected_observe` observes a configured exit, while
`always_observe` observes all configured exits. Observations are collected
without an extra transformer forward pass, but hooks, projections, confidence
calculation, and synchronization still cost energy and time.

In the latest FP32 run, selected observation used 3.112311 GPU J/token and
always observation used 3.214446 J/token. Both were worse than
`never_observe` at 2.764236 J/token. Their paired overheads were substantial,
so the measurements do not show that serving-time observation is worthwhile.

Feedback remains useful as calibration information for the controller, but the
current experiments do not isolate that information value from configuration
selection. A proper ablation must hold the action space and controller fixed
while varying only no feedback, selective feedback, and always-on feedback.

### 4.4 Adaptive selector and cost bandit

The adaptive selector minimizes measured executed cost per token across
ordinary and speculative actions, then optionally buys observations when its
heuristic estimates that uncertainty reduction justifies the measured
overhead. The independent bandit uses executed-cost arms as a comparison.

Latest FP32 results:

- Bandit: 2.791580 J/token and 0.065241 s/token, 32.4% and 38.0% below
  ordinary respectively.
- Adaptive: 2.671720 J/token and 0.060672 s/token, 35.3% and 42.3% below
  ordinary respectively.

The adaptive result is better than the bandit in this workload, but the
comparison is descriptive. The adaptive action space and the held-out fixed
baseline are not yet matched sufficiently to establish that the heuristic,
rather than a favorable fixed configuration, caused the gain.

### 4.5 DEL comparison

[del_policy.py](../hardware_aware_selective_feedback/del_policy.py) is an
independent greedy port of DEL: it tracks decayed acceptance/confidence
statistics, chooses an exit using the TPL objective, dynamically stops drafts,
and updates only on valid committed context. The comparison conventions and
limitations are documented in
[DEL_BASELINE.md](../hardware_aware_selective_feedback/DEL_BASELINE.md).

In the latest available DEL pilot,
[pilot.md](../hardware_aware_selective_feedback/results/del-comparison/pilot.md),
all 97 executed requests matched greedy:

| Policy | Energy J/token | Latency s/token |
|---|---:|---:|
| adaptive | 3.658246 | 0.076396 |
| fixed depth 12 | 3.452809 | 0.070973 |
| DEL | 4.714074 | 0.093979 |

The matched pair comparison reported adaptive versus DEL as 22.40% lower GPU
energy and 18.71% lower latency. Adaptive was 5.95% higher energy and 7.64%
higher latency than fixed depth 12. This is a comparison against this local
DEL port, not against the upstream implementation or its published speedups.
It is also a small pilot, and DEL's shadow projections and decisions are
inside the generation meter while shared calibration and request-level
controller work are treated separately.

### 4.6 Decoder-engine optimization

The engine optimization reuses rotary embeddings, avoids unnecessary
all-ones masks for unpadded input, and keeps token tensors on the device.
The latest pinned comparison is
[rtx3050-engine-optimization-final.md](../hardware_aware_selective_feedback/results/engine-optimization/rtx3050-engine-optimization-final.md).

| Path | Baseline s/token | Optimized s/token | Baseline J/token | Optimized J/token |
|---|---:|---:|---:|---:|
| ordinary | 0.034797 | 0.029404 | 2.141505 | 1.921201 |
| depth 4/draft 1 | 0.030464 | 0.026257 | 1.742505 | 1.742492 |

All 96 measured requests matched native greedy output. Ordinary latency fell
15.5% and ordinary GPU energy fell 10.3%. Speculative latency fell 13.8%, but
speculative energy was effectively unchanged. This is a useful implementation
optimization, but it does not explain the latest adaptive result because the
latest report is a different experiment and software state.

The separate mismatch regression reproduces the existing FP16 mismatch for
depth/length `(4,2)`, `(8,1)`, and `(8,2)` before and after the optimization;
the optimization did not introduce that mismatch.

### 4.7 Memory optimization

The memory optimization projects only the final prompt position into the
vocabulary and releases obsolete forward outputs. Its latest report is
[rtx3050-memory-optimization.md](../hardware_aware_selective_feedback/results/memory-optimization/rtx3050-memory-optimization.md).

| Metric, depth 4/draft 1 | Before | After |
|---|---:|---:|
| Maximum peak allocated GPU memory | 2428.69 MiB | 2387.37 MiB |
| Working memory above pre-request allocation | 62.44 MiB | 21.12 MiB |
| GPU energy/token | 1.510766 J | 1.515699 J |
| Latency/token | 0.024812 s | 0.024573 s |

Working memory fell approximately 66%, while total peak allocated memory fell
41.32 MiB (1.7%). Energy increased approximately 0.3%; therefore this method
was successful as a working-memory optimization but **did not demonstrate an
energy saving**. All 64 requests matched native greedy output.

### 4.8 Precision change: FP32 to FP16

FP16 reduces model storage and can improve throughput on supported CUDA
hardware, but the project correctly treats it as a separate correctness
condition. The 16-token FP16 runs show lower absolute costs than FP32, but
they are not a controlled precision-only comparison because workload and run
conditions differ. More importantly, several speculative FP16 configurations
mismatch greedy output on held-out prompts. FP16 therefore cannot be claimed
as an exact-output efficiency improvement for those configurations.

### 4.9 Earlier distillation and prototype work

[edge_llm_inference/distill.py](../edge_llm_inference/distill.py) contains an
earlier Qwen3 1.7B-to-0.6B knowledge-distillation script. It uses a mixed
soft teacher KL loss and hard language-model loss, but the repository does not
contain a completed energy benchmark proving that the distilled student is
more energy efficient or preserves task quality. It is research background,
not part of the latest LayerSkip result.

[prototype/](../prototype/) is a CPU-only synthetic controller demonstration. Its
energy profile is hand-authored and explicitly not a measurement. It is useful
for testing controller bookkeeping, not for reporting GPU efficiency.

The repository also mentions quantization, pruning, CALM, and HAQA-related
ideas. Quantization flags exist in the executable system, but the latest run
used FP32 with quantization disabled. The repository does not implement a
completed HAQA quantization-search agent or a validated end-to-end pruning/
CALM experiment. Those methods should not be presented as measured results.

## 5. Overall assessment

| Method | Exact-output evidence | Efficiency outcome | Assessment |
|---|---|---|---|
| Latest adaptive FP32 | 381/381 matches | -35.3% GPU J/token, -42.3% latency | Strongest feasibility result; not causal proof of feedback benefit |
| Fixed FP16 depth 4/draft 1, 16 tokens | Valid confirmation | About -11.7% energy, -13.1% latency | Promising short-run exact-output setting |
| Fixed FP16 depth 4/draft 1, 100 tokens | 30/32 matches | Apparent -18.9% energy | Invalid exact-output claim due to mismatch |
| Fixed FP16 200-token candidates | 28/32 overall matches | Very low candidate costs | Invalid candidates; correctness failure |
| Selective/always observers | Exact in latest run | More costly than never observing | Feedback overhead outweighed benefit in measured policies |
| Bandit | Exact in latest run | -32.4% energy, -38.0% latency | Strong comparator, slightly behind adaptive |
| DEL local port | 97/97 pilot matches | Worse than adaptive in pilot | Useful baseline; not an upstream DEL reproduction |
| Decoder-engine optimization | 96/96 matches | Ordinary energy -10.3%; speculative energy unchanged | Valid implementation optimization |
| Memory optimization | 64/64 matches | Working memory -66%; energy +0.3% | Memory success, not energy success |
| Qwen distillation | No energy/quality result | Not established | Unvalidated earlier direction |
| Synthetic prototype | Synthetic agreement only | No measured saving | Demonstration, not benchmark |

The most defensible conclusion is:

> Hardware-measured adaptive self-speculative decoding is feasible and produced
> a large exact-greedy cost reduction in one RTX 3050 FP32 workload. The
> repository does not yet prove that selective feedback itself improves
> efficiency, that the result generalizes across hardware/workloads, or that
> the adaptive controller beats the best deployable fixed configuration.

## 6. Required next experiments

1. Evaluate fixed depth 12/draft 3 on exactly the latest held-out prompts,
   repeats, token limit, and randomized order.
2. Compare the same controller and action set with no feedback, selective
   feedback, and always-on feedback.
3. Use calibration-only selection for the deployable fixed baseline; label
   held-out-selected settings as oracle analyses.
4. Increase independent prompt IDs and seeds, and report prompt-level paired
   variation or confidence intervals.
5. Report calibration and controller preparation costs using a clearly stated
   amortization policy.
6. Repeat on additional GPUs and energy boundaries, preserving exact runtime
   versions and driver information.
7. Diagnose and resolve FP16 mismatches before making longer-output claims.
8. Score GSM8K answers and execute/grade HumanEval code separately from
   token-equivalence checks.

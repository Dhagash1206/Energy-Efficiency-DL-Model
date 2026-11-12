# Publication evaluation protocol

These changes add evaluation tools. A successful pilot does not establish
publication readiness, task-quality parity, energy superiority, or novelty.
Historical raw reports are preserved; no unmeasured baseline or idle component
has been inserted into their tables.

For proposed work beyond these tools, see the [publication build plan](publication-build-plan.md)
and [coding-agent handoff](agent-handoff.md).

## What changed

| Component | Current implementation | Evidence and remaining work |
|---|---|---|
| Depth-12 baseline | `fixed_d12_k3` is a default policy alongside the previous seven | New runs measure eight policies; `--include-del` adds only DEL as a ninth |
| Calibration overhead | `publication.py` plots cumulative cost and break-even with all shared calibration charged to adaptive | Stationary-workload projection; future workload drift and excluded setup work remain limitations |
| Idle and active energy | `--idle-seconds S --idle-samples K` records resident-model idle counter samples and estimates request idle/dynamic excess energy | Reference subtraction is not a physical decomposition of GPU components |
| FP16 and task quality | `--task-quality --precision fp16` keeps full prompts, scores GSM8K, and saves HumanEval completions | HumanEval tests run separately in constrained Docker containers; use larger held-out sets for claims |
| Observer overhead | `--observer-period M` invokes paid LM-head observers only on cycles 0, M, 2M, ... | The same schedule is used for paired calibration and held-out requests; M=1 preserves dense probing |

The main table has **eight** default policies because depth 12 was added without
removing an existing comparison. Historical DEL reports call the same fixed
configuration `fixed_depth12`; new reports call it `fixed_d12_k3`.

## Calibration-inclusive break-even

For a candidate's mean request energy `E_a`, a baseline's mean request energy
`E_b`, and calibration energy `C`, the projections are:

```text
candidate cumulative energy(N) = C + N * E_a
baseline cumulative energy(N)  = N * E_b
N_breakeven = ceil(C / (E_b - E_a)), provided E_b > E_a
```

Break-even denotes candidate cost less than or equal to baseline cost. There
is no finite crossing when candidate mean request energy is not lower.
The tool rejects comparisons with missing matched requests, output mismatches,
missing energy, or mixed measurement boundaries. It charges all shared
calibration, including the bandit's calibration and idle sample energy, once
to adaptive. Fixed baselines receive zero calibration charge. This deliberately
conservative accounting is explicit; it is not a deployment measurement.

Model loading, native-greedy reference generation, and selector decisions
outside the generation meter remain excluded. The charts project measured
mean costs onto future requests from the same distribution; they do not
measure an empirical lifetime or provide a confidence interval. Prompt repeats
are correlated; they must not be treated as independent tasks.

Install optional plotting dependencies and analyze the preserved FP32 reference:

```powershell
python -m pip install -r requirements-analysis.txt
python -m hardware_aware_selective_feedback.publication --report hardware_aware_selective_feedback/results/latest.json --output-prefix hardware_aware_selective_feedback/results/publication/my-fp32-break-even --max-requests 2000
```

The tool writes JSON, Markdown, PNG, and PDF to unused paths. The saved
[FP32 projection](../hardware_aware_selective_feedback/results/publication/fp32-historical-break-even.md)
gives 97 requests versus ordinary, 1,241 versus fixed depth 4, and 1,519 versus
depth 8. Depth 12 is absent from that historical run and has no retrospective
crossing estimate. See its [chart](../hardware_aware_selective_feedback/results/publication/fp32-historical-break-even.png)
and [PDF](../hardware_aware_selective_feedback/results/publication/fp32-historical-break-even.pdf).

## Idle-reference energy accounting

After loading and warming the model, the runner synchronizes the device and
samples its cumulative energy counter while sleeping with the model resident.
GPU clock/power settings, when requested, are already applied. The idle estimate
is total sampled energy divided by total sampled duration. Each request keeps:

```text
energy_j              = measured total counter delta
static_idle_energy_j  = estimated_idle_power_w * measured_request_duration_s
dynamic_energy_j      = energy_j - static_idle_energy_j
```

All three values and their per-token aggregates appear in the new report.
Negative excess values are retained rather than clamped; they can indicate
noise, frequency-state differences, or an unsuitable idle reference. Without
idle calibration, both components remain null, including in historical files.
Policy selection and break-even continue using total measured energy.
Recalibrate for each run, precision, and device setting. The sensor boundary
remains GPU board or CPU package; subtraction does not produce whole-laptop
energy or isolate the energy of individual kernels.

## Full-prompt quality evaluation

Run a new FP16 experiment with a separate output path. This example is a small
workflow check with four held-out tasks, not a powered benchmark:

```powershell
python -m hardware_aware_selective_feedback.run --device cuda --precision fp16 --objective energy --calibration 2 --test 2 --tokens 256 --repeats 1 --observer-pairs 3 --observer-period 4 --idle-seconds 2 --idle-samples 3 --task-quality --output hardware_aware_selective_feedback/results/publication/my-fp16-m4.json
```

Counts are per dataset. The model sees full question/code prompts. GSM8K prompts
instruct it to finish with `#### <numeric answer>`; reference answers and hidden
HumanEval tests are saved as scoring metadata and never added to model input.
Quality evaluation is outside the generation energy meter. Raw completions,
token IDs, prompt token counts, and token-budget exhaustion flags are retained.
The fixed-size CUDA Graph option is rejected with full-prompt quality runs.

GSM8K uses the [upstream strict answer extraction](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py):
the first `#### ` numeric answer with commas removed must match the reference.
A missing delimiter is incorrect; there is no last-number fallback.
HumanEval uses prompt + raw completion + tests + `check(entry_point)`, matching
the [upstream test assembly](https://github.com/openai/human-eval/blob/master/human_eval/execution.py).
Completions receive no code repair or fence removal. This local container
runner is not the upstream multiprocessing runtime; the image and time limit
are recorded so runtime differences are visible.

One greedy completion per unique held-out task, from repeat 0, determines
accuracy and pass@1. Timing repeats are not new sampled candidates. Mismatching
outputs are still scored; they are not discarded to improve quality metrics.
HumanEval scores remain null until tests actually execute. Identical outputs
for a task across policies share the same test result.

Pull a Linux Python image once, then evaluate completions separately:

```powershell
docker pull python:3.12-slim
python -m hardware_aware_selective_feedback.quality --report hardware_aware_selective_feedback/results/publication/my-fp16-m4.json --docker-image python:3.12-slim --timeout 3 --output hardware_aware_selective_feedback/results/publication/my-fp16-m4-quality.json
```

The evaluator resolves the tag to an immutable image ID. Containers have no
host mounts, no network, a read-only root filesystem, an unprivileged user,
no capabilities, and limits on memory, CPUs, process count, and runtime. Host
Python never executes generated completions. Docker infrastructure errors
abort scoring instead of being counted as wrong answers. Containers are
removed even on timeout. Use trusted evaluation machines and retain the
container image ID with reported scores.

## Saved validation evidence

The [new FP16 pilot](../hardware_aware_selective_feedback/results/publication/fp16-m4-quality-pilot.md)
used four calibration prompts, four held-out prompts (two per dataset), one
held-out repeat, a 256-token limit, three observer calibration pairs, and M=4.
All 57 requests matched native greedy. The estimated idle reference was
10.464551 W. Both observer baselines recorded 69 probe cycles each across
their four held-out requests; adaptive scheduled no paid observations.

The [container-executed scores](../hardware_aware_selective_feedback/results/publication/fp16-m4-quality-scores.md)
were GSM8K 0/2 and HumanEval 0/2 for all policies. All 32 policy/task completions
hit the token limit. The GSM8K answers lacked the strict marker; the extraction
score therefore also measures format adherence. These identical outputs show
no observed difference between decoding policies on this tiny sample; they
do not show FP32/FP16 task-quality equivalence or a satisfactory task baseline.

Adaptive total GPU energy was 0.964938 J/token versus 0.948633 for fixed depth 4
and 1.205459 for fixed depth 12. Its [break-even projection](../hardware_aware_selective_feedback/results/publication/fp16-m4-break-even.md)
crossed ordinary at 64 requests and depth 12 at 136 requests, with no crossing
against fixed depth 4. The [PNG](../hardware_aware_selective_feedback/results/publication/fp16-m4-break-even.png)
and [PDF](../hardware_aware_selective_feedback/results/publication/fp16-m4-break-even.pdf)
include calibration and idle sampling. Do not present this as adaptive beating
every fixed baseline, an M=1 versus M=4 ablation, or evidence that paid feedback helps.

## Observer subsampling and larger evaluation

Here, a step means a **verification cycle**, not a token. Accepted drafts can
commit several tokens per cycle; M=4 therefore is not every fourth generated
token. Skipped cycles register no observer hooks and perform no paid observer
LM-head projections. The required draft and verifier heads still run normally.
Reports record the period, actual probe-cycle counts, and valid/excluded
feedback. Paid adaptive observations still require sufficient paired
calibration and may remain unused when their estimated cost is not justified.

For an ablation, use M=1, 2, 4, and 8 with the same seed, disjoint task split,
precision, attention backend, token budget, and sample counts. Recalibrate each
period; dense overhead estimates cannot price sparse probes. Compare observed
overhead, feedback use, energy, and task scores, not just projection counts.

Before publication, expand the held-out sample and seeds, report per-task
paired uncertainty, compare FP32 and FP16 quality, inspect token-budget
exhaustion, and repeat across device/thermal conditions. The runner uses a
balanced subset count per dataset and reserves HumanEval IDs for calibration;
its subset scores must not be described as full-benchmark scores. Prespecify
the evaluation protocol and any prompt-format changes. Saved pilots and the
historical break-even extrapolation do not replace that larger evaluation.

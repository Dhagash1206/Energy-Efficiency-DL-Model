# Saved experiment results

Raw JSON measurements and generated Markdown reports are preserved in place.
Use each report's precision, workload, hardware, correctness checks, and measurement
boundary when interpreting its numbers. Different runs are not interchangeable.

The main [project README](../../README.md) describes `latest.json` / `latest.md`.
The FP16 held-out run in `final/` is a separate experiment and records three
greedy mismatches. The `bandit` and `adaptive` policies are included in the main
runner reports; DEL comparison runs are grouped separately.

## Reports and measurements

New evaluation tools and artifacts are described in the
[publication guide](../../docs/publication.md). The preserved FP32 reference
now also has a [calibration-inclusive break-even chart](publication/fp32-historical-break-even.png),
[PDF](publication/fp32-historical-break-even.pdf),
[projection table](publication/fp32-historical-break-even.md), and
[raw projection inputs/results](publication/fp32-historical-break-even.json).

The new [FP16 M=4 pilot](publication/fp16-m4-quality-pilot.md)
([JSON](publication/fp16-m4-quality-pilot.json)) includes the depth-12 baseline,
idle-reference energy, and full task prompts. Its
[executed quality scores](publication/fp16-m4-quality-scores.md)
([JSON](publication/fp16-m4-quality-scores.json)) are 0/2 on each dataset for
every policy; all completions hit the output limit. Its
[calibration-inclusive chart](publication/fp16-m4-break-even.png)
([PDF](publication/fp16-m4-break-even.pdf), [details](publication/fp16-m4-break-even.md),
[JSON](publication/fp16-m4-break-even.json)) shows no energy crossing versus
fixed depth 4. These are workflow-validation results, not publication claims.

| Group | Experiment | Report | Raw measurements |
|---|---|---|---|
| benchmark | `rtx3050-fp32-energy-safe` | [Markdown](benchmark/rtx3050-fp32-energy-safe.md) | [JSON](benchmark/rtx3050-fp32-energy-safe.json) |
| del-comparison | `pilot` | [Markdown](del-comparison/pilot.md) | [JSON](del-comparison/pilot.json) |
| del-comparison | `smoke` | [Markdown](del-comparison/smoke.md) | [JSON](del-comparison/smoke.json) |
| engine-optimization | `rtx3050-engine-optimization` | ? | [JSON](engine-optimization/rtx3050-engine-optimization.json) |
| engine-optimization | `rtx3050-engine-optimization-final` | [Markdown](engine-optimization/rtx3050-engine-optimization-final.md) | [JSON](engine-optimization/rtx3050-engine-optimization-final.json) |
| engine-optimization | `rtx3050-mismatch-regression` | ? | [JSON](engine-optimization/rtx3050-mismatch-regression.json) |
| final | `rtx3050-heldout` | [Markdown](final/rtx3050-heldout.md) | [JSON](final/rtx3050-heldout.json) |
| fixed-sweep | `rtx3050-fp16-fixed-confirm` | [Markdown](fixed-sweep/rtx3050-fp16-fixed-confirm.md) | [JSON](fixed-sweep/rtx3050-fp16-fixed-confirm.json) |
| fixed-sweep | `rtx3050-fp16-fixed-sweep` | [Markdown](fixed-sweep/rtx3050-fp16-fixed-sweep.md) | [JSON](fixed-sweep/rtx3050-fp16-fixed-sweep.json) |
| fp16-100tokens | `rtx3050-fp16-100tokens` | [Markdown](fp16-100tokens/rtx3050-fp16-100tokens.md) | [JSON](fp16-100tokens/rtx3050-fp16-100tokens.json) |
| fp16-100tokens | `rtx3050-fp16-100tokens-confirm` | [Markdown](fp16-100tokens/rtx3050-fp16-100tokens-confirm.md) | [JSON](fp16-100tokens/rtx3050-fp16-100tokens-confirm.json) |
| fp16-100tokens | `rtx3050-fp16-100tokens-efficiency` | [Markdown](fp16-100tokens/rtx3050-fp16-100tokens-efficiency.md) | ? |
| fp16-100tokens | `rtx3050-fp16-100tokens-fixed` | [Markdown](fp16-100tokens/rtx3050-fp16-100tokens-fixed.md) | [JSON](fp16-100tokens/rtx3050-fp16-100tokens-fixed.json) |
| fp16-200tokens | `rtx3050-fp16-200tokens` | [Markdown](fp16-200tokens/rtx3050-fp16-200tokens.md) | [JSON](fp16-200tokens/rtx3050-fp16-200tokens.json) |
| fp16-200tokens | `rtx3050-fp16-200tokens-sweep` | [Markdown](fp16-200tokens/rtx3050-fp16-200tokens-sweep.md) | [JSON](fp16-200tokens/rtx3050-fp16-200tokens-sweep.json) |
| fp16-run | `rtx3050-fp16-repeat2` | [Markdown](fp16-run/rtx3050-fp16-repeat2.md) | [JSON](fp16-run/rtx3050-fp16-repeat2.json) |
| fp16-run | `rtx3050-fp16-repeat3` | [Markdown](fp16-run/rtx3050-fp16-repeat3.md) | [JSON](fp16-run/rtx3050-fp16-repeat3.json) |
| fp16-run | `rtx3050-fp16-run` | [Markdown](fp16-run/rtx3050-fp16-run.md) | [JSON](fp16-run/rtx3050-fp16-run.json) |
| Main snapshot | `latest` | [Markdown](latest.md) | [JSON](latest.json) |
| memory-optimization | `rtx3050-memory-optimization` | [Markdown](memory-optimization/rtx3050-memory-optimization.md) | [JSON](memory-optimization/rtx3050-memory-optimization.json) |
| pilot | `layerskip-fp32-pilot` | [Markdown](pilot/layerskip-fp32-pilot.md) | [JSON](pilot/layerskip-fp32-pilot.json) |
| pilot | `rtx3050-pilot` | [Markdown](pilot/rtx3050-pilot.md) | [JSON](pilot/rtx3050-pilot.json) |
| pilot | `smollm-fp32-pilot` | [Markdown](pilot/smollm-fp32-pilot.md) | [JSON](pilot/smollm-fp32-pilot.json) |
| pilot | `validation` | ? | [JSON](pilot/validation.json) |
| smoke | `rtx3050-fp16-smoke` | [Markdown](smoke/rtx3050-fp16-smoke.md) | [JSON](smoke/rtx3050-fp16-smoke.json) |
| smoke | `rtx3050-smoke` | [Markdown](smoke/rtx3050-smoke.md) | [JSON](smoke/rtx3050-smoke.json) |

## Preserving results

Use an unused `--output` path for new runs. The main runner requires this path,
writes JSON and a Markdown companion, and refuses to overwrite either file.
The plotting and task-quality tools also require unused output paths.
Formatting tools exclude this directory.

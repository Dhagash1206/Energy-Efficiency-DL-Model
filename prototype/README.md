# Small controller prototype

This is a **synthetic, CPU-only demonstration**, not an LLM benchmark. It runs
with Python's standard library; no downloads, GPU, or API key are required.

From the repository root:

```powershell
python -m prototype.demo
python -m unittest prototype.test_prototype -v
```

The demo prints a comparison and writes `prototype/results/demo.json`, including
the decision trace. You can change the workload seed and output token budget:

## Output fields and what they mean

Each request result includes the following fields:

- prompt_id: stable dataset identifier for the request
- task: task family such as reasoning, summarization, or general QA
- policy: policy used for this run (ordinary, best_fixed, latency, or energy)
- source: whether the backend is synthetic or a real execution backend
- output_tokens: number of committed output tokens produced
- energy_j: total request energy in joules
- j_per_token: energy per committed token
- elapsed_seconds: request wall-clock runtime
- tokens_per_second: generation throughput
- controller_wall_seconds: time spent selecting actions
- draft_acceptance: fraction of speculative tokens accepted during draft verification
- ordinary_fraction: fraction of decisions that fell back to ordinary decoding
- action_counts: number of times each action was chosen
- trace: per-step decision trace with context length, action, and reason
- matches_ordinary: whether the generated tokens match ordinary decoding exactly
- energy_scope: notes what energy source is being reported

For a real run, these numbers come from actual backend execution. For the current
portable demo, they remain simulation values so the controller logic can be tested
without installing PyTorch or CUDA tooling.

```powershell
python -m prototype.demo --seed 7 --tokens 128
```

## What works

- Loads a clearly labeled synthetic cost profile for ordinary decoding and three
  exit-depth/draft-length pairs.
- Estimates energy and latency per committed token from context length and a
  recent acceptance history, with an independent-acceptance approximation.
- Chooses speculation only if its estimated upper energy margin is below the
  ordinary action's lower margin. The margins are illustrative, **not calibrated
  confidence intervals**. They do not capture acceptance-prediction uncertainty.
- Uses one toy token source for drafting and mandatory verification. Rejected
  draft suffixes are discarded; a correction or bonus token is committed.
- Compares ordinary, one fixed speculative configuration, latency selection,
  and energy selection. All must produce identical toy output.
- Includes illustrative controller overhead in both action estimates and reported
  simulated costs. Base cycle costs stand in for drafting, verification, and
  state handling together.

The toy workload becomes harder after token 48 to demonstrate how acceptance
feedback can change decisions. Its behavior and costs are invented. The fixed
baseline is an example, not the best fixed configuration. No measured saving or
claim of algorithmic novelty follows from these results.

## What is still missing

There are no neural model weights, actual model layers, quantization kernels,
parallel verification, KV-cache implementation, GPU measurements, or fitted
energy predictor. The sequential toy verifier only checks token bookkeeping.
The JSON profile stands in for offline preparation; it is hand-authored, not
fitted from experiments. Precision stays fixed as a label, not an executed
quantization mode. The prototype has no latency constraint or exploration policy;
after fallback, stale acceptance estimates can prevent it from resuming
speculation. These are explicit limits of this small demonstration.

The next milestone is a real LayerSkip backend with greedy correctness tests,
validated repeated-request energy measurements, fitted profiles with calibrated
uncertainty, and held-out evaluation. Do not feed real profiles to this toy
runner and interpret its output as measurements.

## Current laptop

The development machine was identified as a Dell Inspiron 13 5330 with an Intel
Core Ultra 7 155H, approximately 16 GB installed RAM, and Intel Arc integrated
graphics. It can run this demo. No NVIDIA GPU was reported by Windows, so this
machine does not provide the CUDA/NVML platform planned for RTX experiments.
CPU/Intel inference would require a separate backend and validation effort.

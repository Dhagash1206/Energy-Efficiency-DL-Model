# Selective-feedback feasibility pilot

This subpackage preserves the existing prototype. It implements cached greedy
drafting/verification, intermediate-exit observation, a prefix-validity filter,
descriptive feedback statistics, and a randomized fixed-configuration pilot.
The proposed joint selector is deliberately not implemented before feasibility
is established. No novelty or energy saving is claimed.

## Run from the repository root

Use an environment containing requirements.txt from this directory:

```powershell
python -B -m unittest prototype.selective_feedback.tests.test_pilot -v
python -B -m prototype.selective_feedback.pilot --model tiny
python -B -m prototype.selective_feedback.pilot --model smollm --prompts 2 --tokens 12 --repeats 3
```

The tiny model is randomly initialized and tests mechanics only. SmolLM2-135M
is an ungated Apache-2.0 pretrained Llama-family model. It is NOT exit-trained.
Its intermediate-exit acceptance cannot establish viability of an exit-trained
research system. LayerSkip remains a supported optional research checkpoint:

```powershell
python -B -m prototype.selective_feedback.pilot --model layerskip --device cuda --precision fp16
```

LayerSkip access requires the user's own Hugging Face approval/token. This code
does not accept license terms, share personal information or bypass gating.
If access remains unavailable, exit-training an open model is a separate
scope change, not something silently substituted into this pilot.

Downloads, dataset snapshots, caches and results live below this directory.
Hugging Face weights and the token are stored under the repository-root
`layerskip/cache` folder (see paths.py), which is git-ignored.
HF model revisions are resolved to immutable commits and recorded. GSM8K test
data is downloaded from a recorded Git commit with a SHA-256 digest. These
small samples are implementation checks, not task-quality evaluation.
Pass --precision int8 for real CPU dynamic quantization of decoder linear
layers; the shared head and embeddings remain floating point. Compare decoding
against the same quantized model. INT8 does not imply an exit-trained model.

## Interfaces and definitions

- engine.generate(model, prompt_ids, ...): committed tokens, cycle traces,
  validated match statistics. Batch size one; greedy only; EOS included.
- engine.native_greedy: independent Transformers cached baseline.
- engine.observe: selected layer hooks; normalized hidden states are projected
  using the shared vocabulary head. Lower-layer observations occur across
  draft calls and the remainder call; upper-layer observations occur in
  verification. No extra full-model pass is performed for observation.
- feedback.validate_feedback: retains exact conditioning token IDs. A prediction
  at the first rejected token is on a valid prefix; later rejected-branch
  predictions are excluded. Outputs after EOS are also excluded.
- feedback.LearningState: one-step matches/counts, NOT estimates of independent
  alternative rollouts, energy, or calibrated uncertainty.
- measurement.Meter: whole-request latency and optional NVML GPU cumulative
  joules. CPU energy is null, never estimated from TDP or time.
- pilot: fixed actions with none/one/all/periodic observation; randomized repeat
  order, warmup, native-output comparisons and paired observation deltas.

A layer is a model processing stage. A hidden state is its numerical token
representation. An exit projects an intermediate representation into vocabulary
scores (logits). Greedy decoding chooses the largest score. Draft depth counts
layers; draft length counts proposed tokens. Verification accepts consecutive
matches and a target correction/bonus. The KV cache stores attention keys and
values; rollback removes rejected suffix entries. A prefix is preceding tokens.
Counterfactual outcomes concern actions not executed. A validity mask records
which learning statements an observation supports. Joules measure energy;
watts measure energy per second. Latency is elapsed time. A baseline is a
comparison method. Held-out prompts are excluded from algorithm tuning.
An ablation replaces/removes a component to test its contribution.

## Pilot protocol and go/no-go

1. Correctness: compare cached ordinary/speculative outputs to native model
   execution, including EOS, zero/short budgets, and cache rollback. Independently
   check observer predictions against full-prefix hidden states.
2. Observation overhead: hold draft depth/length fixed; compare none/one/all/
   periodic modes. Include projections, hooks and trace processing. Never make
   the all-feedback path deliberately inefficient. This Python reference is not
   an optimized serving backend; rerun on an optimized path before final claims.
3. Opportunity: expand predeclared prompts/tasks; compare repeated configuration
   costs and the best fixed choice with an optimistic per-request oracle. Correct
   for measurement noise; changing winning labels alone is insufficient.
4. Feedback utility: compare valid same-prefix signals with actual alternative
   executions. Do not substitute inferred full rollout rewards.
5. Continue to algorithm design only when useful variation and observation
   savings can plausibly exceed selection error and controller costs.

No arbitrary percentage threshold guarantees feasibility. CPU timing is not
energy evidence. GPU-only energy excludes CPU controller consumption. Small
counter differences need longer repeated blocks and uncertainty estimates.
This pilot does not automatically choose a winner or declare a go decision.
Before publication add bootstrap confidence intervals, separate calibration/
held-out workloads, observation budget-matched baselines, and optimized logging.

## Existing work versus proposed work

HedgeSpec already evaluates unchosen drafters using target trajectories; paid
observations are an established online-learning problem. The candidate
contribution is a validated method for selecting device-costed observations
under self-speculative prefix constraints, not the mere presence of feedback.
Safety guarantees and transfer across devices remain out of scope.

- LayerSkip source: https://github.com/facebookresearch/LayerSkip
- Model: https://huggingface.co/facebook/layerskip-llama3.2-1B
- Ungated mechanics model: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Dataset: https://github.com/openai/grade-school-math (MIT)
- HedgeSpec: https://arxiv.org/abs/2510.20064
- Paid observations: https://proceedings.mlr.press/v32/seldin14.html

## Provenance

vendor/self_speculation/llama_model_utils.py was copied from the user's existing
LayerSkip checkout in D:/SEM5/ML/Research/LayerSkip, including its Transformers
4.50 compatibility changes. It is not an untouched upstream release.
Its original copyright header and CC-BY-NC-4.0 license are retained in
vendor/LICENSE.LayerSkip. The surrounding new pilot is not a reimplementation
of HedgeSpec. Inspect saved validation results for what has actually run.


## Portable launcher

For embedded Python that ignores the working directory, run the absolute
path to run.py with that interpreter. All writes still stay here.
The selected/all modes exclude the active draft depth when drafting because
its predictions already exist. Terminal ordinary steps can observe any exit.
The pilot writes both JSON traces and a readable Markdown report.

## Completed local validation

See results/validation.json and results/smollm-fp32-pilot.md.
All 102 FP32 SmolLM smoke runs matched the native greedy baseline. The five
unit tests passed. The subsequent LayerSkip FP32 CPU smoke test also passed all 34 runs (2 prompts,
8-token maximum, one repeat); see results/layerskip-fp32-pilot.md.
Neither test validates quantization, CUDA, energy savings or algorithmic novelty. With two candidate exits, selected and all
observation modes are identical during drafting after active-exit exclusion.

## Project-local Python runtime

A copy of the working Python 3.12 runtime and dependencies is now in
`prototype/selective_feedback/.runtime/python` (ignored by Git).
Run from `D:/SEM5/Energy-Efficiency-DL-Model` in PowerShell:

```powershell
& .\prototype\selective_feedback\.runtime\python\python.exe -B -m unittest prototype.selective_feedback.tests.test_pilot -v
& .\prototype\selective_feedback\.runtime\python\python.exe -B .\prototype\selective_feedback\run.py --model layerskip --device cpu --precision fp32 --prompts 2 --tokens 8 --repeats 1
```

This runtime contains CPU-only PyTorch. The runtime copy does not depend on
the old checkout. Its embedded Python path includes this repository location;
update that path if moving the project. Authentication is stored in the ignored
`layerskip/cache/token` file; never include it in source control or
share it with teammates.

## LayerSkip download fallback

The initial whole-file transfer stalled on this connection. The local
`download_layerskip.py` helper fetched bounded HTTP ranges and verified the
weights against the Hugging Face SHA-256 metadata before publishing the cached
file. It can resume complete saved chunks after an interrupted transfer:

```powershell
& .\prototype\selective_feedback\.runtime\python\python.exe -B .\prototype\selective_feedback\download_layerskip.py
```

The validated model revision is `81deb0f88734409cca506bcefcfb5c6bd2667565`.
Pass `--revision 81deb0f88734409cca506bcefcfb5c6bd2667565` to the pilot to
repeat with this checkpoint. Only the small CPU correctness run is complete;
there are no measured energy results or learned selector yet.

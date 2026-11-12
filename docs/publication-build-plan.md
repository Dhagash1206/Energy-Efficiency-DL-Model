# Build plan: credible model quality and energy-efficient inference research

Prepared 2026-11-10 from the current code, saved outputs, and primary-source research.
This is a proposed implementation plan, not a statement that the experiments below
have already run. Start with [the agent handoff](agent-handoff.md). The existing
[publication guide](publication.md) describes the tools already implemented.

## 1. Recommended direction

First establish a trustworthy full-depth task-quality baseline. Then test whether
adaptive self-speculation reduces **total measured cost relative to a strong fixed
configuration chosen on development data**, while preserving that baseline's output
or meeting a prespecified quality constraint.

Treat three questions separately:

1. **Model capability:** Can the full-depth checkpoint answer the task correctly?
2. **Decoding correctness:** Does self-speculation preserve that checkpoint's output?
3. **Systems benefit:** Does the controller save energy after its own overhead,
   calibration, and paid probes, compared with the strongest fair baseline?

Changing an exact-output decoding controller cannot improve the correctness of an
unchanged target model's answers. Improve task quality through an appropriate
evaluation protocol, checkpoint, or training; improve efficiency through decoding
and control. Measure both together after each change.

The most feasible first research track on this machine is **measurement and
inference with an existing exit-trained model**. Exit-aware fine-tuning is an
optional second track with its own controls and resource needs. Adding all possible
optimizations at once would make the contribution difficult to identify.

No plan guarantees acceptance at a publication venue. A negative or inconclusive
result against a well-tuned fixed baseline is a valid outcome to retain.

## 2. What the repository currently establishes

Source artifacts:

- [FP16 pilot JSON](../hardware_aware_selective_feedback/results/publication/fp16-m4-quality-pilot.json)
  and [report](../hardware_aware_selective_feedback/results/publication/fp16-m4-quality-pilot.md).
- [Executed quality scores](../hardware_aware_selective_feedback/results/publication/fp16-m4-quality-scores.json).
- [FP32 historical reference](../hardware_aware_selective_feedback/results/latest.json).

| Finding | Supported interpretation |
|---|---|
| FP16 pilot: 57/57 measured requests match native greedy | Local decoding agreement on this small workload |
| Four held-out tasks: two GSM8K, two HumanEval | Workflow validation, not a reliable benchmark estimate |
| Both task scores are 0/2 for every policy | No useful quality claim from this pilot; ordinary greedy also fails |
| All 32 held-out policy/task completions reach 256 tokens | Stopping, repetition, and budget effects require diagnosis |
| Adaptive 0.964938 J/token; fixed depth 4 0.948633 | Adaptive uses about 1.7% more energy on this sample |
| Fixed depth 12 is measured by default | One important baseline gap has been addressed |
| Adaptive purchases zero observations | No evidence yet that buying intermediate feedback is beneficial |
| M=4 observer baselines execute | Subsampling works; a matched M=1/M=4 energy ablation is still missing |
| Idle estimate 10.464551 W | A resident-model reference, not physical separation of chip components |
| 89 software tests passed at this checkpoint | Software validation, not 89 correctly answered benchmark problems |

The machine reports an RTX 3050 Laptop GPU with **4096 MiB** memory. The new pilot
records PyTorch `2.6.0+cu124` and Transformers `4.50.0`; the older FP32 report records
a different runtime. They cannot serve as a controlled FP32/FP16 comparison.

### Concrete failure diagnosis

These examples were inspected and must be treated as development material:

| Task | Observed ordinary output | Diagnosis supported by inspection |
|---|---|---|
| `gsm8k-test-208` | Repeats the question with answer `5`; reference is `76` | Wrong reasoning/answer plus repetition and missing marker |
| `gsm8k-test-817` | Repeatedly subtracts 8 and repeats the question; reference is `15` | Wrong initial reasoning plus repetition; a later incidental `15` must not be cherry-picked |
| `HumanEval/154` | Checks equal length and character equality | Wrong function semantics: the task asks about rotated substrings |
| `HumanEval/49` | Returns values based on `p` and ends in an incomplete branch | Wrong modular-exponentiation logic plus truncation/syntax risk |

Therefore, neither increasing the token limit nor accepting any number found in
the completion is a demonstrated fix. Better stopping can remove unwanted trailing
text; it cannot make an incorrect function correct. The present evaluator collapses
many HumanEval exceptions into a generic failure, so exact failure categories still
need instrumentation.

The official checkpoint is continually pretrained from **Llama-3.2-1B base**, using
LayerSkip's exit-training recipe; it is not identified as an instruction model.
The LayerSkip paper's v4 Table 2 reports final-layer scores of **3.79 GSM8K** and
**9.15 HumanEval** for its Llama3.2 1B LayerSkip condition. These are context about
checkpoint capability, not local targets: the model card says its harness-based
evaluations and the paper's internal evaluation differ. [S1, S2]

### Repository issues to audit explicitly

- `native_greedy()` is a handwritten loop. Agreement with it needs an independent
  check against a standard generation path using identical inputs and settings.
- Current `--task-quality` uses a short zero-shot math instruction, strict `####`
  parsing, and EOS/token-limit stopping. It is not the standard harness protocol.
- `load_prompts()` couples the two datasets' sample counts and reserves HumanEval
  tasks for calibration. It cannot directly express a full benchmark plus a
  separate development protocol.
- `Selector.update()` pools executed draft evidence and teacher-forced observer
  evidence in `_matches`. These are not automatically interchangeable observations
  of rollout acceptance or energy.
- `_arm_score()` adds uncertainty to cost, while the cost bandit subtracts an
  exploration bonus. Keep their intended behavior distinct; do not call both UCB
  or claim either heuristic has a theorem that has not been established.
- `prompt_context()` uses dataset identity. This is privileged task metadata unless
  the deployment interface actually provides it; add a label-free control.
- Calibration information differs between adaptive and bandit: the selector sees
  paired observer updates that the bandit does not. Audit information and cost
  budgets before claiming an algorithmic advantage.
- A mismatch in any executed baseline can change the global exclusion set used by
  other policies. Separate a preflight correctness whitelist from deployment-like
  online policy feedback; document this distinction.
- Metered generation currently excludes request-level choose/update work. The
  break-even plot is a stationary mean-cost projection, not an observed cumulative
  trajectory of an online deployment.

## 3. Stage order and decision gates

| Stage | Deliverable | Gate before proceeding |
|---|---|---|
| P0: freeze evidence | Environment/source manifests and split registry | Existing artifacts preserved; every run attributable |
| P1: reference evaluation | Pinned benchmark adapters and failure diagnostics | Reference prompts, generation, extraction, and scores agree on fixtures |
| P2: shared stopping | Correct boundaries across all decoders | Token/output agreement survives EOS and multi-token stop edge cases |
| P3: model-quality decision | Full-depth reference report and model choice | Quality is measurable, failures understood, and model suitable for the stated task |
| P4: fair baseline study | Development-selected fixed configuration and upstream baseline | Held-out data did not select settings; baseline costs are competitive |
| P5: measurement | Full-request energy and calibration accounting | Overheads, sensor limits, and replay behavior are audited |
| P6: adaptive/feedback study | Minimal controller and controlled ablations | Any advantage survives overhead and equal-information controls |
| P7: final evaluation | Frozen protocol, paired statistics, reproducibility bundle | Claims match uncertainty and scope; negative results remain visible |

If P3 fails, take the model-improvement branch in section 6 before spending time on
P6. If P6 does not beat the fair fixed baseline, retain the fixed method as the
deployment choice and reconsider the paper's hypothesis.

## 4. P0–P2: fix the evaluation foundation

### P0 — Preserve evidence and define development versus test

Existing files: `run.py`, `quality.py`, `paths.py`, dependency files, saved reports.
Proposed files: `protocol.py`, `configs/`, `data_splits/`, `run_manifest.py`.

1. Record the current Git commit **and working-tree diff/hash**. The publication
   implementation and reports currently include uncommitted changes; the commit ID
   alone is insufficient provenance.
2. Hash every existing result file and create a manifest. New experiments write new
   paths. Never revise an old report to insert results that were not measured.
3. Record model/tokenizer revisions, configuration hashes, dataset hashes, source
   hashes, Python/PyTorch/CUDA/Transformers versions, driver, GPU, power/clock state,
   attention implementation, seed, prompt template, and generation/scoring config.
   Resolve harness/EvalPlus dependencies to immutable releases or commit IDs.
4. Build an ID registry from existing reports. This audit found references to 22
   HumanEval IDs, 17 GSM8K-test IDs, and 8 GSM8K-train IDs in 29 JSON reports. That is
   a minimum known exposure list, not proof of absence of pretraining contamination.
5. Separate few-shot exemplars/training, controller calibration, development/tuning,
   and final evaluation. Record overlaps intentionally allowed by the chosen
   benchmark protocol. Exclude inspected/tuned tasks from a claimed fresh final
   subset. A full benchmark including previously seen development tasks must be
   labeled transparently as a full-suite report, not untouched confirmation.
6. Add independent dataset counts or explicit ID lists. Fail on duplicate IDs,
   unexpected overlap, missing tasks, or silently shortened input prompts.

Acceptance: the same manifest recreates identical rendered prompts and task IDs;
all result paths are unique; an environment mismatch is visible before inference.

### P1 — Make the reference evaluator authoritative

Existing files: `run.py:load_prompts`, `quality.py`, `engine.py:native_greedy`.
Proposed files: `benchmark_protocols.py`, `reference_eval.py`, scorer fixtures.

**GSM8K:** start with a pinned upstream `gsm8k` harness configuration as the primary
protocol. At research time it specifies five training examples, deterministic
generation, a `Question: … Answer:` format, stop strings including the next question,
and separately named strict and flexible extraction filters. The 8-shot CoT task
is a different configuration. Save the resolved config; do not combine their
prompts and scores under one name. [S3]

Export raw outputs, task-visible stopped text, extracted answers from both frozen
filters, reference answers, finish reasons, and token counts. Choose the primary
metric before development; keep any alternate filter as a labeled diagnostic.
Never select a parser, a number, or a stopping position using the reference answer.

**HumanEval:** pin a base-model completion protocol and its code-boundary treatment.
Use one greedy completion per task for primary pass@1. Run original HumanEval and,
optionally, HumanEval+ on the **same saved completions**. The latter has extra tests
and is a separate metric. Do not expect different tool defaults to yield identical
prompts. Preserve container isolation and record syntax, assertion, runtime,
timeout, and infrastructure failures separately. [S4, S5]

**Independent checks:** compare the local full-depth loop with an established
generation path for the exact same tokenized prompt, checkpoint, precision,
attention implementation, BOS/EOS configuration, and deterministic settings.
Then compare speculative decoders with that verified target. The official LayerSkip
repository already integrates with lm-eval and applies task stop processing; reuse
or adapt that interface with attribution rather than designing benchmark semantics
from scratch. [S6]

Acceptance tests:

- Correct, incorrect, absent-marker, comma, negative, decimal, multiple-number, and
  repeated-question fixtures match the pinned upstream math filter behavior.
- Known-correct HumanEval fixtures pass, known-wrong fixtures fail, and an unavailable
  Docker daemon is an infrastructure error rather than a wrong model answer.
- Prompt rendering and score aggregation match the reference evaluator on the same
  stored samples. Differences have a per-item explanation, not just similar means.
- All special tokens and generation options are explicit. No automatic chat
  template is applied to a base checkpoint merely because a tokenizer contains one.

### P2 — Stop on the same committed output in every decoder

Existing files: `engine.py`, `del_policy.py`, `run.py`, quality export.
Proposed file: `stopping.py` with a shared, versioned stop policy.

1. Use the pinned task protocol's EOS and stop sequences. Keep raw tokens and the
   exact task-visible text separately; record which boundary ended generation.
2. Evaluate stops only on committed generated text. A rejected draft containing a
   stop marker must not stop the request.
3. Handle multi-token stop strings crossing cycles and a stop occurring inside a
   multi-token accepted block. Crop tokens/cache consistently; charge any speculative
   computation already performed beyond the eventual output boundary.
4. Test native greedy, ordinary custom decoding, each fixed configuration, adaptive,
   and DEL with the same stop policy. Do not delete arbitrary helper functions or
   repair code based on test outcomes.
5. Expose per-dataset output budgets and diagnostic completion/repetition rates.
   Test 256, 512, and, if memory permits, 1024 output tokens on development only.
   Larger limits are diagnostic options, not presumed improvements.
6. Profile longer few-shot prompts before large runs. Avoid materializing full
   vocabulary logits for every prefill position when a supported last-position
   path suffices; prove output equivalence for the installed runtime. Log memory
   peaks and OOMs. Do not silently remove few-shot examples to make a run fit.

Acceptance: stops, output IDs, and visible text agree under the declared target;
all terminal edge cases pass. Finish reasons are trustworthy. Repetitive generations
are classified as failures, not interpreted as evidence that all tasks need more tokens.

## 5. P3: establish model quality before optimizing it

Evaluate these separately named conditions under the same applicable protocol:

| Condition | Purpose | Speculative role |
|---|---|---|
| Current LayerSkip checkpoint, full depth | Quality ceiling of the existing exact-output system | Target reference |
| Same checkpoint, fixed/adaptive self-speculation | Test preservation and runtime benefit | Candidate decoders |
| Original Llama-3.2-1B base, full depth | Separate checkpoint capability from exit-training effects | Quality control; no assumed useful early exits |
| A suitable instruction/task model, full depth | Determine whether the deployment task needs a different model | Separately prompted quality control, not a drop-in LayerSkip target |

Use model-appropriate prompts for base versus instruction conditions and identify
that difference explicitly. A model comparison with different templates is not a
controlled decoding-method comparison. Access/license requirements still apply.

Begin with a small development set, expand to a larger fixed set, and only then run
the final benchmark protocol. A suggested engineering progression is 8 tasks per
dataset for debugging, then at least 32 per dataset for variance and failure analysis;
these counts are planning sizes, not evidence of statistical sufficiency.

Quality gate: the reference evaluator is correct; task capability is measured on
enough tasks to support the intended claim; truncation and format errors are
reported; the full-depth model is suitable for the proposed application. Merely
getting one successful answer does not satisfy this gate.

Compare FP32/FP16 within a matched software stack and dataset. Where FP32 does not
fit locally, use an appropriate machine or a clearly labeled CPU correctness-only
check; do not compare CPU and GPU energy as a precision effect. Exact equality is
required between exact-output methods and their **same-precision** target, not
necessarily between FP32 and FP16. Use paired quality analysis across precisions.

## 6. Optional model-improvement branch

LayerSkip's useful early exits depend on its training recipe. Generic instruction
weights or ordinary LoRA/SFT may improve full-depth quality, but preserving useful
intermediate exits must be measured. The published recipe and torchtune's early-exit
training recipe are the starting points, not a guarantee for a new adapter.
[S1, S2, S7]

### Local 4-GB track

- Prioritize reliable inference, reference parity, prompt/stopping diagnostics,
  data preparation, selector experiments, and analysis with the 1B checkpoint.
- Keep FP16 unquantized as the initial deployment condition. Explore weight
  quantization only as a separately labeled study with renewed correctness,
  quality, acceptance, and energy checks. Do not assume int4 saves net energy.
- A tiny parameter-efficient training smoke test is optional only after a measured
  memory budget. Optimizer state, activation memory, sequence length, and KV/logits
  buffers all matter. Do not promise full training will fit because weights fit.

### Larger-GPU track

Prefer access to an existing stronger exit-trained checkpoint for a second-model
replication before undertaking a new training project. The official collection
includes larger Llama and code-oriented checkpoints; verify their own quality,
access, and memory requirements. A 7B FP16 checkpoint alone exceeds this laptop's
4-GB capacity. Quantization is a new experimental condition, not an equivalent
substitute for an unquantized replication. [S6]

If training is needed, compare:

1. Original exit-trained checkpoint.
2. Full-depth task SFT/adapter control with no exit-preserving objective.
3. Exit-aware adaptation with the same data and a prespecified training budget.

A possible **research proposal**, not an implemented recipe, is a final-layer task
loss plus weighted intermediate-exit supervision and/or distillation from a stronger
teacher. Distinguish teacher-generated training examples, internal exit targets,
and the deployed verifier. If using layer dropout, implement and validate the
recipe rather than enabling unrelated generic dropout. Choose exit weights,
distillation temperature, adapter placement, and other hyperparameters on
development data only. Teacher sampling and model-training energy belong to a
separate amortization ledger if deployment savings are claimed.

Save adapter/base hashes, trained layers, target exits, data source/license,
deduplication method, teacher provenance, token budget, optimizer, seed, loss weights,
and merge state. Never use final-test answers/tests as training or repair feedback.

Gate: improved held-out full-depth quality **and** measured exit/rollout acceptance
that leads to a worthwhile runtime trade-off. If SFT improves quality but destroys
draft acceptance, report that trade-off. A newly fine-tuned model is verified against
its own full-depth target, not the old checkpoint's token sequence.

## 7. P4–P6: make the energy/control comparison fair

### P4 — Strong fixed and prior-work baselines

Existing files: `sweep.py`, `run.py`, `selector.py`, `del_policy.py`.
Proposed files: baseline-selection manifest and serialized frozen policy state.

- Sweep depth `{4,8,12}` and draft length `{1,2,3,4}` plus ordinary decoding on
  development data, subject to checkpoint depth constraints. Start smaller when
  profiling resources, but use the same permitted action set for final comparisons.
- Select one fixed winner by a prespecified objective and freeze it before final
  test. Keep `d4/k3`, `d8/k3`, and `d12/k3` as named anchors. A per-prompt or
  whole-test hindsight winner is an oracle diagnostic, not a deployable baseline.
- Measure an established optimized full-depth autoregressive path and, where
  feasible, upstream LayerSkip self-speculation. Comparing only against a slow
  custom ordinary implementation can exaggerate the practical gain.
- Retain the current bandit and independent DEL port. Where a reference method is
  ported, label differences from the authors' runtime; do not call it a reproduction
  of published speedups without the corresponding validation.
- Audit calibration access: either provide a shared calibration table to applicable
  methods or assign equal per-policy acquisition budgets. Record the actual energy
  each method would pay. Fixed-policy tuning is not intrinsically free.

Gate: settings are chosen without final-test outcomes, exact-output configurations
are validated, all intended methods receive the declared information/budget, and
the chosen fixed winner is included even when it beats adaptive.

### P5 — Measure full-request costs and online trajectories

Existing files: `measurement.py`, `run.py`, `publication.py`.
Proposed files: request/stream cost ledger and paired analysis module.

Record two boundaries: a generation-only diagnostic and an explicitly defined
end-to-end request window covering policy selection, generation, paid probes, and
online updates. Avoid inserting extra synchronization inside one boundary solely
to obtain the other; use a counter-read schedule whose overhead is calibrated.
State whether tokenization and output decoding are inside the deployment boundary.
Do not count offline gold/reference evaluation as deployed work, but report its
separate experimental cost. [S10, S11]

Use total GPU-board energy as primary. Keep idle-subtracted excess as a secondary
reference estimate, retaining negative values and idle samples. Bracket sessions
with idle checks, warm the model consistently, randomize/counterbalance policy
order, log temperature/clock/power/memory, and repeat independent sessions. Validate
the counter's resolution/update behavior against these request durations. GPU-board
telemetry alone does not establish laptop/battery energy savings. [S10, S11]

Report J/request, total-J/total-token, latency distributions, useful output tokens,
acceptance, verifier work, memory, and task scores. Energy per correctly solved task
may be a secondary metric: count **all** request energy in the numerator and solved
tasks in the denominator. A zero-success run has no finite energy-per-solution
estimate; never discard failed tasks to improve that number.

For candidate and baseline with setup costs `C_a` and `C_b`, use:

```text
observed cumulative candidate(N) = C_a + sum(actual request costs 1..N)
observed cumulative baseline(N)  = C_b + sum(actual request costs 1..N)
stationary projected crossing   = ceil((C_a-C_b)/(mean(E_b)-mean(E_a)))
```

Handle the signs and boundary cases explicitly. Display the observed curve only
over measured requests; label any extrapolation. The existing chart charges all
shared calibration to adaptive and zero to fixed. Keep it as a labeled conservative
scenario and add a realistic tuning-cost allocation. Uncertainty that includes
zero/negative savings permits an unbounded/no-crossing result; do not discard those
replicates to manufacture a finite break-even interval.

Gate: the primary comparison includes controller/probe costs; ledger totals reconcile;
sensor failures and OOMs are recorded; setup-cost assumptions are visible; no
per-token or idle adjustment replaces the primary total-energy result silently.

### P6 — Improve the controller only after the fixed baseline is credible

Build the simplest controls first:

1. Development-selected fixed policy.
2. A context policy trained offline and frozen for evaluation.
3. Online cost bandit using only its own executed feedback.
4. Adaptive policy with paid observations disabled.
5. The same adaptive policy with the paid-observation decision enabled.

Use features available before executing a request: prompt length, declared deployment
metadata, and the policy's own prior measurements. Treat dataset-label routing as a
separately identified oracle/context condition unless actually supplied in deployment.
Do not use future length, the reference answer, task ID, or costs measured by another
policy on the same test request as hidden features or feedback.

Keep one-step observer correctness, accepted rollout prefixes, and measured action
costs as separate statistics. Learn/validate any mapping between them on development
data. Correlated token observations do not justify an automatically calibrated
Beta-posterior uncertainty on request-level savings.

For an optional contextual cost model, start with a small auditable predictor or
table before a neural controller. Compare total J/request and J/token objectives
explicitly. Charge feature collection and decision cost; require an out-of-sample
benefit over the simpler alternatives.

For paid feedback, a proposed decision rule is:

```text
buy a probe only when estimated future cost reduction exceeds
measured probe cost + decision cost + a declared uncertainty margin
```

This is an empirical acquisition policy unless a formal guarantee is actually
proved. Its units must be consistent, its planning horizon must not reveal future
test outcomes, and its estimated benefit must predict realized improvements.
Our observers are **internal teacher-forced intermediate LM-head predictions**, not
external quality labels. Do not relabel them as verified task correctness.

Ablate no probes, M=1/2/4/8 fixed schedules, paid acquisition, cost-blind acquisition,
and matched random probe placement with a similar budget. Keep the no-paid-feedback
controller identical otherwise. Distinguish cycle-based M from generated-token
sampling. Generalizing probes beyond `(depth=8,k=3)` is a proposed change requiring
calibrated coverage and cache/alignment tests, not an existing capability.

Study a stationary task mix first. If adaptation is motivated by workload change,
predeclare realistic stream changes, independently seed/order streams, and charge
relearning. Do not manufacture shifts after inspecting which configuration wins.

Gate: improvement exceeds measured noise and survives overhead and information
controls. If useful probes are never selected, or all fixed configurations have
nearly the same ranking across requests, report that result and reconsider whether
paid adaptation is warranted for this workload.

## 8. P7: final experiments, uncertainty, and publication gate

Freeze the checkpoint, prompt/scoring protocol, action space, controller settings,
allowed features, primary baseline/metric, split IDs, and analysis script before
final evaluation. Choose a smallest meaningful effect and quality noninferiority
margin based on the intended application and development variance, not the final
result. Use those assumptions to plan the sample size. Avoid invented universal
accuracy or savings targets.

| Study | Minimum design purpose | Main outputs |
|---|---|---|
| Reference quality | Pinned full-depth protocol; independent scorer parity | Per-task scores, finish reasons, strict/flexible math diagnostics, HumanEval Base/+ |
| Fixed sweep | Development only; same action set and precision | Chosen fixed baseline and tuning cost |
| Frozen policies | Same final prompts, randomized order, repeated sessions | Paired energy/latency deltas and task-quality comparison |
| Online policies | Independent stream seeds/orders; isolated state | Cumulative cost, adaptation cost, action and probe trajectories |
| Probe ablation | Matched settings and acquisition budgets | Realized value of feedback versus no-feedback control |
| Precision/model replication | Separate controlled conditions | Quality/acceptance/energy trade-offs; limitations of portability |

For frozen policies, bootstrap paired differences by prompt, keeping all timing
repeats for a prompt together; stratify by dataset and include session structure as
appropriate. Do not count repeats as additional benchmark problems. Use paired
quality intervals/tests across precisions or models. Exact token-identical outputs
under the same scorer necessarily have identical task scores.

Online policies couple later decisions to earlier observations. A naive IID bootstrap
of their request rows breaks that dependency. Prefer repeated independent streams
and stream-level analysis; justify any block/hierarchical bootstrap. Keep uncertainty
over benchmark sampling separate from runtime/session variability.

For the main energy claim, report a paired estimate and interval relative to the
development-selected fixed winner, with the full measurement boundary and calibration
allocation. A clearly positive, practically meaningful lower confidence bound is a
possible prespecified success criterion; an interval crossing zero means the evidence
is inconclusive. Correct for multiple confirmatory comparisons or declare one primary
comparison and label the rest exploratory.

Complete GSM8K and HumanEval reports are desirable once the implementation is stable,
but a public full-suite score and a previously untouched confirmation subset are
different things. If compute permits only a subset, disclose IDs, selection rules,
sample-size rationale, and uncertainty; do not label it the full benchmark. A second
device/model strengthens scope but is not a guarantee of acceptance. Claims about
consumer hardware generally need evidence beyond a single laptop.

Publication checklist:

- A specific hypothesis and an up-to-date comparison with the closest prior work.
- Credible task-quality baseline and reference evaluator parity.
- Strong fixed and appropriate adaptive/upstream baselines, with fair information.
- Complete costs, useful output lengths, calibration, and observer accounting.
- Prespecified held-out analysis with uncertainty and preserved negative results.
- Ablations that identify what causes any gain; no attribution to features that
  never activate.
- Versioned code/config/data/model provenance, environment instructions, raw logs,
  plots, licenses/attribution, and executable reproduction steps.
- A limitations section covering model competence, workload/hardware scope,
  sensor boundary, checkpoint training, and any differences from upstream methods.

## 9. Closest prior work and the novelty test

| Primary source | Existing overlap | What this project must establish beyond that overlap |
|---|---|---|
| LayerSkip [S1, S6] | Exit training and self-speculative decoding | A measured control/measurement contribution beyond using the checkpoint |
| DEL [S8] | Context-aware dynamic exit/speculation decisions | A distinct mechanism and a fair matched comparison, not a renamed heuristic |
| BanditSpec [S9] | Bandit-based adaptation of speculative configurations | Why this acquisition/control problem or objective needs a new method |
| Not-a-Bandit [S12] | Online drafter selection with additional feedback for unchosen drafts | How paid internal observer information differs from feedback already obtainable cheaply |
| CLaSp [S13] | Context-adaptive self-speculative layer skipping | Clear differences in architecture, controls, training, and cost evidence |
| ML.ENERGY [S11] | Systematic LLM energy measurement/benchmarking | A scoped experimental contribution rather than an unvalidated energy label |

A candidate research question is: **When does paying for sparse intermediate
predictions reduce cumulative GPU energy relative to a development-selected fixed
policy and an otherwise identical no-probe controller?** This is a proposal to test,
not a claim of novelty. Check newer citations and implementations before writing
the introduction. A carefully measured negative answer may support a narrower
empirical report, depending on contribution and venue.

## 10. Hardware-conscious work budget

The 4-GB laptop should run one GPU experiment at a time. Use small correctness and
development batches first. Estimate a larger job from measured per-request durations
and explicit request counts, adding warmup, reference generation, calibration, and
container evaluation separately. Preserve failed runs and checkpoint progress so
an interruption does not require rerunning completed tasks.

For the current runner, held-out request count is `8 * (2 * test_per_dataset) * repeats`
without DEL. A full grid of 13 actions costs more than that default comparison.
Do not multiply tasks, seeds, precisions, probe periods, and models into one huge job
before the earlier gates pass. New per-dataset/ID-list support will change this count.

Keep model training and large-checkpoint evaluation on an optional larger-GPU track.
Record actual peak memory and wall time before budgeting a full run. No specific
training size, cloud cost, or speedup is assumed by this plan.

## 11. Primary sources and research provenance

Pin moving repository links to immutable revisions during P0/P1; links here identify
the sources inspected, not a frozen execution environment.

- **S1:** [LayerSkip paper, v4](https://arxiv.org/html/2404.16710v4), especially training,
  Table 2, and limitations.
- **S2:** [Official LayerSkip Llama3.2 1B model card](https://huggingface.co/facebook/layerskip-llama3.2-1B),
  including model origin and evaluation-method caveats.
- **S3:** [lm-evaluation-harness GSM8K configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)
  and [separate CoT configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml).
- **S4:** [HumanEval task definitions](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/humaneval)
  and [official execution code](https://github.com/openai/human-eval/blob/master/human_eval/execution.py).
- **S5:** [EvalPlus](https://github.com/evalplus/evalplus) and its [CLI documentation](https://github.com/evalplus/evalplus/blob/master/docs/cli.md).
- **S6:** [Official LayerSkip code](https://github.com/facebookresearch/LayerSkip)
  and [evaluation adapter](https://github.com/facebookresearch/LayerSkip/blob/main/eval.py).
- **S7:** [torchtune early-exit training recipe](https://github.com/pytorch/torchtune/blob/main/recipes/dev/early_exit_finetune_distributed.py)
  and [official release notes](https://github.com/meta-pytorch/torchtune/releases).
  Check recipe status, supported model/precision, and resource requirements at the pinned revision.
- **S8:** [DEL paper](https://arxiv.org/abs/2504.05598) and [conference record](https://openreview.net/forum?id=cAFxSuXQvT).
- **S9:** [BanditSpec, ICML 2025](https://proceedings.mlr.press/v267/hou25h.html).
- **S10:** [NVIDIA NVML device-query documentation](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html).
- **S11:** [ML.ENERGY benchmark paper](https://arxiv.org/abs/2505.06371).
- **S12:** [Not-a-Bandit, ICLR 2026 conference record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/20563b8508ba42e1b688d922e926ee26-Abstract-Conference.html).
- **S13:** [CLaSp, ACL 2025](https://aclanthology.org/2025.acl-long.1525/).

Research used the connected Parallel deep-research and search tools, followed by
direct primary-source inspection. Research response:
`resp_0b861821-2379-4081-b516-79155dff3623`. Paper/model-card results are external
evidence; phase order, file proposals, gates, and statistical design above are
recommendations for this repository and must be evaluated rather than assumed.

# Implementation prompt: energy-aware selective feedback for self-speculative decoding

Copy everything between `BEGIN IMPLEMENTATION PROMPT` and `END IMPLEMENTATION PROMPT` into a coding-agent session opened at this repository. The objective is a complete, reviewable implementation and an executable research workflow. Positive experimental results and publication acceptance are not assumed.

## BEGIN IMPLEMENTATION PROMPT

You are implementing a research system in the existing `Energy-Efficiency-DL-Model` repository. Work autonomously through implementation, meaningful tests, an offline demonstration, and whatever small real-device validation the environment supports. Do not stop after providing a design, scaffolding, or TODO list. Do not manufacture measurements, theoretical guarantees, successful transfer, novelty, or paper acceptance. Distinguish software completion, experiment completion, and scientific success.

Read applicable repository instructions and inspect the current worktree before changing files. Preserve unrelated user changes. Treat this document as a grounded implementation specification, not evidence that any proposed method already works. Use focused parallel agents for independent audits or implementation areas when available, with clear file ownership and integration review.

### 1. Objective and scope

Implement one coherent research direction with three connected components:

1. **Selective information acquisition:** jointly choose a self-speculative decoding action and whether/which intermediate-layer predictions to compute, using predicted future energy benefit versus measured acquisition cost.
2. **Valid partial-feedback estimation:** use only causally available, same-prefix evidence to improve action predictions; quantify uncertainty and validate predictions against actual executed alternatives.
3. **Decision-focused calibration:** spend startup and recalibration energy only when the likely deployment benefit can repay it; report total energy, including the cost of learning.

Primary question:

> Can energy-priced selective acquisition of intermediate-exit feedback improve net GPU energy efficiency over strong fixed and adaptive decoding policies while retaining full-depth verification?

Primary checkpoint: `facebook/layerskip-llama3.2-1B`, revision `81deb0f88734409cca506bcefcfb5c6bd2667565`. Derive layer count from model configuration; do not assume 16 outside compatibility code. Main hardware: RTX 4060 Laptop GPU; transfer target: an actual RTX 3050 Laptop GPU. Detect the available device instead of inferring it from a filename. Use GSM8K and pinned HumanEval prompts; do not execute HumanEval code. Optional additional model/domain experiments must be isolated and must not delay core completion.

Primary energy boundary: the selected NVIDIA GPU cumulative NVML counter, in joules. CPU mechanics and synthetic runs are separate. Record controller CPU time; do not describe GPU-only energy as whole-laptop savings. Keep model precision, attention backend, and stopping policy identical across each comparison.

Initially disable quantization, compilation, CUDA graphs, clock/power writes, and batching in the primary experiment. Retain existing features as separate baselines/extensions. No backbone training, early-exit retraining, or revival of failed `joulerexit` training is required. A neural controller is optional distillation after a simpler policy demonstrably works; it is not the contribution by itself.

### 2. Prior art and allowed claims

Read and cite the primary sources before finalizing the comparison matrix:

- LayerSkip: https://aclanthology.org/2024.acl-long.681/
- DEL: https://arxiv.org/abs/2504.05598 and https://arxiv.org/html/2504.05598v2
- BanditSpec: https://arxiv.org/abs/2505.15141
- HedgeSpec / Not-a-Bandit: https://arxiv.org/abs/2510.20064
- PELM: https://arxiv.org/abs/2609.09662
- Zeus: https://www.usenix.org/conference/nsdi23/presentation/you
- Conservative Bandits: https://proceedings.mlr.press/v48/wu16.html
- Speculative-decoding energy study: https://arxiv.org/abs/2602.09113
- NVML: https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html

Adaptive depth/length, intermediate-state reuse, feedback on unchosen drafters, online bandits, conservative fallback, and energy-aware profiling already exist. The candidate distinction is the validated decoding-specific policy for selectively purchasing information according to its net energy value. Generic value-of-information equations and confidence thresholds are known ingredients. Do not label them new theorems. Label adaptations of published methods as adaptations, not exact reproductions. The repository's old PELM-named network is not the published PELM implementation.

Write original documentation with source attribution and preserve `vendor/LICENSE.LayerSkip`. Do not promise a plagiarism score. Do not rename borrowed mechanisms and claim originality. Literature searches establish related work, not proof that no similar work exists.

### 3. Ground truth about the repository

Read `docs/REPOSITORY_RESEARCH_AUDIT.md` if present and recheck the files it cites. The audit baseline is Git commit `e3ef469c763a3e3d9dfbd312afa496714976a226`; current files may have changed.

Reuse:

- `engine.py`: independent `native_greedy`, cached `generate`, rollback checks, cycle callbacks and observation plumbing.
- `feedback.py`: committed-prefix validity and explicit one-step-only evidence.
- `measurement.py`: cumulative counters, synchronization, explicit scope and cleanup.
- `run.py`: pinned model/data loaders and disjoint prompt-loading concepts.
- `sweep.py`: randomized paired execution and completeness/uniqueness validation.
- Existing focused tests and attributed vendor helper.

Current defects to address before interpreting new results:

1. `device_energy.fit_device` ignores matrix rank. Saved fixed-length startup rows satisfy `verify_steps + accepted = 64`, so `[1, verify_steps, accepted, rejected]` is rank deficient. Do not use these coefficients as separately identified physical costs.
2. `fit_exit_profiles` includes mismatching rows and groups different prompts as repeats. Preserve complete repeat identity and exclude unqualified actions from controller fitting.
3. `startup_sweep.py` uses unpinned dataset loading, fixed action order, very few repeats, and `generate(0,0)` instead of independent native references.
4. `run.py` hard-codes the fixed baseline `(4,3)`, sets `greedy_match=True` in rows, and raises on mismatch before recording it. Replace these behaviors in the research workflow.
5. `run.py` updates during held-out requests/repeats. This is online evaluation, not frozen generalization. Support both explicitly.
6. `sweep.py --adaptive-drafts` changes speculative actions; such runs cannot be labelled fixed-policy comparisons.
7. `pelm_round.py` offers shallow verification, uses synthetic teacher states, ignores some advertised telemetry in teacher costs, and updates fits that its frozen network does not use. Preserve it as legacy; do not promote its training label accuracy to energy-optimality evidence.
8. `_pad_stopped_cache` in the vendor helper clones shallow cache entries into skipped layers. This cannot support the new full-verification contract or transitions to deeper verification. Prohibit its use by new exact-output policies.
9. `_observe` casts hidden states to float before a possibly FP16 `lm_head`; implement and test a dtype-consistent observer projection. Casting already-computed FP16 logits to FP32 cannot recover lost precision or change their ordering.
10. `diagnose_mismatch.py` calls an FP16 model in its purported FP32 reference, has a prefill/cycle position-offset issue, and makes unsupported near-tie verdicts. Repair labels/indexing and use controlled matched-prefix diagnostics.
11. Root requirements/setup declare torch 2.6.0 and transformers 4.50.0, but installed metadata seen during audit identifies torch 2.11.0+cu128 and transformers 5.21.0, and current vendor code uses newer cache APIs. Verify the actual executable environment. Select and test one compatible stack, then synchronize requirements/setup/docs. Do not blindly reinstall stale pins into a functioning environment or add untested version ranges.
12. `download_layerskip.py` must use the same pinned model revision as the runner.
13. `pelm_second_gpu.py` is a status guard, not an implemented transfer experiment.
14. Old docs, test counts, and performance percentages are historical. Remove unsupported savings from new CLI help and current-facing documentation.

Evidence constraints:

- `results/latest.*` is a tiny FP32 CPU feasibility run, not GPU success.
- Historical 3050 measurements exist in `fp16-run/` and `smoke/`; the specific 3050 startup and PELM-transfer status files say `not_measured`.
- The 4060 per-round network did not beat fixed depth 4 / length 1 in its saved pilot evaluation.
- The 200-token fixed sweep records 36 mismatches out of 144 requests and no eligible speculative action. Correctness at 16 tokens does not establish correctness at 200 tokens.
- Do not rewrite old results, reinterpret synthetic data as measured energy, or silently pool incompatible devices/backends/precisions.

### 4. Working architecture and contracts

Prefer small composable modules under `hardware_aware_selective_feedback/`, rather than another independent experiment stack. Suggested responsibilities, with names adjustable:

| Module | Responsibility |
|---|---|
| `research_schema.py` | Versioned actions, observation plans, decisions, windows, records, manifests |
| `qualification.py` | Native-reference stability, cache transitions, qualification registry |
| `cost_model.py` | Executed-cost prediction and calibrated uncertainty |
| `side_feedback.py` | Prefix-valid evidence, provenance, contextual agreement features |
| `acquisition.py` | Predictive information value, cost/budget gate, acquisition baselines |
| `calibration.py` | Balanced seed profiling, additional measurements, stopping, bounded recovery |
| `research_data.py` | Immutable pinned datasets, splits, tokenization/stopping manifests |
| `research.py` | Stage-driven CLI and orchestration |
| `research_analysis.py` | Artifact validation, statistics, tables and plots |

Avoid unnecessary dependencies. Reuse code rather than duplicating loaders, meters, reporting, and model execution. Keep old entry points working or delegate them through explicit adapters. Existing experimental result formats must remain readable as historical formats; do not pretend missing provenance can be reconstructed.

Define distinct identities:

```text
DecodingAction = (draft_depth, draft_length, verification_depth)
ObservationPlan = (unique exit depths, positions/cadence, projection mode)
PolicyAction = (DecodingAction, ObservationPlan)
Decision = (policy action, state fingerprint, predictions, uncertainty,
            acquisition value/cost, horizon, legality, reason)
MeasurementWindow = (start/end event, raw counter endpoints, duration,
                     committed work, decisions covered, quality flags)
```

An action key must not accidentally merge observed and unobserved policies. Separate shared underlying decoding-cost estimates from measured observation overhead explicitly.

Qualified initial grid: ordinary `(0,0)` plus depths `(4,8,12)` below actual model depth and draft lengths `(1,2,3,4)`. Allow a smaller configured pilot grid without silently excluding stronger alternatives from final comparisons. Verification depth is always the actual full layer count. Clamp drafts to remaining committed-token budget, handle EOS, and validate observer depths/positions. Do not charge again for an identical prediction already naturally computed; also do not assume cached hidden states make vocabulary projection free.

Extend the engine compatibly with a structured decision callback or explicit observation callback. Legacy tuple action callbacks must retain their behavior. Apply the chosen observation plan before the relevant forward pass, log the decision before execution, and deliver valid outcomes only after verification/commit. Observation results cannot influence an already-selected past action.

Implement action changes at existing completed-cycle boundaries. For measurement-friendly blocks, keep one action/observation schedule stable for a declared committed-token block, allowing overshoot only by an explicitly recorded cycle boundary. Use a step/session abstraction only if needed; preserve the existing forward mathematics and a `generate` compatibility wrapper. Verify cache content/rollback across depth changes, not merely shape.

The primary controller can act at block boundaries while using recent cycle feedback. It must not require per-round measured joules. If block energy cannot be resolved, use longer fixed-action profiling windows for model fitting and request-level measurements for evaluation. Do not divide request energy across rounds and call the allocations measured labels.

### 5. Correctness and numerical qualification

Use `native_greedy` under the same weights, dtype, attention backend, tokenization, tie-breaking and stopping settings. First check reference self-consistency across repeats. Then qualify actions and observer combinations on calibration/qualification prompts at intended input/output lengths, including regression IDs when available: `gsm8k-test-245`, `gsm8k-test-880`, and `gsm8k-test-411`. These exposed examples are regression fixtures, not untouched final-test evidence.

Full-depth verification preserves the algorithm's exact-arithmetic logic; finite-precision native identity must still be tested. A near-tie is a diagnostic, not proof of cause or proof that an FP32 cast fixes it. An actual FP32 experiment must execute the declared higher-precision computation and report its different cost/reference contract.

Quarantine any failing action/backend/dtype/observer combination. Never commit tokens justified only by a shallow verifier or fabricated cache layers. A final-test mismatch invalidates the exact-output claim for that tested policy, even if calibration qualified it. Persist mismatches before continuing to other independent policies. Do not silently drop mismatching prompts to improve energy statistics.

Fallback on unavailable telemetry/acquisition model is the qualified static calibration/validation winner. If no speculative action qualifies, use ordinary decoding and record the research outcome. A fallback after detecting an already-emitted mismatch is not a correctness guarantee. Any implemented recovery must restore/recompute correct state and include all failed and recovery work in reported cost; otherwise mark the request/policy invalid.

Preserve true zero acceptance. Ordinary steps have no draft acceptance label; record missing/not-applicable and feature age rather than replacing it with zero or a fictitious success. Validate `clip_commit` output as a prefix and align feedback with actual committed positions.

### 6. Measurement and energy accounting

Reuse `Meter` and add serializable raw endpoints, device identity, window identity and quality flags. Use context managers on every path. Preserve cleanup failures without hiding the original experiment exception.

Device probing must measure counter behavior separately: read latency, observed nonzero increments, update cadence distribution under a small controlled workload, reset/decrease behavior and repeated-window variability. Do not assume a universal update interval or universal sensor accuracy. Missing/unsupported measurements remain null with a reason. Repeated zero deltas on short windows indicate inadequate resolution, not free computation.

Select a reliable minimum measurement window from the observed probe and validation, with a configurable cap. Aggregate multiple executions when necessary; store aggregate energy and denominator. Do not manufacture per-request or per-round independent energy from a window covering several requests. Flag unresolved windows and keep them out of action fitting and energy claims.

Define two complementary protocols:

- Controlled fixed-work profiling: declared prompt lengths and fixed generated-token budgets, with EOS behavior explicitly recorded. This identifies action costs and observation overhead without conflating different output amounts.
- End-to-end evaluation: identical natural EOS/stopping policy and cap for every policy. Report actual lengths and censored/capped requests. Do not claim task completion quality from an intentionally fixed-work microbenchmark.

Report prefill-plus-decode request energy as the primary practical boundary. Optional decode-only profiling must identify and measure its boundary independently. Use a separate profiling ledger for matched-prefix diagnostics, reference generation, warmups and calibration. Reference/diagnostic execution used only for evaluation is not deployed controller energy, but must be recorded as research overhead and scheduled so it does not unevenly warm policies.

Measure acquisition overhead with randomized same-prompt/action paired windows (base vs observed), holding actual committed work constant. Retain signed raw differences and confidence intervals; a negative noisy estimate is not proof that observation generates energy. Use a conservative nonnegative bound for acquisition decisions when supported. Calibrate overhead across relevant action/context buckets and combinations; do not assume additivity without checking.

Measured request GPU energy already contains in-request acquisition, GPU idle time while the CPU decides, and synchronization overhead inside the boundary. Do not add their estimated components again. Record CPU decision latency separately. External CPU energy, when available, is a separate scope.

For deployment comparisons, allocate each policy its actual startup/calibration/warmup costs once under a declared cold-start protocol. Also report warm steady-state results. If request windows exclude service-idle gaps, call cumulative totals active-work energy; use a continuous boundary for any claim about complete session energy. Keep hardware clock writes optional and disabled by default; never silently change them to manufacture a gain.

### 7. Identifiable cost prediction

Build the primary model from **actual executed cost labels**, using only pre-decision features at prediction time. Prefer a small regularized direct predictor of energy and latency for an action/context over interpreting unidentified accepted/rejected coefficients physically.

Practical initial features:

- Prompt/effective context length and planned committed-work horizon.
- Action depth/length and observation plan or separately identified overhead.
- Recent executed acceptance statistics, their action provenance and age.
- Contextual agreement summaries from valid acquired feedback, their sample counts and age.
- Power, temperature and clock only when available, with missingness/staleness.

Use a low-dimensional regularized model or pooled/bucketed empirical estimates with shrinkage, explicit fallback for sparse/unseen contexts, and positive predictions. Compare against a simple per-action mean/lookup table. Do not add a large learned model merely to appear novel.

Use fixed-work windows for direct comparable energy prediction. When remaining generation length is uncertain, predict energy and committed work jointly and justify the chosen expected-cost-per-work objective; do not substitute unknowable future generated length or acceptance. Realized acceptance can label an outcome, but cannot be a current-action input unless obtained earlier and marked as such.

If retaining the decomposition `E = a + bV + cA + dR` as an ablation, vary independent work/length conditions; check design rank, singular values, condition number, coefficient stability and held-out error. Reject nonidentifiable interpretations. Do not include algebraically redundant features such as N, V and A when `N=V+A`. A predictive reduced model is acceptable; physical attribution requires stronger evidence.

Group repeats by prompt, action, observer plan, token cap/stopping, dtype/backend, device and measurement configuration. Validate complete balanced coverage for action comparisons. Fit on calibration data; tune on validation only. Estimate uncertainty with prompt-clustered resampling or another explicitly justified method. Calibrate and report predictive-interval coverage on held-out calibration/validation groups. Predictions and model intervals must never be labelled measured joules or rigorous confidence sequences without justification.

### 8. Valid partial-feedback estimator

Preserve `validate_feedback` semantics: a prediction is a valid one-step comparison only when its input prefix equals the committed target prefix and its predicted position is within the committed outcome. Rejected suffixes and after-stop positions are invalid and have no match label. Store provenance sufficient to reconstruct this without duplicating full prefixes in every row.

Maintain contextual per-exit agreement summaries, using a documented beta-binomial shrinkage model or a similarly simple calibrated model. Treat these as one-step predictive features. They are not independently measured alternate-action energy and not automatically complete rollout acceptance. Cluster correlated positions when assessing uncertainty; raw token count is not an independent sample count.

If using a survival model for draft length, specify conditional match probability given prior matches, depth and context. Validate predicted acceptance-length distributions against executed drafts. Do not silently assume independent, identically distributed acceptance across positions. A simple independent approximation may be an explicitly labelled baseline.

During calibration, randomize a bounded subset of observation plans with known nonzero propensities. Log action and acquisition propensities. Use propensity weighting only where overlap and conditioning assumptions hold, report effective sample size/weight sensitivity, and never use it to turn rejected-branch evidence into valid labels. For unsupported regions, fall back to the pooled model or decline acquisition.

Create offline matched-prefix validation fixtures: reconstruct an identical committed prefix and execute candidate continuations independently, with no shared mutable cache contamination. Account for reconstruction/profiling work separately. Compare same-prefix agreement features, predicted action rankings, and actual measured action outcomes. Full alternative rollouts are measured only if actually run. Do not fabricate them by replaying a shallow prediction on an invalid branch.

### 9. Concrete acquisition heuristic to implement

Implement a practical model-based value-of-information heuristic, not merely acceptance entropy multiplied by horizon.

Let `h` be an explicit useful future committed-token/request horizon, `a` a legal executing action, and `S` an observation plan. Condition observation outcomes on the actual action, plan and pre-decision state: `p(o | a, S, state)`. Different actions expose different prefixes/positions; an action-independent outcome distribution is not generally valid. Estimate:

```text
V(a,S | state,h) = predicted future energy with currently available information
                 - expected future energy after observing outcomes from S
C(a,S | state)  = incremental observation energy + incremental decision overhead
                 + acquisition-block action opportunity cost, if applicable
```

Implement an initial finite-scenario estimator:

1. From valid calibration evidence, estimate a small distribution of possible contextual agreement outcomes for each eligible `(a,S)`. Use empirical/shrunk scenarios; do not enumerate exponentially many exit subsets.
2. Identify the acquisition block whose execution produces the observation. Its committed work cannot benefit retrospectively from that observation. Subtract this work from the useful future horizon. In frozen request-reset mode, a probe collected at request completion has no value for that request or another request; it needs later within-request decisions to be useful.
3. For each scenario, update a copy of the feedback belief, predict costs for legal future actions, and choose a future action under the same latency constraint. Keep fitted model parameters frozen in frozen evaluation. Scenario feature values must have calibration support; do not insert arbitrary belief values into an unsupported regressor.
4. Use the same scenario distribution to evaluate the no-acquisition incumbent and the scenario-informed action, so estimated benefit is their expected decision-loss difference rather than an inconsistency between unrelated predictors. Check this consistency on validation. Limit or penalize extrapolation beyond a validated context/persistence horizon; current-prefix feedback is not automatically useful on future prompts.
5. If acquiring information forces an action different from the no-acquisition incumbent during the acquisition block, include its predicted extra execution energy as an opportunity cost, in addition to the observation itself. Do not hide expensive exploratory actions under a cheap observer-head estimate.
6. Validate predicted benefit against independent calibration continuations with actually acquired observations and actually executed alternatives. Use prompt-grouped fitting/validation or cross-fitting so scenario/benefit evaluation is not merely an in-sample replay of its own training labels.
7. Estimate acquisition-value uncertainty by clustered calibration resampling, including cost-model and observation-model uncertainty. Penalize disagreement/miscalibration observed on validation.

Concrete initial estimator: collect randomized **calibration transitions**. Execute an acquisition block with base action `a0` and optional `S`; record its free acceptance/telemetry feedback `F` and valid paid observations `O`. Independently randomize a qualified successor-block action `a1` with logged positive propensity, execute it, and measure its actual resolved block energy. Fit a small cross-fitted model `mu(a1,x,F,O)` from these executed successor outcomes, and an empirical/shrunk scenario model `p(F,O | x,a0,S)`. Here `x` contains only pre-acquisition-block information. Use a few calibration-selected outcome bins and pooled fallbacks rather than a high-dimensional model. Do not assign labels to unexecuted successor actions.

For a fixed useful successor block, compute the same-scenario comparison:

```text
V1 = E_F [ min_a E_(O|F) mu(a,x,F,O) ]
     - E_(F,O) [ min_a mu(a,x,F,O) ]
```

The first term lets the no-paid-information policy use all naturally available `F`; the second additionally uses `O`. This prevents attributing the value of free acceptance feedback to a paid observation. Apply the same qualification/latency constraints to both minimizations. Missing paid data are missing, not failed matches. Default the benefit horizon to **one successor block** and validate the predicted benefit on held-out executed transitions. Extend to multiple blocks only after measuring feedback persistence/decay. If sensor windows cannot resolve successor blocks, lengthen the blocks or use an explicitly supported coarser transition design; do not invent block labels. This is a deployable heuristic with testable prediction error, not an oracle or a new generic information-value theorem.

Acquisition gate:

```text
eligible(a,S) = qualified observation path
              AND h > 0
              AND enough calibration support
              AND acquisition and exploration budgets remain
              AND predicted latency is feasible

acquire (a,S) only when lower_bound(V(a,S)) > upper_bound(C(a,S)) + delta
```

Choose the feasible S with the largest conservative predicted net value. Candidates initially: none, one relevant alternative exit, or all eligible exits; add subsets only if justified. The margin `delta`, horizons, minimum support, and bounds are validation-selected and frozen. Bounds are empirical heuristic intervals unless a valid sequential construction is implemented and proved.

Default rules:

- Decision-insensitive uncertainty: no observation when all plausible outcomes select the same action.
- Zero useful horizon, exhausted budget, missing model support, stale/inaccessible telemetry: no paid observation and qualified static fallback as appropriate.
- Previous observations are used only with recorded age/context; do not pretend their validity transfers perfectly across prompts.
- Low acceptance may favor shorter drafts, but compare predicted costs instead of hard-coding temperature or acceptance labels as ground truth.
- Acquire now for future decisions. Do not charge a probe to one policy and let another policy use it for free.

Action selection minimizes conservative predicted energy per comparable committed work, subject to an optional latency constraint. A practical initial constraint is a configurable upper predicted latency ratio, default 1.05 relative to the frozen static baseline for that context. Report interval coverage and observed violations; this is not a hard SLA guarantee.

Optional stability rule: switch from the incumbent only when predicted benefit clears uncertainty and measured switching/decision overhead; add a bounded minimum dwell interval and max observation cadence. Tune on validation and compare with a version without these restrictions.

### 10. Calibration stopping and drift recovery

Start with balanced randomized seed measurements across prompts/actions and observation plans, at varied prompt/output lengths. Require enough distinct prompts to estimate variability; two repeats on two prompts are insufficient. A pilot default can use 16-32 calibration prompts and at least three repeats for paired overhead measurements, with explicit total time/energy limits. Final counts depend on pilot variability and the smallest meaningful effect.

After the seed phase, prioritize additional executed comparisons between plausible top actions in contexts where decisions remain uncertain. Use estimated deployment-horizon energy loss, not coefficient magnitude, to allocate measurements.

Continue only when estimated recoverable future decision loss plausibly exceeds the measured cost of the proposed comparison. Stop on separated decisions, no plausible repayable benefit, exhausted calibration energy/time budget, insufficient sensor quality, or unavailable qualified candidates. Preserve every raw measurement and stopping reason. Do not claim repeated looks at ordinary bootstrap intervals provide an anytime-valid guarantee; either use a justified sequential method or label stopping heuristic and evaluate it on fresh held-out data.

Compare adaptive calibration with a full fixed-grid calibration and a matched-budget random calibration. Measure selection agreement and held-out decision/energy performance versus total calibration energy. Fixing regression rank alone is not the research contribution.

Detect drift at completed reliable measurement windows using residuals against frozen validation residual distributions. Require persistent evidence, use a cooldown, and cap recalibration energy. Triggers cannot use future labels. Recovery measurements use new executed data with provenance. Fallback to the frozen qualified baseline during unsupported states. Compare against periodic recalibration and no recalibration under controlled workload/state shifts. Do not generate artificial drift by silently changing model/dtype/backend.

### 11. Experiment design and policies

Create immutable calibration, validation and final-test manifests with pinned sources, IDs, content hashes, tokenizer revision, effective input IDs/hash, truncation policy and all seeds. Existing exposed prompts belong to pilot/regression sets. Do not reuse the original startup/test slices as a fresh final test. Reject split overlap and train/test contamination of acquisition or uncertainty fitting.

A practical starting allocation is 32 calibration, 32 validation and at least 128 final-test prompts per principal workload when available. Treat these as starting counts, not a journal standard. Use pilot effect/variance to justify final counts before viewing final outcomes. For a small corpus such as HumanEval, account explicitly for finite available IDs and reserve a disjoint final subset. Counts in the new CLI must explicitly say per-dataset or total; never leave this ambiguous.

Implement these named policies with shared qualification and execution settings:

1. Independent ordinary native greedy.
2. Every legal fixed action without optional observers or adaptive shortening.
3. Best calibration/validation-selected fixed action, frozen before final test.
4. Existing acceptance-only adaptive fallback as a separate heuristic baseline.
5. Executed-feedback-only cost bandit, no purchased side information.
6. Context-aware action selector without paid observations.
7. Always-observe controller using the same cost/feedback machinery.
8. Periodic observation controller.
9. Random observation controller at matched acquisition budget/cadence.
10. Uncertainty-only acquisition using the same candidate observations.
11. Proposed energy-priced acquisition with fixed-grid calibration.
12. Proposed acquisition with decision-focused calibration stopping.

Add a faithful DEL comparison where feasible; otherwise document an explicitly named DEL-inspired adapted baseline and limitations. An all-information baseline inspired by HedgeSpec is not an exact reproduction of its multi-drafter setting. PELM includes DVFS/verification-depth controls unavailable or prohibited here; document the distinction and compare a common full-verification subset if executable, without misleading labels. Preserve legacy neural-head runs as optional baselines, not mandatory evidence of novelty.

Separate evaluation modes:

- **Frozen generalization:** learned parameters, normalization and thresholds remain frozen. Causal within-request feedback/belief changes are allowed as part of the fixed algorithm, but reset request-local state for every prompt. No cross-request held-out model fitting or tuning.
- **Online/prequential:** score each request before updating from its own outcome. Maintain independent state per policy/trajectory. Reset to the same frozen calibration snapshot for each replicate stream, and log stream order/state hashes. Use fresh held-out trajectories and avoid training on repeated copies of a prompt then describing later copies as unseen.

Randomize policy order within paired prompt/repeat blocks for frozen evaluation. In online evaluation, preserve each policy's declared stream and independent state while randomizing physical scheduling without sharing outcomes. Explicit warmups before measured blocks, comparable starting thermal conditions and order logs are required. Avoid cross-policy teacher/feedback leakage. Final policies must run the same effective prompts and stopping rules.

Required ablations: no paid feedback; always feedback; no acquisition cost; uncertainty-only versus decision-value acquisition; no horizon term; no contextual agreement; no telemetry beyond acceptance; no uncertainty/stability gate; full versus stopped calibration; no drift recovery. Invalid-prefix feedback is diagnostic-only in a sandboxed analysis, never an eligible exact-output policy. Report failed ablations rather than hiding them.

Transfer workflow: source-device frozen controller; target-device legal qualification and static baselines; source controller zero-shot with explicitly declared target probe; target-refitted controller; target-from-scratch calibration control. Use identical final effective prompt manifests. Record what target information zero-shot permits. Keep source and target uncertainty normalization distinct. A target guard must reject the wrong GPU; missing target hardware writes `not_run` with a reason and executable command. Never relabel 4060 data as 3050 data.

### 12. Persistence, schema and reporting

Use unique run directories, not automatic overwrites of `results/latest.*` or historical profiles. Save an immutable manifest/config, append-only request/decision/window JSONL, a phase/status file, and validated summary JSON/Markdown. Flush failure rows promptly and checkpoint state for deterministic resume. Resume requires compatible config, source/model/dataset/device fingerprints and completed unique keys; reject duplicates and incompatible reuse.

At minimum record:

- Schema/run IDs, UTC and local timezone-aware timestamps, exact argv/config, source revision and dirty-diff hash.
- Executable/package/OS versions; actual GPU UUID/name/VRAM; actual driver and CUDA build/runtime separately; backend/dtype/attention settings.
- Model/tokenizer/dataset revisions and hashes; prompt IDs and effective token IDs/hashes; all split/order/controller seeds.
- Clock/power-limit state, telemetry availability/staleness, warmup/cooldown/repeat protocol, counter source/scope/quality and raw endpoints.
- Policy, phase/mode, trajectory/state hash, request/repeat/block/order keys, qualified actions and selected observation plan.
- Actual tokens/length, reference IDs or content-addressed reference hash, true exact-match status, first divergence, status/error and retry/recovery cost.
- Drafted/accepted/rejected counts, full verifier calls and optional prefill count separately; valid/excluded observation counts and reason.
- Controller/acquisition wall time, memory when available, decision predictions and empirical uncertainty, benefit/cost bounds, horizon and gate/fallback reason.
- Explicit provenance for every numerical value: measured, predicted, simulated, derived or unavailable.

Avoid quadratic trace storage. Store shared input/output IDs once and use prefix offsets/hashes plus bounded diagnostic payloads. Never retain full hidden-state tensors or vocabulary logits in normal traces. Diagnostic collection must have a declared overhead/boundary and limit.

Statistics:

- Primary frozen comparisons: paired per-prompt total energy and J/token changes versus the frozen selected fixed baseline; positive saving convention stated once.
- Report macro mean/median of prompt ratios separately from token-weighted `sum(E)/sum(N)`.
- Compute point estimates from observed data, not the mean of bootstrap replicates.
- Cluster frozen repeats by prompt. For online adaptation, use independent trajectories or justified block/trajectory uncertainty, not independent-token bootstrap.
- If a counter window aggregates multiple requests, pair and resample the actually measured window/block rather than inventing per-prompt energy observations. Per-prompt energy statistics require genuinely resolved prompt-level energy; latency/token agreement can still be reported at their actual finer resolution.
- Report 95% intervals, sample/pair coverage, missing/error/mismatch counts, prompt wins/losses/ties, median and tail latency, memory and interval coverage.
- Do not reuse `joulerexit.stats.paired_savings` as a generic J/token function: it bootstraps joules per correct answer and drops undefined replicates.
- Exact-output agreement and task answer accuracy are different. Quality extraction is optional secondary analysis with tests, and never substitutes for token-ID comparison.
- A best-fixed action selected retrospectively on final test is an additional labelled hindsight reference, not the deployed baseline. Avoid noisy per-prompt minima presented as attainable oracle gains; use independent repetitions/selection splits for diagnostics.
- Break-even: report the first deployment horizon where cumulative savings after startup exceed zero with uncertainty, or `not_reached`. The simple startup-cost divided by steady-state saving formula is a labelled stationary approximation only when denominator is positive.

Produce standalone plots: paired prompt savings; cumulative deployment energy including startup; acquisition rate/cost and action changes; information-value prediction calibration; energy/latency tradeoff; never/selective/always-observe crossover versus workload length; device transfer and recovery. Plots must come from validated raw artifacts and identify scope. Use available plotting tools and declare any added analysis dependency.

### 13. CLI and executable workflow

Provide `python -m hardware_aware_selective_feedback.research` with coherent subcommands (names may be adjusted, but documentation/tests must match):

```text
probe       # no checkpoint download; inspect runtime/device/counter
prepare     # pinned data and immutable split manifest
qualify     # native-reference stability, actions, observation paths, lengths
pilot       # information usefulness and acquisition-cost feasibility
calibrate   # full or adaptive calibration; fit costs/feedback/acquisition
validate    # select hyperparameters and freeze artifacts
evaluate    # frozen or prequential, explicit repeat/stream semantics
transfer    # zero-shot/refit/from-scratch target comparisons
report      # read-only artifact validation and analysis
demo        # deterministic synthetic mechanics, explicitly simulated
```

Common flags/config must cover device selection/guard, dtype/backend, model/data/split paths, action grid, token/stopping protocol, energy source, counts, repeats/seeds, time/energy budgets, acquisition baseline, latency ratio, evaluation mode, output directory and resume. Reject incompatible flags. No model download is allowed during `--help`, `probe`, artifact-only `report`, or `demo`.

Ship runnable JSON configs, for example `configs/research/smoke.json`, `pilot.json`, `final_frozen.json`, `online_shift.json`, and `transfer.json`. Validate unknown/missing keys and serialize the fully resolved configuration. Document exact PowerShell commands from the repo root; avoid line-continuation or quoting traps. Add a plan/dry-run mode estimating work from requested prompts/actions/repeats without claiming measured runtime.

Prefer staged commands over one uninterruptible all-night job. An optional pipeline command should checkpoint stages and respect budgets. A final-test rerun must create a new run with its declared reason; never silently tune and replace unfavorable final artifacts.

### 14. Meaningful verification

Run the existing focused tests with an actual verified interpreter. Do not use historical passing counts. Avoid indiscriminate root test discovery: `edge_llm_inference` includes scripts with import-time model work. Existing intended scope is `python -m pytest -q hardware_aware_selective_feedback prototype`; unittest-based fallbacks are acceptable. Resolve real compatibility failures rather than deleting tests or weakening independent native comparisons.

Add tests that establish behavior and invariants:

- Native agreement, cache rollback/content across action switches, causal feedback ordering, EOS/cap alignment and callback compatibility on tiny random models.
- Rejected/post-stop feedback excluded; zero acceptance preserved; no future labels in decisions.
- Full-depth-only legality; no shallow padded caches in primary policies.
- Dtype-consistent observer projections; real CUDA FP16 smoke test gated on availability.
- Actual fixed-token-count rank-deficient fixture rejected; repeat groups and complete coverage correct; mismatched data cannot train qualified costs.
- Acquisition selects none when information cannot change the decision, benefit is below cost, horizon is zero, support is inadequate or budget is exhausted; acquires in a controlled useful scenario.
- Frozen model/thresholds cannot update across test prompts; within-request belief resets; online score-before-update and independent replicate state.
- Delayed/stale/zero/reset/missing counter behavior; measurement windows and no invented per-round labels.
- Startup/acquisition/recovery energy included exactly once; paired bootstrap preserves prompt clusters.
- Mismatch/OOM/error rows survive interruption; missing/duplicate pairs and invalid baselines invalidate conclusions; resume rejects incompatible fingerprints.
- Small controlled simulator with costly observations, misleading invalid branches, partial information and drift, demonstrating why each component matters. Keep simulated results separate from GPU measurements.
- CLI/config smoke tests with no downloads; legacy adapters and raw artifact validation.

CPU/tiny-model tests establish mechanics, not GPU savings. A real-device smoke run follows only after tests and device qualification. Do not launch full matrices before estimating cost and passing the pilot feasibility gate. Missing hardware/credentials/counter access should produce an honest status and exact next command, while independent implementation work continues.

### 15. Research decision rules and completion

Implement all requested components even if hardware is unavailable; mark real evidence pending. Before a large evaluation, the pilot must establish whether extra observations change action rankings and whether the resulting measured benefit could repay their costs. If not, complete the infrastructure and produce the negative feasibility report instead of inventing useful telemetry or training a network to force apparent novelty.

Freeze practical success thresholds before final evaluation. A possible initial target, subject to pilot-based justification, is at least 5% median paired GPU J/token improvement versus the selected fixed baseline, a paired interval excluding zero, no observed native mismatches, and at most 5% declared latency regression. These are project criteria, not journal requirements. A small repeatable gain may be meaningful; an unmeasured or negative gain is not a success. Any measured mismatch must remain visible and blocks an exact-output success claim for that policy.

Deliver:

1. Integrated code with no placeholder paths for the core acquisition/estimation/calibration workflow.
2. Passing meaningful tests, or exact observed failures with no fabricated passing count.
3. An offline simulated demonstration and supported real-device smoke/qualification artifacts.
4. Versioned configs, schemas, reproducible commands and resumable experiment execution.
5. A research protocol describing frozen and online hypotheses, baselines, split/uncertainty methods and measurement boundaries.
6. Related-work/attribution matrix and an honest claim ledger: implemented, tested, measured, supported, refuted or pending.
7. Validated tables/plots generated from actual available artifacts, preserving old results separately.
8. A final report separating software completion from device experiments and evidence for novelty.

Possible final interpretations must remain available:

- **Positive:** selective acquisition reduces net energy against strong fixed and adaptive baselines under the tested conditions.
- **Limited:** a small decision-focused calibration identifies a good fixed policy, but runtime paid feedback does not repay its overhead.
- **Negative:** no eligible speculative action or reliable adaptation advantage is established; report numerical/measurement limits and reproducible crossover evidence.

Do not claim universal energy optimality, guaranteed per-request savings, a new foundation model, whole-laptop energy reduction, proof of first publication, or guaranteed journal acceptance. Do not stop at saying the work could be done: implement the authorized workflow, validate it as far as evidence permits, and leave concrete runnable artifacts for any external-device stages.

## END IMPLEMENTATION PROMPT

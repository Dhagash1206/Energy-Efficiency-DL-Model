# Repository audit for the selective-feedback research implementation

Audit date: 29 October 2026, Asia/Calcutta.

Source baseline: `e3ef469c763a3e3d9dfbd312afa496714976a226`.

This audit supports [RESEARCH_IMPLEMENTATION_PROMPT.md](RESEARCH_IMPLEMENTATION_PROMPT.md). It distinguishes implemented mechanisms from scientific evidence and specifies the repairs needed before new experiments.

## Coverage and limits

The baseline contains **134 tracked files**, including **73 Python files**, **46 files under result directories**, and **two binary controller checkpoints**. The accompanying [research_repository_inventory.json](research_repository_inventory.json) lists every baseline tracked path, size, content hash, text length where applicable, and JSON parsing status. The inventory excludes the newly written handoff documents themselves.

The review covered the active LayerSkip engine, vendor helper, measurement/data layers, controllers, experiment/report scripts, focused tests, root/setup documentation, synthetic prototype, legacy training track, and saved result artifacts. Semantic review concentrated on code paths relevant to the proposed research and its validity. Binary checkpoints were inventoried without deserializing their contents; their weights are not evidence of successful research. Dependency caches, downloaded model weights, credentials, and virtual-environment package internals are not research source files and were not exhaustively reviewed.

No repository `AGENTS.md` was found by the file search. The initial Git worktree was clean. No existing implementation or historical result was changed for this request.

No training, model download, GPU experiment, or test suite was run during the audit. The Python launcher advertised a Python 3.11 executable, but an attempted lightweight interpreter inspection returned `Access is denied`. Consequently, installed-package metadata and historical test outputs do not establish a functioning current runtime or a current passing test count.

## Source map and disposition

| Area | Files/roles | Disposition |
|---|---|---|
| Root | `README.md`, `requirements.txt`, `run.txt`, `EFFICIENCY_CHANGES.txt`, `.gitignore` | Update current-facing instructions; preserve historical observations with their limits. |
| Windows setup | `scripts/setup_cuda_3050.ps1` | Reconcile interpreter/package versions; validate readiness rather than always printing success. |
| Execution | `engine.py`, `vendor/self_speculation/llama_model_utils.py` | Reuse with full-verification legality, dtype-safe observers, compatible callbacks and transition tests. |
| Optional execution optimizations | `graph_drafter.py`, `gpu_clocks.py` | Keep separate from the initial primary method; correctness/performance support is conditional. |
| Observation evidence | `feedback.py` | Preserve same-prefix validity; extend provenance/uncertainty without inventing alternate rollout costs. |
| Selection | `selector.py` | Reuse action/observation representation and independent cost bandit; replace unsupported information-value heuristic. |
| Old energy decomposition | `device_energy.py`, `reject_energy.py`, `reject_probe.py` | Preserve historical profiles; add identifiability checks and prefer direct executed-cost prediction. |
| Legacy learned control | `config_head.py`, `pelm_round.py`, `train_config_head.py`, `train_pelm.py` | Optional historical baselines; do not use synthetic teacher agreement as evidence of measured optimality. |
| Experiments | `run.py`, `sweep.py`, `startup_sweep.py`, `score_configs.py`, `score_pelm.py`, `pelm_second_gpu.py` | Consolidate on shared pinned data, qualification, persistent records, fixed-baseline selection and explicit evaluation modes. |
| Measurement | `measurement.py` | Reuse counter scopes, UUID mapping, synchronization/reset checks; add quality windows and raw endpoints. |
| Model/data paths | `paths.py`, `download_layerskip.py`, pinned loaders in `run.py` | Unify model revision, hashes, dataset splits and effective input identities. |
| Diagnostics/reporting | `diagnose_mismatch.py`, `pilot.py`, `report.py` | Repair numerical diagnostic claims; keep mechanics pilots distinct from energy evidence. |
| Active tests | `hardware_aware_selective_feedback/tests/` | Preserve native-reference, feedback, cache, measurement and pairing tests; extend missing invariants. |
| Synthetic demonstration | `prototype/` | Synthetic control mechanics only; no measured energy/latency claim. |
| Legacy model-training track | `joulerexit/` | Historical failure/alternative approach; do not mix approximate early-exit quality results with exact speculation. |
| Qwen compression scripts | `edge_llm_inference/` | Separate project stage, with import-time model work; exclude from indiscriminate root test discovery. |
| Licensing/checkpoints | `vendor/LICENSE.LayerSkip`, `checkpoints/*.pt` | Preserve attribution and applicable terms; checkpoint presence alone does not validate a contribution. |

Paths without a package prefix in the middle rows refer to `hardware_aware_selective_feedback/`.

## High-priority findings

### 1. Declared and locally recorded runtime stacks disagree

`requirements.txt` pins torch 2.6.0, transformers 4.50.0 and Hugging Face Hub below 1.0. The setup script installs the older CUDA 12.4 torch build. Local `.venv` distribution metadata instead identifies torch 2.11.0+cu128, transformers 5.21.0 and huggingface_hub 2.3.0. These are metadata observations, not successfully imported runtime versions.

The current vendor helper uses newer cache interfaces such as `cache.layers` and `is_initialized`, while documentation says newer transformers breaks it. Select a tested supported stack and synchronize code, requirements, setup and docs. Blindly reinstalling the declared requirements risks replacing a stack the modified helper expects.

The script creates Python 3.12 but tells users Python 3.11+ is sufficient. It prints readiness even though `prototype.doctor` can report missing CUDA without failing. Actual driver version and `torch.version.cuda` must be reported as separate fields.

### 2. The startup energy decomposition is not identifiable

`device_energy.fit_device` fits `E = a + bV + cA + dR` and ignores rank returned by `numpy.linalg.lstsq`. Every one of the 126 saved 4060 startup rows satisfies `V+A=64`. The intercept, V and A columns are therefore dependent. Fitted coefficients cannot be uniquely interpreted as separate physical costs.

Fix the experimental design or use an identifiable reduced/direct action-cost model. Add singular-value/rank/condition and held-out diagnostics. Fitting error versus a repeat-noise threshold does not establish causal attribution of rejected computation.

`fit_exit_profiles` also accepts mismatching rows and groups different prompts into depth/length repeat groups. `startup_choice` checks only present rows, without requiring complete balanced prompt/repeat coverage. These defects affect fitted costs and apparent noise.

### 3. Current controller labels do not demonstrate telemetry-based optimality

`pelm_round.py` constructs synthetic telemetry examples; its teacher does not use power/clock to determine energy labels, and passed previous-round speed does not directly change the action-cost calculation. The reported label agreement is training-set agreement. `nudge_last` updates fit dictionaries that `choose` does not consult when using the frozen network. A zero-acceptance outcome is replaced with one in a feedback feature.

`selector.py` has a useful joint action/observation representation, but its observer value is an uncertainty heuristic. Acquired observations mostly change a small uncertainty penalty; a validated connection from information to improved action ranking and measured savings is missing. Paid observation is limited to depth 8 / draft length 3.

### 4. Shallow verification is incompatible with the new fidelity contract

The PELM action menu includes verification stopping at layer 12. The vendor `_pad_stopped_cache` duplicates the last executed layer's K/V into skipped layers; its own documentation warns against later switching to full-depth verification. The engine accepts callbacks that can change stop depth without establishing content-correct cache transitions.

New primary policies must verify through the actual final layer and prohibit this padded-cache path. Full verification is necessary for the algorithmic contract but does not by itself prove token identity across different floating-point execution paths.

### 5. Numerical diagnostics contain misleading precision claims

The purported FP32 reference in `diagnose_mismatch.py` executes the FP16 model and casts already-rounded logits to float. This cannot recover prior rounding or resolve their argmax. Its cycle-position cursor also overlooks the initial token emitted by prefill. A small logit margin does not prove a cause or that changing projection precision fixes the mismatch.

The observer path feeds `norm(hidden.float())` into a possibly FP16 vocabulary head. Real FP16 observer behavior needs a dtype-consistent implementation and smoke test; CPU FP32 alignment tests do not establish it.

### 6. Several runner behaviors can overstate evidence

`run.py` selects a hard-coded fixed baseline `(4,3)`, calibrates initial arms on one prompt in fixed order, writes `greedy_match=True`, and raises on mismatch before preserving the failed row. It learns across held-out requests and repeats. The new runner needs a calibration/validation-selected fixed baseline, balanced randomized measurements, explicit frozen/prequential modes, and persistent failure records.

`sweep.py` is the strongest reusable pairing/eligibility foundation: it validates complete unique baseline/candidate pairs and actual token mismatches. However, adaptive-draft flags mean nominally fixed settings may adapt, and it needs persistent error rows, consistent warmups and uncertainty analysis.

`startup_sweep.py` and score scripts use separate unpinned dataset slices, sparse provenance and custom `generate(0,0)` references. `pelm_second_gpu.py` does not execute transfer when the target GPU is present. `download_layerskip.py` must use the same pinned revision as the primary runner.

### 7. Existing statistics are task-specific

`joulerexit.stats.paired_savings` estimates a difference in joules per correct answer, skips bootstrap samples with undefined denominators, and uses a bootstrap mean as its estimate. It is not a generic paired energy/J-token analysis function. New statistics need direct observed point estimates, prompt-clustered frozen repeats, and trajectory-aware online uncertainty.

Task answer extraction and output token agreement are distinct. Low or zero task accuracy makes joules-per-correct unstable; exact-output systems should primarily compare identical generated work and report quality separately.

### 8. Trace and optimization boundaries need care

Current traces can repeat entire prefixes and contexts for many observations, producing large host-memory/serialization overhead. Shared IDs plus offsets/hashes can retain reconstructibility at lower cost.

CUDA graph drafters currently conflict with observer hooks/dynamic action callbacks. Batched ordinary inference exists; batched speculative inference does not. These features cannot be silently enabled for only one policy or treated as proven GPU optimizations from CPU tests. Clock restoration may reset to defaults rather than a known prior custom lock; cleanup should log actual state and preserve original exceptions.

## Saved-result interpretation

| Artifact/family | What the evidence supports |
|---|---|
| `results/latest.*` | FP32 CPU, latency objective, four held-out prompts, one repeat, eight-token cap. All 45 executed requests match native greedy. Adaptive GPU-energy success is not tested; CPU J/token is 11.601508 adaptive versus 11.137473 ordinary. |
| 4060 startup profile | Sixteen prompts, eighteen measurements per action across seven actions, raw counts and energy; preliminary and rank-deficient. Four speculative action groups contain mismatches. |
| `rtx4060-pelm-score.json` | Eight GSM8K and four SVAMP examples. On GSM8K, adaptive uses 1122.202 J versus 1039.185 J for fixed depth 4 / length 1, about 8% more. No supported adaptive advantage. |
| `nvidia-geforce-rtx-4060-laptop-gpu-score.json` | The different learned-head pilot explicitly records `bar_cleared=false`; fixed and external selector consume less than the head. Do not confuse this with the per-round PELM score. |
| `rtx3050-device-profile.json`, `rtx3050-pelm-status.json` | Specific startup/transfer stages are not measured; visible device at those attempts was 4060. |
| `fp16-run/`, `smoke/` | Some historical artifacts explicitly identify an RTX 3050 Laptop GPU. Thus a blanket statement that no 3050 experiment exists is incorrect. |
| Fixed confirmation at 16-token cap | Depth 4 / length 1 is eligible at approximately 1.492836 J/token versus ordinary 1.690754. Other candidate mismatches remain visible; total 74/80 requests match. |
| Fixed sweep at 200-token cap | 108/144 requests match. Every speculative configuration is ineligible for exact-output energy conclusions in that run. |
| Pinned engine optimization comparison | All 96 short requests match. Speculative latency improves 13.8%; speculative GPU energy is effectively unchanged. Ordinary latency/energy also improve, so optimizing a shared engine is not evidence for selective-feedback novelty. |
| Memory optimization | A narrow memory improvement; it does not establish lower GPU power or energy. |
| Pilots and `prototype/synthetic_profile.json` | Mechanics/simulated control evidence; not measured research savings. |
| `joulerexit/results/rtx4060-outcome.md` | Earlier training/gate approach failed its stated success bar. Preserve as historical outcome. |

Device names missing from historical metadata cannot be inferred reliably from filenames. Different caps, prompts, backends, precisions and energy scopes must not be combined as equivalent runs.

## Recommended implementation order

1. Establish a working tested runtime and honest device/sensor diagnostics.
2. Repair full-depth legality, observer dtype, failure persistence and independent native qualification.
3. Introduce immutable splits, versioned schemas, raw JSONL and resumable stage execution.
4. Fit identifiable action costs and contextual valid-feedback features from actual measurements.
5. Test whether extra information changes decisions enough to repay its cost.
6. Implement and validate energy-priced acquisition, then adaptive calibration stopping and bounded drift recovery.
7. Compare against strong fixed/no-observation/always/periodic/random/uncertainty/bandit baselines.
8. Freeze settings; run separate frozen and online evaluations, then honest target-device transfer.
9. Generate a claim ledger and reproducible positive, limited or negative report from raw artifacts.

The implementation prompt contains concrete contracts, gate formulas, experiment modes, artifact schemas, tests, CLI stages and completion criteria. It does not authorize assuming that a future positive result already exists.

## Closest related work and implications

- [LayerSkip](https://aclanthology.org/2024.acl-long.681/) supplies early-exit/self-speculation foundations; preserve attribution.
- [DEL](https://arxiv.org/abs/2504.05598) already selects dynamic exits/lengths and collects intermediate-layer information, including discussion of its overhead. Intermediate feedback itself is not new.
- [BanditSpec](https://arxiv.org/abs/2505.15141) already provides online speculative configuration selection with regret analysis.
- [HedgeSpec](https://arxiv.org/abs/2510.20064) already obtains information about unchosen drafters and analyzes evaluation overhead.
- [PELM](https://arxiv.org/abs/2609.09662) already combines learned hardware control with speculation and variable verification depth.
- [Zeus](https://www.usenix.org/conference/nsdi23/presentation/you) and [Conservative Bandits](https://proceedings.mlr.press/v48/wu16.html) establish relevant energy-learning and conservative-exploration ingredients.

The prospective contribution is a validated decoding-specific method for deciding when extra information is worth its measured energy, supported by valid partial-feedback prediction and decision-focused calibration. This remains a research hypothesis until evidence and further related-work comparison support it.

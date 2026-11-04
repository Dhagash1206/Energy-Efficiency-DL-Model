# Code Review and Architecture - Energy-Efficiency-DL-Model

- Branch reviewed: `Dhagash_patch5` (commit `b3e9482`); fixes for findings 5, 6, 11 and 14 are on `Dhagash_patch6`. File and folder names below use the capitalised layout introduced afterwards.
- Date: 2026-11-02
- Scope: all tracked code (about 5,500 lines of Python), scripts and docs
- Test status: 57 tests pass (5 duplicate pilot tests removed; `python -m pytest -q Layerskip_energy Controller_simulation`)

---

## 1. What the project does

- Goal: reduce the energy and latency of LLM inference on edge GPUs (RTX 3050) without changing the output.
- Main technique: LayerSkip self-speculative decoding.
  - The first `depth` layers draft cheap tokens.
  - The remaining layers verify all drafts in one pass.
  - Every output is checked against plain greedy decoding (exact-output guarantee).
- On top of that, a selector learns per request which (depth, draft length) to use and whether intermediate-layer observations are worth their cost.
- Current best result: the adaptive policy used 20.7% less GPU energy per token and was 32.8% faster than ordinary decoding (FP32, RTX 3050, 133/133 exact matches).
- Two side tracks:
  - `Controller_simulation/`: a synthetic simulation of an energy-aware controller (no real model).
  - `Distillation/`: knowledge distillation of Qwen3-1.7B into Qwen3-0.6B.

---

## 2. Architecture

### 2.1 Layers (top to bottom)

- **Entry points (CLIs)**
  - `Run_experiment.py`: main experiment (7 policies, calibration and test phases)
  - `Fixed_setting_sweep.py`: fixed-setting energy sweep plus batched baseline
  - `Mechanics_pilot.py`: small mechanics check (tiny / SmolLM / LayerSkip)
  - `Diagnose_mismatch.py`: near-tie analysis of greedy mismatches
  - `Download_weights.py`: resumable, hash-verified weight download
  - `Energy_meter.py`: energy-counter probe
  - `Controller_simulation/Demo.py`, `Controller_simulation/Check_gpu.py`: simulation demo and CUDA check
  - `Distillation/Distill.py`, `Benchmark_models.py`: distillation and model comparison
- **Policy layer**
  - `Policy_selector.py`: `Selector` (cost-aware heuristic and observer scheduling) and `CostBandit` (lower-confidence-bound baseline)
- **Decoding engine**
  - `Decoding_engine.py`: `native_greedy`, `batched_greedy`, `generate` (speculative, adaptive drafts, observers)
  - `Cuda_graph_drafter.py`: optional static-cache / CUDA Graph draft step
  - `Observer_feedback.py`: decides which observer predictions count as valid evidence
- **Model internals (vendored)**
  - `Vendor/self_speculation/llama_model_utils.py`: split forward passes (`forward`, `forward_early`, `forward_remainder`) and KV-cache cropping
- **Hardware layer**
  - `Energy_meter.py`: `Meter` (latency plus NVML / RAPL / Windows EMI / file energy counters)
  - `Gpu_pinning.py`: clock and power-limit pinning through `nvidia-smi`
- **Data and storage**
  - `Run_experiment.py` `load_prompts` / `source_file`: GSM8K and HumanEval downloads at pinned commits, SHA-256 recorded
  - `Cache_paths.py`: Hugging Face cache location (`layerskip/cache`, git-ignored)
  - `Results/`: JSON and Markdown reports grouped by experiment

### 2.2 Request flow (one speculative request)

- `run_one` (Run_experiment.py) calls `Meter.run`, which wraps `generate` (Decoding_engine.py).
- `generate`:
  - Prefill: `forward` runs all layers over the prompt and keeps only the last logits.
  - Each cycle:
    - choose the draft length `k` (fixed, or `_adaptive_k`: full length, shorter, or ordinary fallback)
    - draft `k` tokens with `forward_early` (or `GraphDrafter.draft`)
    - verify with one `forward_remainder` pass over `[pending, *drafts]`
    - commit the accepted prefix plus one correction token
    - crop the KV cache with `crop_past_key_values`
  - Observers (optional): forward hooks record intermediate-layer predictions; `validate_feedback` keeps only committed positions.
- `run_one` compares the tokens with the `native_greedy` reference and sets `greedy_match`.
- `Run_experiment.py` `execute` records the row; on a mismatch it records the details and excludes that setting from the selector and bandit.

### 2.3 Experiment flow (`Run_experiment.py main`)

- Load prompts (GSM8K and HumanEval, pinned commits) and the model (pinned LayerSkip revision).
- Calibration:
  - run every arm once on the first prompt
  - run observer pairs (base vs observed) to measure observation cost
  - run adaptive and bandit calibration on the remaining calibration prompts
- Test: for each held-out prompt and repeat, run 7 policies in shuffled order:
  - ordinary
  - fixed
  - never observe
  - selected observe
  - always observe
  - bandit
  - adaptive
- Report: aggregates, paired deltas, feasibility, mismatches, and selector summary, written as JSON and Markdown.

---

## 3. Files, purpose and functions

### 3.1 Repository root

- `README.md`: project goals, planned stages, current layout, setup steps.
- `Requirements.txt`: single pinned dependency list (torch 2.6.0, transformers 4.50.0, NVML, datasets, bitsandbytes on non-macOS).
- `HOW_TO_RUN.txt`: copy-paste commands for setup, tests, demo, pilot, experiment, sweep and distillation.
- `EFFICIENCY_CHANGES.txt`: what each efficiency option does, measured results, RTX 3050 benchmark.
- `CODE_REVIEW.md`: this document.
- `.gitignore`: ignores venvs, caches, `models/`, `layerskip/` (weights and HF token), `.vscode/`.
- `Scripts/Install-CudaEnvironment.ps1`: Windows setup; creates `.venv`, installs CUDA torch 2.6.0 (cu124) and `Requirements.txt`, runs `Layerskip_energy.Check_gpu`.

### 3.2 `Layerskip_energy/` (main experiment)

- `__init__.py`: package marker.
- `README.md`: how to run the experiment, module boundaries, results layout, mismatch handling.
- `.gitignore`: ignores `cache/`, `data/`, `.runtime/`.

- `Decoding_engine.py`: decoding engine.
  - `_check_cache`: asserts every KV-cache layer has the expected length at a cycle boundary.
  - `_observe`: context manager; forward hooks that project intermediate-layer hidden states to tokens.
  - `_adaptive_k`: picks the draft length from recent acceptance (>=50% full, 30-50% shorter, <30% ordinary, with periodic re-probe).
  - `native_greedy`: one-token-at-a-time greedy decoding; the exact-output reference.
  - `batched_greedy`: greedy decoding of several prompts in one left-padded batch.
  - `generate`: ordinary or speculative decoding with observers, adaptive drafts, margin logging and optional `GraphDrafter`.
  - `_resolve_exits`: which observer depths to collect this cycle (none / selected / all / periodic).

- `Vendor/self_speculation/llama_model_utils.py`: attributed LayerSkip helper (CC BY-NC, see `Vendor/LICENSE.LayerSkip`).
  - `ForwardResult`: logits, KV cache and exit-query hidden states.
  - `_make_causal_mask`, `_expand_mask`, `_causal_mask`: attention masks for cached decoding.
  - `_run_decoder`: runs one decoder layer with the transformers 4.50 API (rotary embeddings, `cache_position`).
  - `crop_past_key_values`: trims rejected draft positions from the cache.
  - `forward`: all layers (prefill and ordinary steps).
  - `forward_early`: layers `0..depth-1`; produces a draft token and accumulates exit hidden states.
  - `forward_remainder`: layers `depth..N-1` over the whole draft block; verifies drafts.

- `Cuda_graph_drafter.py`: prototype CUDA Graph draft step.
  - `GraphDrafter.__init__`: allocates fixed-size key/value buffers for the draft layers.
  - `reset` / `commit`: track which buffer positions match the engine cache.
  - `draft`: runs up to `k` draft steps; returns tokens, an extended tuple cache and exit hidden states.
  - `_sync`: copies cache positions written by prefill or ordinary steps into the buffers.
  - `_capture` / `_run_step`: CUDA Graph capture and replay (eager fallback off CUDA).
  - `_step`: one fixed-shape draft step (manual attention with a position mask).

- `Observer_feedback.py`: observer evidence.
  - `validate_feedback`: one row per (depth, position); valid only inside the committed prefix.
  - `LearningState`: running match counts per depth (`update`, `export`).

- `Policy_selector.py`: request-level policies.
  - `Action`: (depth, draft length, observer exits) with validation; `key` gives (depth, length).
  - `Selector`:
    - `update`: stores cost per token (separately for observed runs) and acceptance evidence.
    - `exclude`: removes a setting after a greedy mismatch (never ordinary).
    - `record_observer_pair`: measured observer overhead from a paired run.
    - `choose`: lowest score arm; adds paid exits only at (8, 3) when calibrated and justified.
    - `summary`: JSON statistics.
    - `_uncertainty`: Laplace-smoothed acceptance uncertainty.
    - `_arm_score`: mean cost plus noise and uncertainty penalty; per-context costs when `context_aware`.
  - `CostBandit`: lower-confidence-bound baseline (`exclude`, `update`, `choose`, `summary`).

- `Energy_meter.py`: energy and latency.
  - `Meter`: picks NVML (CUDA), RAPL (Linux CPU), EMI (Windows CPU) or an external counter file.
    - `run`: synchronizes the device, times the call, returns joules or `None` with a reason.
    - `_joules`: counter difference with wrap and reset checks.
    - `describe` / `close`: report and release counters.
  - `main`: `--probe`-style CLI that prints counter availability.

- `Gpu_pinning.py`: benchmark repeatability.
  - `_nvidia_smi`: runs one `nvidia-smi` command with clear permission errors.
  - `pinned_gpu`: context manager; locks clock and/or power limit and always restores them.

- `Run_experiment.py`: main experiment.
  - `resolve_dtype`: fp32 everywhere, fp16 on CUDA only.
  - `source_file`: downloads one dataset file at a pinned commit and records its SHA-256.
  - `load_prompts`: disjoint GSM8K train/test and HumanEval calibration/test prompts.
  - `load_model`: pinned LayerSkip checkpoint; optional int4/int8, `torch.compile`, attention backend.
  - `build_graph_drafters`: one `GraphDrafter` per depth.
  - `first_divergence`: index of the first differing token.
  - `run_one`: one measured request plus greedy check (optionally aborts on mismatch).
  - `prompt_context`: dataset plus short/long label for the context-aware selector.
  - `row`: flattens one request into a report row.
  - `main`: CLI, calibration, test phase, aggregates, JSON and Markdown report.

- `Fixed_setting_sweep.py`: fixed-setting sweep.
  - `summarize`: per-setting energy and latency with paired deltas; valid only if baseline and candidate match greedy.
  - `measure_batched`: batched ordinary-decoding baseline per batch size.
  - `run_sweep`: runs all settings in shuffled order, writes JSON and Markdown.
  - `main`: CLI (depths, lengths, batch sizes, adaptive drafts, CUDA Graphs, GPU pinning).

- `Mechanics_pilot.py`: mechanics check merged from the old `prototype/selective_feedback`.
  - `get_model`: random tiny Llama, SmolLM2-135M or LayerSkip; optional fp16 or CPU int8.
  - `prompts_for`: fixture prompts or GSM8K test prompts.
  - `main`: runs every (depth, length, observer mode) case, writes `Results/Pilot/`.
- `Pilot_report.py`: `render` turns a pilot result into Markdown; `main` re-renders a saved JSON.
- `Cache_paths.py`: cache paths (`REPO`, `CACHE`, `HUB`, `TOKEN`); `ensure` creates them.
- `Download_weights.py`: `main` downloads `model.safetensors` in 8 MiB ranges, resumes, verifies SHA-256.
- `Diagnose_mismatch.py`:
  - `cases_from_report`: reads unique mismatches and prompt-set arguments from a report.
  - `top2_margin_at`: top-1 vs top-2 logit gap.
  - `run_fp16_with_margins`: speculative run with margin logging.
  - `run_fp32_reference`: greedy with FP32 logits.
  - `main`: verdict per case (near-tie vs structural).

- `Tests/`:
  - `test_decoding_engine.py`: greedy agreement, observer alignment, EOS, cache rollback, adaptive thresholds and fallback, batched greedy, deep-model positions.
  - `test_policy_selector.py`: feedback validity, exploration, observer cost, exclusion of paid runs, context-aware selection.
  - `test_energy_meter.py`: RAPL wrap, external counter rules, missing counter.
  - `test_fixed_setting_sweep.py`: paired validity rules, meter cleanup, batched baseline.
  - `test_run_experiment.py`: precision rules; end-to-end mismatch recording and `--stop-on-mismatch`.
  - `test_pilot.py`: pilot-style agreement and feedback checks.
  - `test_gpu_pinning.py`: apply and restore with a fake `nvidia-smi`.
  - `test_cuda_graph_drafter.py`: static-cache step equals `forward_early`; generation equals greedy.
  - `test_diagnose_mismatch.py`: reading cases from run and sweep reports.

- `Results/`:
  - `latest.*`: default `Run_experiment.py` output.
  - `benchmark/`: reference FP32 RTX 3050 run.
  - `smoke/`, `fp16-run/`, `fp16-100tokens/`, `fp16-200tokens/`, `fixed-sweep/`, `engine-optimization/`, `memory-optimization/`, `pilot/`: earlier runs by experiment.

### 3.3 `Controller_simulation/` (synthetic simulation)

- `README.md`: how to run the demo and its limits (all numbers fictional).
- `Synthetic_profile.json`: invented action costs used by the demo.
- `Controller.py`:
  - `Action`: validated action cost profile.
  - `State`: context, remaining tokens, acceptance history.
  - `expected_committed_tokens`: expected tokens per cycle under independent acceptance.
  - `Controller`: estimates and chooses actions with an energy margin.
- `Demo.py`:
  - `fraction`: deterministic pseudo-random number.
  - `ToyModel`: token source with simulated draft errors.
  - `load_actions`: reads the profile.
  - `run_policy`: runs one policy.
  - `positive_int`: argument check.
  - `main`: prints the policy comparison.
- `Check_gpu.py`: `check_cuda` and `main` report whether the RTX 3050 CUDA path is usable.
- `Archive/` (no entry point, kept for reference):
  - `Experiment.py`: calibration, validation tuning and held-out comparison (`calibrate`, `estimate_action`, `choose`, `run_request`, `aggregate`, `experiment`).
  - `Backends.py`: `SimulatedBackend` (fictional costs) and `TransformersBackend` (slow real-model reference without KV cache).
  - `Data.py`: prompt fingerprints, stratified splits, manifests, pinned public dataset fetch.
  - `Devices.py`: `detect` and `select_device` capability checks.
  - `Measurement.py`: `EnergyMeter` (NVML only), `start_measurement`, `stop_measurement`.
- `Tests/test_controller.py`: controller maths and toy policy correctness.

### 3.4 `Distillation/` (distillation track)

- `Distill.py`: script (runs at import).
  - `format_example`: builds an Alpaca instruction prompt.
  - `tokenize`: adds EOS, pads, masks padding with `IGNORE_INDEX`.
  - `distillation_loss`: shifted, padding-masked KL plus cross-entropy, both per token.
- `Benchmark_models.py`: script; `run` loads a model, times 50 tokens, prints tokens/s, latency and memory.

---

## 4. Code review findings (most important first)

- **1. `--compile` has no effect on measured requests (High)**
  - Where: `Run_experiment.py:183` wraps the model with `torch.compile`.
  - Why: `generate` never calls `model(...)`. The vendored `forward`, `forward_early` and `forward_remainder` call `model.model.layers[...]` and `model.lm_head` directly, which bypass the compiled wrapper. Only `native_greedy` (`Decoding_engine.py:169`) and `batched_greedy` (`Decoding_engine.py:218`) use it.
  - Impact: the help text promises about 5-10% lower latency, but measured requests never get it. Only the unmeasured greedy reference is compiled, which could also change its rounding and cause false mismatches.
  - Fix: compile the per-layer modules used by the vendored functions, or remove the flag and its help text.

- **2. FP16 exact-match check is fragile by design (High, research impact)**
  - Where: `forward_remainder` verifies `k+1` tokens in one matrix multiply; `native_greedy` decodes one token at a time.
  - Why: different matrix shapes round differently in FP16, so near-tie tokens can flip. This explains the `depth=4, length=3` failure in the large FP16 run.
  - Impact: FP16 runs, the most energy-efficient setting, keep failing correctness for reasons unrelated to the method.
  - Fix:
    - run the new mismatch logging, then `Diagnose_mismatch.py --report`
    - if the failures are near-ties, report them as a separate category, or use an FP32 reference for FP16 runs.

- **3. `Download_weights.py` does not prepare what `Run_experiment.py` loads (Medium)**
  - Where: `Download_weights.py:26` downloads only `model.safetensors`, at the current `main` revision. `Run_experiment.py:143` treats the pinned snapshot as complete when `model.safetensors` exists.
  - Impact:
    - if `main` equals the pinned revision, `Run_experiment.py:152` fails because `config.json` and the tokenizer files are missing
    - otherwise the 2.5 GB download is unused and `Run_experiment.py` downloads again.
  - Fix: download with `snapshot_download(revision=REVISION, allow_patterns=[...])`, or also fetch the config and tokenizer files at the pinned revision.

- **4. Pilot prompts are not pinned (Medium)**
  - Where: `Mechanics_pilot.py:63` downloads GSM8K from the latest `master` commit.
  - Impact: pilot results are not reproducible, unlike `Run_experiment.py`, which pins the dataset commit and records its SHA-256.
  - Fix: reuse `run.DATASETS` / `source_file`.

- **5. Unused prototype modules (Medium, maintainability)** - FIXED: moved to `Controller_simulation/Archive/`
  - Where: `Controller_simulation/Experiment.py`, `Backends.py`, `Data.py`, `Devices.py`, `Measurement.py`.
  - Why: no entry point imports them; there is no `Controller_simulation/__main__.py`. `Devices.py:59` tells users to run `python -m prototype doctor`, which does not exist.
  - Fix: add a small CLI, or move them to an `Archive/` folder and say so in the README.

- **6. `Run_experiment.py main()` is too long (Medium, maintainability)** - FIXED: split into `build_parser`, `resolve_device`, `Experiment.calibrate`, `Experiment.run_test_phase`, `aggregate_results`, `render_markdown`
  - Where: `Run_experiment.py:240-610` (about 370 lines) mixes CLI parsing, calibration, the test phase, statistics and report writing.
  - Impact: hard to test or change one phase safely.
  - Fix: split into `calibrate()`, `run_test_phase()`, `aggregate()` and `write_report()`.

- **7. The sweep cannot run without an energy counter (Low-Medium)**
  - Where: `Fixed_setting_sweep.py:318`.
  - Impact: the sweep and the batched baseline cannot run on a Mac or CPU, even for latency only.
  - Fix: add `--objective latency`, like `Run_experiment.py`.

- **8. Observer calibration is hard-coded to depth 8, length 3 (Low)**
  - Where: `Run_experiment.py:379`, `Policy_selector.py` `choose`.
  - Impact: if (8, 3) mismatches and is excluded, the observer pairs still run it; the depths also assume a 16-layer model.
  - Fix: derive the observed arm from the selector's remaining arms and the model's layer count.

- **9. `Meter.close` leaves `available` as True (Low)**
  - Where: `Energy_meter.py:286`.
  - Impact: `run()` after `close()` would call NVML after shutdown.
  - Fix: set `self.backend = None` in `close`.

- **10. Pilot report text is wrong on CUDA (Low)**
  - Where: `Pilot_report.py:19` always says "CPU energy is unavailable", even when NVML energy was measured.
  - Fix: use the report's real energy boundary.

- **11. `Distillation/test_models.py` looks like a test (Low)** - FIXED: renamed to `Benchmark_models.py`
  - Why: it matches pytest's `test_*.py` pattern but loads two models at import time.
  - Impact: running `pytest` from the repository root fails or loads gigabytes.
  - Fix: rename to `Benchmark_models.py`.

- **12. Long prompts lose their end (Low, experiment validity)**
  - Where: `MAX_PROMPT_TOKENS = 192` keeps the first 192 tokens.
  - Impact: long HumanEval prompts can lose the function signature, so generation continues from the middle of a docstring.
  - Fix: keep the last 192 tokens, or skip prompts that are too long and record it.

- **13. No confidence intervals (Low, evidence)**
  - Where: `Run_experiment.py` aggregates report totals and medians only.
  - Impact: the 20.7% energy saving comes from 16 runs of 16 tokens per policy, with no error bars.
  - Fix: add bootstrap confidence intervals on the paired deltas against ordinary decoding.

- **14. Remaining lint (Low)** - FIXED: `ruff format` applied, `ruff check` passes (settings in `pyproject.toml`)
  - Unused `ForwardResult` import at `Decoding_engine.py:28`.
  - `raise` without `from` at `Run_experiment.py:138`.
  - Unused imports in `Diagnose_mismatch.py`.
  - `zip()` without `strict=` in `Fixed_setting_sweep.py`.

---

## 5. Strengths

- Every request is checked against plain greedy decoding, and mismatches are recorded, not hidden.
- Energy scope is explicit (GPU, CPU package, whole device); missing sensors give `None` with a reason, never estimates.
- Model and dataset revisions are pinned; dataset hashes are recorded in reports, and the weight download is verified with SHA-256.
- Credentials are handled safely: the HF token is never printed or stored in results, and `trust_remote_code=False` is used everywhere.
- Policies run in shuffled order with paired comparisons, which reduces order and thermal bias.
- Good test coverage of engine invariants: 57 tests, including cache rollback, EOS and position-ID regressions.
- New efficiency features are opt-in, so earlier results stay reproducible.

---

## 6. Recommended next steps

- Fix findings 1 (`--compile`) and 3 (download) before the next RTX 3050 run.
- Rerun the large FP16 experiment with mismatch logging, then run `Diagnose_mismatch.py --report`.
- Repeat the FP32 benchmark with 64 tokens, more prompts and confidence intervals.
- Test `--adaptive-drafts`, `--cuda-graph-drafts`, `--batch-sizes` and GPU pinning on the RTX 3050.
- Split `Run_experiment.py main()`, and decide whether to keep or archive the unused prototype modules.
- Add a GitHub Actions job that runs the 57 tests on every push.

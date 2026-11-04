# Hardware-aware selective-feedback research

This is the executable LayerSkip experiment, not model training. It loads the
pretrained `facebook/layerskip-llama3.2-1B` checkpoint at pinned revision
`81deb0f88734409cca506bcefcfb5c6bd2667565`. The code downloads a missing
snapshot into ignored `layerskip/cache` using the machine's authorized Hugging
Face login. It never opens, prints, or stores a token in results. Each teammate
needs their own checkpoint access. Run `hf auth login` privately if download
fails; never put a token in Git.

`Decoding_engine.py` implements ordinary greedy and cached self-speculative decoding.
`Observer_feedback.py` excludes observations outside the committed prefix.
`Policy_selector.py` chooses a decoding action and whether to buy intermediate-exit
observations, with an independent-arm bandit comparison. `Energy_meter.py`
selects a real cumulative energy counter or reports why none is available.
`Run_experiment.py` loads pinned GSM8K and HumanEval prompt workloads, checks every
output against ordinary greedy token IDs, and writes `Results/latest.json`
and `Results/latest.md`. `Vendor/self_speculation/llama_model_utils.py` is the attributed LayerSkip
helper under `Vendor/LICENSE.LayerSkip` (CC BY-NC); retain the license and follow the
gated checkpoint's terms. `Tests/test_*.py` are focused correctness tests.

Install Python, then install a PyTorch build for the hardware you intend to
run (CPU, CUDA, Intel XPU, or macOS MPS). Install the root `Requirements.txt`
in the same environment. The repository's existing Windows CPU runtime can be
used for local checks:

    .\Layerskip_energy\.runtime\python\python.exe -B -m unittest Layerskip_energy.Tests.test_decoding_engine Layerskip_energy.Tests.test_policy_selector Layerskip_energy.Tests.test_energy_meter -v
    .\Layerskip_energy\.runtime\python\python.exe -B -m Layerskip_energy.Run_experiment --probe --device auto
    .\Layerskip_energy\.runtime\python\python.exe -B -m Layerskip_energy.Run_experiment --device cpu --calibration 2 --test 2 --tokens 8 --repeats 1

With a suitable Python on another machine, replace that interpreter path with
`python`. `--device auto` chooses CUDA, then Intel XPU, then Mac MPS, then CPU
when each is available in PyTorch. You may force `--device cpu`, `cuda`, `xpu`,
or `mps`. Use `--probe` to check the selected device and counter without
loading the 1B model or datasets. A backend being available does not prove
this custom LayerSkip kernel runs on it; run the focused tests and a small
experiment on each new device first.

The runner defaults to FP32. Use `--precision fp16` for CUDA inference on
lower-memory GPUs; FP16 is currently restricted to CUDA, and output correctness
is checked against ordinary greedy decoding using the same precision. A 4 GB
GPU may still run out of memory, depending on available VRAM and runtime use.
Start with a small smoke run and save it separately to avoid replacing
`Results/latest.json`:

    python -m Layerskip_energy.Run_experiment --device cuda --precision fp16 --objective energy --calibration 1 --test 1 --tokens 4 --repeats 1 --observer-pairs 1 --output Layerskip_energy/Results/Smoke/rtx3050-fp16-smoke.json

The Markdown report is written beside the JSON report with a `.md` extension.
Only increase prompts, tokens, repeats, or calibration after the smoke run
completes and all greedy comparisons pass.

To measure fixed speculative settings without feedback observers, run a paired
energy sweep. It randomizes setting order for each prompt and repeat, checks
every output against native greedy tokens, and records mismatches explicitly:

    python -m Layerskip_energy.Fixed_setting_sweep --device cuda --precision fp16 --test 4 --tokens 16 --repeats 2 --output Layerskip_energy/Results/Fixed-sweep/fixed-sweep.json

The default grid is depths 4, 8, 12 and draft lengths 1, 2, 3, 4, plus
ordinary decoding. Use `--depths` and `--lengths` to narrow the grid, and a
different `--seed` for a fresh prompt split. A setting with any greedy mismatch
is unsuitable for an exact-output energy comparison even if its energy is low.

The module boundaries are: `Decoding_engine.py` owns token generation and cache rollback;
`Vendor/self_speculation/llama_model_utils.py` owns split model execution; `Observer_feedback.py` validates
observations; `Policy_selector.py` chooses actions; `Energy_meter.py` owns hardware
counters; and `Run_experiment.py` / `Fixed_setting_sweep.py` coordinate experiments and reporting.
The sweep CLI delegates execution to `run_sweep`, and `summarize` independently
validates complete, unique baseline/candidate pairs. A failed ordinary baseline
invalidates comparisons for every setting. `Meter` supports context management
so initialization and execution errors still release counter resources.

The decoding helper reuses rotary position embeddings across layers and creates
only causal masks for these unpadded inputs. The engine keeps selected token
tensors on the device for subsequent steps; host reads still support EOS and
acceptance decisions. These changes reduce repeated work, but energy savings
must be established by measurements. FP16 speculative output mismatches were
observed on `gsm8k-test-245` with depth/length `(4, 2)`, `(8, 1)` and `(8, 2)`;
those settings require further diagnosis before exact-output claims.

Prompt prefill now projects vocabulary scores only for the final prompt token;
all prompt positions still populate the KV cache. Temporary forward results are
released before later draft/verification steps. The 32-token CUDA comparison in
`Results/Memory-optimization/rtx3050-memory-optimization.md` measured about 41 MiB lower total peak
allocated memory for depth 4 / draft length 1. Model weights remain resident;
this does not establish a reduction in GPU power or utilization. Changing the
projection shape can affect floating-point rounding, so keep checking native
greedy agreement on new workloads.

Energy is **measured only when a working cumulative counter exists**:

- NVIDIA CUDA GPU: `--energy-source auto` uses NVML GPU joules. Its boundary is
  the NVIDIA GPU, not the whole laptop.
- Linux CPU: `--device cpu --energy-source auto` uses readable top-level
  `/sys/class/powercap/*rapl*` package counters. Its boundary is CPU packages;
  integrated-GPU activity may share a package, but other laptop components
  are outside the measurement. RAPL access depends on OS permissions.
- Windows CPU: `--device cpu --energy-source auto` uses a readable Windows EMI
  V2 `_PKG` channel when present. This measures CPU-package energy, not the
  whole laptop. If absent or inaccessible, the run uses latency.
- Windows XPU and macOS CPU/MPS: this code has no built-in per-request energy
  counter. They still run the latency experiment. A supported external meter
  may provide a **cumulative** numeric text file:

      python -m Layerskip_energy.Run_experiment --device mps --energy-source file --counter-file /path/to/cumulative.txt --counter-unit joules --energy-scope whole_device --objective energy --probe

  The file must be updated by the actual meter throughout the experiment;
  a static number is not a sensor. Set the scope to what that meter truly
  covers. `--counter-unit` also accepts `millijoules` or `microjoules`.
  `--objective energy` stops if the counter is unavailable. `--objective auto`
  otherwise uses labeled latency and records `energy_j: null`. Never compare
  joules across different counter boundaries as if they measured the same
  hardware.

A larger run on an energy-capable device, after the small check, is:

    python -m Layerskip_energy.Run_experiment --device cuda --objective energy --calibration 4 --test 8 --tokens 64 --repeats 3 --observer-pairs 6

Adjust `--device` and sensor flags for your machine. These counts mean prompts
**per dataset**, so that example has 8 calibration and 16 held-out prompts.
The data sources, revisions, SHA-256 hashes, item IDs, and disjoint split are
saved in JSON. GSM8K train and separate HumanEval IDs calibrate; GSM8K test
and other HumanEval IDs are held out. HumanEval code is never executed or
scored. Calibration pairs are spread over distinct prompts when available,
with execution order randomized. Fixed, never/selected/always observe,
independent bandit, adaptive, and ordinary policies run on the same held-out
prompts. The report includes raw runs, paired cost deltas, observer overhead,
and whether the best **executed** fixed configuration varies by prompt.

The selector buys observations only for the measured depth-8, draft-length-3
setting and exit sets `(4,)` or `(4, 12)`. Exit feedback is valid one-step
same-prefix evidence, not another layer's full rollout or imaginary energy.
The selector is a heuristic with no formal safety or optimality guarantee.
Short CPU runs establish correctness and feasibility only. Energy savings or
research novelty require repeated energy runs, consistent sensor boundaries,
more varied prompts, and comparison to the baselines.

## Selective-feedback pilot

`Mechanics_pilot.py` is the earlier feasibility matrix (merged from the former
`prototype/selective_feedback` package); it now runs on this package's engine.
`Cache_paths.py` places the Hugging Face cache in the git-ignored repository-root
`layerskip/cache`, and `Download_weights.py` fetches the checkpoint with
SHA-256 verification. Pilot reports are written to `Results/Pilot/`.

    python -B -m unittest Layerskip_energy.Tests.test_pilot -v
    python -B -m Layerskip_energy.Mechanics_pilot --model tiny
    python -B -m Layerskip_energy.Mechanics_pilot --model smollm --prompts 2 --tokens 12 --repeats 3
    python -B -m Layerskip_energy.Download_weights

## Results layout

`Results/latest.*` is the default `Run_experiment.py` output. Earlier runs are grouped by
experiment: `smoke/`, `fp16-run/`, `fp16-100tokens/`, `fp16-200tokens/`,
`fixed-sweep/`, `engine-optimization/`, `memory-optimization/`, `benchmark/`
and `pilot/`.

`benchmark/rtx3050-fp32-energy-safe.*` is the current reference benchmark:
FP32 on an RTX 3050 with NVML energy, 8 held-out prompts x 2 repeats, 16 tokens,
133/133 requests matching native greedy. Against ordinary decoding, the adaptive
policy measured 3.48 vs 4.39 J/token (20.7% less GPU energy) and 75.5 vs
112.3 ms/token (32.8% lower latency, 1.49x throughput). This is a small
feasibility sample; repeat it with more prompts and longer outputs.

## Greedy mismatches

`Run_experiment.py` now records a greedy-output mismatch instead of stopping: the report
lists it under `mismatches`, the configuration is excluded from adaptive and
bandit selection (`excluded_configurations`), and the policy table shows a
mismatch count. Use `--stop-on-mismatch` for the old abort behaviour. To find
out whether a mismatch is an FP16 near-tie or a real divergence:

    python -m Layerskip_energy.Diagnose_mismatch --report path/to/report.json

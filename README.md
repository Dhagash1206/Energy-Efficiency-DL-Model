# Hardware-Aware Energy-Efficient LLM Inference

This research codebase studies energy-aware adaptive self-speculative decoding on consumer hardware. The main experiment uses pretrained **LayerSkip Llama 3.2 1B** to compare ordinary greedy decoding, fixed speculative settings, observation policies, and request-level adaptive selection using measured energy or latency.

The latest saved CUDA run matched native greedy output on **381/381 measured requests**. Its adaptive policy measured **35.3% lower GPU energy per token** and **42.3% lower latency per token** than ordinary decoding. These are descriptive results from one workload and device, not established general savings.

The original Qwen3 distillation work remains in `edge_llm_inference/`. The earlier compression/pruning/CALM pipeline is research background, not the implemented end-to-end pipeline.

## How the experiment works

1. Load `facebook/layerskip-llama3.2-1B` at pinned revision `81deb0f88734409cca506bcefcfb5c6bd2667565`.
2. Load pinned GSM8K and HumanEval prompt workloads with disjoint calibration and held-out IDs. Prompts are truncated to 192 tokens.
3. Calibrate ordinary decoding and speculative actions with exit depths 4, 8, and 12 and draft lengths 1 and 3. Measure observer overhead with paired executions.
4. Run seven policies on the same held-out prompts, randomizing policy order within each prompt/repeat. The adaptive selector and bandit continue learning from their own executed test requests.
5. Compare each measured output with native greedy token IDs and save raw measurements, comparisons, and summaries.

Self-speculation drafts with early model layers, verifies with the full model, and rolls back rejected cache entries. Intermediate-exit feedback is used only where it belongs to the committed prefix. The selector chooses a configuration per request; optional `--adaptive-drafts` separately changes draft lengths within a request.

| Policy | Behavior |
|---|---|
| `ordinary` | Ordinary full-model greedy decoding |
| `fixed` | Exit depth 4, draft length 3, no observers |
| `never_observe` | Exit depth 8, draft length 3, no observers |
| `selected_observe` | Exit depth 8, draft length 3, observe exit 4 |
| `always_observe` | Exit depth 8, draft length 3, observe exits 4 and 12 |
| `bandit` | Independent-arm cost bandit using executed costs |
| `adaptive` | Cost-aware heuristic selecting a decoding action and, when justified, paid observations |

Paid observations are available to the adaptive selector only for depth 8 / draft length 3, after sufficient paired calibration. The selector has no formal optimality guarantee.

## Latest CUDA energy results

Sources: [full JSON report](hardware_aware_selective_feedback/results/latest.json) and [generated Markdown report](hardware_aware_selective_feedback/results/latest.md).

| Setting | Recorded value |
|---|---|
| Hardware / OS | NVIDIA GeForce RTX 3050 Laptop GPU / Windows 11 |
| Precision / device | FP32 / `cuda:0` |
| Runtime | PyTorch `2.9.0+cu130`, Transformers `5.18.0` |
| Objective / sensor | Energy / NVML cumulative GPU energy counter (mJ converted to J) |
| Measurement boundary | NVIDIA GPU, not the whole laptop |
| Prompt split | 8 calibration and 16 held-out prompts, equally divided between GSM8K and HumanEval |
| Held-out repeats / token limit | 3 repeats / 64 maximum generated tokens |
| Observer calibration | 6 pairs per observation mode across 6 distinct calibration prompts |
| Seed / CPU threads | 42 / 4 |
| Correctness | 381/381 measured requests matched native greedy; no excluded configurations |

There are 45 calibration requests and 336 held-out requests: 48 per policy. Each policy generated 3,072 held-out tokens. The command used default FP32 and `--attn auto` (SDPA on CUDA). Quantization, compilation, adaptive draft lengths, context-specific selection, CUDA Graphs, and GPU clock/power pinning were disabled.

| Policy | Runs | Latency (ms/token) | GPU energy (J/token) | Greedy mismatches |
|---|---:|---:|---:|---:|
| `ordinary` | 48 | 105.173 | 4.129220 | 0 |
| `fixed` | 48 | 64.210 | 2.784979 | 0 |
| `never_observe` | 48 | 63.219 | 2.764236 | 0 |
| `selected_observe` | 48 | 70.298 | 3.112311 | 0 |
| `always_observe` | 48 | 72.359 | 3.214446 | 0 |
| `bandit` | 48 | 65.241 | 2.791580 | 0 |
| `adaptive` | 48 | 60.672 | 2.671720 | 0 |

Per-token values are total measured request cost divided by total generated tokens, not medians of per-request ratios. Measurements include prefill and generation; model loading, native-greedy reference generation, and selector decisions outside the measured generation call are excluded.

Compared with ordinary decoding, adaptive decoding measured 35.3% less GPU energy per token, 42.3% less time per token, and 1.73x throughput. It used depth 12 / draft length 3 for 47 held-out requests and depth 8 / draft length 3 for one, with **no paid observations during held-out adaptive requests**. This run therefore does not demonstrate that buying feedback improves efficiency.

The median paired observer overhead relative to `never_observe` was **21.680 J/request** for `selected_observe` and **28.613 J/request** for `always_observe`. The best executed fixed baseline varied across prompts. Calibration consumed **8,991.701 J** separately; that cost is not amortized into the policy table.

Hardware variability, small prompt counts, and the absence of task-quality scoring limit interpretation. HumanEval code is neither executed nor scored; greedy agreement checks output equivalence, not answer quality.

## Setup

Run commands from the repository root. Use a virtual environment and install a PyTorch build appropriate for your hardware, followed by the root dependency file:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. The Windows NVIDIA setup helper creates a Python 3.12 environment when needed and installs the CUDA 12.4 PyTorch stack:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cuda_3050.ps1
.\.venv\Scripts\Activate.ps1
python -m prototype.doctor
python -m hardware_aware_selective_feedback.run --probe --device cuda
```

**Reproducibility gap:** `requirements.txt` currently pins PyTorch `2.6.0` and Transformers `4.50.0`, whereas the latest report records `2.9.0+cu130` and `5.18.0`. Installing the dependency file or running the setup helper does not recreate that recorded environment. Preserve runtime versions with results and check correctness when changing them.

The LayerSkip checkpoint requires authorized Hugging Face access. After obtaining access, authenticate locally with `hf auth login`. A missing snapshot is downloaded into the ignored `layerskip/cache/` directory; do not commit credentials or model weights. This experiment uses an already exit-trained checkpoint and does not require local model training or llama.cpp.

## Run the experiment

The command used for the latest saved results was:

```powershell
python -m hardware_aware_selective_feedback.run `
  --device cuda `
  --objective energy `
  --calibration 4 `
  --test 8 `
  --tokens 64 `
  --repeats 3 `
  --observer-pairs 6
```

`--calibration` and `--test` count prompts **per dataset**. `--tokens` is the maximum output length; generation can stop at EOS. PowerShell uses backticks for continuation; other shells can run the command on one line.

Default output paths are `hardware_aware_selective_feedback/results/latest.json` and `latest.md`. Use `--output path/to/run.json` to preserve an existing report; the Markdown report is written beside it.

Start with a separately saved smoke run on a new environment:

```powershell
python -m hardware_aware_selective_feedback.run --device cuda --objective energy --calibration 1 --test 1 --tokens 4 --repeats 1 --observer-pairs 1 --output hardware_aware_selective_feedback/results/smoke/local-check.json
```

FP32 is the runner default. `--precision fp16` is available on CUDA to reduce memory needs, but earlier FP16 experiments recorded greedy mismatches. By default, mismatches are recorded and the affected configuration is excluded from future adaptive/bandit selection; fixed baselines may still execute it. Policies with mismatches are unsuitable for exact-output comparisons. Add `--stop-on-mismatch` to abort instead.

```powershell
python -m hardware_aware_selective_feedback.diagnose_mismatch --report hardware_aware_selective_feedback/results/latest.json
```

### Device and energy support

`--device auto` tries CUDA, Intel XPU, macOS MPS, then CPU. Device availability alone does not validate the custom decoding path.

| Device / platform | Built-in cumulative energy source |
|---|---|
| NVIDIA CUDA | NVML GPU energy, when supported |
| Linux CPU | Readable RAPL CPU-package counters |
| Windows CPU | Readable EMI V2 `_PKG` channels, when available |
| XPU or macOS CPU/MPS | No built-in per-request energy counter |

`--objective energy` requires a working counter. `--objective auto` falls back to latency when energy is unavailable, recording `energy_j: null`. An external meter can supply a continuously updated cumulative text counter through `--energy-source file --counter-file <path> --counter-unit joules --energy-scope <scope>`. Compare energy only across equivalent measurement boundaries. See the [experiment README](hardware_aware_selective_feedback/README.md) for sensor details.

### Optional experiments

| Option | Purpose / current scope |
|---|---|
| `--adaptive-drafts` | Shorten drafts below 50% recent acceptance; fall back below 30% with periodic probes |
| `--context-selector` | Use dataset and prompt-length-specific selector costs after enough observations |
| `--quantize int4` / `int8` | Optional bitsandbytes weight quantization on CUDA |
| `--compile` | Optional `torch.compile` model wrapper |
| `--attn` | Choose `auto`, `sdpa`, `eager`, or `flash_attention_2` |
| `--cuda-graph-drafts` | Prototype static-cache draft replay; CUDA only, incompatible with `--compile` and `--quantize`; observer requests use the regular path |
| `--gpu-clock-mhz`, `--gpu-power-limit-w` | Apply GPU settings during execution and restore them afterward; requires permissions and hardware support |

These options were not benchmarked by the latest command and carry no measured improvement claim here. Recheck greedy agreement after changing precision or execution settings. CUDA Graph capture/performance still needs device validation; see [efficiency implementation notes](EFFICIENCY_CHANGES.txt).

To compare a fixed grid without observers, use the paired sweep. Unlike `run.py`, its precision defaults to FP16; this example explicitly uses FP32:

```powershell
python -m hardware_aware_selective_feedback.sweep --device cuda --precision fp32 --test 4 --tokens 16 --repeats 2 --output hardware_aware_selective_feedback/results/fixed-sweep/local-fp32.json
```

The default sweep covers depths 4/8/12 and draft lengths 1/2/3/4 plus ordinary decoding. `--batch-sizes 2 4` adds batched ordinary baselines; batched speculative decoding is not implemented.

## Repository map

| Path | Purpose |
|---|---|
| `hardware_aware_selective_feedback/run.py` | Dataset loading, calibration, policy comparison, and JSON/Markdown reporting |
| `hardware_aware_selective_feedback/engine.py` | Greedy/self-speculative generation, cache rollback, adaptive drafts, batched greedy |
| `hardware_aware_selective_feedback/selector.py` | Request-level selector and independent-arm bandit |
| `hardware_aware_selective_feedback/feedback.py` | Committed-prefix observation validation |
| `hardware_aware_selective_feedback/measurement.py` | Device synchronization and cumulative energy counters |
| `hardware_aware_selective_feedback/sweep.py` | Paired fixed-configuration and batched-baseline measurements |
| `hardware_aware_selective_feedback/graph_drafter.py`, `gpu_clocks.py` | Optional graph drafting and GPU settings |
| `hardware_aware_selective_feedback/pilot.py` | Earlier decoding/feedback feasibility matrix |
| `hardware_aware_selective_feedback/vendor/` | Attributed LayerSkip inference helper and license |
| `hardware_aware_selective_feedback/results/` | Latest reports and grouped historical experiments |
| `hardware_aware_selective_feedback/tests/` | Focused correctness and measurement tests |
| `prototype/` | Dependency-free synthetic controller demo and supporting utilities |
| `edge_llm_inference/` | Earlier Qwen3 distillation and model timing scripts using local weights; currently select MPS or CPU |
| `scripts/setup_cuda_3050.ps1` | Windows CUDA environment setup |
| `requirements.txt` | Shared dependencies |

## Checks and synthetic demo

Run the unit tests without starting a full benchmark:

```powershell
python -m unittest discover -s hardware_aware_selective_feedback/tests -v
python -m unittest prototype.tests.test_prototype -v
```

For a lightweight controller demonstration requiring only Python's standard library:

```powershell
python -m prototype.demo
```

Its energy and timing are synthetic. See [prototype documentation](prototype/README.md) for its assumptions, and the [experiment documentation](hardware_aware_selective_feedback/README.md) for pilot commands and historical results.

## Attribution

The vendored LayerSkip helper is covered by [LICENSE.LayerSkip](hardware_aware_selective_feedback/vendor/LICENSE.LayerSkip) (CC BY-NC). Retain its attribution and comply with the checkpoint's access and license terms. Dataset revisions, source URLs, hashes, and licenses are recorded in the experiment JSON.

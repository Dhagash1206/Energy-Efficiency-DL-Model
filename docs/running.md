# Running the project

Run commands from the repository root. See the [main README](../README.md)
for hardware requirements, checkpoint access, measurement limits, and the
recorded runtime's differences from the pinned dependency versions.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. The Windows NVIDIA
setup helper is `scripts/setup_cuda_3050.ps1`; review the main README's runtime
version warning before using it to reproduce a saved experiment.

```powershell
python -m prototype.doctor
python -m hardware_aware_selective_feedback.run --probe --device auto
```

## Tests and formatting

```powershell
python -B -m unittest discover -s hardware_aware_selective_feedback/tests -t . -v
python -B -m unittest prototype.tests.test_prototype -v
python -m ruff format --check .
python -m ruff check .
```

Apply formatting with `python -m ruff format .`. Results and vendored code
are excluded. Formatting tools are pinned in `requirements-dev.txt`.

## Offline demonstrations

```powershell
python -m prototype.demo
python -B -m hardware_aware_selective_feedback.pilot --model tiny
```

The prototype uses synthetic costs. The tiny-model pilot checks mechanics;
it does not establish real-model efficiency. A SmolLM pilot is also available:

```powershell
python -B -m hardware_aware_selective_feedback.pilot --model smollm --prompts 2 --tokens 12 --repeats 3
```

## LayerSkip experiments

Obtain authorized Hugging Face checkpoint access, then authenticate and download:

```powershell
hf auth login
python -B -m hardware_aware_selective_feedback.download_layerskip
```

Choose an unused `--output` filename for each experiment to preserve existing
results. The main runner requires an unused explicit output path and refuses
to overwrite either companion file. These examples explicitly use FP32; FP16 runs have recorded
greedy mismatches. Energy mode requires a supported counter.

```powershell
python -m hardware_aware_selective_feedback.run --device cuda --precision fp32 --objective energy --calibration 1 --test 1 --tokens 4 --repeats 1 --observer-pairs 1 --output hardware_aware_selective_feedback/results/smoke/local-check.json
python -m hardware_aware_selective_feedback.run --device cuda --precision fp32 --objective energy --calibration 4 --test 8 --tokens 64 --repeats 3 --observer-pairs 6 --output hardware_aware_selective_feedback/results/benchmark/local-energy.json
python -m hardware_aware_selective_feedback.sweep --device cuda --precision fp32 --test 4 --tokens 16 --repeats 2 --output hardware_aware_selective_feedback/results/fixed-sweep/local-sweep.json
```

The main runner includes ordinary, fixed, `fixed_d12_k3`, never/selected/always-observe,
bandit, and adaptive policies by default. Add `--include-del` for DEL as a ninth
policy. See the [DEL guide](../hardware_aware_selective_feedback/DEL_BASELINE.md).
For idle subtraction, observer subsampling, break-even plots, and task-quality
scores, follow the [publication evaluation guide](publication.md).

```powershell
python -m hardware_aware_selective_feedback.diagnose_mismatch --report hardware_aware_selective_feedback/results/latest.json
```

## Optional experiments

| Flag | Purpose |
|---|---|
| `--adaptive-drafts` | Adjust draft lengths within a request |
| `--context-selector` | Use prompt-context-specific selector costs in the main runner |
| `--batch-sizes 2 4` | Add batched ordinary baselines in the sweep |
| `--cuda-graph-drafts` | Prototype CUDA Graph draft execution |
| `--stop-on-mismatch` | Abort on the first greedy mismatch |
| `--gpu-clock-mhz`, `--gpu-power-limit-w` | Apply GPU clock/power settings where supported and permitted |
| `--quantize int8` / `--quantize int4` | Optional CUDA weight quantization |
| `--compile` | Optional model compilation |

Check each module's `--help` for combinations and defaults. See
[efficiency implementation notes](efficiency-changes.txt) for historical details.

## Earlier Qwen distillation

The scripts in `edge_llm_inference/` use local model weights and currently
select MPS or CPU. From that directory:

```powershell
python distill.py
python test_models.py
```

The scripts expect `models/qwen3-1.7b` and `models/qwen3-0.6b` relative to
their working directory. See the source before running a long training job.

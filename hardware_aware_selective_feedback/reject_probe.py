"""Measure whether rejected drafts cost more GPU joules than committed tokens.

Example:
    python -m hardware_aware_selective_feedback.reject_probe
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from hardware_aware_selective_feedback.engine import generate
from hardware_aware_selective_feedback.measurement import Meter
from hardware_aware_selective_feedback.reject_energy import RejectRow, fit_reject

REVISION = "81deb0f88734409cca506bcefcfb5c6bd2667565"
PROMPTS = (
    "Question: Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4 into muffins. She sells the rest for $2 each. How much does she make every day?\nAnswer:",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts does it take?\nAnswer:",
)


def find_snapshot() -> Path:
    name = f"models--facebook--layerskip-llama3.2-1B/snapshots/{REVISION}"
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / name,
        Path(__file__).resolve().parents[1] / "layerskip" / "cache" / "hub" / name,
    ]
    for path in candidates:
        if (path / "model.safetensors").exists():
            return path
    raise FileNotFoundError("LayerSkip snapshot not found in the Hugging Face cache")


def counts(result: dict) -> tuple[int, int]:
    committed = len(result["tokens"])
    rejected = sum(len(cycle["draft"]) - cycle["accepted"] for cycle in result["cycles"])
    if committed < 1 or rejected < 0:
        raise RuntimeError("generation produced no committed tokens")
    return committed, rejected


def main():
    snapshot = find_snapshot()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().to(device)
    meter = Meter(device, energy_source="nvml")
    if not meter.available:
        raise SystemExit(meter.reason)
    # One prompt keeps prefill inside the intercept. Two lengths make the
    # committed coefficient identifiable. eos is empty so the limit is reached.
    prompt = tokenizer(PROMPTS[0], add_special_tokens=True).input_ids
    settings = ((0, 0), (4, 1), (4, 4), (8, 2))
    limits = (32, 64)
    for _ in range(2):
        generate(model, prompt, limit=32, depth=0, drafts=0, eos=())
    rows: list[RejectRow] = []
    details = []
    for limit in limits:
        for depth, drafts in settings:
            for repeat in range(3):
                result, reading = meter.run(
                    lambda limit=limit, depth=depth, drafts=drafts: generate(
                        model, prompt, limit=limit, depth=depth, drafts=drafts, eos=()
                    )
                )
                committed, rejected = counts(result)
                joules = reading["energy_j"]
                if joules is None:
                    raise SystemExit("NVML returned no joules")
                setting = f"n{limit}-d{depth}-k{drafts}"
                rows.append(RejectRow(committed, rejected, joules, setting))
                details.append({
                    "setting": setting,
                    "repeat": repeat,
                    "committed": committed,
                    "rejected": rejected,
                    "joules": joules,
                    "latency_s": reading["latency_s"],
                })
                print(
                    f"{setting} repeat {repeat}: committed {committed} rejected {rejected} "
                    f"joules {joules:.4f} latency {reading['latency_s']:.3f}s",
                    flush=True,
                )
    fitted = fit_reject(rows)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "energy_scope": "nvidia_gpu",
        "model": "facebook/layerskip-llama3.2-1B",
        "revision": REVISION,
        "verdict": "TRAIN" if fitted["reject_is_real"] else "STOP",
        **fitted,
        "rows": details,
    }
    output = Path("hardware_aware_selective_feedback/results/rtx4060-reject-probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['verdict']}: reject_is_real={fitted['reject_is_real']}")
    print(
        f"token-count RMSE {fitted['token_count_rmse_j']:.4f} J, "
        f"full RMSE {fitted['full_rmse_j']:.4f} J, noise {fitted['repeat_noise_j']:.4f} J"
    )
    print(f"committed {fitted['committed_j']:.4f} J/token, rejected {fitted['rejected_j']:.4f} J/token")
    print(f"wrote {output}")
    if not fitted["reject_is_real"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

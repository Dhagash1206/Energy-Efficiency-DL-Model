"""Measure GPU joules at forced exit depths and fit the energy residual.

Example:
    python -m joulerexit.probe --model HuggingFaceTB/SmolLM2-135M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from joulerexit.energy import EnergyWindow, NvmlEnergy
from joulerexit.exits import exit_blocks
from joulerexit.forward import forward_depth
from joulerexit.residual import ProbeRow, fit_profile


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_prefill(model, meter, token_row, depth: int, inner: int) -> float:
    """Joules per prompt token, averaged over `inner` fresh prefills."""
    length = token_row.shape[1]
    _sync()
    with torch.inference_mode():
        with EnergyWindow(meter) as window:
            for _ in range(inner):
                forward_depth(model, token_row, depth, None)
            _sync()
    return window.joules / (inner * length)


def measure_decode(model, meter, prompt, depth: int, new_tokens: int) -> float:
    """Joules per generated token. The prompt prefill is outside the window."""
    with torch.inference_mode():
        logits, cache = forward_depth(model, prompt, depth, None)
        _sync()
        with EnergyWindow(meter) as window:
            for _ in range(new_tokens):
                next_id = logits[:, -1:].argmax(dim=-1, keepdim=True)
                logits, cache = forward_depth(model, next_id, depth, cache)
            _sync()
    return window.joules / new_tokens


def run_probe(model, meter, exits, prefill_lengths, decode_tokens, inner_prefills, repeats, vocab):
    rows = []
    device = next(model.parameters()).device
    for length in prefill_lengths:
        token_row = torch.randint(0, vocab, (1, length), device=device)
        for depth in exits:
            for _ in range(repeats):
                joules = measure_prefill(model, meter, token_row, depth, inner_prefills)
                rows.append(ProbeRow(depth, length, "prefill", joules))
    prompt = torch.randint(0, vocab, (1, prefill_lengths[0]), device=device)
    for depth in exits:
        for _ in range(repeats):
            joules = measure_decode(model, meter, prompt, depth, decode_tokens)
            rows.append(ProbeRow(depth, decode_tokens, "decode", joules))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Probe GPU joules at forced exit depths")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--inner-prefills", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("joulerexit/results/probe.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("The probe needs a CUDA GPU. This PyTorch build cannot see one.")

    meter = NvmlEnergy(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    exits = exit_blocks(args.model, model.config.num_hidden_layers)
    vocab = model.config.vocab_size

    # One unmeasured warmup so the first timed call is not a cold start.
    warmup = torch.randint(0, vocab, (1, 16), device="cuda")
    with torch.inference_mode():
        forward_depth(model, warmup, exits[-1], None)
    _sync()

    with torch.inference_mode():
        rows = run_probe(
            model,
            meter,
            exits,
            args.prefill_lengths,
            args.decode_tokens,
            args.inner_prefills,
            args.repeats,
            vocab,
        )
    profile = fit_profile(rows)
    report = {
        "model": args.model,
        "gpu": meter.name,
        "joules_are": "per token",
        "energy_scope": "nvidia_gpu",
        "counter": "NVML cumulative GPU energy (mJ)",
        "exits": list(exits),
        "residual_is_real": profile.residual_is_real,
        "depth_only_rmse_j": profile.depth_only_rmse,
        "full_rmse_j": profile.full_rmse,
        "repeat_noise_j": profile.repeat_noise_j,
        "coefficients": {
            "intercept": profile.intercept,
            "depth_slope": profile.depth_slope,
            "log_length_slope": profile.log_length_slope,
            "decode_offset": profile.decode_offset,
            "depth_only_intercept": profile.depth_only_intercept,
            "depth_only_slope": profile.depth_only_slope,
        },
        "rows": [
            {"depth": row.depth, "seq_len": row.seq_len, "phase": row.phase, "joules": row.joules}
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    verdict = "CONTINUE" if profile.residual_is_real else "STOP"
    print(f"{verdict}: residual_is_real={profile.residual_is_real}")
    print(f"depth-only RMSE {profile.depth_only_rmse:.6f} J, full RMSE {profile.full_rmse:.6f} J")
    print(f"repeat noise {profile.repeat_noise_j:.6f} J")
    print(f"wrote {args.output}")
    if not profile.residual_is_real:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

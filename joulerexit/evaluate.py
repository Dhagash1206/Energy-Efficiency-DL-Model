"""Score GSM8K, SVAMP, and BoolQ with NVML joules per correct answer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from joulerexit.controller import fit_controller
from joulerexit.decode import all_exit_logits, generate
from joulerexit.energy import EnergyWindow, NvmlEnergy
from joulerexit.gate import ExitGates
from joulerexit.profile_io import load_profile
from joulerexit.skip_forward import _entropy
from joulerexit.stats import joules_per_correct, paired_savings
from joulerexit.tasks import load_boolq, load_gsm8k, load_svamp, normalize_number, yes_no


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(model_name: str, backbone, device):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="eager")
    if backbone is not None:
        model.load_state_dict(backbone)
    model.to(device).eval()
    return model


def load_checkpoint(path: Path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = load_model(saved["model_name"], saved["backbone"], device)
    exits = tuple(saved["exits"])
    gates = ExitGates(model.config.hidden_size, len(exits)).to(device)
    gates.load_state_dict(saved["gates"])
    gates.eval()
    return model, gates, saved


def _correct(task: str, text: str, gold: str) -> bool:
    if task == "boolq":
        return yes_no(text) == gold and gold in ("yes", "no")
    predicted = normalize_number(text)
    return predicted != "" and predicted == gold


def _tune_tau(records: list[tuple[list[float], list[int], int]]) -> float:
    best = None
    for tau in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        correct = 0
        depth = 0
        for entropies, preds, gold in records:
            chosen = len(entropies) - 1
            for index, entropy in enumerate(entropies[:-1]):
                if entropy < tau:
                    chosen = index
                    break
            correct += int(preds[chosen] == gold)
            depth += chosen
        key = (correct, -depth)
        if best is None or key > best[0]:
            best = (key, tau)
    return 1.5 if best is None else best[1]


def calibrate(model, profile, exits, rows, device):
    features = []
    labels = []
    records = []
    for row in rows:
        input_ids = torch.tensor([row.input_ids], device=device)
        label_row = row.labels
        logits = all_exit_logits(model, input_ids, exits)
        for position in range(input_ids.shape[1] - 1):
            gold = label_row[position + 1]
            if gold < 0:
                continue
            entropies = []
            preds = []
            for index, logit in enumerate(logits):
                entropies.append(float(_entropy(logit[:, position]).item()))
                preds.append(int(logit[:, position].argmax().item()))
            matches = [pred == gold for pred in preds]
            target = matches.index(True) if any(matches) else len(exits) - 1
            for index, depth in enumerate(exits[:-1]):
                residual = profile.residual(depth, position + 1, "decode")
                features.append((entropies[index], residual, float(depth)))
                labels.append(1 if index == target else 0)
            records.append((entropies, preds, gold))
    controller = fit_controller(features, labels)
    return controller, _tune_tau(records)


def _score_method(model, gates, profile, exits, rows, tokenizer, mode, tau, controller, meter, repeats, max_new):
    per_example = []
    for row in rows:
        prompt = torch.tensor([row.prompt_ids], device=next(model.parameters()).device)
        _sync()
        generate(
            model, gates, profile, prompt, exits, mode, max_new_tokens=1,
            tau=tau, controller=controller, eos_token_id=tokenizer.eos_token_id,
        )
        readings = []
        text = ""
        used = []
        for _ in range(repeats):
            _sync()
            with EnergyWindow(meter) as window:
                tokens, used = generate(
                    model, gates, profile, prompt, exits, mode, max_new_tokens=max_new,
                    tau=tau, controller=controller, eos_token_id=tokenizer.eos_token_id,
                )
                _sync()
            readings.append(window.joules)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        per_example.append(
            {
                "task": row.task,
                "gold": row.gold,
                "prediction": text,
                "correct": _correct(row.task, text, row.gold),
                "joules": float(np.median(readings)),
                "mean_exit": float(np.mean([exits[index] for index in used])) if used else None,
            }
        )
    return per_example


def _summary(rows: list[dict]) -> dict:
    joules = np.asarray([row["joules"] for row in rows], dtype=np.float64)
    correct = np.asarray([1.0 if row["correct"] else 0.0 for row in rows], dtype=np.float64)
    exits = [row["mean_exit"] for row in rows if row["mean_exit"] is not None]
    return {
        "n": len(rows),
        "accuracy": float(correct.mean()) if len(rows) else 0.0,
        "correct": int(correct.sum()),
        "joules_per_correct": joules_per_correct(joules, correct),
        "mean_joules": float(joules.mean()) if len(rows) else None,
        "mean_exit_block": float(np.mean(exits)) if exits else None,
    }


def _judge(by_task: dict, paired: dict) -> dict:
    """Success on the 4060 numbers. The 3050 rerun is still required for a journal claim."""
    reasons = []
    ok = True
    for task in ("gsm8k", "boolq"):
        residual = by_task[task].get("residual-s0", {}).get("accuracy")
        depth = by_task[task].get("depth-s0", {}).get("accuracy")
        if residual is None or depth is None:
            ok = False
            reasons.append(f"{task} is missing residual-s0 or depth-s0")
            continue
        if residual + 0.02 < depth:
            ok = False
            reasons.append(f"{task} residual accuracy {residual:.3f} is more than 0.02 below depth {depth:.3f}")
        for baseline in ("entropy-s0", "depth-s0", "controller"):
            interval = paired.get(task, {}).get(baseline)
            if interval is None or interval["low"] is None or interval["low"] <= 0:
                ok = False
                reasons.append(f"{task} joules interval versus {baseline} does not sit above zero: {interval}")
    full_svamp = by_task["svamp"]["full"]["accuracy"]
    residual_svamp = by_task["svamp"]["residual-s0"]["accuracy"]
    if full_svamp >= 0.1 and residual_svamp < 0.5 * full_svamp:
        ok = False
        reasons.append(f"SVAMP collapsed from {full_svamp:.3f} to {residual_svamp:.3f}")
    if full_svamp > 0 and residual_svamp == 0:
        ok = False
        reasons.append("SVAMP residual accuracy is zero while full depth is not")
    if not reasons:
        reasons.append("4060 bar holds. A journal claim still needs the RTX 3050 measurement.")
    return {"success_on_4060": ok, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    device = torch.device("cuda")
    profile = load_profile(args.profile)
    meter = NvmlEnergy(0)
    folder = Path(args.checkpoint_dir)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed != ""]
    primary = load_checkpoint(folder / "smollm-residual-s0.pt", device)
    model, _gates, saved = primary
    tokenizer = AutoTokenizer.from_pretrained(saved["model_name"])
    exits = tuple(saved["exits"])
    _train, val, gsm_test = load_gsm8k(tokenizer, args.max_length, 1, 16, args.limit)
    _bool_train, bool_test = load_boolq(tokenizer, args.max_length, 1, args.limit)
    svamp_test = load_svamp(tokenizer, args.max_length, args.limit)
    controller, tau = calibrate(model, profile, exits, val, device)
    print(f"controller {controller.to_dict()} tau {tau}", flush=True)

    tasks = {"gsm8k": gsm_test, "boolq": bool_test, "svamp": svamp_test}
    new_tokens = {"gsm8k": 48, "boolq": 8, "svamp": 32}
    jobs = [
        ("full", model, None, "full", args.repeats),
        ("fixed", model, None, "fixed", args.repeats),
        ("entropy-threshold", model, None, "entropy", args.repeats),
        ("controller", model, None, "controller", args.repeats),
        ("residual-s0", model, _gates, "residual", args.repeats),
    ]
    records = {}

    def score_job(name, method_model, gates, mode, repeats):
        records[name] = {}
        for task, rows in tasks.items():
            print(f"scoring {name} {task}", flush=True)
            records[name][task] = _score_method(
                method_model, gates, profile, exits, rows, tokenizer, mode, tau, controller,
                meter, repeats, new_tokens[task],
            )

    for job in jobs:
        score_job(*job)
    del model, _gates
    torch.cuda.empty_cache()

    for mode in ("residual", "depth", "entropy", "raw"):
        for seed in seeds:
            name = f"{mode}-s{seed}"
            if name == "residual-s0":
                continue
            path = folder / f"smollm-{mode}-s{seed}.pt"
            if not path.exists():
                continue
            method_model, gates, _saved = load_checkpoint(path, device)
            repeats = args.repeats if seed == seeds[0] else 1
            score_job(name, method_model, gates, mode, repeats)
            del method_model, gates
            torch.cuda.empty_cache()

    by_task = {}
    paired = {}
    for task in tasks:
        by_task[task] = {name: _summary(records[name][task]) for name in records}
        residual_rows = records["residual-s0"][task]
        residual_j = np.asarray([row["joules"] for row in residual_rows])
        residual_c = np.asarray([1.0 if row["correct"] else 0.0 for row in residual_rows])
        paired[task] = {}
        for baseline in ("entropy-s0", "depth-s0", "controller", "entropy-threshold", "full", "raw-s0"):
            if baseline not in records:
                continue
            base_rows = records[baseline][task]
            paired[task][baseline] = paired_savings(
                np.asarray([row["joules"] for row in base_rows]),
                np.asarray([1.0 if row["correct"] else 0.0 for row in base_rows]),
                residual_j,
                residual_c,
            )
    report = {
        "gpu": meter.name,
        "energy_scope": "nvidia_gpu",
        "tau": tau,
        "controller": controller.to_dict(),
        "summary": by_task,
        "paired_baseline_minus_residual": paired,
        "examples": records,
        "judge": _judge(by_task, paired),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": by_task, "judge": report["judge"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

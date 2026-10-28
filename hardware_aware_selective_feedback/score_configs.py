"""Score the trained head against greedy, the fixed device choice, and the outside selector.

Example:
    python -m hardware_aware_selective_feedback.score_configs
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from hardware_aware_selective_feedback.config_head import ConfigHead, probe_hidden
from hardware_aware_selective_feedback.engine import generate
from hardware_aware_selective_feedback.measurement import Meter
from hardware_aware_selective_feedback.reject_probe import find_snapshot
from hardware_aware_selective_feedback.selector import Action, Selector
from joulerexit.stats import joules_per_correct, paired_savings
from joulerexit.tasks import gsm8k_gold, normalize_number

PROFILE = Path("hardware_aware_selective_feedback/results/nvidia-geforce-rtx-4060-laptop-gpu-device-profile.json")
CHECKPOINT = Path("hardware_aware_selective_feedback/checkpoints/config-head.pt")
OUTPUT = Path("hardware_aware_selective_feedback/results/nvidia-geforce-rtx-4060-laptop-gpu-score.json")


def examples():
    gsm = list(load_dataset("openai/gsm8k", "main", split="test"))[16:32]
    rows = []
    for row in gsm:
        rows.append(("gsm8k", "Question: " + row["question"] + "\nAnswer:", gsm8k_gold(row["answer"]), 48))
    try:
        svamp = list(load_dataset("ChilleD/SVAMP", split="test"))[:8]
    except Exception:
        svamp = list(load_dataset("ChilleD/SVAMP", split="train"))[:8]
    for row in svamp:
        question = row.get("question") or row.get("Question") or ""
        body = row.get("body") or row.get("Body") or ""
        if body and body not in question:
            question = f"{body} {question}".strip()
        gold = normalize_number(str(row.get("answer", row.get("Answer", ""))))
        rows.append(("svamp", f"Question: {question}\nAnswer:", gold, 32))
    return rows


def run_method(model, meter, prompt, limit, depth, length, choose=None):
    result, reading = meter.run(
        lambda: generate(
            model, prompt, limit=limit, depth=depth, drafts=length, eos=(), choose_draft=choose
        )
    )
    actions = Counter((cycle["chosen_depth"], cycle["chosen_drafts"]) for cycle in result["cycles"])
    return result, reading["energy_j"], {f"{d}/{k}": count for (d, k), count in actions.items()}


def main():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    choice = profile["startup_choice"]
    snapshot = find_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().to("cuda")
    saved = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    head = ConfigHead(saved["hidden_size"], actions=tuple(tuple(item) for item in saved["actions"])).to("cuda").eval()
    head.load_state_dict(saved["state"])
    meter = Meter("cuda", energy_source="nvml")
    if not meter.available:
        raise SystemExit(meter.reason)
    selector = Selector(depths=(4, 8), lengths=(1, 2, 4), objective="energy")

    def choose_draft(pending, cache):
        hidden = probe_hidden(model, pending, cache, 4).float()
        return head.action_at(hidden)

    records = []
    for task, text, gold, limit in examples():
        prompt = tokenizer(text, add_special_tokens=True).input_ids
        greedy, greedy_j, _ = run_method(model, meter, prompt, limit, 0, 0)
        fixed, fixed_j, _ = run_method(model, meter, prompt, limit, choice["depth"], choice["length"])
        action = selector.choose(remaining_requests=1)
        selected, selected_j, _ = run_method(model, meter, prompt, limit, action.depth, action.length)
        selector.update(Action(action.depth, action.length), {"energy_j": selected_j}, selected)
        adapted, adapted_j, actions = run_method(model, meter, prompt, limit, 0, 0, choose_draft)
        decoded = {
            "greedy": tokenizer.decode(greedy["tokens"]),
            "fixed": tokenizer.decode(fixed["tokens"]),
            "selector": tokenizer.decode(selected["tokens"]),
            "head": tokenizer.decode(adapted["tokens"]),
        }
        record = {
            "task": task,
            "gold": gold,
            "joules": {"greedy": greedy_j, "fixed": fixed_j, "selector": selected_j, "head": adapted_j},
            "correct": {name: normalize_number(text) == gold and gold != "" for name, text in decoded.items()},
            "token_match": {
                "fixed": fixed["tokens"] == greedy["tokens"],
                "selector": selected["tokens"] == greedy["tokens"],
                "head": adapted["tokens"] == greedy["tokens"],
            },
            "head_actions": actions,
            "selector_action": {"depth": action.depth, "length": action.length},
        }
        records.append(record)
        print(
            f"{task} gold={gold} head_correct={record['correct']['head']} "
            f"greedy_j={greedy_j:.2f} fixed_j={fixed_j:.2f} head_j={adapted_j:.2f} "
            f"head_match={record['token_match']['head']}",
            flush=True,
        )

    def arrays(task, name):
        subset = [row for row in records if row["task"] == task]
        joules = np.asarray([row["joules"][name] for row in subset], dtype=np.float64)
        correct = np.asarray([row["correct"][name] for row in subset], dtype=np.float64)
        return joules, correct

    summary = {}
    for task in ("gsm8k", "svamp"):
        summary[task] = {}
        for name in ("greedy", "fixed", "selector", "head"):
            joules, correct = arrays(task, name)
            summary[task][name] = {
                "correct": int(correct.sum()),
                "count": int(len(correct)),
                "joules": float(joules.sum()),
                "joules_per_correct": joules_per_correct(joules, correct),
            }
        base_j, base_c = arrays(task, "greedy")
        for name in ("fixed", "selector", "head"):
            other_j, other_c = arrays(task, name)
            summary[task][f"greedy_minus_{name}"] = paired_savings(base_j, base_c, other_j, other_c)

    head_actions = Counter()
    for row in records:
        if row["task"] == "gsm8k":
            head_actions.update(row["head_actions"])
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "energy_scope": "nvidia_gpu",
        "fixed_action": {"depth": choice["depth"], "length": choice["length"]},
        "head_collapsed": len(head_actions) <= 1,
        "head_actions": dict(head_actions),
        "summary": summary,
        "selector": selector.summary(),
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"head_collapsed": report["head_collapsed"], "summary": summary}, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

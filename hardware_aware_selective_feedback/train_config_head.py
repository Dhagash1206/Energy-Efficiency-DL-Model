"""Train the draft-config head on acceptance-prefix labels.

Example:
    python -m hardware_aware_selective_feedback.train_config_head
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from hardware_aware_selective_feedback.config_head import ACTIONS, ConfigHead, action_costs, head_loss
from hardware_aware_selective_feedback.device_energy import step_winners
from hardware_aware_selective_feedback.engine import generate
from hardware_aware_selective_feedback.reject_probe import find_snapshot

PROFILE = Path("hardware_aware_selective_feedback/results/nvidia-geforce-rtx-4060-laptop-gpu-device-profile.json")
OUTPUT = Path("hardware_aware_selective_feedback/checkpoints/config-head.pt")


def load_profile() -> dict:
    report = json.loads(PROFILE.read_text(encoding="utf-8"))
    if not report["decision"]["train_head"]:
        raise SystemExit(report["decision"]["reason"])
    return report


def capture_cycles(model, prompt: list[int], depth: int, limit: int) -> tuple[list[torch.Tensor], list[dict]]:
    captured: list[torch.Tensor] = []
    layer = model.model.layers[depth - 1]

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] == 1:
            captured.append(hidden[0, -1].detach())

    handle = layer.register_forward_hook(hook)
    try:
        result = generate(model, prompt, limit=limit, depth=depth, drafts=4, eos=())
    finally:
        handle.remove()
    cursor = 0
    states = []
    for cycle in result["cycles"]:
        drafted = len(cycle["draft"])
        # k early passes, then one remainder pass over the pending token.
        # A greedy cycle is one full pass of that token.
        need = drafted + 1
        if cursor + need > len(captured):
            raise RuntimeError(
                f"layer hook captured {len(captured)} states but cycle needs index {cursor + need}"
            )
        states.append(captured[cursor])
        cursor += need
    if cursor != len(captured):
        raise RuntimeError(f"layer hook left {len(captured) - cursor} unused states")
    return states, result["cycles"]


def collect(model, tokenizer, profile: dict, prompts: int, limit: int):
    depth = 4
    rows = list(load_dataset("openai/gsm8k", "main", split="train"))[:prompts]
    hiddens = []
    labels = []
    costs = []
    counted = Counter()
    for row in rows:
        prompt = tokenizer("Question: " + row["question"] + "\nAnswer:", add_special_tokens=True).input_ids
        states, cycles = capture_cycles(model, prompt, depth, limit)
        for hidden, cycle in zip(states, cycles):
            if len(cycle["draft"]) != 4:
                continue
            accepted = int(cycle["accepted"])
            winner = step_winners(profile, depth, [accepted])[0]
            hiddens.append(hidden)
            labels.append(ACTIONS.index(winner))
            costs.append(action_costs(profile, accepted))
            counted[winner] += 1
    if len(labels) < 8:
        raise RuntimeError("not enough labeled decode steps")
    return (
        torch.stack(hiddens),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(costs, dtype=torch.float32),
        counted,
    )


def main():
    profile = load_profile()
    snapshot = find_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print("collecting acceptance labels", flush=True)
    hidden, labels, costs, counted = collect(model, tokenizer, profile, prompts=24, limit=32)
    print({f"{depth}/{length}": count for (depth, length), count in sorted(counted.items())}, flush=True)
    head = ConfigHead(hidden.shape[1]).to("cuda")
    hidden = hidden.to("cuda").float()
    labels = labels.to("cuda")
    costs = costs.to("cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    last = None
    for step in range(1, 81):
        logits = head(hidden)
        loss = head_loss(logits, labels, costs)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite head loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = float(loss)
        if step % 20 == 0:
            predicted = logits.argmax(dim=-1)
            accuracy = float((predicted == labels).float().mean())
            print(f"step {step}: loss {last:.4f} label_acc {accuracy:.3f}", flush=True)
    predicted = head(hidden).argmax(dim=-1).tolist()
    used = sorted({ACTIONS[index] for index in predicted})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": head.state_dict(),
            "actions": ACTIONS,
            "hidden_size": hidden.shape[1],
            "gpu": profile["gpu"],
            "profile_choice": profile["startup_choice"],
            "label_counts": {f"{depth}/{length}": count for (depth, length), count in counted.items()},
            "train_actions_used": [{"depth": depth, "length": length} for depth, length in used],
            "final_loss": last,
        },
        OUTPUT,
    )
    print(f"wrote {OUTPUT}; actions used in the fit {used}", flush=True)


if __name__ == "__main__":
    main()

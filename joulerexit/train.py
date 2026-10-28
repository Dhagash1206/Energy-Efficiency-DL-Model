"""Train exit gates. SmolLM updates the backbone lightly. LayerSkip does not."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from joulerexit.exits import exit_blocks
from joulerexit.gate import ExitGates
from joulerexit.profile_io import energy_scale, load_profile
from joulerexit.skip_forward import forward_exits
from joulerexit.tasks import load_boolq, load_gsm8k


def _token_accuracy(model, gates, profile, rows, exits, mode, device, lambda_energy, detach_backbone) -> float:
    correct = 0.0
    total = 0.0
    model.eval()
    gates.eval()
    with torch.no_grad():
        for row in rows:
            input_ids = torch.tensor([row.input_ids], device=device)
            labels = torch.tensor([row.labels], device=device)
            _loss, stats = forward_exits(
                model,
                gates,
                profile,
                input_ids,
                labels,
                exits,
                mode,
                lambda_energy,
                detach_backbone=detach_backbone,
            )
            weight = float((labels[:, 1:] != -100).sum())
            correct += stats["token_acc"] * weight
            total += weight
    model.train()
    gates.train()
    return correct / max(total, 1.0)


def train_one(args) -> Path:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    profile = load_profile(args.profile)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.to(device)
    exits = exit_blocks(args.model, model.config.num_hidden_layers)
    gates = ExitGates(model.config.hidden_size, len(exits)).to(device)
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    lambda_energy = 0.0 if args.mode == "entropy" else energy_scale(profile, exits)
    gsm_train, gsm_val, _gsm_test = load_gsm8k(tokenizer, args.max_length, args.train_limit, args.val_limit, 1)
    bool_train, _bool_test = load_boolq(tokenizer, args.max_length, max(1, args.train_limit // 4), 1)
    rows = gsm_train + bool_train
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    reference = _full_depth_accuracy(model, gsm_val[: args.val_limit], device)

    parameters = gates.parameters() if args.freeze_backbone else [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.backbone_lr},
        {"params": gates.parameters(), "lr": args.gate_lr},
    ]
    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(parameters, lr=args.gate_lr)
    else:
        optimizer = torch.optim.AdamW(parameters)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    model.train()
    gates.train()
    step = 0
    while step < args.steps:
        row = rows[step % len(rows)]
        input_ids = torch.tensor([row.input_ids], device=device)
        labels = torch.tensor([row.labels], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = forward_exits(
            model,
            gates,
            profile,
            input_ids,
            labels,
            exits,
            args.mode,
            lambda_energy,
            generator,
            detach_backbone=args.freeze_backbone,
        )
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            print(f"step {step + 1} skipped a non-finite loss", flush=True)
            step += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        if not args.freeze_backbone:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step} loss {stats['loss']:.3f} ce {stats['ce']:.3f} "
                f"energy {stats['energy']:.4f} acc {stats['token_acc']:.3f} "
                f"exit {stats['mean_exit_block']:.1f}",
                flush=True,
            )

    trained_acc = _token_accuracy(
        model, gates, profile, gsm_val[: args.val_limit], exits, args.mode, device, lambda_energy, args.freeze_backbone
    )
    print(f"val token acc {trained_acc:.3f} full-depth reference {reference:.3f}", flush=True)
    accuracy_bar_met = args.mode == "entropy" or trained_acc + 0.02 >= reference
    if not accuracy_bar_met:
        print(
            "Validation token accuracy is more than 0.02 below full depth. "
            "The checkpoint is kept and the result is not treated as an accuracy match.",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "mode": args.mode,
            "seed": args.seed,
            "gates": gates.state_dict(),
            "backbone": None if args.freeze_backbone else {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "lambda_energy": lambda_energy,
            "exits": exits,
            "frozen_backbone": args.freeze_backbone,
            "val_token_acc": trained_acc,
            "full_depth_token_acc": reference,
            "accuracy_bar_met": accuracy_bar_met,
        },
        output,
    )
    print(f"saved {output}", flush=True)
    return output


def _full_depth_accuracy(model, rows, device) -> float:
    correct = 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for row in rows:
            input_ids = torch.tensor([row.input_ids], device=device)
            labels = torch.tensor([row.labels], device=device)
            logits = model(input_ids=input_ids).logits
            pred = logits[:, :-1, :].argmax(dim=-1)
            gold = labels[:, 1:]
            mask = gold != -100
            correct += float((pred[mask] == gold[mask]).sum())
            total += float(mask.sum())
    return correct / max(total, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("residual", "depth", "raw", "entropy"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--val-limit", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--allow-accuracy-drop", action="store_true")
    train_one(parser.parse_args())


if __name__ == "__main__":
    main()

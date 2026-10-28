"""Score the per-round network against greedy, the locked draft, and a 100 ms hold.

Example:
    python -m hardware_aware_selective_feedback.score_pelm
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from hardware_aware_selective_feedback.engine import generate
from hardware_aware_selective_feedback.measurement import Meter
from hardware_aware_selective_feedback.pelm_round import (
    FULL_STOP,
    RoundNet,
    RoundRuntime,
    teach,
    train_network,
)
from hardware_aware_selective_feedback.reject_probe import find_snapshot
from joulerexit.tasks import gsm8k_gold, normalize_number

PROFILE = Path("hardware_aware_selective_feedback/results/nvidia-geforce-rtx-4060-laptop-gpu-device-profile.json")
CHECKPOINT = Path("hardware_aware_selective_feedback/checkpoints/pelm-round.pt")
OUTPUT = Path("hardware_aware_selective_feedback/results/rtx4060-pelm-score.json")
METHODS = ("greedy", "fixed", "short", "round", "hold")
TOKEN_CAP = 256
ANSWER_SCORE = re.compile(r"The answer is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE)
ANSWER_DONE = re.compile(r"The answer is\s+-?\d+(?:,\d{3})*(?:\.\d+)?[^\n]*\n", re.IGNORECASE)
CALCULATOR = re.compile(r"<<.*?>>", re.DOTALL)


def format_worked(question: str, answer: str) -> str:
    """One solved question. The solution ends with The answer is and the gold number."""
    body = CALCULATOR.sub("", answer).replace("####", "The answer is").strip()
    if "The answer is" not in body:
        raise ValueError("a worked example must end with The answer is")
    return f"Question: {question}\nAnswer: {body}\n"


def few_shot_prompt(demos: list[tuple[str, str]], question: str) -> str:
    """Eight train solutions, then the question. The same block is used for every method."""
    if len(demos) != 8:
        raise ValueError("the prompt uses eight worked examples")
    block = "\n".join(format_worked(item_question, item_answer) for item_question, item_answer in demos)
    return block + f"\nQuestion: {question}\nAnswer:"


def scored_answer(text: str) -> str:
    """The number after the first 'The answer is'. A later question does not count."""
    match = ANSWER_SCORE.search(text)
    if not match:
        return ""
    return normalize_number(match.group(1))


def answer_finished(text: str) -> bool:
    """True once a newline follows the line that states the answer number."""
    return ANSWER_DONE.search(text) is not None


def clip_committed(decode, before: list[int], committed: list[int]) -> list[int]:
    """Keep tokens only through the newline after the answer number."""
    kept: list[int] = []
    for token in committed:
        kept.append(token)
        window = (before + kept)[-32:]
        if answer_finished(decode(window)):
            return kept
    return committed


def train_examples() -> list[tuple[str, str]]:
    """First eight GSM8K train solutions. They are not in the scored test slice."""
    train = list(load_dataset("openai/gsm8k", "main", split="train"))[:8]
    demos = [(row["question"], row["answer"]) for row in train]
    scored = {row["question"] for row in list(load_dataset("openai/gsm8k", "main", split="test"))[16:24]}
    if any(question in scored for question, _ in demos):
        raise ValueError("a scored test question was used as an example")
    return demos


def examples():
    demos = train_examples()
    rows = []
    for row in list(load_dataset("openai/gsm8k", "main", split="test"))[16:24]:
        rows.append(("gsm8k", few_shot_prompt(demos, row["question"]), gsm8k_gold(row["answer"]), TOKEN_CAP))
    try:
        svamp = list(load_dataset("ChilleD/SVAMP", split="test"))[:4]
    except Exception:
        svamp = list(load_dataset("ChilleD/SVAMP", split="train"))[:4]
    for row in svamp:
        question = row.get("question") or row.get("Question") or ""
        body = row.get("body") or row.get("Body") or ""
        if body and body not in question:
            question = f"{body} {question}".strip()
        gold = normalize_number(str(row.get("answer", row.get("Answer", ""))))
        rows.append(("svamp", few_shot_prompt(demos, question), gold, TOKEN_CAP))
    return rows


def calibrate(model, meter, prompt, runtime: RoundRuntime) -> dict:
    """Nudge one exit with round joules measured as their own requests.

    Each measurement is one counter read before and one after. Those reads are
    not taken inside the scored answers. A delta outside a single-cycle range
    is left out so a prefill or a second cycle cannot move the fit.
    """
    _, prefill = meter.run(lambda: generate(model, prompt, limit=1, depth=0, drafts=0, eos=()))
    deltas = []
    for length in (4, 1):
        _, reading = meter.run(
            lambda length=length: generate(
                model,
                prompt,
                limit=length + 2,
                depth=4,
                drafts=length,
                eos=(),
                on_cycle=runtime.on_cycle,
            )
        )
        delta = float(reading["energy_j"] - prefill["energy_j"])
        if 0.2 <= delta <= 15.0:
            runtime.nudge_last(delta)
            deltas.append(delta)
    return {"prefill_j": float(prefill["energy_j"]), "cycle_deltas_j": deltas}


def run_method(model, meter, tokenizer, prompt, limit, depth, length, runtime=None):
    def clip(before, committed):
        return clip_committed(tokenizer.decode, before, committed)

    def halt(tokens):
        return answer_finished(tokenizer.decode(tokens[-32:]))

    result, reading = meter.run(
        lambda: generate(
            model,
            prompt,
            limit=limit,
            depth=depth,
            drafts=length,
            eos=(),
            choose_draft=None if runtime is None else runtime.choose,
            on_cycle=None if runtime is None else runtime.on_cycle,
            clip_commit=clip,
            halt=halt,
        )
    )
    actions = Counter(
        (cycle["chosen_depth"], cycle["chosen_drafts"], cycle["chosen_stop"])
        for cycle in result["cycles"]
    )
    return result, reading["energy_j"], {f"{d}/{k}/stop{s}": count for (d, k, s), count in actions.items()}


def mean_savings(base: np.ndarray, other: np.ndarray) -> float:
    return float(np.mean(base - other))


def main():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    teacher = teach(profile)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = RoundNet(tuple(tuple(item) for item in saved["actions"]))
    net.load_state_dict(saved["state"])
    net.eval()
    choice = teacher["locked"]
    noise = float(teacher["repeat_noise_j"])
    snapshot = find_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().to("cuda")
    meter = Meter("cuda", energy_source="nvml")
    if not meter.available:
        raise SystemExit(meter.reason)
    generate(model, tokenizer("2 + 2 =", add_special_tokens=True).input_ids, limit=8, depth=0, drafts=0, eos=())
    calibration_runtime = RoundRuntime(net, teacher, meter)
    calibration = calibrate(
        model, meter, tokenizer("1 + 1 =", add_special_tokens=True).input_ids, calibration_runtime
    )
    if calibration["cycle_deltas_j"]:
        original_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        original_fits = teacher["fits"]
        teacher["fits"] = calibration_runtime.fits
        _, tune = train_network(teacher, steps=30, net=net)
        calibration["tune"] = tune
        keep_tune = tune["cool_action"]["length"] > 0 and tune["label_match"] >= 0.9
        calibration["tune_kept"] = keep_tune
        if not keep_tune:
            net.load_state_dict(original_state)
            teacher["fits"] = original_fits
        net.eval()

    records = []
    for task, text, gold, limit in examples():
        prompt = tokenizer(text, add_special_tokens=True).input_ids
        greedy, greedy_j, _ = run_method(model, meter, tokenizer, prompt, limit, 0, 0)
        fixed, fixed_j, _ = run_method(model, meter, tokenizer, prompt, limit, choice["depth"], choice["length"])
        short, short_j, _ = run_method(model, meter, tokenizer, prompt, limit, 4, 1)
        round_run = RoundRuntime(net, teacher, meter)
        adapted, adapted_j, round_actions = run_method(model, meter, tokenizer, prompt, limit, 0, 0, round_run)
        hold_run = RoundRuntime(net, teacher, meter, hold_s=0.1)
        held, held_j, _ = run_method(model, meter, tokenizer, prompt, limit, 0, 0, hold_run)
        decoded = {
            "greedy": tokenizer.decode(greedy["tokens"]),
            "fixed": tokenizer.decode(fixed["tokens"]),
            "short": tokenizer.decode(short["tokens"]),
            "round": tokenizer.decode(adapted["tokens"]),
            "hold": tokenizer.decode(held["tokens"]),
        }
        outputs = {
            "greedy": greedy,
            "fixed": fixed,
            "short": short,
            "round": adapted,
            "hold": held,
        }
        record = {
            "task": task,
            "gold": gold,
            "joules": {
                "greedy": greedy_j,
                "fixed": fixed_j,
                "short": short_j,
                "round": adapted_j,
                "hold": held_j,
            },
            "tokens": {name: len(output["tokens"]) for name, output in outputs.items()},
            "answers": {name: scored_answer(text) for name, text in decoded.items()},
            "correct": {
                name: scored_answer(text) == gold and gold != ""
                for name, text in decoded.items()
            },
            "token_match": {
                name: output["tokens"] == greedy["tokens"]
                for name, output in outputs.items()
                if name != "greedy"
            },
            "round_actions": round_actions,
        }
        records.append(record)
        print(
            f"{task} gold={gold} answer={record['answers']['round']} "
            f"round_j={adapted_j:.2f} fixed_j={fixed_j:.2f} short_j={short_j:.2f} "
            f"greedy_j={greedy_j:.2f} hold_j={held_j:.2f} "
            f"correct={record['correct']['round']} round_match={record['token_match']['round']}",
            flush=True,
        )

    def column(task, name, field):
        subset = [row for row in records if row["task"] == task]
        return np.asarray([row[field][name] for row in subset], dtype=np.float64)

    summary = {}
    for task in ("gsm8k", "svamp"):
        summary[task] = {}
        for name in METHODS:
            joules = column(task, name, "joules")
            tokens = column(task, name, "tokens")
            correct = column(task, name, "correct")
            summary[task][name] = {
                "correct": int(correct.sum()),
                "count": int(len(correct)),
                "accuracy_points": float(100.0 * correct.mean()) if len(correct) else None,
                "joules": float(joules.sum()),
                "joules_per_token": float(joules.sum() / tokens.sum()) if tokens.sum() else None,
            }
        greedy_j = column(task, "greedy", "joules")
        for name in ("fixed", "short", "round", "hold"):
            summary[task][f"mean_greedy_minus_{name}_j"] = mean_savings(greedy_j, column(task, name, "joules"))

    gsm_short = column("gsm8k", "short", "joules")
    gsm_round = column("gsm8k", "round", "joules")
    beats_short = mean_savings(gsm_short, gsm_round) > noise
    per_token_lower = (
        summary["gsm8k"]["round"]["joules_per_token"] < summary["gsm8k"]["short"]["joules_per_token"]
    )
    request_higher = summary["gsm8k"]["round"]["joules"] > summary["gsm8k"]["short"]["joules"]
    kept = bool(beats_short and not (per_token_lower and request_higher))
    round_actions = Counter()
    for row in records:
        if row["task"] == "gsm8k":
            round_actions.update(row["round_actions"])
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "energy_scope": "nvidia_gpu",
        "repeat_noise_j": noise,
        "locked": choice,
        "full_stop": FULL_STOP,
        "token_cap": TOKEN_CAP,
        "worked_examples": 8,
        "shallow_check": "not_run",
        "clock_written": False,
        "energy_counter_polled_inside_request": False,
        "calibration": calibration,
        "kept": kept,
        "beats_short": beats_short,
        "short_draft": {"depth": 4, "length": 1},
        "mean_short_minus_round_j": mean_savings(gsm_short, gsm_round),
        "per_token_win_rejected": bool(per_token_lower and request_higher),
        "round_actions": dict(round_actions),
        "summary": summary,
        "records": records,
        "result": (
            "the per-round network spends fewer request joules than the fixed layer-4 length-1 draft"
            if kept
            else "the fixed layer-4 length-1 draft remains the energy result"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        key: report[key]
        for key in ("kept", "beats_short", "result", "summary")
    }, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

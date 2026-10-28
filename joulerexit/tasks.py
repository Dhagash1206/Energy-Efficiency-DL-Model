"""Task prompts. GSM8K and BoolQ are trained. SVAMP is evaluation only."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Example:
    task: str
    prompt: str
    answer: str
    gold: str
    prompt_ids: list[int]
    input_ids: list[int]
    labels: list[int]


def normalize_number(text: str) -> str:
    found = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not found:
        return ""
    value = float(found[-1])
    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return normalize_number(answer.split("####")[-1])
    return normalize_number(answer)


def yes_no(text: str) -> str:
    match = re.findall(r"\b(yes|no)\b", text.lower())
    return match[-1] if match else ""


def pack(tokenizer, prompt: str, answer: str, max_length: int) -> tuple[list[int], list[int], list[int]] | None:
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False).input_ids
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    budget = max_length - len(answer_ids) - len(eos)
    if budget < 16:
        answer_ids = answer_ids[: max(1, max_length // 4)]
        eos = eos[: max(0, max_length // 4)]
        budget = max_length - len(answer_ids) - len(eos)
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]
    input_ids = prompt_ids + answer_ids + eos
    labels = [-100] * len(prompt_ids) + answer_ids + eos
    if not any(label >= 0 for label in labels):
        return None
    return prompt_ids, input_ids, labels


def _example(task, prompt, answer, gold, tokenizer, max_length) -> Example | None:
    packed = pack(tokenizer, prompt, answer, max_length)
    if packed is None:
        return None
    prompt_ids, input_ids, labels = packed
    return Example(task, prompt, answer, gold, prompt_ids, input_ids, labels)


def load_gsm8k(tokenizer, max_length: int, train_limit: int, val_limit: int, test_limit: int):
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for row in train:
        item = _example(
            "gsm8k",
            f"Question: {row['question']}\nAnswer:",
            row["answer"].replace("####", "The answer is"),
            gsm8k_gold(row["answer"]),
            tokenizer,
            max_length,
        )
        if item is not None:
            rows.append(item)
    val = rows[-val_limit:]
    trainable = rows[:-val_limit][:train_limit]
    heldout = []
    for row in test:
        if len(heldout) >= test_limit:
            break
        item = _example(
            "gsm8k",
            f"Question: {row['question']}\nAnswer:",
            row["answer"].replace("####", "The answer is"),
            gsm8k_gold(row["answer"]),
            tokenizer,
            max_length,
        )
        if item is not None:
            heldout.append(item)
    return trainable, val, heldout


def load_boolq(tokenizer, max_length: int, train_limit: int, test_limit: int):
    from datasets import load_dataset

    try:
        train = load_dataset("google/boolq", split="train")
        val = load_dataset("google/boolq", split="validation")
    except Exception:
        train = load_dataset("boolq", split="train")
        val = load_dataset("boolq", split="validation")
    trainable = []
    for row in train:
        if len(trainable) >= train_limit:
            break
        answer = "yes" if row["answer"] else "no"
        item = _example(
            "boolq",
            f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer:",
            answer,
            answer,
            tokenizer,
            max_length,
        )
        if item is not None:
            trainable.append(item)
    heldout = []
    for row in val:
        if len(heldout) >= test_limit:
            break
        answer = "yes" if row["answer"] else "no"
        item = _example(
            "boolq",
            f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer:",
            answer,
            answer,
            tokenizer,
            max_length,
        )
        if item is not None:
            heldout.append(item)
    return trainable, heldout


def load_svamp(tokenizer, max_length: int, test_limit: int):
    from datasets import load_dataset

    try:
        data = load_dataset("ChilleD/SVAMP", split="test")
    except Exception:
        data = load_dataset("ChilleD/SVAMP", split="train")
    heldout = []
    for row in data:
        if len(heldout) >= test_limit:
            break
        question = row.get("question") or row.get("Question") or ""
        body = row.get("body") or row.get("Body") or ""
        if body and body not in question:
            question = f"{body} {question}".strip()
        raw_answer = row.get("answer", row.get("Answer", ""))
        item = _example(
            "svamp",
            f"Question: {question}\nAnswer:",
            str(raw_answer),
            normalize_number(str(raw_answer)),
            tokenizer,
            max_length,
        )
        if item is not None and item.gold:
            heldout.append(item)
    return heldout

"""Local prompt data, reproducible splits, and optional public-dataset import."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


STARTER = Path(__file__).with_name("data") / "starter.jsonl"


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_dataset(path=STARTER):
    records, ids, prompts = [], set(), set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object")
        for field in ("id", "task", "prompt", "source"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Line {line_number}: missing nonempty {field}")
        normalized = " ".join(row["prompt"].casefold().split())
        if row["id"] in ids or normalized in prompts:
            raise ValueError(f"Line {line_number}: duplicate ID or normalized prompt")
        ids.add(row["id"])
        prompts.add(normalized)
        records.append(row)
    if not records:
        raise ValueError("Dataset is empty")
    return records


def split_dataset(records, seed=42):
    """Stratified 50/25/25 split; input ordering does not affect membership."""
    groups = defaultdict(list)
    for row in records:
        groups[row["task"]].append(row)
    splits = {name: [] for name in ("calibration", "validation", "test")}
    for task, group in sorted(groups.items()):
        if len(group) < 4:
            raise ValueError(f"Task {task!r} needs at least four unique prompts")
        group = sorted(group, key=lambda row: fingerprint(f"{seed}:{row['id']}"))
        first = len(group) // 2
        second = first + max(1, len(group) // 4)
        splits["calibration"].extend(group[:first])
        splits["validation"].extend(group[first:second])
        splits["test"].extend(group[second:])
    return splits


def manifest(records, splits, seed):
    canonical = json.dumps(sorted(records, key=lambda row: row["id"]), sort_keys=True, ensure_ascii=False)
    return {"sha256": fingerprint(canonical), "seed": seed,
            "total_prompts": len(records), "tasks": dict(Counter(row["task"] for row in records)),
            "splits": {name: [row["id"] for row in group] for name, group in splits.items()},
            "quality_scoring": "Reference answers retained for inspection; task accuracy is not automatically scored."}


def export_splits(records, output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = split_dataset(records, seed)
    for name, rows in splits.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    info = manifest(records, splits, seed)
    (output / "manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def fetch_public(name, revision, limit, output):
    """Explicit network operation. Pin a dataset commit for reproducibility."""
    if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("Provide the dataset's full 40-character commit SHA as --revision")
    if limit < 12:
        raise ValueError("Fetch at least 12 rows")
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise RuntimeError("Install optional requirements-hf.txt to import public datasets") from exc
    choices = {"gsm8k": ("openai/gsm8k", "main", "reasoning"),
               "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "summarization")}
    repo, config, task = choices[name]
    stream = hf_load(repo, config, split="train", revision=revision, streaming=True)
    records, seen = [], set()
    for index, row in enumerate(stream):
        prompt = (row["question"] if name == "gsm8k" else
                  "Summarize the following article in two sentences:\n\n" + row["article"])
        normalized = " ".join(prompt.casefold().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        records.append({"id": f"{name}-{index}", "task": task, "prompt": prompt,
                        "reference": row["answer"] if name == "gsm8k" else row["highlights"],
                        "source": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
                        "source_split": "train", "source_revision": revision})
        if len(records) >= limit:
            break
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return {"rows": len(records), "path": str(path), "source": repo, "revision": revision,
            "sampling": "first unique rows from upstream train; prototype subset, not official benchmark scores"}

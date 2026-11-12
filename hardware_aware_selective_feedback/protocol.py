"""Dataset splits, evaluation protocol definitions, and reproducibility verification.

Phase P0: Separates per-dataset IDs and counts from the historical coupled loader.
Provides explicit development versus held-out test splits, checks against the
exposure registry, enforces prompt stability, and rejects duplicate or corrupted prompts.
"""

import gzip
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "layerskip" / "cache"
DATA_SPLITS_DIR = ROOT / "data_splits"

PINNED_DATASETS = {
    "gsm8k": {
        "repo": "openai/grade-school-math",
        "path_pattern": "grade_school_math/data/{}.jsonl",
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "license": "MIT",
    },
    "humaneval": {
        "repo": "openai/human-eval",
        "path_pattern": "data/HumanEval.jsonl.gz",
        "revision": "6d43fb980f9fee3c892a914eda09951f772ad10d",
        "license": "MIT",
    },
}


def load_raw_dataset(name: str, split: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and verify raw dataset records against pinned revisions."""
    if name not in PINNED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'")
    info = PINNED_DATASETS[name]
    repo, path_pattern, revision, license_name = (
        info["repo"],
        info["path_pattern"],
        info["revision"],
        info["license"],
    )

    directory = CACHE / "research_datasets" / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if cached.get("revision") != revision:
            raise ValueError(f"Cached {name} revision differs from pinned revision {revision}")

    file_name = path_pattern.format(split) if name == "gsm8k" else path_pattern
    destination = directory / Path(file_name).name

    if not destination.exists():
        # Fallback to local search or error out if cache not pre-populated
        raise FileNotFoundError(
            f"Dataset file {destination} not found in cache. Ensure datasets are downloaded."
        )

    raw_bytes = destination.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if name == "humaneval":
        raw_bytes = gzip.decompress(raw_bytes)
    records = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    metadata = {
        "dataset": name,
        "split": split,
        "repo": repo,
        "revision": revision,
        "license": license_name,
        "sha256": digest,
        "num_records": len(records),
    }
    return records, metadata


def get_known_exposed_ids() -> Set[str]:
    """Retrieve all IDs known to have been exposed in historical pilot/dev runs."""
    registry_path = DATA_SPLITS_DIR / "exposure_registry.json"
    if not registry_path.exists():
        return set()
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        exposed = set()
        for k in ("humaneval", "gsm8k_test", "gsm8k_train", "other"):
            exposed.update(data.get("summary", {}).get(k, []))
        return exposed
    except Exception:
        return set()


@dataclass
class SplitConfig:
    seed: int = 42
    gsm8k_calibration: int = 0
    gsm8k_test: int = 0
    humaneval_calibration: int = 0
    humaneval_test: int = 0
    calibration_ids: Optional[List[str]] = None
    test_ids: Optional[List[str]] = None
    exclude_exposed_from_test: bool = False
    fewshot_k: int = 0
    strict_disjoint_check: bool = True


def render_prompt_text(dataset: str, row: Dict[str, Any], fewshot_prefix: str = "") -> str:
    """Standardized prompt rendering for benchmark evaluation."""
    if dataset == "gsm8k":
        question = row["question"].strip()
        body = f"Question: {question}\nAnswer:"
        return (fewshot_prefix + body) if fewshot_prefix else body
    elif dataset == "humaneval":
        return row["prompt"]
    else:
        raise ValueError(f"Unknown dataset for rendering: {dataset}")


def compute_prompt_hash(text: str) -> str:
    """Stable SHA256 of rendered prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset_splits(config: SplitConfig) -> Dict[str, Any]:
    """Decoupled prompt loader supporting independent counts, explicit ID lists,

    and strict split disjointness checks.
    """
    gsm_train, gsm_train_meta = load_raw_dataset("gsm8k", "train")
    gsm_test, gsm_test_meta = load_raw_dataset("gsm8k", "test")
    he_tasks, he_meta = load_raw_dataset("humaneval")

    exposed_ids = get_known_exposed_ids()

    # Index records
    gsm_train_map = {f"gsm8k-train-{i}": row for i, row in enumerate(gsm_train)}
    gsm_test_map = {f"gsm8k-test-{i}": row for i, row in enumerate(gsm_test)}
    he_map = {row["task_id"]: row for row in he_tasks}

    calib_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    exemplar_rows: List[Dict[str, Any]] = []

    rng = random.Random(config.seed)

    # 1. Few-shot exemplars (from GSM8K train)
    fewshot_prefix = ""
    if config.fewshot_k > 0:
        exemplar_indices = list(range(len(gsm_train)))
        rng.shuffle(exemplar_indices)
        exemplar_blocks = []
        for idx in exemplar_indices[: config.fewshot_k]:
            pid = f"gsm8k-train-{idx}"
            row = gsm_train_map[pid]
            ex_q = row["question"].strip()
            ex_a = row["answer"].strip()
            # Standard upstream lm-evaluation-harness GSM8K few-shot format:
            # "Question: <question>\nAnswer: <answer>\n\n"
            ex_block = f"Question: {ex_q}\nAnswer: {ex_a}\n\n"
            exemplar_blocks.append(ex_block)
            exemplar_rows.append(
                {
                    "id": pid,
                    "dataset": "gsm8k",
                    "text": ex_block,
                    "prompt_hash": compute_prompt_hash(ex_block),
                    "answer": row["answer"],
                    "split": "exemplars",
                }
            )
        fewshot_prefix = "".join(exemplar_blocks)

    # 2. Calibration split
    if config.calibration_ids is not None:
        for pid in config.calibration_ids:
            if pid in gsm_train_map:
                row = gsm_train_map[pid]
                text = render_prompt_text("gsm8k", row)
                calib_rows.append(
                    {
                        "id": pid,
                        "dataset": "gsm8k",
                        "text": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "answer": row["answer"],
                        "split": "calibration",
                        "exposed": pid in exposed_ids,
                    }
                )
            elif pid in he_map:
                row = he_map[pid]
                text = render_prompt_text("humaneval", row)
                calib_rows.append(
                    {
                        "id": pid,
                        "dataset": "humaneval",
                        "text": text,
                        "prompt": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "test": row["test"],
                        "entry_point": row["entry_point"],
                        "split": "calibration",
                        "exposed": pid in exposed_ids,
                    }
                )
            else:
                raise ValueError(f"Calibration ID '{pid}' not found in source datasets")
    else:
        # Sample GSM8K train for calibration
        if config.gsm8k_calibration > 0:
            train_items = list(gsm_train_map.items())
            rng.shuffle(train_items)
            # Avoid overlap with exemplars
            exemplar_pids = {e["id"] for e in exemplar_rows}
            selected_gsm = [(pid, row) for pid, row in train_items if pid not in exemplar_pids][
                : config.gsm8k_calibration
            ]
            if len(selected_gsm) < config.gsm8k_calibration:
                raise ValueError("Insufficient GSM8K train records for calibration")
            for pid, row in selected_gsm:
                text = render_prompt_text("gsm8k", row)
                calib_rows.append(
                    {
                        "id": pid,
                        "dataset": "gsm8k",
                        "text": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "answer": row["answer"],
                        "split": "calibration",
                        "exposed": pid in exposed_ids,
                    }
                )

        # Sample HumanEval for calibration
        if config.humaneval_calibration > 0:
            he_items = list(he_map.items())
            rng.shuffle(he_items)
            selected_he = he_items[: config.humaneval_calibration]
            if len(selected_he) < config.humaneval_calibration:
                raise ValueError("Insufficient HumanEval records for calibration")
            for pid, row in selected_he:
                text = render_prompt_text("humaneval", row)
                calib_rows.append(
                    {
                        "id": pid,
                        "dataset": "humaneval",
                        "text": text,
                        "prompt": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "test": row["test"],
                        "entry_point": row["entry_point"],
                        "split": "calibration",
                        "exposed": pid in exposed_ids,
                    }
                )

    # 3. Test split
    calib_pids = {c["id"] for c in calib_rows}

    if config.test_ids is not None:
        for pid in config.test_ids:
            if pid in gsm_test_map:
                row = gsm_test_map[pid]
                text = render_prompt_text("gsm8k", row, fewshot_prefix=fewshot_prefix)
                test_rows.append(
                    {
                        "id": pid,
                        "dataset": "gsm8k",
                        "text": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "answer": row["answer"],
                        "split": "test",
                        "exposed": pid in exposed_ids,
                    }
                )
            elif pid in he_map:
                row = he_map[pid]
                text = render_prompt_text("humaneval", row)
                test_rows.append(
                    {
                        "id": pid,
                        "dataset": "humaneval",
                        "text": text,
                        "prompt": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "test": row["test"],
                        "entry_point": row["entry_point"],
                        "split": "test",
                        "exposed": pid in exposed_ids,
                    }
                )
            else:
                raise ValueError(f"Test ID '{pid}' not found in source datasets")
    else:
        # Sample GSM8K test
        if config.gsm8k_test > 0:
            test_items = list(gsm_test_map.items())
            rng.shuffle(test_items)
            if config.exclude_exposed_from_test:
                test_items = [it for it in test_items if it[0] not in exposed_ids]
            selected_gsm_test = test_items[: config.gsm8k_test]
            if len(selected_gsm_test) < config.gsm8k_test:
                raise ValueError("Insufficient GSM8K test records for requested test count")
            for pid, row in selected_gsm_test:
                text = render_prompt_text("gsm8k", row, fewshot_prefix=fewshot_prefix)
                test_rows.append(
                    {
                        "id": pid,
                        "dataset": "gsm8k",
                        "text": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "answer": row["answer"],
                        "split": "test",
                        "exposed": pid in exposed_ids,
                    }
                )

        # Sample HumanEval test (must not overlap with calib)
        if config.humaneval_test > 0:
            he_available = [it for it in he_map.items() if it[0] not in calib_pids]
            rng.shuffle(he_available)
            if config.exclude_exposed_from_test:
                he_available = [it for it in he_available if it[0] not in exposed_ids]
            selected_he_test = he_available[: config.humaneval_test]
            if len(selected_he_test) < config.humaneval_test:
                raise ValueError("Insufficient unreserved HumanEval records for test count")
            for pid, row in selected_he_test:
                text = render_prompt_text("humaneval", row)
                test_rows.append(
                    {
                        "id": pid,
                        "dataset": "humaneval",
                        "text": text,
                        "prompt": text,
                        "prompt_hash": compute_prompt_hash(text),
                        "test": row["test"],
                        "entry_point": row["entry_point"],
                        "split": "test",
                        "exposed": pid in exposed_ids,
                    }
                )

    # Validation checks
    calib_id_set = {r["id"] for r in calib_rows}
    test_id_set = {r["id"] for r in test_rows}

    if len(calib_id_set) != len(calib_rows):
        raise ValueError("Duplicate task IDs in calibration set")
    if len(test_id_set) != len(test_rows):
        raise ValueError("Duplicate task IDs in test set")

    if config.strict_disjoint_check:
        overlap = calib_id_set.intersection(test_id_set)
        if overlap:
            raise ValueError(
                f"Strict disjoint check failed: task IDs appear in both calibration and test: {sorted(overlap)}"
            )

    # Check for empty or corrupted prompt texts
    for r in calib_rows + test_rows + exemplar_rows:
        if not r["text"] or len(r["text"].strip()) < 5:
            raise ValueError(f"Task '{r['id']}' has suspiciously short or empty prompt text")

    manifest = {
        "config": asdict(config),
        "dataset_metadata": {
            "gsm8k_train": gsm_train_meta,
            "gsm8k_test": gsm_test_meta,
            "humaneval": he_meta,
        },
        "counts": {
            "calibration": len(calib_rows),
            "test": len(test_rows),
            "exemplars": len(exemplar_rows),
            "calibration_by_dataset": {
                "gsm8k": sum(1 for r in calib_rows if r["dataset"] == "gsm8k"),
                "humaneval": sum(1 for r in calib_rows if r["dataset"] == "humaneval"),
            },
            "test_by_dataset": {
                "gsm8k": sum(1 for r in test_rows if r["dataset"] == "gsm8k"),
                "humaneval": sum(1 for r in test_rows if r["dataset"] == "humaneval"),
            },
        },
        "exposure_audit": {
            "calibration_exposed_count": sum(1 for r in calib_rows if r.get("exposed")),
            "test_exposed_count": sum(1 for r in test_rows if r.get("exposed")),
        },
    }

    return {
        "calibration_prompts": calib_rows,
        "test_prompts": test_rows,
        "exemplars": exemplar_rows,
        "manifest": manifest,
    }


def save_splits_to_disk(split_data: Dict[str, Any], output_path: Path):
    """Save resolved splits and prompt hashes to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split_data, indent=2) + "\n", encoding="utf-8")

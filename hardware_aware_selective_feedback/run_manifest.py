"""Reproducibility manifest, environment provenance, and exposure registry.

Phase P0: Records exact git commit, working-tree dirty status, diff hash,
runtime/hardware environment, hashes all existing result files, and tracks
known task exposures from historical reports.
"""

import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_SPLITS_DIR = ROOT / "data_splits"


def get_environment_provenance(repo_dir=None):
    """Capture complete Git, runtime, Python, PyTorch, and hardware environment."""
    repo = Path(repo_dir or ROOT)
    env = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
        },
    }

    # Git provenance
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status_raw = subprocess.check_output(
            ["git", "status", "--porcelain", "-uall"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff_raw = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
        )
        diff_sha256 = hashlib.sha256(diff_raw).hexdigest()
        dirty = len(status_raw) > 0
        status_lines = [line.strip() for line in status_raw.splitlines() if line.strip()]

        # Collect and hash untracked files to protect untracked module contents
        untracked_files = {}
        hasher = hashlib.sha256(diff_raw)
        for line in status_lines:
            if line.startswith("??"):
                rel_path = line[2:].strip()
                full_file = repo / rel_path
                if full_file.is_file():
                    content = full_file.read_bytes()
                    f_hash = hashlib.sha256(content).hexdigest()
                    untracked_files[rel_path] = f_hash
                    hasher.update(f"{rel_path}:{f_hash}".encode("utf-8"))

        working_tree_sha256 = hasher.hexdigest()

        env["git"] = {
            "commit": commit,
            "dirty": dirty,
            "diff_sha256": diff_sha256,
            "untracked_files_count": len(untracked_files),
            "untracked_files": untracked_files,
            "working_tree_sha256": working_tree_sha256,
            "status_summary": status_lines,
        }
    except Exception as e:
        env["git"] = {"error": str(e), "commit": "unknown", "dirty": True}

    # PyTorch and CUDA
    try:
        import torch

        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            device_idx = 0
            props = torch.cuda.get_device_properties(device_idx)
            env["torch"]["gpu_device"] = {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
    except ImportError:
        env["torch"] = {"available": False}

    # Transformers
    try:
        import transformers

        env["transformers"] = {"version": transformers.__version__}
    except ImportError:
        env["transformers"] = {"available": False}

    return env


def create_results_manifest(results_dir=None, output_path=None):
    """Compute SHA256, size, and relative paths for all existing results files."""
    results_path = Path(results_dir or RESULTS_DIR)
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "root_directory": str(results_path),
        "files": {},
    }

    if results_path.exists():
        for file_path in sorted(results_path.rglob("*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(results_path).as_posix()
                content = file_path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                manifest["files"][rel_path] = {
                    "sha256": digest,
                    "size_bytes": len(content),
                    "extension": file_path.suffix.lower(),
                }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return manifest


def build_exposure_registry(results_dir=None, output_path=None):
    """Scan all historical result JSON files to extract any exposed prompt IDs."""
    results_path = Path(results_dir or RESULTS_DIR)
    registry = {
        "description": "Registry of prompt/task IDs present in historical evaluation and pilot reports",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "per_file": {},
        "summary": {
            "humaneval": [],
            "gsm8k_test": [],
            "gsm8k_train": [],
            "other": [],
        },
    }

    all_humaneval = set()
    all_gsm8k_test = set()
    all_gsm8k_train = set()
    all_other = set()

    if results_path.exists():
        manifest_file = DATA_SPLITS_DIR / "historical_results_manifest.json"
        if results_dir is None and manifest_file.exists():
            hist_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            json_files = [
                results_path / rel_p
                for rel_p in sorted(hist_manifest.get("files", {}).keys())
                if rel_p.endswith(".json") and (results_path / rel_p).exists()
            ]
        else:
            json_files = sorted(results_path.rglob("*.json"))

        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            file_ids = set()
            for row in data.get("rows", []):
                pid = row.get("prompt_id") or row.get("id")
                if pid:
                    file_ids.add(pid)
            for q in data.get("quality_tasks", []):
                qid = q.get("id") or q.get("prompt_id")
                if qid:
                    file_ids.add(qid)
            for p in data.get("calibration_prompts", []):
                if isinstance(p, dict) and "id" in p:
                    file_ids.add(p["id"])
            for p in data.get("test_prompts", []):
                if isinstance(p, dict) and "id" in p:
                    file_ids.add(p["id"])

            rel_name = json_file.relative_to(results_path).as_posix()
            registry["per_file"][rel_name] = sorted(file_ids)

            for pid in file_ids:
                if pid.startswith("HumanEval/"):
                    all_humaneval.add(pid)
                elif pid.startswith("gsm8k-test"):
                    all_gsm8k_test.add(pid)
                elif pid.startswith("gsm8k-train"):
                    all_gsm8k_train.add(pid)
                else:
                    all_other.add(pid)

    registry["summary"]["humaneval"] = sorted(all_humaneval)
    registry["summary"]["gsm8k_test"] = sorted(all_gsm8k_test)
    registry["summary"]["gsm8k_train"] = sorted(all_gsm8k_train)
    registry["summary"]["other"] = sorted(all_other)
    registry["summary"]["counts"] = {
        "humaneval": len(all_humaneval),
        "gsm8k_test": len(all_gsm8k_test),
        "gsm8k_train": len(all_gsm8k_train),
        "total_exposed": len(all_humaneval) + len(all_gsm8k_test) + len(all_gsm8k_train),
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    return registry


def assert_unique_output_path(output_path, historical_manifest=None):
    """Ensure output path does not exist and does not overwrite historical results."""
    out = Path(output_path).resolve()
    if out.exists():
        raise FileExistsError(
            f"Output file already exists at '{out}'. Historical reports must never be overwritten."
        )

    if historical_manifest:
        if isinstance(historical_manifest, (str, Path)):
            manifest_data = json.loads(Path(historical_manifest).read_text(encoding="utf-8"))
        else:
            manifest_data = historical_manifest

        # Check if basename or relative path matches any registered historical file
        for rel_path in manifest_data.get("files", {}):
            if Path(rel_path).name == out.name:
                raise ValueError(
                    f"Output filename '{out.name}' matches historical result in manifest. "
                    "Use a fresh, unique output path."
                )


def verify_environment(expected_manifest):
    """Verify that current environment matches expected manifest parameters."""
    current = get_environment_provenance()
    mismatches = []

    exp_git = expected_manifest.get("git", {})
    curr_git = current.get("git", {})
    if exp_git.get("commit") and exp_git.get("commit") != curr_git.get("commit"):
        mismatches.append(
            f"Git commit mismatch: expected {exp_git.get('commit')}, got {curr_git.get('commit')}"
        )
    if exp_git.get("working_tree_sha256") and exp_git.get("working_tree_sha256") != curr_git.get(
        "working_tree_sha256"
    ):
        mismatches.append(
            f"Git working-tree digest mismatch (tracked+untracked): expected {exp_git.get('working_tree_sha256')}, got {curr_git.get('working_tree_sha256')}"
        )
    elif exp_git.get("diff_sha256") and exp_git.get("diff_sha256") != curr_git.get("diff_sha256"):
        mismatches.append(
            f"Git dirty-tree diff mismatch: expected {exp_git.get('diff_sha256')}, got {curr_git.get('diff_sha256')}"
        )

    exp_torch = expected_manifest.get("torch", {})
    curr_torch = current.get("torch", {})
    if exp_torch.get("version") and exp_torch.get("version") != curr_torch.get("version"):
        mismatches.append(
            f"PyTorch version mismatch: expected {exp_torch.get('version')}, got {curr_torch.get('version')}"
        )

    exp_trans = expected_manifest.get("transformers", {})
    curr_trans = current.get("transformers", {})
    if exp_trans.get("version") and exp_trans.get("version") != curr_trans.get("version"):
        mismatches.append(
            f"Transformers version mismatch: expected {exp_trans.get('version')}, got {curr_trans.get('version')}"
        )

    return len(mismatches) == 0, mismatches

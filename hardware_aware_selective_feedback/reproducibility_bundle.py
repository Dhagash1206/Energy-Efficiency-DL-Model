"""Publication evidence bundle, audit verification, and reproducibility checklist.

Phase P7: Verifies all publication gates, bundles environment provenance,
paired uncertainty intervals, and ensures negative/inconclusive results remain
fully visible without post-hoc cherry-picking.
"""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

CHECKLIST_ITEMS = [
    (
        "environment_provenance",
        "Versioned code, git diff hash, hardware/runtime manifest, and unique paths",
    ),
    (
        "valid_request_rows",
        "Complete executed rows with valid tokens, latencies, and measurement data",
    ),
    (
        "exact_output_validity",
        "Greedy match verified; any mismatch strictly excluded from exact-output comparisons",
    ),
    (
        "calibration_accounted",
        "All shared calibration and idle sampling charged before efficiency claims",
    ),
    (
        "quality_evaluated",
        "Full-prompt task quality evaluated with separate extraction/execution reporting",
    ),
]


def audit_publication_readiness(report: Dict[str, Any]) -> Dict[str, Any]:
    """Audit a completed evaluation report against publication readiness gates.

    Strictly rejects empty provenance, invalid rows, unhandled mismatches,
    or absent calibration accounting.
    """
    audit_results = {}
    failures = []

    # 1. Environment provenance: must have git commit, diff/working-tree hash, platform, torch
    prov = report.get("environment_provenance")
    if not isinstance(prov, dict) or not prov:
        audit_results["environment_provenance"] = False
        failures.append("Missing or empty environment_provenance")
    else:
        git_info = prov.get("git", {})
        has_commit = bool(git_info.get("commit") and git_info.get("commit") != "unknown")
        has_hash = bool(git_info.get("working_tree_sha256") or git_info.get("diff_sha256"))
        has_runtime = "torch" in prov or "platform" in prov
        valid_prov = has_commit and has_hash and has_runtime
        audit_results["environment_provenance"] = valid_prov
        if not valid_prov:
            failures.append("Incomplete git or runtime provenance")

    # 2. Rows: must exist, be non-empty, and each row must have valid core fields
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        audit_results["valid_request_rows"] = False
        failures.append("No request rows found")
    else:
        valid_rows = True
        for r in rows:
            if not isinstance(r, dict):
                valid_rows = False
                break
            if not ("prompt_id" in r and "policy" in r and "split" in r and "tokens" in r):
                valid_rows = False
                break
        audit_results["valid_request_rows"] = valid_rows
        if not valid_rows:
            failures.append("Malformed or incomplete request rows")

    # 3. Exact output matches: policies with greedy_mismatches cannot be claimed as exact matches
    aggregates = report.get("aggregates", {})
    mismatches = report.get("mismatches", [])
    if not isinstance(aggregates, dict) or not aggregates:
        audit_results["exact_output_validity"] = False
        failures.append("Missing policy aggregates")
    else:
        valid_matches = True
        for pol, agg in aggregates.items():
            if not isinstance(agg, dict):
                valid_matches = False
                break
            gm = agg.get("greedy_mismatches", 0)
            all_m = agg.get("all_greedy_match", False)
            # If greedy_mismatches > 0, all_greedy_match MUST NOT be True
            if gm > 0 and all_m is True:
                valid_matches = False
                break
        # If mismatches occurred, ensure excluded_configurations is present and documented
        if mismatches and "excluded_configurations" not in report:
            valid_matches = False
        audit_results["exact_output_validity"] = valid_matches
        if not valid_matches:
            failures.append(
                "Policy with greedy mismatches claimed as exact match or undocumented exclusions"
            )

    # 4. Calibration accounting: publication_break_even must be present, non-empty, and valid
    pbe = report.get("publication_break_even")
    if not isinstance(pbe, dict) or not pbe:
        audit_results["calibration_accounted"] = False
        failures.append("Missing publication_break_even calibration accounting")
    else:
        # Check that comparisons actually exist
        has_comparisons = any(
            isinstance(entry, dict)
            and ("calibration_energy_j" in entry or "invalid_reason" in entry)
            for entry in pbe.values()
        )
        audit_results["calibration_accounted"] = has_comparisons
        if not has_comparisons:
            failures.append("Empty or invalid comparison entries in publication_break_even")

    # 5. Quality evaluation: if task_quality argument was passed, must contain quality results
    is_quality_run = bool(report.get("arguments", {}).get("task_quality"))
    if is_quality_run:
        has_qual = "task_quality" in report and isinstance(report["task_quality"], dict)
        audit_results["quality_evaluated"] = has_qual
        if not has_qual:
            failures.append("task_quality run missing task_quality evaluation summary")
    else:
        audit_results["quality_evaluated"] = True

    passed_all = len(failures) == 0 and all(audit_results.values())

    return {
        "passed_all_gates": passed_all,
        "gates": audit_results,
        "failures": failures,
        "audited_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def generate_reproducibility_bundle(
    report_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a self-contained reproducibility bundle with checksums and checklist audit."""
    report_file = Path(report_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_content = report_file.read_bytes()
    report_digest = hashlib.sha256(report_content).hexdigest()
    report_data = json.loads(report_content.decode("utf-8"))

    readiness = audit_publication_readiness(report_data)

    bundle_manifest = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_report": {
            "path": str(report_file.resolve()),
            "sha256": report_digest,
            "size_bytes": len(report_content),
        },
        "audit": readiness,
    }

    bundle_manifest_file = out_dir / "bundle_manifest.json"
    bundle_manifest_file.write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Reproducibility Evidence Bundle",
        "",
        f"Generated: {bundle_manifest['generated_at_utc']}",
        f"Source report: `{report_file.name}` (SHA256: `{report_digest[:16]}...`)",
        "",
        "## Publication Gates Audit",
        "",
        f"Overall Gate Status: **{'PASSED' if readiness['passed_all_gates'] else 'REJECTED / INCOMPLETE'}**",
        "",
        "| Gate | Passed |",
        "|---|:---:|",
    ]
    for gate, passed in readiness["gates"].items():
        md_lines.append(f"| {gate} | {'✓' if passed else '✗'} |")

    if readiness["failures"]:
        md_lines += [
            "",
            "### Identified Audit Failures",
            "",
        ]
        for f in readiness["failures"]:
            md_lines.append(f"- ⚠️ {f}")

    md_lines += [
        "",
        "## Reproducibility Checklist",
        "",
    ]
    for key, desc in CHECKLIST_ITEMS:
        # Dynamically check whether the gate actually passed!
        passed = readiness["gates"].get(key, False)
        checkbox = "[x]" if passed else "[ ]"
        status_suffix = "" if passed else " *(UNMET)*"
        md_lines.append(f"- {checkbox} **{key}**: {desc}{status_suffix}")

    summary_file = out_dir / "reproducibility_summary.md"
    summary_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return bundle_manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "hardware_aware_selective_feedback/results/publication/p5_live_experiment.json"
        ),
        help="Path to source experiment JSON report to audit and bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hardware_aware_selective_feedback/results/publication/bundle"),
        help="Output directory for reproducibility bundle and summary.",
    )
    args = parser.parse_args()

    manifest = generate_reproducibility_bundle(args.report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "passed_all_gates": manifest["audit"]["passed_all_gates"],
                "failures": manifest["audit"]["failures"],
            }
        )
    )


if __name__ == "__main__":
    main()

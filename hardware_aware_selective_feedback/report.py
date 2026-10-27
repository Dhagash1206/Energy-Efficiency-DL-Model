"""Descriptive mechanics report; does not make an energy feasibility decision."""
import json
import statistics
from collections import defaultdict
from pathlib import Path


def render(result):
    rows = result["rows"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["depth"], row["draft_length"], row["observation_mode"])].append(row)
    lines = ["# Mechanics pilot report", "",
             f"Model: {result['model']}",
             f"Revision: {result['model_revision']}",
             f"Precision: {result['arguments']['precision']}",
             f"Correctness: {result['status']} ({sum(r['agreement'] for r in rows)}/{len(rows)} runs agree)",
             "", "**This is a mechanics check, not evidence of research energy savings.**",
             ("LayerSkip is exit-trained. CPU energy is unavailable. Timing includes prefill," if result["exit_trained"] else "This model is not exit-trained. CPU energy is unavailable. Timing includes prefill,"),
             "decoding, observation and trace processing; paired differences are noisy.",
             "", "| Depth | Draft length | Observation | Runs | Median request seconds | Median joules |",
             "|---|---|---|---|---|---|"]
    for (depth, length, mode), items in sorted(grouped.items()):
        energy = [r["energy_j"] for r in items if r["energy_j"] is not None]
        energy_text = f"{statistics.median(energy):.6f}" if len(energy) == len(items) else "unavailable"
        lines.append(f"| {depth} | {length} | {mode} | {len(items)} | "
                     f"{statistics.median(r['latency_s'] for r in items):.6f} | {energy_text} |")
    lines += ["", "## Feedback checks", ""]
    valid = invalid = 0
    for row in rows:
        for cycle in row["cycles"]:
            for observation in cycle["observations"]:
                valid += int(observation["valid"])
                invalid += int(not observation["valid"])
    lines += [f"Valid one-step observations: {valid}.",
              f"Excluded observations: {invalid}; these are not counted as failures.",
              "", "## Next decision", "",
              "No go/no-go conclusion is justified from this smoke test.",
              "Use repeated sensor-backed measurements with an exit-trained model before",
              "deciding whether selective observation can repay its costs.",
              "The selected mode observes the first available alternative exit; it is a",
              "fixed baseline, not the proposed adaptive selector.",
              "", "## Limits", ""]
    lines += ["- " + value for value in result["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = args.report.resolve()
    if not source.is_relative_to(root):
        parser.error("Report must be inside this prototype subpackage")
    result = json.loads(source.read_text(encoding="utf-8-sig"))
    output = source.with_suffix(".md")
    output.write_text(render(result), encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()


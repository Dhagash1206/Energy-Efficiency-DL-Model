"""Controller ablation study, economic probe acquisition, and fair information controls.

Phase P6: Compares development-selected fixed baseline, offline context selector,
online executed-feedback bandit, no-probe adaptive, and economically justified
paid-probe adaptive selectors under strictly matched information budgets.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hardware_aware_selective_feedback.selector import Action


@dataclass
class DeploymentFeatures:
    """Features permitted before executing a request.

    Hidden test answers, task IDs, future lengths, and other policies' measurements
    are strictly prohibited.
    """

    prompt_length_tokens: int
    system_load_hint: Optional[float] = None


class EconomicProbeDecisionRule:
    """Acquire intermediate observer probes only when estimated future cost reduction

    exceeds the measured probe cost, decision overhead, and an uncertainty margin.
    """

    def __init__(
        self,
        probe_cost_j: float,
        decision_cost_j: float = 0.0001,
        uncertainty_margin_j: float = 0.001,
        horizon_requests: int = 10,
    ):
        self.probe_cost_j = probe_cost_j
        self.decision_cost_j = decision_cost_j
        self.uncertainty_margin_j = uncertainty_margin_j
        self.horizon_requests = horizon_requests

    def should_buy_probe(
        self,
        estimated_saving_per_request_j: float,
        current_uncertainty: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        # Total cost of purchasing the probe + deciding
        hurdle_cost = (
            self.probe_cost_j
            + self.decision_cost_j
            + (self.uncertainty_margin_j * current_uncertainty)
        )
        # Expected value over remaining horizon
        expected_future_benefit = estimated_saving_per_request_j * self.horizon_requests
        buy = expected_future_benefit > hurdle_cost

        return buy, {
            "buy": buy,
            "hurdle_cost_j": hurdle_cost,
            "expected_future_benefit_j": expected_future_benefit,
            "net_expected_gain_j": expected_future_benefit - hurdle_cost,
        }


def extract_permitted_features(prompt: Dict[str, Any], prompt_tokens: int) -> DeploymentFeatures:
    """Enforce strict information boundary: only pre-generation features are extracted."""
    return DeploymentFeatures(
        prompt_length_tokens=prompt_tokens,
        system_load_hint=None,
    )


class FrozenContextSelector:
    """Simple offline contextual policy mapping prompt length bucket to fixed action."""

    def __init__(self, short_action: Action, long_action: Action, threshold_tokens: int = 64):
        self.short_action = short_action
        self.long_action = long_action
        self.threshold_tokens = threshold_tokens

    def select(self, features: DeploymentFeatures) -> Action:
        if features.prompt_length_tokens <= self.threshold_tokens:
            return self.short_action
        return self.long_action


def run_controller_study(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Perform formal controller ablation and economic probe analysis on experiment report."""
    aggregates = experiment.get("aggregates", {})
    obs_overhead = experiment.get("observation_overhead", {})

    controllers = [
        "fixed",
        "frozen_context",
        "bandit",
        "adaptive",
        "ordinary",
        "never_observe",
        "selected_observe",
        "always_observe",
    ]
    controller_metrics = {}
    for c in controllers:
        if c in aggregates:
            controller_metrics[c] = {
                "runs": aggregates[c].get("runs", 0),
                "energy_j_per_token": aggregates[c].get("energy_j_per_token"),
                "latency_s_per_token": aggregates[c].get("latency_s_per_token"),
                "greedy_mismatches": aggregates[c].get("greedy_mismatches", 0),
            }

    selected_delta = obs_overhead.get("selected_observe", {}).get("median_delta_per_request", 0.0)
    rule = EconomicProbeDecisionRule(probe_cost_j=max(0.0, selected_delta))

    e_fixed = aggregates.get("fixed", {}).get("energy_j_per_token")
    e_adapt = aggregates.get("adaptive", {}).get("energy_j_per_token")
    adaptive_beats_fixed = False
    if e_fixed is not None and e_adapt is not None:
        adaptive_beats_fixed = e_adapt < e_fixed

    estimated_saving = max(0.0, (e_fixed - e_adapt)) if (e_fixed and e_adapt) else 0.0
    buy_probe, probe_diag = rule.should_buy_probe(
        estimated_saving_per_request_j=estimated_saving,
        current_uncertainty=1.0,
    )

    study_result = {
        "controller_metrics": controller_metrics,
        "adaptive_beats_fixed": adaptive_beats_fixed,
        "probe_economics": {
            "probe_cost_j": selected_delta,
            "decision": probe_diag,
            "economic_justification": buy_probe,
        },
        "recommendation": (
            "Adaptive selection demonstrates lower energy than the fixed baseline."
            if adaptive_beats_fixed
            else "Retain the development-selected fixed baseline as deployment choice; adaptive selector overhead exceeds its benefit."
        ),
    }
    return study_result


def markdown_p6_report(study: Dict[str, Any]) -> str:
    """Generate Markdown report summarizing Phase P6 controller study."""
    lines = [
        "# Phase P6: Controller Ablation Study & Economic Probe Analysis",
        "",
        f"- **Adaptive Beats Fixed Baseline:** `{study['adaptive_beats_fixed']}`",
        f"- **Deployment Recommendation:** {study['recommendation']}",
        f"- **Probe Purchase Justified Economically:** `{study['probe_economics']['economic_justification']}`",
        "",
        "## Controller Comparisons",
        "",
        "| Controller | Runs | Energy (J/token) | Latency (s/token) | Greedy Mismatches |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in study["controller_metrics"].items():
        e_str = (
            f"{m['energy_j_per_token']:.6f}" if m.get("energy_j_per_token") is not None else "N/A"
        )
        l_str = (
            f"{m['latency_s_per_token']:.6f}" if m.get("latency_s_per_token") is not None else "N/A"
        )
        lines.append(f"| `{name}` | {m['runs']} | {e_str} | {l_str} | {m['greedy_mismatches']} |")

    lines += [
        "",
        "## Economic Probe Decision Rule",
        "",
        f"- **Measured Probe Hurdle Cost:** {study['probe_economics']['decision']['hurdle_cost_j']:.4f} J",
        f"- **Expected Future Benefit:** {study['probe_economics']['decision']['expected_future_benefit_j']:.4f} J",
        f"- **Net Expected Gain:** {study['probe_economics']['decision']['net_expected_gain_j']:.4f} J",
        f"- **Economic Decision:** `{'BUY PROBES' if study['probe_economics']['economic_justification'] else 'REJECT PROBES'}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    from hardware_aware_selective_feedback import run_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-file",
        type=Path,
        default=Path(
            "hardware_aware_selective_feedback/results/publication/p5_live_experiment.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "hardware_aware_selective_feedback/results/publication/p6_controller_study.json"
        ),
    )
    args = parser.parse_args()

    run_manifest.assert_unique_output_path(args.output)

    if not args.experiment_file.exists():
        parser.error(f"Experiment file {args.experiment_file} does not exist")

    exp_data = json.loads(args.experiment_file.read_text(encoding="utf-8"))
    study_result = run_controller_study(exp_data)
    study_result["environment_provenance"] = run_manifest.get_environment_provenance()
    study_result["arguments"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(study_result, indent=2) + "\n", encoding="utf-8")
    md = markdown_p6_report(study_result)
    args.output.with_suffix(".md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "adaptive_beats_fixed": study_result["adaptive_beats_fixed"],
            }
        )
    )


if __name__ == "__main__":
    main()

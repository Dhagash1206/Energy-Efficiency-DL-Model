"""Request-level cost accounting, cumulative online trajectories, and setup cost allocation.

Phase P5: Measures both generation-only and end-to-end request boundaries
(including policy selection, paid probes, and online updates). Computes
observed cumulative trajectories with fair setup cost allocation and handles
crossing edge cases cleanly.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RequestCostEntry:
    request_index: int
    prompt_id: str
    policy: str
    repeat: int
    generated_tokens: int
    generation_energy_j: Optional[float]
    overhead_energy_j: float  # selection + update + paid probes
    total_request_energy_j: Optional[float]
    generation_latency_s: float
    overhead_latency_s: float
    total_request_latency_s: float
    greedy_match: bool


class CostLedger:
    """Detailed stream ledger tracking all request energy, latencies, and overheads."""

    def __init__(self):
        self.entries: List[RequestCostEntry] = []

    def record(
        self,
        request_index: int,
        prompt_id: str,
        policy: str,
        repeat: int,
        generated_tokens: int,
        generation_energy_j: Optional[float],
        overhead_energy_j: float,
        generation_latency_s: float,
        overhead_latency_s: float,
        greedy_match: bool = True,
    ):
        total_e = (
            generation_energy_j + overhead_energy_j if generation_energy_j is not None else None
        )
        total_lat = generation_latency_s + overhead_latency_s
        entry = RequestCostEntry(
            request_index=request_index,
            prompt_id=prompt_id,
            policy=policy,
            repeat=repeat,
            generated_tokens=generated_tokens,
            generation_energy_j=generation_energy_j,
            overhead_energy_j=overhead_energy_j,
            total_request_energy_j=total_e,
            generation_latency_s=generation_latency_s,
            overhead_latency_s=overhead_latency_s,
            total_request_latency_s=total_lat,
            greedy_match=greedy_match,
        )
        self.entries.append(entry)

    def reconcile(self) -> Dict[str, Any]:
        """Verify that all totals reconcile and check for sensor faults or missing values."""
        valid_energy_count = sum(1 for e in self.entries if e.total_request_energy_j is not None)
        missing_energy_count = len(self.entries) - valid_energy_count

        total_gen_e = sum(e.generation_energy_j or 0.0 for e in self.entries)
        total_over_e = sum(e.overhead_energy_j for e in self.entries)
        total_req_e = sum(e.total_request_energy_j or 0.0 for e in self.entries)

        discrepancy = abs(total_req_e - (total_gen_e + total_over_e))

        return {
            "total_entries": len(self.entries),
            "valid_energy_entries": valid_energy_count,
            "missing_energy_entries": missing_energy_count,
            "total_generation_energy_j": total_gen_e,
            "total_overhead_energy_j": total_over_e,
            "total_request_energy_j": total_req_e,
            "reconciliation_discrepancy_j": discrepancy,
            "is_reconciled": discrepancy < 1e-6,
        }

    def compute_cumulative_trajectory(
        self,
        candidate_policy: str,
        baseline_policy: str,
        setup_cost_candidate_j: float = 0.0,
        setup_cost_baseline_j: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute the empirical cumulative energy trajectory for candidate vs baseline

        over strictly matched requests.
        """
        cand_entries = [e for e in self.entries if e.policy == candidate_policy]
        base_entries = [e for e in self.entries if e.policy == baseline_policy]

        # Map by (prompt_id, repeat)
        cand_map = {(e.prompt_id, e.repeat): e for e in cand_entries}
        base_map = {(e.prompt_id, e.repeat): e for e in base_entries}

        common_keys = [k for k in cand_map if k in base_map]
        if not common_keys:
            raise ValueError(
                f"No matched requests found between {candidate_policy} and {baseline_policy}"
            )

        cand_cum = setup_cost_candidate_j
        base_cum = setup_cost_baseline_j

        observed_curve: List[Dict[str, Any]] = []
        observed_crossing_step = None

        cand_costs = []
        base_costs = []

        for step, key in enumerate(common_keys, start=1):
            c_e = cand_map[key].total_request_energy_j
            b_e = base_map[key].total_request_energy_j

            if c_e is None or b_e is None:
                continue

            cand_costs.append(c_e)
            base_costs.append(b_e)

            cand_cum += c_e
            base_cum += b_e

            if (
                observed_crossing_step is None
                and cand_cum <= base_cum
                and setup_cost_candidate_j > setup_cost_baseline_j
            ):
                observed_crossing_step = step

            observed_curve.append(
                {
                    "step": step,
                    "prompt_id": key[0],
                    "candidate_cumulative_j": cand_cum,
                    "baseline_cumulative_j": base_cum,
                    "cumulative_saving_j": base_cum - cand_cum,
                }
            )

        # Projected stationary crossing:
        # N_projected = ceil((C_a - C_b) / (mean(E_b) - mean(E_a)))
        mean_cand = sum(cand_costs) / len(cand_costs) if cand_costs else 0.0
        mean_base = sum(base_costs) / len(base_costs) if base_costs else 0.0
        mean_saving = mean_base - mean_cand
        setup_diff = setup_cost_candidate_j - setup_cost_baseline_j

        if mean_saving > 0:
            if setup_diff <= 0:
                n_projected = 0
                no_crossing_reason = None
            else:
                n_projected = math.ceil(setup_diff / mean_saving)
                no_crossing_reason = None
        else:
            n_projected = None
            no_crossing_reason = (
                "Candidate mean request energy is not lower than baseline; break-even is unbounded"
            )

        return {
            "candidate_policy": candidate_policy,
            "baseline_policy": baseline_policy,
            "matched_requests": len(observed_curve),
            "setup_cost_candidate_j": setup_cost_candidate_j,
            "setup_cost_baseline_j": setup_cost_baseline_j,
            "mean_candidate_request_energy_j": mean_cand,
            "mean_baseline_request_energy_j": mean_base,
            "mean_saving_per_request_j": mean_saving,
            "observed_crossing_step": observed_crossing_step,
            "projected_stationary_crossing": n_projected,
            "no_crossing_reason": no_crossing_reason,
            "trajectory": observed_curve,
        }

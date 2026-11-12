"""Generate high-resolution publication figures for the research paper draft."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)

FIG_DIR = Path("hardware_aware_selective_feedback/results/publication/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
p4 = json.load(
    open(
        "hardware_aware_selective_feedback/results/publication/p4_baseline_sweep.json",
        encoding="utf-8",
    )
)
p5 = json.load(
    open(
        "hardware_aware_selective_feedback/results/publication/p5_live_experiment.json",
        encoding="utf-8",
    )
)
p6 = json.load(
    open(
        "hardware_aware_selective_feedback/results/publication/p6_controller_study.json",
        encoding="utf-8",
    )
)
p7 = json.load(
    open(
        "hardware_aware_selective_feedback/results/publication/p7_paired_analysis.json",
        encoding="utf-8",
    )
)


def generate_figure1_pareto():
    """Figure 1: Energy vs. Latency Pareto Frontier."""
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    # 1. P4 fixed sweep candidates
    records = p4["sweep_records"]
    for r in records:
        d = r["depth"]
        k = r["draft_length"]
        lat = r["latency_s_per_token"] * 1000  # ms
        nrg = r["energy_j_per_token"]
        if d == 0:
            continue  # plot ordinary separately
        ax.scatter(lat, nrg, color="#4A90E2", alpha=0.6, s=60, edgecolors="none")
        ax.annotate(
            f"d{d}/k{k}",
            (lat, nrg),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=8,
            color="#555",
        )

    # 2. Main evaluated policies from P5
    agg = p5["aggregates"]
    policy_styles = {
        "ordinary": ("#D0021B", "s", "Ordinary Greedy (Base)", 100),
        "fixed": ("#2E7D32", "*", "Fixed Baseline (d4/k1, Frozen Winner)", 150),
        "fixed_d12_k3": ("#8E44AD", "D", "Fixed Anchor (d12/k3)", 90),
        "adaptive": ("#E67E22", "o", "Adaptive Feedback Selector", 120),
        "bandit": ("#F39C12", "^", "Contextual Cost Bandit", 90),
        "frozen_context": ("#16A085", "v", "Frozen Context Selector", 90),
        "del": ("#7F8C8D", "X", "DEL Shadow Head Port", 90),
    }

    for pol, (color, marker, label, size) in policy_styles.items():
        if pol in agg:
            lat = agg[pol]["latency_s_per_token"] * 1000
            nrg = agg[pol]["energy_j_per_token"]
            ax.scatter(
                lat,
                nrg,
                color=color,
                marker=marker,
                s=size,
                label=label,
                zorder=5,
                edgecolors="black",
                linewidths=0.8,
            )

    ax.set_xlabel("Latency per Generated Token (ms / token)")
    ax.set_ylabel("Measured GPU Board Energy (J / token)")
    ax.set_title("Energy–Latency Optimization Landscape (NVIDIA RTX 3050 Laptop GPU, FP16)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_pareto_energy_latency.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
    print("Saved:", out_path)


def generate_figure2_cumulative_trajectories():
    """Figure 2: Empirical cumulative energy trajectories & break-even dynamics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Panel A: Adaptive vs Ordinary
    traj_ord = p5["cumulative_trajectories"]["adaptive_vs_ordinary"]["trajectory"]
    steps = [p["step"] for p in traj_ord]
    cand_cum_ord = [p["candidate_cumulative_j"] for p in traj_ord]
    base_cum_ord = [p["baseline_cumulative_j"] for p in traj_ord]

    # Extrapolate beyond observed requests to show stationary break-even projection
    N_proj = 150
    step_ext = np.arange(1, N_proj + 1)
    mean_cand = p5["aggregates"]["adaptive"]["energy_j_per_token"] * 32
    mean_ord = p5["aggregates"]["ordinary"]["energy_j_per_token"] * 32
    setup_cost = 10.97  # J (shared calibration overhead)

    cand_ext = setup_cost + step_ext * mean_cand
    base_ext = step_ext * mean_ord

    ax1.plot(
        step_ext, base_ext, color="#D0021B", linestyle="--", label="Ordinary Decoding (No Setup)"
    )
    ax1.plot(step_ext, cand_ext, color="#E67E22", label="Adaptive (With Setup C_a = 10.97 J)")
    ax1.axvline(x=120, color="#27AE60", linestyle=":", label="Break-Even Crossing (N* = 120)")
    ax1.scatter(steps, cand_cum_ord, color="#E67E22", s=25, zorder=4)
    ax1.scatter(steps, base_cum_ord, color="#D0021B", s=25, zorder=4)

    ax1.set_xlabel("Deployment Request Count (N)")
    ax1.set_ylabel("Cumulative GPU Energy (Joules)")
    ax1.set_title("(a) Adaptive vs. Ordinary Decoding Baseline")
    ax1.legend(loc="upper left")

    # Panel B: Adaptive vs Fixed Baseline (d4/k1)
    traj_fix = p5["cumulative_trajectories"]["adaptive_vs_fixed"]["trajectory"]
    steps_fix = [p["step"] for p in traj_fix]
    cand_cum_fix = [p["candidate_cumulative_j"] for p in traj_fix]
    base_cum_fix = [p["baseline_cumulative_j"] for p in traj_fix]

    mean_fix = p5["aggregates"]["fixed"]["energy_j_per_token"] * 32
    cand_ext_fix = setup_cost + step_ext * mean_cand
    base_ext_fix = step_ext * mean_fix

    ax2.plot(
        step_ext,
        base_ext_fix,
        color="#2E7D32",
        linestyle="-",
        label="Frozen Fixed Baseline (d4/k1)",
    )
    ax2.plot(
        step_ext, cand_ext_fix, color="#E67E22", linestyle="--", label="Adaptive Feedback Selector"
    )
    ax2.scatter(steps_fix, cand_cum_fix, color="#E67E22", s=25, zorder=4)
    ax2.scatter(steps_fix, base_cum_fix, color="#2E7D32", s=25, zorder=4)

    ax2.text(
        75,
        4500,
        "No Crossing Supported\n(Fixed is strictly lower rate:\n1.201 J/tok < 1.235 J/tok)",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", edgecolor="#E57373"),
        fontsize=9,
        color="#C62828",
    )

    ax2.set_xlabel("Deployment Request Count (N)")
    ax2.set_ylabel("Cumulative GPU Energy (Joules)")
    ax2.set_title("(b) Adaptive vs. Fair Frozen Fixed Baseline")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_cumulative_trajectories.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
    print("Saved:", out_path)


def generate_figure3_bootstrap_ci():
    """Figure 3: Clustered bootstrap paired energy differences (95% CIs)."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)

    comparisons = [
        ("fixed_vs_ordinary", "Fixed (d4/k1) vs. Ordinary", "#2E7D32"),
        ("adaptive_vs_ordinary", "Adaptive vs. Ordinary", "#E67E22"),
        ("adaptive_vs_fixed", "Adaptive vs. Fixed (d4/k1)", "#4A90E2"),
    ]

    y_pos = np.arange(len(comparisons))
    labels = []

    for i, (key, label, color) in enumerate(comparisons):
        comp = p7["comparisons"][key]
        mean_diff = comp["mean_difference"]
        ci_low = comp["ci_lower"]
        ci_high = comp["ci_upper"]
        ax.errorbar(
            mean_diff,
            i,
            xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.5,
            capsize=6,
            capthick=2,
            markersize=8,
        )
        labels.append(label)

    ax.axvline(x=0, color="#777", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Top-down order

    ax.set_xlabel(
        "Paired Energy Difference per Request (Joules, Baseline − Candidate)\n[> 0 indicates candidate is more energy efficient]"
    )
    ax.set_title(
        "Prompt-Clustered Bootstrap Paired Differences (95% Confidence Intervals, 2,000 resamples)"
    )

    # Annotation of significance
    ax.text(
        9.8,
        0 - 0.2,
        "+9.80 J [5.95, 14.89] (p < 0.05, Significant)",
        color="#2E7D32",
        fontweight="bold",
        fontsize=9,
    )
    ax.text(
        8.7,
        1 - 0.2,
        "+8.74 J [5.41, 12.07] (p < 0.05, Significant)",
        color="#E67E22",
        fontweight="bold",
        fontsize=9,
    )
    ax.text(
        -1.06,
        2 - 0.2,
        "-1.06 J [-5.27, +3.14] (Crosses zero, Not Significant)",
        color="#C62828",
        fontweight="bold",
        fontsize=9,
    )

    plt.tight_layout()
    out_path = FIG_DIR / "fig3_bootstrap_ci.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
    print("Saved:", out_path)


def generate_figure4_controller_comparison():
    """Figure 4: Energy per token across all controllers and baselines."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

    policies = [
        ("fixed", "Fixed (d4/k1)\n[Winner]", "#2E7D32"),
        ("adaptive", "Adaptive\nSelector", "#E67E22"),
        ("frozen_context", "Context\nSelector", "#16A085"),
        ("bandit", "Online\nBandit", "#F39C12"),
        ("fixed_d12_k3", "Fixed\n(d12/k3)", "#8E44AD"),
        ("ordinary", "Ordinary\nGreedy", "#D0021B"),
        ("never_observe", "d8/k3\n(No probe)", "#34495E"),
        ("del", "DEL Shadow\nPaper Port", "#7F8C8D"),
        ("selected_observe", "d8/k3 (Probe 4)\n[Dense]", "#C0392B"),
        ("always_observe", "d8/k3 (Probe 4,12)\n[Dense]", "#962D22"),
    ]

    names = [p[1] for p in policies]
    values = [p5["aggregates"][p[0]]["energy_j_per_token"] for p in policies]
    colors = [p[2] for p in policies]

    bars = ax.bar(
        range(len(names)), values, color=colors, edgecolor="black", linewidth=0.7, width=0.65
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("GPU Board Energy (Joules / Generated Token)")
    ax.set_title("Controller & Policy Energy Comparison (Held-Out Test Prompts)")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    out_path = FIG_DIR / "fig4_controller_comparison.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    generate_figure1_pareto()
    generate_figure2_cumulative_trajectories()
    generate_figure3_bootstrap_ci()
    generate_figure4_controller_comparison()
    print("All figures successfully generated in", FIG_DIR)

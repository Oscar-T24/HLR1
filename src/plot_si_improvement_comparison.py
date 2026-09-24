"""Plot clear SI-improvement comparisons for RL, PSO, and RL-init PSO."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .config import CHECKPOINT_DIR, ENV_CONFIG, FIGURES_DIR, HFSS_FILE, SAC_CONFIG

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .sac_agent import SACAgent
from .plot_pso_convergence_theta_disabled import sample_theta_disabled_case
from .test_3d_beam_map import (
    evaluate_pso_seed_case,
    evaluate_rl_initialized_pso_seed_case,
    make_env,
    rollout_best_focus,
    summarize_results,
)
from .utils import load_hfss_channel


METHODS = ("RL", "RL-init PSO", "PSO")
SI_VALUE_METHODS = ("Ref SI", "RL", "RL-init PSO", "PSO")
COLORS = {
    "Ref SI": "tab:gray",
    "RL": "tab:blue",
    "RL-init PSO": "tab:green",
    "PSO": "tab:orange",
}

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
}

POSTER_GRID_STYLE = {"color": "#8a7f85", "linewidth": 0.9, "alpha": 0.42}


def load_agent():
    agent_config = SAC_CONFIG.copy()
    agent_config["chkpt_dir"] = str(CHECKPOINT_DIR)
    agent = SACAgent(**agent_config)
    agent.load_models()
    return agent


def evaluate_modes(seed_start, num_seeds):
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    agent = load_agent()

    rl_results = []
    pso_results = []
    rlpso_results = []
    for seed in range(seed_start, seed_start + num_seeds):
        env = make_env(hsi, seed)
        case = sample_theta_disabled_case(env, np.random.default_rng(seed), seed)
        start = time.perf_counter()
        si_ref_db, best_si_db, best_info, final_info = rollout_best_focus(env, agent, case)
        rl_result = {
            "seed": int(seed),
            "case": case,
            "si_ref_db": float(si_ref_db),
            "best_si_db": float(best_si_db),
            "si_improvement_db": float(si_ref_db - best_si_db),
            "dl_gain_loss_db": float(best_info["DL_gain_loss_dB"]),
            "ul_gain_loss_db": float(best_info["UL_gain_loss_dB"]),
            "selection_mode": best_info.get("selection_mode", "best SI"),
            "gain_violation_score": float(best_info.get("gain_violation_score", 0.0)),
            "runtime_s": float(time.perf_counter() - start),
            "best_info": best_info,
            "final_info": final_info,
        }
        pso_result = evaluate_pso_seed_case(env, rl_result["case"])
        rlpso_result = evaluate_rl_initialized_pso_seed_case(env, rl_result)
        rl_results.append(rl_result)
        pso_results.append(pso_result)
        rlpso_results.append(rlpso_result)
        print(
            f"seed {seed} | "
            f"RL {rl_result['si_improvement_db']:.2f} dB | "
            f"RL-init PSO {rlpso_result['si_improvement_db']:.2f} dB | "
            f"PSO {pso_result['si_improvement_db']:.2f} dB"
        )

    return selected_freq, {
        "RL": rl_results,
        "RL-init PSO": rlpso_results,
        "PSO": pso_results,
    }


def improvements_by_method(results_by_method):
    return {
        method: np.asarray([item["si_improvement_db"] for item in results_by_method[method]], dtype=np.float64)
        for method in METHODS
    }


def si_values_by_method(results_by_method):
    return {
        "Ref SI": np.asarray([item["si_ref_db"] for item in results_by_method["RL"]], dtype=np.float64),
        "RL": np.asarray([item["best_si_db"] for item in results_by_method["RL"]], dtype=np.float64),
        "RL-init PSO": np.asarray([item["best_si_db"] for item in results_by_method["RL-init PSO"]], dtype=np.float64),
        "PSO": np.asarray([item["best_si_db"] for item in results_by_method["PSO"]], dtype=np.float64),
    }


def plot_box_with_points(si_values, output_path, seed_start, num_seeds):
    os.makedirs(output_path.parent, exist_ok=True)
    rng = np.random.default_rng(12345)
    data = [si_values[method] for method in SI_VALUE_METHODS]
    positions = np.arange(1, len(SI_VALUE_METHODS) + 1)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.55))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.46,
            patch_artist=True,
            showmeans=True,
            medianprops={"color": "black", "linewidth": 1.45},
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.5},
            whiskerprops={"linewidth": 1.1, "color": "0.25"},
            capprops={"linewidth": 1.1, "color": "0.25"},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "0.35",
                "markersize": 2.6,
                "alpha": 0.75,
            },
        )
        for patch, method in zip(box["boxes"], SI_VALUE_METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.23)
            patch.set_edgecolor(COLORS[method])
            patch.set_linewidth(1.35)

        for pos, method in zip(positions, SI_VALUE_METHODS):
            values = si_values[method]
            jitter = rng.uniform(-0.10, 0.10, size=values.size)
            ax.scatter(
                np.full(values.size, pos) + jitter,
                values,
                s=11,
                color=COLORS[method],
                alpha=0.52,
                edgecolors="none",
            )

        all_values = np.concatenate(data)
        y_min = np.floor(np.nanmin(all_values) / 10.0) * 10.0
        y_max = np.ceil(np.nanmax(all_values) / 10.0) * 10.0
        ax.set_xticks(positions)
        ax.set_xticklabels(SI_VALUE_METHODS)
        ax.set_ylabel("Beam-level SI [dB]")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", **POSTER_GRID_STYLE)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, transparent=True)
        white_output_path = output_path.with_name(f"{output_path.stem}_white_background{output_path.suffix}")
        fig.patch.set_alpha(1.0)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fig.savefig(white_output_path, dpi=300, transparent=False, facecolor="white", edgecolor="white")
        plt.close(fig)


def plot_sorted_curves(improvements, output_path, seed_start, num_seeds):
    os.makedirs(output_path.parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ranks = np.arange(1, num_seeds + 1)
    for method in METHODS:
        values = np.sort(improvements[method])
        ax.plot(ranks, values, linewidth=2.4, color=COLORS[method], label=method)

    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_xlabel("Seed rank, sorted by SI improvement")
    ax.set_ylabel("SI improvement [dB]")
    ax.set_title(f"Sorted SI Improvement Curves, seeds {seed_start}-{seed_start + num_seeds - 1}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ref_si_by_sorted_improvement(results_by_method, improvements, output_path, seed_start, num_seeds):
    os.makedirs(output_path.parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ranks = np.arange(1, num_seeds + 1)
    for method in METHODS:
        order = np.argsort(improvements[method])
        ref_si = np.asarray([item["si_ref_db"] for item in results_by_method[method]], dtype=np.float64)
        ax.plot(ranks, ref_si[order], linewidth=2.4, color=COLORS[method], label=method)

    ax.set_xlabel("Seed rank, sorted by each method's SI improvement")
    ax.set_ylabel("Ref SI [dB]")
    ax.set_title(f"Reference SI in Improvement-Sorted Order, seeds {seed_start}-{seed_start + num_seeds - 1}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(results_by_method):
    for method in METHODS:
        summary = summarize_results(results_by_method[method])
        print(
            f"{method:<12} feasible {summary['feasible_count']:3d}/{summary['count']:<3d} | "
            f"mean SI reduction {summary['mean_si_improvement']:7.2f} +/- {summary['std_si_improvement']:.2f} dB | "
            f"mean SI {summary['mean_si']:7.2f} dB"
        )


def run(seed_start, num_seeds, output_prefix):
    selected_freq, results_by_method = evaluate_modes(seed_start, num_seeds)
    improvements = improvements_by_method(results_by_method)
    si_values = si_values_by_method(results_by_method)

    prefix = Path(output_prefix)
    if not prefix.is_absolute():
        prefix = FIGURES_DIR / prefix
    box_path = prefix.with_name(prefix.name + "_box_points.png")
    sorted_path = prefix.with_name(prefix.name + "_sorted_curves.png")
    sorted_ref_si_path = prefix.with_name(prefix.name + "_sorted_ref_si.png")

    plot_box_with_points(si_values, box_path, seed_start, num_seeds)
    plot_sorted_curves(improvements, sorted_path, seed_start, num_seeds)
    plot_ref_si_by_sorted_improvement(results_by_method, improvements, sorted_ref_si_path, seed_start, num_seeds)

    print()
    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Saved box/points plot: {box_path}")
    print(f"Saved sorted-curve plot: {sorted_path}")
    print(f"Saved sorted ref-SI plot: {sorted_ref_si_path}")
    print_summary(results_by_method)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot clearer SI-improvement comparisons for RL, PSO, and RL-init PSO.")
    parser.add_argument("--seed-start", type=int, default=5000, help="First seed in the evaluation range.")
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of consecutive shared theta-disabled cases to evaluate.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="si_improvement_comparison",
        help="Output filename prefix. Relative paths are written under the figures directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seed_start, args.num_seeds, args.output_prefix)

"""Plot SI-improvement histograms for RL, PSO, and RL-initialized PSO."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CHECKPOINT_DIR, ENV_CONFIG, FIGURES_DIR, HFSS_FILE, SAC_CONFIG
from .sac_agent import SACAgent
from .test_3d_beam_map import (
    evaluate_pso_seed_case,
    evaluate_rl_initialized_pso_seed_case,
    evaluate_seed_case,
    make_env,
    summarize_results,
)
from .utils import load_hfss_channel


def load_agent():
    agent_config = SAC_CONFIG.copy()
    agent_config["chkpt_dir"] = str(CHECKPOINT_DIR)
    agent = SACAgent(**agent_config)
    agent.load_models()
    return agent


def _improvements(results):
    return np.asarray([item["si_improvement_db"] for item in results], dtype=np.float64)


def evaluate_modes(seed_start, num_seeds):
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    env = make_env(hsi, seed_start)
    agent = load_agent()

    rl_results = []
    pso_results = []
    rlpso_results = []
    for seed in range(seed_start, seed_start + num_seeds):
        rl_result = evaluate_seed_case(env, agent, seed)
        pso_result = evaluate_pso_seed_case(env, rl_result["case"])
        rlpso_result = evaluate_rl_initialized_pso_seed_case(env, rl_result)
        rl_results.append(rl_result)
        pso_results.append(pso_result)
        rlpso_results.append(rlpso_result)
        print(
            f"seed {seed} | "
            f"RL {rl_result['si_improvement_db']:.2f} dB | "
            f"PSO {pso_result['si_improvement_db']:.2f} dB | "
            f"RL-init PSO {rlpso_result['si_improvement_db']:.2f} dB"
        )

    return selected_freq, rl_results, pso_results, rlpso_results


def plot_histogram(rl_results, pso_results, rlpso_results, output_path, seed_start, num_seeds, bin_width):
    os.makedirs(output_path.parent, exist_ok=True)
    rl_improvement = _improvements(rl_results)
    pso_improvement = _improvements(pso_results)
    rlpso_improvement = _improvements(rlpso_results)

    plt.figure(figsize=(9, 5))
    hist_range = (-10.0, 90.0)
    bins = np.arange(hist_range[0], hist_range[1] + bin_width, bin_width, dtype=np.float64)
    plt.hist(
        [rl_improvement, pso_improvement, rlpso_improvement],
        bins=bins,
        label=["RL", "PSO", "RL-init PSO"],
        color=["tab:blue", "tab:orange", "tab:green"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
    )
    plt.xlim(hist_range)
    plt.xlabel(f"SI improvement [dB], bin width {bin_width:g} dB")
    plt.ylabel("Count")
    plt.title(f"SI Improvement Distribution, seeds {seed_start}-{seed_start + num_seeds - 1}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_summary(label, results):
    summary = summarize_results(results)
    print(
        f"{label:<12} feasible {summary['feasible_count']:3d}/{summary['count']:<3d} | "
        f"mean SI reduction {summary['mean_si_improvement']:7.2f} +/- {summary['std_si_improvement']:.2f} dB | "
        f"mean SI {summary['mean_si']:7.2f} dB"
    )


def run(seed_start, num_seeds, output, bin_width):
    selected_freq, rl_results, pso_results, rlpso_results = evaluate_modes(seed_start, num_seeds)
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = FIGURES_DIR / output_path
    plot_histogram(rl_results, pso_results, rlpso_results, output_path, seed_start, num_seeds, bin_width)

    print()
    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Saved SI-improvement histogram: {output_path}")
    print_summary("RL", rl_results)
    print_summary("PSO", pso_results)
    print_summary("RL-init PSO", rlpso_results)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SI-improvement histograms for RL, PSO, and RL-initialized PSO.")
    parser.add_argument("--seed-start", type=int, default=5000, help="First seed in the 100-seed evaluation range.")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of consecutive seeds to evaluate.")
    parser.add_argument("--bin-width", type=float, default=5.0, help="Histogram bin width in dB over the fixed -10 to 90 dB range.")
    parser.add_argument(
        "--output",
        type=str,
        default="si_improvement_histogram_rl_pso_rlinit_pso.png",
        help="Output PNG path. Relative paths are written under the figures directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seed_start, args.num_seeds, args.output, args.bin_width)

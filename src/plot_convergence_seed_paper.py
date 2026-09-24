"""Generate paper-style theta-disabled convergence figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ENV_CONFIG, FIGURES_DIR, HFSS_FILE
from .params import DEFAULT_PSO_CONFIG, RL_INIT_PSO_CONFIG
from .plot_pso_convergence import (
    _full_pso_x,
    _rlinit_pso_x,
    load_agent,
    make_env,
    rollout_best_focus,
    trace_full_pso,
    trace_rl_initialized_pso,
)
from .plot_pso_convergence_theta_disabled import _normalized_history, sample_theta_disabled_case
from .utils import load_hfss_channel


def _prepare_axis():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def _right_sawtooth_points(x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0.0)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size <= 1:
        return x_values, y_values

    drop_idx = np.flatnonzero(y_values[1:] < y_values[:-1] * (1.0 - 1e-10))
    knot_idx = np.unique(np.concatenate(([0], drop_idx, [y_values.size - 1])))
    return x_values[knot_idx], y_values[knot_idx]


def _unique_increasing_points(x_values, y_values):
    order = np.argsort(x_values, kind="stable")
    x_sorted = np.asarray(x_values, dtype=np.float64)[order]
    y_sorted = np.asarray(y_values, dtype=np.float64)[order]

    unique_x = []
    unique_y = []
    for x_value in np.unique(x_sorted):
        same_x = x_sorted == x_value
        unique_x.append(float(x_value))
        unique_y.append(float(np.min(y_sorted[same_x])))
    return np.asarray(unique_x), np.asarray(unique_y)


def _smooth_log_curve(x_values, y_values, num_points=300):
    x_values, y_values = _unique_increasing_points(x_values, y_values)
    if x_values.size < 2:
        return x_values, y_values

    dense_x = np.linspace(float(x_values[0]), float(x_values[-1]), num_points)
    log_y = np.log10(np.maximum(y_values, 1e-300))
    try:
        from scipy.interpolate import PchipInterpolator

        dense_log_y = PchipInterpolator(x_values, log_y)(dense_x)
    except Exception:
        dense_log_y = np.interp(dense_x, x_values, log_y)
    return dense_x, 10.0**dense_log_y


def _plot_seed_curves(full, rlpso, matched, ref_si_db, rl_init_si_db, output_path):
    _prepare_axis()
    full_history = np.concatenate(([ref_si_db, ref_si_db], full))
    rlpso_history = np.concatenate(([ref_si_db, rl_init_si_db], rlpso))
    matched_history = np.concatenate(([ref_si_db, rl_init_si_db], matched))

    curves = [
        ("PSO", _full_pso_x(len(full_history), 0.0), _normalized_history(full_history, ref_si_db), "tab:blue"),
        (
            "RL-init PSO",
            _rlinit_pso_x(len(rlpso_history), 0.0),
            _normalized_history(rlpso_history, ref_si_db),
            "tab:orange",
        ),
        (
            "RL-init PSO matched",
            _rlinit_pso_x(len(matched_history), 0.0),
            _normalized_history(matched_history, ref_si_db),
            "tab:green",
        ),
    ]

    fig, ax = plt.subplots(figsize=(4.6, 3.35))
    for label, x_values, y_values, color in curves:
        ax.step(x_values, y_values, where="post", color=color, linewidth=2.0, label=label)
        knot_x, knot_y = _right_sawtooth_points(x_values, y_values)
        smooth_x, smooth_y = _smooth_log_curve(knot_x, knot_y)
        ax.plot(
            smooth_x,
            smooth_y,
            color=color,
            alpha=0.35,
            linestyle="--",
            linewidth=1.5,
        )
        ax.plot(
            knot_x,
            knot_y,
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor="0.85",
            alpha=0.65,
            markeredgecolor=color,
            markeredgewidth=0.9,
        )

    ax.set_yscale("log")
    ax.set_xlim(-2.0, 92.0)
    ax.set_ylim(1e-10, 1.2)
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Normalized linear SI")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", frameon=True, edgecolor="0.25", fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _pso_configs():
    full_config = DEFAULT_PSO_CONFIG.copy()
    rlpso_config = DEFAULT_PSO_CONFIG.copy()
    rlpso_config.update(RL_INIT_PSO_CONFIG)
    matched_config = rlpso_config.copy()
    matched_config["num_particles"] = full_config["num_particles"]
    matched_config["num_iters"] = full_config["num_iters"]
    return full_config, rlpso_config, matched_config


def _trace_seed(hsi, agent, seed, full_config, rlpso_config, matched_config):
    env = make_env(hsi, seed)
    case = sample_theta_disabled_case(env, np.random.default_rng(seed), seed)

    rl_init_si_db, rl_best_info = rollout_best_focus(env, agent, case)
    full_history, full_result = trace_full_pso(env, case, seed, full_config)
    rlpso_history, rlpso_result = trace_rl_initialized_pso(env, case, rl_best_info, seed, rlpso_config)
    matched_history, matched_result = trace_rl_initialized_pso(env, case, rl_best_info, seed, matched_config)
    return case, rl_init_si_db, full_history, rlpso_history, matched_history, full_result, rlpso_result, matched_result


def _mean_curve(histories):
    return np.nanmean(np.asarray(histories, dtype=np.float64), axis=0)


def _plot_mean_curves(full_histories, rlpso_histories, matched_histories, output_path):
    _prepare_axis()
    full_mean = _mean_curve(full_histories)
    rlpso_mean = _mean_curve(rlpso_histories)
    matched_mean = _mean_curve(matched_histories)

    curves = [
        ("PSO", _full_pso_x(len(full_mean), 0.0), full_mean, "tab:blue"),
        ("RL-init PSO", _rlinit_pso_x(len(rlpso_mean), 0.0), rlpso_mean, "tab:orange"),
        (
            "RL-init PSO matched",
            _rlinit_pso_x(len(matched_mean), 0.0),
            matched_mean,
            "tab:green",
        ),
    ]

    fig, ax = plt.subplots(figsize=(4.6, 3.35))
    for label, x_values, y_values, color in curves:
        ax.plot(x_values, y_values, color=color, linewidth=2.6, label=label)

    ax.set_yscale("log")
    ax.set_xlim(-2.0, 92.0)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Mean normalized linear SI")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", frameon=True, edgecolor="0.25", fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_seed(seed, output):
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    agent = load_agent()
    full_config, rlpso_config, matched_config = _pso_configs()
    case, rl_init_si_db, full_history, rlpso_history, matched_history, full_result, rlpso_result, matched_result = _trace_seed(
        hsi,
        agent,
        seed,
        full_config,
        rlpso_config,
        matched_config,
    )

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = FIGURES_DIR / output_path
    os.makedirs(output_path.parent, exist_ok=True)

    _plot_seed_curves(
        full_history,
        rlpso_history,
        matched_history,
        full_result["si_ref_db"],
        rl_init_si_db,
        output_path,
    )

    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Seed {seed} theta-disabled case: {case}")
    print(f"Reference SI: {full_result['si_ref_db']:.2f} dB")
    print(f"RL init SI: {rl_init_si_db:.2f} dB")
    print(f"PSO final best SI: {full_result['best_si_db']:.2f} dB")
    print(f"RL-init PSO final best SI: {rlpso_result['best_si_db']:.2f} dB")
    print(f"RL-init PSO matched final best SI: {matched_result['best_si_db']:.2f} dB")
    print(f"Saved seed convergence plot: {output_path}")


def run_aggregate(seed_start, num_seeds, output):
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    agent = load_agent()
    full_config, rlpso_config, matched_config = _pso_configs()
    full_norm_histories = []
    rlpso_norm_histories = []
    matched_norm_histories = []

    for seed in range(seed_start, seed_start + num_seeds):
        (
            _case,
            rl_init_si_db,
            full_history,
            rlpso_history,
            matched_history,
            full_result,
            rlpso_result,
            matched_result,
        ) = _trace_seed(hsi, agent, seed, full_config, rlpso_config, matched_config)
        ref_si_db = full_result["si_ref_db"]
        full_norm_histories.append(_normalized_history(np.concatenate(([ref_si_db, ref_si_db], full_history)), ref_si_db))
        rlpso_norm_histories.append(_normalized_history(np.concatenate(([ref_si_db, rl_init_si_db], rlpso_history)), ref_si_db))
        matched_norm_histories.append(_normalized_history(np.concatenate(([ref_si_db, rl_init_si_db], matched_history)), ref_si_db))
        print(
            f"seed {seed} | PSO {full_result['best_si_db']:.2f} dB | "
            f"RL-init {rlpso_result['best_si_db']:.2f} dB | "
            f"matched {matched_result['best_si_db']:.2f} dB"
        )

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = FIGURES_DIR / output_path
    os.makedirs(output_path.parent, exist_ok=True)

    _plot_mean_curves(full_norm_histories, rlpso_norm_histories, matched_norm_histories, output_path)
    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Seeds: {seed_start}-{seed_start + num_seeds - 1}")
    print(f"Saved aggregate convergence plot: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot theta-disabled convergence figures.")
    parser.add_argument("--seed", type=int, default=5003, help="Deterministic theta-disabled seed to plot.")
    parser.add_argument(
        "--seed-output",
        type=str,
        default="pso_convergence_seed5003_line_paper.png",
        help="Single-seed PNG path. Relative paths are written under the figures directory.",
    )
    parser.add_argument("--seed-start", type=int, default=5000, help="First aggregate seed.")
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of aggregate seeds.")
    parser.add_argument(
        "--aggregate-output",
        type=str,
        default="pso_vs_rlinit_pso_convergence_theta_disabled_eval_linear.png",
        help="Aggregate PNG path. Relative paths are written under the figures directory.",
    )
    parser.add_argument(
        "--only",
        choices=("both", "seed", "aggregate"),
        default="both",
        help="Which figure to generate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.only in {"both", "seed"}:
        run_seed(args.seed, args.seed_output)
    if args.only in {"both", "aggregate"}:
        run_aggregate(args.seed_start, args.num_seeds, args.aggregate_output)

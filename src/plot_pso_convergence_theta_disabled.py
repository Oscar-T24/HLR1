"""Plot PSO convergence on a theta-disabled deterministic test field.

This companion to plot_pso_convergence.py keeps the same full PSO, RL-init PSO,
and matched-budget RL-init PSO comparison. The only sampling change is that DL
and UL user theta are fixed by THETA_DISABLED_TEST_CONFIG while range and phi
continue to vary over deterministic grids.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ENV_CONFIG, FIGURES_DIR, HFSS_FILE, THETA_DISABLED_TEST_CONFIG
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
from .utils import load_hfss_channel


def si_db_to_linear(si_db):
    return 10.0 ** (np.asarray(si_db, dtype=np.float64) / 10.0)


def middle_rank_indices(values, lower_fraction=0.2, upper_fraction=0.8):
    """Return the stable rank slice [lower_fraction, upper_fraction)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not 0.0 <= lower_fraction < upper_fraction <= 1.0:
        raise ValueError("Rank fractions must satisfy 0 <= lower < upper <= 1.")

    lower = int(round(lower_fraction * len(values)))
    upper = int(round(upper_fraction * len(values)))
    return np.argsort(values, kind="stable")[lower:upper]


def matched_convergence_metrics(full_histories, rlpso_histories, ref_si_values):
    """Compare PSO methods at 45 iterations and at a shared per-case SI reduction."""
    full_histories = np.asarray(full_histories, dtype=np.float64)
    rlpso_histories = np.asarray(rlpso_histories, dtype=np.float64)
    ref_si_values = np.asarray(ref_si_values, dtype=np.float64)
    matched_iteration = min(full_histories.shape[1], rlpso_histories.shape[1])

    full_reductions = ref_si_values[:, None] - full_histories
    rlpso_reductions = ref_si_values[:, None] - rlpso_histories
    common_targets = np.minimum(full_reductions[:, -1], rlpso_reductions[:, -1])

    def first_reaching_iterations(reductions):
        reached = reductions >= common_targets[:, None] - 1e-12
        return np.argmax(reached, axis=1) + 1

    full_matched_iterations = float(np.mean(first_reaching_iterations(full_reductions)))
    rlpso_matched_iterations = float(np.mean(first_reaching_iterations(rlpso_reductions)))
    return {
        "matched_iteration": matched_iteration,
        "full_mean_reduction_at_matched_iteration": float(
            np.mean(full_reductions[:, matched_iteration - 1])
        ),
        "rlpso_mean_reduction_at_matched_iteration": float(
            np.mean(rlpso_reductions[:, matched_iteration - 1])
        ),
        "full_mean_iterations_at_matched_reduction": full_matched_iterations,
        "rlpso_mean_iterations_at_matched_reduction": rlpso_matched_iterations,
        "rlpso_mean_iteration_saving": full_matched_iterations - rlpso_matched_iterations,
    }


def plot_beam_si_convergence_summary(
    full_histories,
    rlpso_histories,
    matched_histories,
    ref_si_values,
    rl_init_si_values,
    output_path,
    seed_start,
    num_seeds,
    full_config,
    rlpso_config,
    matched_config,
    statistic,
    case_indices=None,
):
    os.makedirs(output_path.parent, exist_ok=True)
    full_values_db = np.asarray(
        [np.concatenate(([ref_si, ref_si], history)) for ref_si, history in zip(ref_si_values, full_histories)]
    )
    rlpso_values_db = np.asarray(
        [
            np.concatenate(([ref_si, rl_init_si], history))
            for ref_si, rl_init_si, history in zip(ref_si_values, rl_init_si_values, rlpso_histories)
        ]
    )
    matched_values_db = np.asarray(
        [
            np.concatenate(([ref_si, rl_init_si], history))
            for ref_si, rl_init_si, history in zip(ref_si_values, rl_init_si_values, matched_histories)
        ]
    )
    if case_indices is None:
        case_indices = np.arange(len(ref_si_values))
    case_indices = np.asarray(case_indices, dtype=int)
    if statistic == "mean":
        reducer = np.nanmean
        statistic_label = "Mean"
    elif statistic == "median":
        reducer = np.nanmedian
        statistic_label = "Median"
    else:
        raise ValueError(f"Unsupported convergence statistic: {statistic}")

    full_summary = reducer(si_db_to_linear(full_values_db[case_indices]), axis=0)
    rlpso_summary = reducer(si_db_to_linear(rlpso_values_db[case_indices]), axis=0)
    matched_summary = reducer(si_db_to_linear(matched_values_db[case_indices]), axis=0)
    selected_case_count = len(case_indices)

    plt.figure(figsize=(9, 6.4))
    plt.step(
        _full_pso_x(len(full_summary), 0.0),
        full_summary,
        where="post",
        color="tab:blue",
        linewidth=2.6,
        label=f"PSO ({full_config['num_iters']} iters, {full_config['num_particles']} particles)",
    )
    plt.step(
        _rlinit_pso_x(len(rlpso_summary), 0.0),
        rlpso_summary,
        where="post",
        color="tab:orange",
        linewidth=2.6,
        label=f"RL-init PSO ({rlpso_config['num_iters']} iters, {rlpso_config['num_particles']} particles)",
    )
    plt.step(
        _rlinit_pso_x(len(matched_summary), 0.0),
        matched_summary,
        where="post",
        color="tab:green",
        linewidth=2.6,
        label=(
            "RL-init PSO matched budget "
            f"({matched_config['num_iters']} iters, {matched_config['num_particles']} particles)"
        ),
    )
    plt.xlabel("PSO iteration", fontsize=16)
    plt.ylabel(f"{statistic_label} beam-level SI (linear)", fontsize=16)
    plt.title(
        f"{statistic_label} Linear Beam-Level SI Convergence ({selected_case_count} Cases)",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    plt.yscale("log")
    plt.tick_params(axis="both", labelsize=14)
    # Paper version: restore with plt.grid(True, which="both", alpha=0.3).
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_case_convergence(
    full_history,
    rlpso_history,
    matched_history,
    ref_si_db,
    rl_init_si_db,
    seed,
    output_path,
    full_config,
    rlpso_config,
    matched_config,
):
    os.makedirs(output_path.parent, exist_ok=True)

    full_values = si_db_to_linear(np.concatenate(([ref_si_db, ref_si_db], full_history)))
    rlpso_values = si_db_to_linear(np.concatenate(([ref_si_db, rl_init_si_db], rlpso_history)))
    matched_values = si_db_to_linear(np.concatenate(([ref_si_db, rl_init_si_db], matched_history)))
    full_x = _full_pso_x(len(full_values), 0.0)
    rlpso_x = _rlinit_pso_x(len(rlpso_values), 0.0)
    matched_x = _rlinit_pso_x(len(matched_values), 0.0)

    plt.figure(figsize=(9, 5.2))
    plt.step(
        full_x,
        full_values,
        where="post",
        color="tab:blue",
        linewidth=2.2,
        label=f"PSO ({full_config['num_iters']} iters, {full_config['num_particles']} particles)",
    )
    plt.step(
        rlpso_x,
        rlpso_values,
        where="post",
        color="tab:orange",
        linewidth=2.2,
        label=f"RL-init PSO ({rlpso_config['num_iters']} iters, {rlpso_config['num_particles']} particles)",
    )
    plt.step(
        matched_x,
        matched_values,
        where="post",
        color="tab:green",
        linewidth=2.2,
        label=f"RL-init PSO matched budget ({matched_config['num_iters']} iters, {matched_config['num_particles']} particles)",
    )

    plt.xlabel("PSO iteration")
    plt.ylabel("Beam-level SI (linear)")
    plt.title(f"Theta-Disabled Linear Beam-Level SI Convergence, seed {seed}")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_mean_si_improvement(
    full_histories,
    rlpso_histories,
    matched_histories,
    ref_si_values,
    rl_init_si_values,
    output_path,
    seed_start,
    num_seeds,
    full_config,
    rlpso_config,
    matched_config,
):
    os.makedirs(output_path.parent, exist_ok=True)
    full_improvements = np.asarray(
        [
            ref_si - np.concatenate(([ref_si, ref_si], history))
            for ref_si, history in zip(ref_si_values, full_histories)
        ]
    )
    rlpso_improvements = np.asarray(
        [
            ref_si - np.concatenate(([ref_si, rl_init_si], history))
            for ref_si, rl_init_si, history in zip(ref_si_values, rl_init_si_values, rlpso_histories)
        ]
    )
    matched_improvements = np.asarray(
        [
            ref_si - np.concatenate(([ref_si, rl_init_si], history))
            for ref_si, rl_init_si, history in zip(ref_si_values, rl_init_si_values, matched_histories)
        ]
    )
    full_mean = np.nanmean(full_improvements, axis=0)
    rlpso_mean = np.nanmean(rlpso_improvements, axis=0)
    matched_mean = np.nanmean(matched_improvements, axis=0)
    comparison_iteration = min(
        full_config["num_iters"],
        rlpso_config["num_iters"],
        matched_config["num_iters"],
    )
    full_x = _full_pso_x(len(full_mean), 0.0)
    rlpso_x = _rlinit_pso_x(len(rlpso_mean), 0.0)
    matched_x = _rlinit_pso_x(len(matched_mean), 0.0)

    figure = plt.figure(figsize=(9, 5.2))
    axes = plt.gca()
    figure.patch.set_alpha(0.0)
    axes.set_facecolor("none")
    plt.step(
        full_x,
        full_mean,
        where="post",
        color="tab:orange",
        linewidth=2.6,
        label=f"PSO ({full_config['num_iters']} iters, {full_config['num_particles']} particles)",
    )
    plt.step(
        rlpso_x,
        rlpso_mean,
        where="post",
        color="tab:green",
        linewidth=2.6,
        label=f"RL-init PSO ({rlpso_config['num_iters']} iters, {rlpso_config['num_particles']} particles)",
    )
    plt.step(
        matched_x,
        matched_mean,
        where="post",
        color="tab:purple",
        linewidth=2.6,
        label=(
            "RL-init PSO matched budget "
            f"({matched_config['num_iters']} iters, {matched_config['num_particles']} particles)"
        ),
    )
    plt.axhline(
        0.0,
        color="tab:gray",
        linestyle="--",
        linewidth=3.5,
        alpha=0.6,
        label="Reference SI baseline",
    )
    plt.axvline(
        comparison_iteration,
        color="0.25",
        linestyle=":",
        linewidth=1.8,
        label=f"{comparison_iteration}-iteration comparison",
    )
    comparison_curves = (
        ("PSO", full_x, full_mean, "tab:orange", -18),
        ("RL-init PSO", rlpso_x, rlpso_mean, "tab:green", 0),
        ("RL-init matched", matched_x, matched_mean, "tab:purple", 18),
    )
    for label, x_values, mean_values, color, y_offset in comparison_curves:
        value_index = np.flatnonzero(np.isclose(x_values, comparison_iteration))[0]
        value = mean_values[value_index]
        plt.scatter(comparison_iteration, value, color=color, edgecolor="white", linewidth=0.8, s=42, zorder=5)
        plt.annotate(
            f"{label}: {value:.2f} dB",
            xy=(comparison_iteration, value),
            xytext=(8, y_offset),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "alpha": 0.88},
        )
    plt.xlabel("PSO iteration", fontsize=16, fontfamily="serif")
    plt.ylabel("Mean SI reduction from reference [dB]", fontsize=16, fontfamily="serif")
    plt.title("Mean Beam-Level SI Reduction", fontsize=17, fontweight="bold", fontfamily="serif", pad=10)
    axes.tick_params(axis="both", labelsize=13)
    for tick_label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
        tick_label.set_fontfamily("serif")
    plt.grid(True, color="#8a7f85", linewidth=0.9, alpha=0.42)
    plt.legend(
        loc="upper left",
        prop={"family": "serif", "size": 10},
        facecolor="none",
        framealpha=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, transparent=True)
    white_output_path = output_path.with_name(f"{output_path.stem}_white_background{output_path.suffix}")
    figure.patch.set_alpha(1.0)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    plt.savefig(white_output_path, dpi=200, transparent=False, facecolor="white", edgecolor="white")
    plt.close()


def plot_case_si_improvement(
    full_history,
    rlpso_history,
    matched_history,
    ref_si_db,
    rl_init_si_db,
    seed,
    output_path,
    full_config,
    rlpso_config,
    matched_config,
):
    os.makedirs(output_path.parent, exist_ok=True)
    full_improvement = ref_si_db - np.concatenate(([ref_si_db, ref_si_db], full_history))
    rlpso_improvement = ref_si_db - np.concatenate(([ref_si_db, rl_init_si_db], rlpso_history))
    matched_improvement = ref_si_db - np.concatenate(([ref_si_db, rl_init_si_db], matched_history))

    plt.figure(figsize=(9, 5.2))
    plt.step(
        _full_pso_x(len(full_improvement), 0.0),
        full_improvement,
        where="post",
        color="tab:blue",
        linewidth=2.2,
        label=f"PSO ({full_config['num_iters']} iters, {full_config['num_particles']} particles)",
    )
    plt.step(
        _rlinit_pso_x(len(rlpso_improvement), 0.0),
        rlpso_improvement,
        where="post",
        color="tab:orange",
        linewidth=2.2,
        label=f"RL-init PSO ({rlpso_config['num_iters']} iters, {rlpso_config['num_particles']} particles)",
    )
    plt.step(
        _rlinit_pso_x(len(matched_improvement), 0.0),
        matched_improvement,
        where="post",
        color="tab:green",
        linewidth=2.2,
        label=(
            "RL-init PSO matched budget "
            f"({matched_config['num_iters']} iters, {matched_config['num_particles']} particles)"
        ),
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Reference SI baseline")
    plt.xlabel("PSO iteration")
    plt.ylabel("SI improvement from reference [dB]")
    plt.title(f"Theta-Disabled SI Improvement, seed {seed}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_case_seeds(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def sample_theta_disabled_case(env, rng, seed):
    dl_r_grid = np.asarray(THETA_DISABLED_TEST_CONFIG["dl_r_grid"], dtype=np.float64)
    ul_r_grid = np.asarray(THETA_DISABLED_TEST_CONFIG["ul_r_grid"], dtype=np.float64)
    phi_grid = np.arange(
        THETA_DISABLED_TEST_CONFIG["phi_grid_min"],
        THETA_DISABLED_TEST_CONFIG["phi_grid_max"] + 0.1 * THETA_DISABLED_TEST_CONFIG["phi_grid_step"],
        THETA_DISABLED_TEST_CONFIG["phi_grid_step"],
    )
    theta_fixed = float(THETA_DISABLED_TEST_CONFIG["theta_fixed"])

    while True:
        case = {
            "r_dl_user": float(rng.choice(dl_r_grid)),
            "r_ul_user": float(rng.choice(ul_r_grid)),
            "phi_dl_user": float(rng.choice(phi_grid)),
            "phi_ul_user": float(rng.choice(phi_grid)),
            "theta_dl_user": theta_fixed,
            "theta_ul_user": theta_fixed,
            "seed": int(seed),
        }
        angular_sep = abs(case["phi_dl_user"] - case["phi_ul_user"])
        if angular_sep >= env.min_angular_sep:
            return case


def run(
    seed_start,
    num_seeds,
    output,
    median_output,
    improvement_output,
    case_seeds=None,
    case_output_dir=None,
    improvement_case_output_dir=None,
):
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")

    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    agent = load_agent()
    full_config = DEFAULT_PSO_CONFIG.copy()
    rlpso_config = DEFAULT_PSO_CONFIG.copy()
    rlpso_config.update(RL_INIT_PSO_CONFIG)
    matched_config = rlpso_config.copy()
    matched_config["num_particles"] = full_config["num_particles"]
    matched_config["num_iters"] = full_config["num_iters"]

    full_histories = []
    rlpso_histories = []
    matched_histories = []
    ref_si_values = []
    rl_init_si_values = []
    case_histories = {}
    full_final_si = []
    rlpso_final_si = []
    matched_final_si = []

    for seed in range(seed_start, seed_start + num_seeds):
        env = make_env(hsi, seed)
        case = sample_theta_disabled_case(env, np.random.default_rng(seed), seed)
        rl_init_si_db, rl_best_info = rollout_best_focus(env, agent, case)
        full_history, full_result = trace_full_pso(env, case, seed, full_config)
        rlpso_history, rlpso_result = trace_rl_initialized_pso(env, case, rl_best_info, seed, rlpso_config)
        matched_history, matched_result = trace_rl_initialized_pso(env, case, rl_best_info, seed, matched_config)
        full_histories.append(full_history)
        rlpso_histories.append(rlpso_history)
        matched_histories.append(matched_history)
        ref_si_values.append(full_result["si_ref_db"])
        rl_init_si_values.append(rl_init_si_db)
        case_histories[seed] = {
            "full": full_history,
            "rlpso": rlpso_history,
            "matched": matched_history,
            "ref_si_db": full_result["si_ref_db"],
            "rl_init_si_db": rl_init_si_db,
        }
        full_final_si.append(full_result["best_si_db"])
        rlpso_final_si.append(rlpso_result["best_si_db"])
        matched_final_si.append(matched_result["best_si_db"])
        print(
            f"seed {seed} | theta {case['theta_dl_user']:.1f}/{case['theta_ul_user']:.1f} deg | "
            f"PSO {full_result['best_si_db']:.2f} dB | "
            f"RL-init {rlpso_result['best_si_db']:.2f} dB | "
            f"RL-init matched {matched_result['best_si_db']:.2f} dB"
        )

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = FIGURES_DIR / output_path

    middle_indices = middle_rank_indices(ref_si_values, 0.2, 0.8)
    plot_beam_si_convergence_summary(
        full_histories,
        rlpso_histories,
        matched_histories,
        ref_si_values,
        rl_init_si_values,
        output_path,
        seed_start,
        num_seeds,
        full_config,
        rlpso_config,
        matched_config,
        statistic="mean",
        case_indices=middle_indices,
    )

    median_output_path = Path(median_output)
    if not median_output_path.is_absolute():
        median_output_path = FIGURES_DIR / median_output_path
    plot_beam_si_convergence_summary(
        full_histories,
        rlpso_histories,
        matched_histories,
        ref_si_values,
        rl_init_si_values,
        median_output_path,
        seed_start,
        num_seeds,
        full_config,
        rlpso_config,
        matched_config,
        statistic="median",
    )

    improvement_output_path = Path(improvement_output)
    if not improvement_output_path.is_absolute():
        improvement_output_path = FIGURES_DIR / improvement_output_path
    plot_mean_si_improvement(
        full_histories,
        rlpso_histories,
        matched_histories,
        ref_si_values,
        rl_init_si_values,
        improvement_output_path,
        seed_start,
        num_seeds,
        full_config,
        rlpso_config,
        matched_config,
    )

    selected_seeds = list(
        dict.fromkeys(case_seeds or [seed_start, seed_start + min(3, num_seeds - 1), seed_start + num_seeds - 1])
    )
    invalid_seeds = [seed for seed in selected_seeds if seed not in case_histories]
    if invalid_seeds:
        raise ValueError(f"Selected case seeds are outside the evaluated range: {invalid_seeds}")

    cases_dir = Path(case_output_dir) if case_output_dir else output_path.parent / f"{output_path.stem}_cases"
    if not cases_dir.is_absolute():
        cases_dir = FIGURES_DIR / cases_dir
    improvement_cases_dir = (
        Path(improvement_case_output_dir)
        if improvement_case_output_dir
        else improvement_output_path.parent / f"{improvement_output_path.stem}_cases"
    )
    if not improvement_cases_dir.is_absolute():
        improvement_cases_dir = FIGURES_DIR / improvement_cases_dir
    for seed in selected_seeds:
        case_data = case_histories[seed]
        case_path = cases_dir / f"theta_disabled_convergence_seed_{seed}.png"
        plot_case_convergence(
            case_data["full"],
            case_data["rlpso"],
            case_data["matched"],
            case_data["ref_si_db"],
            case_data["rl_init_si_db"],
            seed,
            case_path,
            full_config,
            rlpso_config,
            matched_config,
        )
        print(f"Saved selected-case convergence plot: {case_path}")
        improvement_case_path = improvement_cases_dir / f"theta_disabled_si_improvement_seed_{seed}.png"
        plot_case_si_improvement(
            case_data["full"],
            case_data["rlpso"],
            case_data["matched"],
            case_data["ref_si_db"],
            case_data["rl_init_si_db"],
            seed,
            improvement_case_path,
            full_config,
            rlpso_config,
            matched_config,
        )
        print(f"Saved selected-case SI-improvement plot: {improvement_case_path}")

    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Seeds: {seed_start}-{seed_start + num_seeds - 1} ({num_seeds} theta-disabled cases)")
    print(f"Theta fixed at: {THETA_DISABLED_TEST_CONFIG['theta_fixed']:.1f} deg")
    print(
        f"Saved {len(middle_indices)}-case mean beam-level SI plot "
        f"(reference-SI ranks 0.2-0.8): {output_path}"
    )
    print(f"Saved {num_seeds}-case median beam-level SI plot: {median_output_path}")
    print(f"Saved {num_seeds}-case mean SI-improvement plot: {improvement_output_path}")
    print(f"PSO mean final SI: {np.mean(full_final_si):.2f} +/- {np.std(full_final_si):.2f} dB")
    print(f"RL-init PSO mean final SI: {np.mean(rlpso_final_si):.2f} +/- {np.std(rlpso_final_si):.2f} dB")
    print(f"RL-init PSO matched-budget mean final SI: {np.mean(matched_final_si):.2f} +/- {np.std(matched_final_si):.2f} dB")
    comparison = matched_convergence_metrics(full_histories, rlpso_histories, ref_si_values)
    print(
        f"PSO mean SI reduction @ {comparison['matched_iteration']} iters: "
        f"{comparison['full_mean_reduction_at_matched_iteration']:.2f} dB"
    )
    print(
        f"RL-init PSO mean SI reduction @ {comparison['matched_iteration']} iters: "
        f"{comparison['rlpso_mean_reduction_at_matched_iteration']:.2f} dB"
    )
    print(
        "Mean iterations @ matched per-case SI reduction: "
        f"PSO {comparison['full_mean_iterations_at_matched_reduction']:.2f}, "
        f"RL-init PSO {comparison['rlpso_mean_iterations_at_matched_reduction']:.2f}"
    )
    print(
        "RL-init PSO mean iteration saving @ matched per-case SI reduction: "
        f"{comparison['rlpso_mean_iteration_saving']:.2f} iterations"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PSO convergence with theta fixed in the test field.")
    parser.add_argument("--seed", type=int, default=5000, help="First seed in the Monte Carlo evaluation range.")
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of consecutive Monte Carlo seeds to average.")
    parser.add_argument(
        "--output",
        type=str,
        default="pso_vs_rlinit_pso_convergence_theta_disabled_mean_linear.png",
        help=(
            "Mean linear beam-level SI PNG path for the middle 0.2-0.8 reference-SI ranks. "
            "Relative paths are written under the figures directory."
        ),
    )
    parser.add_argument(
        "--median-output",
        type=str,
        default="pso_vs_rlinit_pso_convergence_theta_disabled_median_linear.png",
        help="Median linear beam-level SI PNG path for all cases.",
    )
    parser.add_argument(
        "--case-seeds",
        type=parse_case_seeds,
        default=None,
        help="Comma-separated evaluated seeds for separate one-case figures (default: first, fourth, and last).",
    )
    parser.add_argument(
        "--case-output-dir",
        type=str,
        default=None,
        help="Directory for selected-case PNGs (default: a sibling directory named from --output).",
    )
    parser.add_argument(
        "--improvement-output",
        type=str,
        default="pso_vs_rlinit_pso_theta_disabled_mean_improvement_db.png",
        help="Mean reference-relative SI-improvement PNG path.",
    )
    parser.add_argument(
        "--improvement-case-output-dir",
        type=str,
        default=None,
        help="Directory for selected-case SI-improvement PNGs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.seed,
        args.num_seeds,
        args.output,
        args.median_output,
        args.improvement_output,
        case_seeds=args.case_seeds,
        case_output_dir=args.case_output_dir,
        improvement_case_output_dir=args.improvement_case_output_dir,
    )

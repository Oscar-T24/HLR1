"""Build a standalone PSO search book for deterministic seed cases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .config import ENV_CONFIG, HFSS_FILE, PSO_BOOK_CONFIG, PSO_BOOK_FILE
from .env import NearFieldBeamFocusingEnv
from .utils import load_hfss_channel


def make_env(hsi, seed):
    env_config = ENV_CONFIG.copy()
    env_config["seed"] = seed
    return NearFieldBeamFocusingEnv(Hsi=hsi, **env_config)


def sample_seed_case(env, seed):
    env.reset(seed=seed)
    return np.array(
        [
            env.r_dl_user,
            env.phi_dl_user,
            env.theta_dl_user,
            env.r_ul_user,
            env.phi_ul_user,
            env.theta_ul_user,
        ],
        dtype=np.float64,
    )


def set_case(env, case):
    env.r_dl_user = float(case[0])
    env.phi_dl_user = float(case[1])
    env.theta_dl_user = float(case[2])
    env.r_ul_user = float(case[3])
    env.phi_ul_user = float(case[4])
    env.theta_ul_user = float(case[5])


def set_focus(env, focus):
    env.r_tx_focus = float(focus[0])
    env.phi_tx_focus = float(focus[1])
    env.theta_tx_focus = float(focus[2])
    env.r_rx_focus = float(focus[3])
    env.phi_rx_focus = float(focus[4])
    env.theta_rx_focus = float(focus[5])


def evaluate_focus(env, focus):
    set_focus(env, focus)
    si_db = env.compute_SI_dB(*focus)
    gain_metrics = env.compute_gain_metrics()
    dl_violation = max(0.0, gain_metrics["DL_gain_loss_dB"] - env.dl_gain_margin_db)
    ul_violation = max(0.0, gain_metrics["UL_gain_loss_dB"] - env.ul_gain_margin_db)
    violation = env.lambda_dl_gain * dl_violation + env.lambda_ul_gain * ul_violation
    objective = si_db + PSO_BOOK_CONFIG["penalty_db"] * violation
    return float(objective), float(si_db), float(gain_metrics["DL_gain_loss_dB"]), float(gain_metrics["UL_gain_loss_dB"]), float(violation)


def pso_search(env, rng, n_particles=PSO_BOOK_CONFIG["num_particles"], n_iters=PSO_BOOK_CONFIG["num_iters"]):
    lower = np.array([env.r_min, env.phi_min, env.theta_min, env.r_min, env.phi_min, env.theta_min], dtype=np.float64)
    upper = np.array([env.r_max, env.phi_max, env.theta_max, env.r_max, env.phi_max, env.theta_max], dtype=np.float64)
    span = upper - lower

    positions = lower + rng.random((n_particles, 6)) * span
    velocity_fraction = PSO_BOOK_CONFIG["initial_velocity_fraction"]
    velocities = rng.uniform(-velocity_fraction, velocity_fraction, size=(n_particles, 6)) * span
    personal_best = positions.copy()
    personal_best_obj = np.full(n_particles, np.inf, dtype=np.float64)
    global_best = positions[0].copy()
    global_best_obj = np.inf
    global_best_metrics = None

    for _ in range(n_iters):
        for idx in range(n_particles):
            metrics = evaluate_focus(env, positions[idx])
            objective = metrics[0]
            if objective < personal_best_obj[idx]:
                personal_best_obj[idx] = objective
                personal_best[idx] = positions[idx].copy()
            if objective < global_best_obj:
                global_best_obj = objective
                global_best = positions[idx].copy()
                global_best_metrics = metrics

        r1 = rng.random((n_particles, 6))
        r2 = rng.random((n_particles, 6))
        velocities = (
            PSO_BOOK_CONFIG["inertia"] * velocities
            + PSO_BOOK_CONFIG["cognitive"] * r1 * (personal_best - positions)
            + PSO_BOOK_CONFIG["social"] * r2 * (global_best - positions)
        )
        velocity_clip = PSO_BOOK_CONFIG["velocity_clip_fraction"]
        velocities = np.clip(velocities, -velocity_clip * span, velocity_clip * span)
        positions = np.clip(positions + velocities, lower, upper)

    if global_best_metrics is None:
        raise RuntimeError("PSO did not evaluate any particle.")
    return global_best, global_best_metrics


def build_pso_book(
    seed_start=PSO_BOOK_CONFIG["seed_start"],
    seed_count=PSO_BOOK_CONFIG["seed_count"],
    n_particles=PSO_BOOK_CONFIG["num_particles"],
    n_iters=PSO_BOOK_CONFIG["num_iters"],
    output_file=PSO_BOOK_FILE,
):
    output_file = Path(output_file)
    os.makedirs(output_file.parent, exist_ok=True)
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    env = make_env(hsi, seed_start)

    seeds = np.arange(seed_start, seed_start + seed_count, dtype=np.int64)
    cases = np.zeros((seed_count, 6), dtype=np.float64)
    best_focuses = np.zeros((seed_count, 6), dtype=np.float64)
    metrics = np.zeros((seed_count, 5), dtype=np.float64)

    for idx, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        case = sample_seed_case(env, int(seed))
        set_case(env, case)
        best_focus, best_metrics = pso_search(env, rng, n_particles=n_particles, n_iters=n_iters)
        cases[idx] = case
        best_focuses[idx] = best_focus
        metrics[idx] = np.asarray(best_metrics, dtype=np.float64)
        print(
            f"seed {int(seed)} | "
            f"SI={best_metrics[1]:.2f} dB | "
            f"DL/UL loss={best_metrics[2]:.2f}/{best_metrics[3]:.2f} dB | "
            f"violation={best_metrics[4]:.2f}"
        )

    np.savez_compressed(
        output_file,
        selected_freq=np.array([selected_freq], dtype=np.float64),
        seeds=seeds,
        cases=cases,
        best_focuses=best_focuses,
        metrics=metrics,
        metric_columns=np.array(["objective", "si_db", "dl_gain_loss_db", "ul_gain_loss_db", "violation"]),
        case_columns=np.array(["r_dl_user", "phi_dl_user", "theta_dl_user", "r_ul_user", "phi_ul_user", "theta_ul_user"]),
        focus_columns=np.array(["r_tx_focus", "phi_tx_focus", "theta_tx_focus", "r_rx_focus", "phi_rx_focus", "theta_rx_focus"]),
    )
    print(f"Saved PSO book to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a PSO search book for seed-sampled near-field cases.")
    parser.add_argument("--seed-start", type=int, default=PSO_BOOK_CONFIG["seed_start"])
    parser.add_argument("--seed-count", type=int, default=PSO_BOOK_CONFIG["seed_count"])
    parser.add_argument("--particles", type=int, default=PSO_BOOK_CONFIG["num_particles"])
    parser.add_argument("--iters", type=int, default=PSO_BOOK_CONFIG["num_iters"])
    parser.add_argument("--output", type=str, default=str(PSO_BOOK_FILE))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = PSO_BOOK_FILE.parent / output_file
    build_pso_book(
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        n_particles=args.particles,
        n_iters=args.iters,
        output_file=output_file,
    )

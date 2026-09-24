"""Plot Monte Carlo SI convergence of full PSO versus RL-init PSO variants.

The y-axis records SI in dB, starting from the reference user-focused SI before
optimization. RL-init curves also include the learned-model initialization point
before PSO iterations. Lower SI is better. The comparison includes the
configured RL-init PSO and an RL-init PSO variant whose search budget matches
full PSO.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import CHECKPOINT_DIR, ENV_CONFIG, FIGURES_DIR, HFSS_FILE, SAC_CONFIG, TEST_3D_CONFIG

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .env import NearFieldBeamFocusingEnv
from .params import DEFAULT_PSO_CONFIG, RL_INIT_PSO_CONFIG
from .pso_optimizer import _build_result, _focus_vector_bounds, _gain_violation_score, _info_from_focus_vector, _objective
from .rl_initialized_pso import _local_bounds, focus_vector_from_info
from .sac_agent import SACAgent
from .utils import load_hfss_channel


def make_env(hsi, seed):
    env_config = ENV_CONFIG.copy()
    env_config["seed"] = seed
    return NearFieldBeamFocusingEnv(Hsi=hsi, **env_config)


def load_agent():
    agent_config = SAC_CONFIG.copy()
    agent_config["chkpt_dir"] = str(CHECKPOINT_DIR)
    agent = SACAgent(**agent_config)
    agent.load_models()
    return agent


def sample_3d_case(env, rng, seed):
    dl_r_grid = np.asarray(TEST_3D_CONFIG["dl_r_grid"], dtype=np.float64)
    ul_r_grid = np.asarray(TEST_3D_CONFIG["ul_r_grid"], dtype=np.float64)
    phi_grid = np.arange(
        TEST_3D_CONFIG["phi_grid_min"],
        TEST_3D_CONFIG["phi_grid_max"] + 0.1 * TEST_3D_CONFIG["phi_grid_step"],
        TEST_3D_CONFIG["phi_grid_step"],
    )
    theta_grid = np.arange(
        TEST_3D_CONFIG["theta_grid_min"],
        TEST_3D_CONFIG["theta_grid_max"] + 0.1 * TEST_3D_CONFIG["theta_grid_step"],
        TEST_3D_CONFIG["theta_grid_step"],
    )

    while True:
        case = {
            "r_dl_user": float(rng.choice(dl_r_grid)),
            "r_ul_user": float(rng.choice(ul_r_grid)),
            "phi_dl_user": float(rng.choice(phi_grid)),
            "phi_ul_user": float(rng.choice(phi_grid)),
            "theta_dl_user": float(rng.choice(theta_grid)),
            "theta_ul_user": float(rng.choice(theta_grid)),
            "seed": int(seed),
        }
        angular_sep = np.hypot(
            case["phi_dl_user"] - case["phi_ul_user"],
            case["theta_dl_user"] - case["theta_ul_user"],
        )
        if angular_sep >= env.min_angular_sep:
            return case


def rollout_best_focus(env, agent, case):
    obs, _ = env.reset_fixed(**case)
    best_feasible_info = None
    best_feasible_si_db = np.inf
    fallback_info = None
    fallback_si_db = np.inf
    fallback_violation = np.inf
    done = False

    while not done:
        action = agent.choose_action(obs, evaluate=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        violation = _gain_violation_score(env, info)
        if violation <= 0.0:
            if info["SI_dB"] < best_feasible_si_db:
                best_feasible_si_db = info["SI_dB"]
                best_feasible_info = info.copy()
        elif violation < fallback_violation or (np.isclose(violation, fallback_violation) and info["SI_dB"] < fallback_si_db):
            fallback_violation = violation
            fallback_si_db = info["SI_dB"]
            fallback_info = info.copy()

    if best_feasible_info is not None:
        best_feasible_info["selection_mode"] = "gain-feasible best SI"
        best_feasible_info["gain_violation_score"] = 0.0
        return best_feasible_si_db, best_feasible_info

    if fallback_info is None:
        raise RuntimeError("RL rollout did not produce a valid focus state.")
    fallback_info["selection_mode"] = "lowest gain violation fallback"
    fallback_info["gain_violation_score"] = float(fallback_violation)
    return fallback_si_db, fallback_info


def _history_value(best_feasible_si_db, fallback_si_db):
    if np.isfinite(best_feasible_si_db):
        return float(best_feasible_si_db)
    if np.isfinite(fallback_si_db):
        return float(fallback_si_db)
    return np.nan


def _final_info(env, global_pos, best_feasible_info, best_feasible_si_db, fallback_info, fallback_si_db, fallback_violation, config, selection_mode):
    if best_feasible_info is not None:
        best_feasible_info["pso_objective_score"] = float(best_feasible_si_db)
        best_feasible_info["selection_mode"] = selection_mode
        best_feasible_info["gain_violation_score"] = 0.0
        return best_feasible_si_db, best_feasible_info

    if fallback_info is None:
        fallback_info = _info_from_focus_vector(env, global_pos)
        fallback_violation = _gain_violation_score(env, fallback_info)
        fallback_si_db = fallback_info["SI_dB"]
    fallback_info["selection_mode"] = "lowest gain violation fallback"
    fallback_info["gain_violation_score"] = float(fallback_violation)
    fallback_info["pso_objective_score"] = float(fallback_si_db + config["penalty_db"] * fallback_violation)
    return fallback_si_db, fallback_info


def trace_full_pso(env, case, seed, config):
    pso_config = DEFAULT_PSO_CONFIG.copy()
    pso_config.update(config or {})

    rng = np.random.default_rng(seed)
    env.reset_fixed(**case)
    si_ref_db = env.compute_SI_reference_dB()
    lower, upper = _focus_vector_bounds(env)
    span = upper - lower
    num_particles = int(pso_config["num_particles"])
    num_iters = int(pso_config["num_iters"])

    particles = rng.uniform(lower, upper, size=(num_particles, lower.size))
    particles[0] = np.array(
        [env.r_dl_user, env.phi_dl_user, env.theta_dl_user, env.r_ul_user, env.phi_ul_user, env.theta_ul_user],
        dtype=np.float64,
    )
    velocities = rng.uniform(-0.10 * span, 0.10 * span, size=particles.shape)

    personal_pos = particles.copy()
    personal_score = np.full(num_particles, np.inf, dtype=np.float64)
    global_pos = particles[0].copy()
    global_score = np.inf
    best_feasible_info = None
    best_feasible_si_db = np.inf
    fallback_info = None
    fallback_si_db = np.inf
    fallback_violation = np.inf
    history = []

    for _ in range(num_iters):
        for idx in range(num_particles):
            score, violation, info = _objective(env, particles[idx], pso_config)
            if score < personal_score[idx]:
                personal_score[idx] = score
                personal_pos[idx] = particles[idx].copy()
            if score < global_score:
                global_score = score
                global_pos = particles[idx].copy()
            if violation <= 0.0:
                if info["SI_dB"] < best_feasible_si_db:
                    best_feasible_si_db = info["SI_dB"]
                    best_feasible_info = info.copy()
            elif violation < fallback_violation or (np.isclose(violation, fallback_violation) and info["SI_dB"] < fallback_si_db):
                fallback_violation = violation
                fallback_si_db = info["SI_dB"]
                fallback_info = info.copy()

        history.append(_history_value(best_feasible_si_db, fallback_si_db))
        rp = rng.random(size=particles.shape)
        rg = rng.random(size=particles.shape)
        velocities = (
            pso_config["inertia"] * velocities
            + pso_config["cognitive"] * rp * (personal_pos - particles)
            + pso_config["social"] * rg * (global_pos - particles)
        )
        velocities = np.clip(velocities, -0.25 * span, 0.25 * span)
        particles = np.clip(particles + velocities, lower, upper)

    best_si_db, best_info = _final_info(
        env,
        global_pos,
        best_feasible_info,
        best_feasible_si_db,
        fallback_info,
        fallback_si_db,
        fallback_violation,
        pso_config,
        "gain-feasible best SI",
    )
    return np.asarray(history, dtype=np.float64), _build_result(env, case, seed, si_ref_db, best_si_db, best_info)


def trace_rl_initialized_pso(env, case, rl_best_info, seed, config):
    pso_config = DEFAULT_PSO_CONFIG.copy()
    pso_config.update(RL_INIT_PSO_CONFIG)
    pso_config.update(config or {})

    rng = np.random.default_rng(seed)
    env.reset_fixed(**case)
    si_ref_db = env.compute_SI_reference_dB()
    initial_focus = focus_vector_from_info(rl_best_info)
    global_lower, global_upper = _focus_vector_bounds(env)
    lower, upper = _local_bounds(env, initial_focus, pso_config)
    span = np.maximum(upper - lower, 1e-12)
    num_particles = int(pso_config["num_particles"])
    num_iters = int(pso_config["num_iters"])

    particles = rng.uniform(lower, upper, size=(num_particles, lower.size))
    particles[0] = np.clip(initial_focus, global_lower, global_upper)
    velocities = rng.uniform(-0.10 * span, 0.10 * span, size=particles.shape)

    personal_pos = particles.copy()
    personal_score = np.full(num_particles, np.inf, dtype=np.float64)
    global_pos = particles[0].copy()
    global_score = np.inf
    best_feasible_info = None
    best_feasible_si_db = np.inf
    fallback_info = None
    fallback_si_db = np.inf
    fallback_violation = np.inf
    history = []

    for _ in range(num_iters):
        for idx in range(num_particles):
            score, violation, info = _objective(env, particles[idx], pso_config)
            if score < personal_score[idx]:
                personal_score[idx] = score
                personal_pos[idx] = particles[idx].copy()
            if score < global_score:
                global_score = score
                global_pos = particles[idx].copy()
            if violation <= 0.0:
                if info["SI_dB"] < best_feasible_si_db:
                    best_feasible_si_db = info["SI_dB"]
                    best_feasible_info = info.copy()
            elif violation < fallback_violation or (np.isclose(violation, fallback_violation) and info["SI_dB"] < fallback_si_db):
                fallback_violation = violation
                fallback_si_db = info["SI_dB"]
                fallback_info = info.copy()

        history.append(_history_value(best_feasible_si_db, fallback_si_db))
        rp = rng.random(size=particles.shape)
        rg = rng.random(size=particles.shape)
        velocities = (
            pso_config["inertia"] * velocities
            + pso_config["cognitive"] * rp * (personal_pos - particles)
            + pso_config["social"] * rg * (global_pos - particles)
        )
        velocities = np.clip(velocities, -0.25 * span, 0.25 * span)
        particles = np.clip(particles + velocities, lower, upper)

    best_si_db, best_info = _final_info(
        env,
        global_pos,
        best_feasible_info,
        best_feasible_si_db,
        fallback_info,
        fallback_si_db,
        fallback_violation,
        pso_config,
        "RL-initialized local PSO feasible best SI",
    )
    return np.asarray(history, dtype=np.float64), _build_result(env, case, seed, si_ref_db, best_si_db, best_info)


def smooth_history(history, points_per_iter=12, window_fraction=0.08):
    history = np.asarray(history, dtype=np.float64)
    finite = np.isfinite(history)
    if history.size < 3 or np.count_nonzero(finite) < 3:
        x = np.arange(1, history.size + 1, dtype=np.float64)
        return x, history

    x = np.arange(1, history.size + 1, dtype=np.float64)
    x_finite = x[finite]
    y_finite = history[finite]
    x_dense = np.linspace(x_finite[0], x_finite[-1], max(history.size * points_per_iter, history.size))
    y_dense = np.interp(x_dense, x_finite, y_finite)

    window = max(3, int(round(y_dense.size * window_fraction)))
    if window % 2 == 0:
        window += 1
    if window >= y_dense.size:
        window = y_dense.size - 1 if y_dense.size % 2 == 0 else y_dense.size
    if window < 3:
        return x_dense, y_dense

    kernel = np.hanning(window)
    kernel = kernel / np.sum(kernel)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y_dense, (pad_left, pad_right), mode="edge")
    y_smooth = np.convolve(padded, kernel, mode="valid")

    y_smooth[0] = y_dense[0]
    y_smooth[-1] = y_dense[-1]
    return x_dense, y_smooth


def _summary_band(histories):
    values = np.asarray(histories, dtype=np.float64)
    return np.nanmean(values, axis=0), np.nanmin(values, axis=0), np.nanmax(values, axis=0)


def _full_pso_x(length, ref_x):
    if length <= 1:
        return np.full(length, ref_x, dtype=np.float64)
    return np.concatenate(([ref_x], np.arange(length - 1, dtype=np.float64)))


def _rlinit_pso_x(length, ref_x):
    if length <= 1:
        return np.full(length, ref_x, dtype=np.float64)
    return np.concatenate(([ref_x, 0.0], np.arange(1, length - 1, dtype=np.float64)))


def _smooth_full_after_ref(mean_values):
    x_smooth, y_smooth = smooth_history(mean_values[1:])
    return x_smooth - 1.0, y_smooth


def _smooth_rlinit_after_ref(mean_values):
    if mean_values.size <= 2:
        return np.arange(mean_values.size - 1, dtype=np.float64), mean_values[1:]
    iter_x, iter_y = smooth_history(mean_values[2:])
    return np.concatenate(([0.0], iter_x)), np.concatenate(([mean_values[1]], iter_y))


def plot_convergence(
    full_histories,
    rlpso_histories,
    matched_histories,
    ref_si_values,
    rl_init_si_values,
    matched_seed_history,
    matched_seed,
    output_path,
    seed_start,
    num_seeds,
    full_config,
    rlpso_config,
    matched_config,
    show_raw=True,
    title_note=None,
):
    os.makedirs(output_path.parent, exist_ok=True)
    plt.figure(figsize=(9, 5.2))
    ref_x = 0.0
    full_histories = [np.concatenate(([ref_si, ref_si], history)) for ref_si, history in zip(ref_si_values, full_histories)]
    rlpso_histories = [
        np.concatenate(([ref_si, rl_init_si], history))
        for ref_si, rl_init_si, history in zip(ref_si_values, rl_init_si_values, rlpso_histories)
    ]
    matched_histories = [
        np.concatenate(([ref_si, rl_init_si], history))
        for ref_si, rl_init_si, history in zip(ref_si_values, rl_init_si_values, matched_histories)
    ]
    full_mean, full_min, full_max = _summary_band(full_histories)
    rlpso_mean, rlpso_min, rlpso_max = _summary_band(rlpso_histories)
    matched_mean, matched_min, matched_max = _summary_band(matched_histories)
    full_x = _full_pso_x(len(full_mean), ref_x)
    rlpso_x = _rlinit_pso_x(len(rlpso_mean), ref_x)
    matched_x = _rlinit_pso_x(len(matched_mean), ref_x)
    full_smooth_x, full_smooth = _smooth_full_after_ref(full_mean)
    rlpso_smooth_x, rlpso_smooth = _smooth_rlinit_after_ref(rlpso_mean)
    matched_smooth_x, matched_smooth = _smooth_rlinit_after_ref(matched_mean)

    if show_raw:
        plt.step(full_x[1:], full_mean[1:], where="post", color="tab:blue", alpha=0.20, linewidth=1.1)
        plt.step(rlpso_x[1:], rlpso_mean[1:], where="post", color="tab:orange", alpha=0.20, linewidth=1.1)
        plt.step(matched_x[1:], matched_mean[1:], where="post", color="tab:green", alpha=0.20, linewidth=1.1)

    plt.fill_between(full_x, full_min, full_max, color="tab:blue", alpha=0.09)
    plt.fill_between(rlpso_x, rlpso_min, rlpso_max, color="tab:orange", alpha=0.09)
    plt.fill_between(matched_x, matched_min, matched_max, color="tab:green", alpha=0.09)
    matched_seed_x = _rlinit_pso_x(len(matched_seed_history), ref_x)
    plt.plot(
        matched_seed_x[1:],
        matched_seed_history[1:],
        color="tab:green",
        alpha=0.28,
        linewidth=1.0,
        label=f"Example seed {matched_seed} matched-budget trace",
    )

    full_particles = int(full_config["num_particles"])
    rlpso_particles = int(rlpso_config["num_particles"])
    matched_particles = int(matched_config["num_particles"])
    full_iters = int(full_config["num_iters"])
    rlpso_iters = int(rlpso_config["num_iters"])
    matched_iters = int(matched_config["num_iters"])
    plt.plot(
        full_smooth_x,
        full_smooth,
        color="tab:blue",
        linewidth=2.6,
        label=f"PSO ({full_iters} iters, {full_particles} particles)",
    )
    plt.plot(
        rlpso_smooth_x,
        rlpso_smooth,
        color="tab:orange",
        linewidth=2.6,
        label=f"RL-init PSO ({rlpso_iters} iters, {rlpso_particles} particles)",
    )
    plt.plot(
        matched_smooth_x,
        matched_smooth,
        color="tab:green",
        linewidth=2.6,
        label=f"RL-init PSO matched budget ({matched_iters} iters, {matched_particles} particles)",
    )
    max_iter = max(full_iters, rlpso_iters, matched_iters)
    tick_offsets = np.arange(20, max_iter + 1, 20)
    tick_positions = np.concatenate(([ref_x], tick_offsets))
    tick_labels = ["0 / Ref SI"] + [str(int(value)) for value in tick_offsets]
    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("PSO iteration")
    plt.ylabel("Mean SI [dB]")
    title = f"Monte Carlo Mean SI Convergence, seeds {seed_start}-{seed_start + num_seeds - 1}"
    if title_note:
        title = f"{title} ({title_note})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run(seed_start, num_seeds, output, show_raw):
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
    matched_seed_histories = []
    seed_values = []
    full_final_si = []
    rlpso_final_si = []
    matched_final_si = []
    for seed in range(seed_start, seed_start + num_seeds):
        env = make_env(hsi, seed)
        case = sample_3d_case(env, np.random.default_rng(seed), seed)
        rl_init_si_db, rl_best_info = rollout_best_focus(env, agent, case)
        full_history, full_result = trace_full_pso(env, case, seed, full_config)
        rlpso_history, rlpso_result = trace_rl_initialized_pso(env, case, rl_best_info, seed, rlpso_config)
        matched_history, matched_result = trace_rl_initialized_pso(env, case, rl_best_info, seed, matched_config)
        full_histories.append(full_history)
        rlpso_histories.append(rlpso_history)
        matched_histories.append(matched_history)
        ref_si_values.append(full_result["si_ref_db"])
        rl_init_si_values.append(rl_init_si_db)
        matched_seed_histories.append(np.concatenate(([full_result["si_ref_db"], rl_init_si_db], matched_history)))
        seed_values.append(seed)
        full_final_si.append(full_result["best_si_db"])
        rlpso_final_si.append(rlpso_result["best_si_db"])
        matched_final_si.append(matched_result["best_si_db"])
        print(
            f"seed {seed} | PSO {full_result['best_si_db']:.2f} dB | "
            f"RL-init {rlpso_result['best_si_db']:.2f} dB | "
            f"RL-init matched {matched_result['best_si_db']:.2f} dB"
        )

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = FIGURES_DIR / output_path
    rl_init_array = np.asarray(rl_init_si_values, dtype=np.float64)
    matched_final_array = np.asarray(matched_final_si, dtype=np.float64)
    init_scale = max(float(np.std(rl_init_array)), 1e-12)
    final_scale = max(float(np.std(matched_final_array)), 1e-12)
    representative_score = (
        np.abs(rl_init_array - np.mean(rl_init_array)) / init_scale
        + np.abs(matched_final_array - np.mean(matched_final_array)) / final_scale
    )
    representative_idx = int(np.argmin(representative_score))
    matched_seed_history = matched_seed_histories[representative_idx]
    matched_seed = seed_values[representative_idx]
    plot_convergence(
        full_histories,
        rlpso_histories,
        matched_histories,
        ref_si_values,
        rl_init_si_values,
        matched_seed_history,
        matched_seed,
        output_path,
        seed_start,
        num_seeds,
        full_config,
        rlpso_config,
        matched_config,
        show_raw=show_raw,
    )

    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Seeds: {seed_start}-{seed_start + num_seeds - 1} ({num_seeds} cases)")
    print(f"Saved convergence plot: {output_path}")
    print(f"PSO mean final best SI: {np.mean(full_final_si):.2f} +/- {np.std(full_final_si):.2f} dB")
    print(f"RL-init PSO mean final best SI: {np.mean(rlpso_final_si):.2f} +/- {np.std(rlpso_final_si):.2f} dB")
    print(f"RL-init PSO matched-budget mean final best SI: {np.mean(matched_final_si):.2f} +/- {np.std(matched_final_si):.2f} dB")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Monte Carlo mean best-SI convergence for PSO and RL-init PSO.")
    parser.add_argument("--seed", type=int, default=5000, help="First seed in the Monte Carlo evaluation range.")
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of consecutive Monte Carlo seeds to average.")
    parser.add_argument(
        "--output",
        type=str,
        default="pso_vs_rlinit_pso_convergence.png",
        help="Output PNG path. Relative paths are written under the figures directory.",
    )
    parser.add_argument("--hide-raw", action="store_true", help="Hide the faint raw step curves and show only smoothed curves.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seed, args.num_seeds, args.output, show_raw=not args.hide_raw)

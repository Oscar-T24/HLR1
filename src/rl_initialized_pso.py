"""RL-initialized local PSO for near-field focus optimization."""

from __future__ import annotations

import numpy as np

from .params import DEFAULT_PSO_CONFIG, RL_INIT_PSO_CONFIG
from .pso_optimizer import _build_result, _focus_vector_bounds, _gain_violation_score, _info_from_focus_vector, _objective


def focus_vector_from_info(info):
    return np.array(
        [
            info["r_tx_focus"],
            info["phi_tx_focus"],
            info["theta_tx_focus"],
            info["r_rx_focus"],
            info["phi_rx_focus"],
            info["theta_rx_focus"],
        ],
        dtype=np.float64,
    )


def _local_bounds(env, center, config):
    lower, upper = _focus_vector_bounds(env)
    radius = np.array(
        [
            config["radius_r"],
            config["radius_phi"],
            config["radius_theta"],
            config["radius_r"],
            config["radius_phi"],
            config["radius_theta"],
        ],
        dtype=np.float64,
    )
    return np.maximum(lower, center - radius), np.minimum(upper, center + radius)


def _run_local_swarm(env, initial_focus, seed, config):
    rng = np.random.default_rng(seed)
    global_lower, global_upper = _focus_vector_bounds(env)
    lower, upper = _local_bounds(env, initial_focus, config)
    span = np.maximum(upper - lower, 1e-12)
    num_particles = int(config["num_particles"])
    num_iters = int(config["num_iters"])

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

    for _ in range(num_iters):
        for idx in range(num_particles):
            score, violation, info = _objective(env, particles[idx], config)
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

        rp = rng.random(size=particles.shape)
        rg = rng.random(size=particles.shape)
        velocities = (
            config["inertia"] * velocities
            + config["cognitive"] * rp * (personal_pos - particles)
            + config["social"] * rg * (global_pos - particles)
        )
        velocities = np.clip(velocities, -0.25 * span, 0.25 * span)
        particles = np.clip(particles + velocities, lower, upper)

    if best_feasible_info is not None:
        best_feasible_info["pso_objective_score"] = float(best_feasible_si_db)
        best_feasible_info["selection_mode"] = "RL-initialized local PSO feasible best SI"
        best_feasible_info["gain_violation_score"] = 0.0
        return best_feasible_si_db, best_feasible_info

    if fallback_info is None:
        fallback_info = _info_from_focus_vector(env, global_pos)
        fallback_violation = _gain_violation_score(env, fallback_info)
        fallback_si_db = fallback_info["SI_dB"]
    fallback_info["selection_mode"] = "RL-initialized local PSO lowest violation fallback"
    fallback_info["gain_violation_score"] = float(fallback_violation)
    fallback_info["pso_objective_score"] = float(fallback_si_db + config["penalty_db"] * fallback_violation)
    return fallback_si_db, fallback_info


def run_rl_initialized_pso(env, case, rl_best_info, seed=None, config=None):
    """Run a short constrained PSO in a local box around the RL-selected focus."""
    pso_config = DEFAULT_PSO_CONFIG.copy()
    pso_config.update(RL_INIT_PSO_CONFIG)
    if config:
        pso_config.update(config)

    pso_seed = int(case["seed"] if seed is None else seed)
    env.reset_fixed(**case)
    si_ref_db = env.compute_SI_reference_dB()
    initial_focus = focus_vector_from_info(rl_best_info)
    best_si_db, best_info = _run_local_swarm(env, initial_focus, pso_seed, pso_config)
    return _build_result(env, case, pso_seed, si_ref_db, best_si_db, best_info)

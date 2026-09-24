"""Particle swarm baseline for near-field focus optimization."""

from __future__ import annotations

import numpy as np

from .params import DEFAULT_PSO_CONFIG
from .utils import near_field_channel_to_user, near_field_conjugate_weights


def _focus_vector_bounds(env):
    lower = np.array([env.r_min, env.phi_min, env.theta_min, env.r_min, env.phi_min, env.theta_min], dtype=np.float64)
    upper = np.array([env.r_max, env.phi_max, env.theta_max, env.r_max, env.phi_max, env.theta_max], dtype=np.float64)
    return lower, upper


def _focus_from_info(info, side):
    return {
        "r": float(info[f"r_{side}_focus"]),
        "phi": float(info[f"phi_{side}_focus"]),
        "theta": float(info[f"theta_{side}_focus"]),
    }


def _set_env_focus(env, focus_vector):
    r_tx, phi_tx, theta_tx, r_rx, phi_rx, theta_rx = np.asarray(focus_vector, dtype=np.float64)
    env.r_tx_focus = float(r_tx)
    env.phi_tx_focus = float(phi_tx)
    env.theta_tx_focus = float(theta_tx)
    env.r_rx_focus = float(r_rx)
    env.phi_rx_focus = float(phi_rx)
    env.theta_rx_focus = float(theta_rx)
    return r_tx, phi_tx, theta_tx, r_rx, phi_rx, theta_rx


def _info_from_focus_vector(env, focus_vector):
    r_tx, phi_tx, theta_tx, r_rx, phi_rx, theta_rx = _set_env_focus(env, focus_vector)
    si_db = env.compute_SI_dB(r_tx, phi_tx, theta_tx, r_rx, phi_rx, theta_rx)
    gain_metrics = env.compute_gain_metrics()
    info = {
        "SI_dB": float(si_db),
        "SI_ref_dB": float(env.compute_SI_reference_dB()),
        "DL_gain_margin_dB": float(env.dl_gain_margin_db),
        "UL_gain_margin_dB": float(env.ul_gain_margin_db),
        "r_tx_focus": float(r_tx),
        "phi_tx_focus": float(phi_tx),
        "theta_tx_focus": float(theta_tx),
        "r_rx_focus": float(r_rx),
        "phi_rx_focus": float(phi_rx),
        "theta_rx_focus": float(theta_rx),
        "r_dl_user": float(env.r_dl_user),
        "phi_dl_user": float(env.phi_dl_user),
        "theta_dl_user": float(env.theta_dl_user),
        "r_ul_user": float(env.r_ul_user),
        "phi_ul_user": float(env.phi_ul_user),
        "theta_ul_user": float(env.theta_ul_user),
    }
    info.update(gain_metrics)
    return info


def _gain_violation_score(env, info):
    dl_violation = max(0.0, info["DL_gain_loss_dB"] - env.dl_gain_margin_db)
    ul_violation = max(0.0, info["UL_gain_loss_dB"] - env.ul_gain_margin_db)
    return float(env.lambda_dl_gain * dl_violation + env.lambda_ul_gain * ul_violation)


def _objective(env, focus_vector, config):
    info = _info_from_focus_vector(env, focus_vector)
    violation = _gain_violation_score(env, info)
    return float(info["SI_dB"] + config["penalty_db"] * violation), violation, info


def _objective_from_info(env, info, config):
    violation = _gain_violation_score(env, info)
    return float(info["SI_dB"] + config["penalty_db"] * violation), violation


def _run_swarm(env, seed, config):
    rng = np.random.default_rng(seed)
    lower, upper = _focus_vector_bounds(env)
    span = upper - lower
    num_particles = int(config["num_particles"])
    num_iters = int(config["num_iters"])

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
        best_feasible_info["selection_mode"] = "gain-feasible best SI"
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


def _user_gain_db(env, pos_elem, focus, user):
    focus_sph = (focus["r"], focus["phi"], focus["theta"])
    user_sph = (user["r"], user["phi"], user["theta"])
    return env.compute_user_gain_dB(pos_elem, focus_sph, user_sph)


def _build_result(env, case, seed, si_ref_db, best_si_db, best_info):
    tx_user = {"r": float(case["r_dl_user"]), "phi": float(case["phi_dl_user"]), "theta": float(case["theta_dl_user"])}
    rx_user = {"r": float(case["r_ul_user"]), "phi": float(case["phi_ul_user"]), "theta": float(case["theta_ul_user"])}
    ref_tx_focus = tx_user.copy()
    ref_rx_focus = rx_user.copy()
    tx_focus = _focus_from_info(best_info, "tx")
    rx_focus = _focus_from_info(best_info, "rx")

    dl_gain_before_db = _user_gain_db(env, env.pos_tx, ref_tx_focus, tx_user)
    dl_gain_after_db = _user_gain_db(env, env.pos_tx, tx_focus, tx_user)
    ul_gain_before_db = _user_gain_db(env, env.pos_rx, ref_rx_focus, rx_user)
    ul_gain_after_db = _user_gain_db(env, env.pos_rx, rx_focus, rx_user)
    tx_to_ul_gain_db = _user_gain_db(env, env.pos_tx, tx_focus, rx_user)
    rx_to_dl_gain_db = _user_gain_db(env, env.pos_rx, rx_focus, tx_user)
    gain_violation = float(best_info.get("gain_violation_score", _gain_violation_score(env, best_info)))

    return {
        "seed": int(seed),
        "case": case,
        "si_ref_db": float(si_ref_db),
        "best_si_db": float(best_si_db),
        "si_improvement_db": float(si_ref_db - best_si_db),
        "dl_gain_loss_db": float(best_info["DL_gain_loss_dB"]),
        "ul_gain_loss_db": float(best_info["UL_gain_loss_dB"]),
        "selection_mode": best_info.get("selection_mode", "best SI"),
        "gain_violation_score": gain_violation,
        "is_gain_feasible": bool(gain_violation <= 0.0),
        "best_tx_focus": tx_focus,
        "best_rx_focus": rx_focus,
        "best_info": best_info,
        "pso_objective_score": float(best_info.get("pso_objective_score", best_si_db)),
        "dl_gain_before_db": float(dl_gain_before_db),
        "dl_gain_after_db": float(dl_gain_after_db),
        "dl_gain_change_db": float(dl_gain_after_db - dl_gain_before_db),
        "ul_gain_before_db": float(ul_gain_before_db),
        "ul_gain_after_db": float(ul_gain_after_db),
        "ul_gain_change_db": float(ul_gain_after_db - ul_gain_before_db),
        "tx_to_ul_gain_db": float(tx_to_ul_gain_db),
        "rx_to_dl_gain_db": float(rx_to_dl_gain_db),
        "dl_gain_margin_db": float(env.dl_gain_margin_db),
        "ul_gain_margin_db": float(env.ul_gain_margin_db),
    }


def run_pso_focus_optimization(env, case, seed=None, config=None):
    """Optimize Tx/Rx focus locations for low SI subject to DL/UL gain margins."""
    pso_config = DEFAULT_PSO_CONFIG.copy()
    if config:
        pso_config.update(config)

    pso_seed = int(case["seed"] if seed is None else seed)
    env.reset_fixed(**case)
    si_ref_db = env.compute_SI_reference_dB()
    best_si_db, best_info = _run_swarm(env, pso_seed, pso_config)
    return _build_result(env, case, pso_seed, si_ref_db, best_si_db, best_info)


def _focus_vector_from_result(result):
    tx = result["best_tx_focus"]
    rx = result["best_rx_focus"]
    return np.array([tx["r"], tx["phi"], tx["theta"], rx["r"], rx["phi"], rx["theta"]], dtype=np.float64)


def _direct_user_gain_db(pos_elem, focus, user, fc):
    weights = near_field_conjugate_weights(pos_elem, r=focus["r"], phi_deg=focus["phi"], theta_deg=focus["theta"], fc=fc)
    channel = near_field_channel_to_user(
        pos_elem,
        np.array([user["r"], user["theta"], user["phi"]], dtype=np.float64),
        fc,
    )
    gain = np.abs(channel @ weights) ** 2
    return float(10.0 * np.log10(max(gain, 1e-15)))


def recompute_pso_metrics(env, case, result):
    """Recompute the selected PSO solution directly from beamforming helpers."""
    tx_focus = result["best_tx_focus"]
    rx_focus = result["best_rx_focus"]
    tx_user = {"r": float(case["r_dl_user"]), "phi": float(case["phi_dl_user"]), "theta": float(case["theta_dl_user"])}
    rx_user = {"r": float(case["r_ul_user"]), "phi": float(case["phi_ul_user"]), "theta": float(case["theta_ul_user"])}
    ref_tx_focus = tx_user.copy()
    ref_rx_focus = rx_user.copy()

    w_tx = near_field_conjugate_weights(
        env.pos_tx,
        r=tx_focus["r"],
        phi_deg=tx_focus["phi"],
        theta_deg=tx_focus["theta"],
        fc=env.fc,
    )
    w_rx = near_field_conjugate_weights(
        env.pos_rx,
        r=rx_focus["r"],
        phi_deg=rx_focus["phi"],
        theta_deg=rx_focus["theta"],
        fc=env.fc,
    )
    si_linear = np.abs((w_rx.conj().T @ env.Hsi) @ w_tx)
    si_db = float(20.0 * np.log10(max(si_linear, 1e-15)))

    dl_gain_before_db = _direct_user_gain_db(env.pos_tx, ref_tx_focus, tx_user, env.fc)
    dl_gain_after_db = _direct_user_gain_db(env.pos_tx, tx_focus, tx_user, env.fc)
    ul_gain_before_db = _direct_user_gain_db(env.pos_rx, ref_rx_focus, rx_user, env.fc)
    ul_gain_after_db = _direct_user_gain_db(env.pos_rx, rx_focus, rx_user, env.fc)
    tx_to_ul_gain_db = _direct_user_gain_db(env.pos_tx, tx_focus, rx_user, env.fc)
    rx_to_dl_gain_db = _direct_user_gain_db(env.pos_rx, rx_focus, tx_user, env.fc)
    dl_gain_loss_db = float(dl_gain_before_db - dl_gain_after_db)
    ul_gain_loss_db = float(ul_gain_before_db - ul_gain_after_db)
    dl_violation = max(0.0, dl_gain_loss_db - env.dl_gain_margin_db)
    ul_violation = max(0.0, ul_gain_loss_db - env.ul_gain_margin_db)

    return {
        "SI_dB": si_db,
        "DL_gain_ref_dB": float(dl_gain_before_db),
        "DL_gain_dB": float(dl_gain_after_db),
        "DL_gain_loss_dB": dl_gain_loss_db,
        "UL_gain_ref_dB": float(ul_gain_before_db),
        "UL_gain_dB": float(ul_gain_after_db),
        "UL_gain_loss_dB": ul_gain_loss_db,
        "tx_to_ul_gain_db": float(tx_to_ul_gain_db),
        "rx_to_dl_gain_db": float(rx_to_dl_gain_db),
        "DL_feasible": bool(dl_gain_loss_db <= env.dl_gain_margin_db),
        "UL_feasible": bool(ul_gain_loss_db <= env.ul_gain_margin_db),
        "total_violation": float(env.lambda_dl_gain * dl_violation + env.lambda_ul_gain * ul_violation),
    }


def local_perturbation_check(env, case, result):
    env.reset_fixed(**case)
    focus = _focus_vector_from_result(result)
    lower, upper = _focus_vector_bounds(env)
    labels = ["tx_r", "tx_phi", "tx_theta", "rx_r", "rx_phi", "rx_theta"]
    steps = np.array([0.01, 0.1, 0.1, 0.01, 0.1, 0.1], dtype=np.float64)
    base_info = _info_from_focus_vector(env, focus)
    perturbations = []
    for idx, label in enumerate(labels):
        for direction in (-1.0, 1.0):
            candidate = focus.copy()
            candidate[idx] = np.clip(candidate[idx] + direction * steps[idx], lower[idx], upper[idx])
            info = _info_from_focus_vector(env, candidate)
            perturbations.append(
                {
                    "parameter": label,
                    "delta": float(direction * steps[idx]),
                    "SI_dB": float(info["SI_dB"]),
                    "delta_SI_dB": float(info["SI_dB"] - base_info["SI_dB"]),
                }
            )
    max_abs_delta = max(abs(item["delta_SI_dB"]) for item in perturbations)
    worst_si = max(item["SI_dB"] for item in perturbations)
    return {
        "base_si_db": float(base_info["SI_dB"]),
        "perturbations": perturbations,
        "max_abs_delta_si_db": float(max_abs_delta),
        "worst_perturbed_si_db": float(worst_si),
        "is_sharp_null": bool(max_abs_delta > 20.0 or worst_si - base_info["SI_dB"] > 20.0),
    }


def multi_run_pso_check(env, case, base_seed, num_runs=5, seed_stride=1009, config=None):
    runs = []
    for idx in range(num_runs):
        run_seed = int(base_seed + (idx + 1) * seed_stride)
        run_result = run_pso_focus_optimization(env, case, seed=run_seed, config=config)
        runs.append(
            {
                "seed": run_seed,
                "best_si_db": float(run_result["best_si_db"]),
                "tx_focus": run_result["best_tx_focus"],
                "rx_focus": run_result["best_rx_focus"],
                "is_gain_feasible": bool(run_result["is_gain_feasible"]),
            }
        )
    si_values = np.asarray([item["best_si_db"] for item in runs], dtype=np.float64)
    return {
        "runs": runs,
        "mean_best_si_db": float(np.mean(si_values)),
        "std_best_si_db": float(np.std(si_values)),
        "best_si_db": float(np.min(si_values)),
        "worst_si_db": float(np.max(si_values)),
        "is_unstable": bool(np.std(si_values) > 6.0 or np.max(si_values) - np.min(si_values) > 15.0),
    }


def random_sanity_check(env, case, seed, num_samples=1000, config=None):
    pso_config = DEFAULT_PSO_CONFIG.copy()
    if config:
        pso_config.update(config)
    env.reset_fixed(**case)
    rng = np.random.default_rng(seed)
    lower, upper = _focus_vector_bounds(env)

    best_feasible_info = None
    best_feasible_si_db = np.inf
    fallback_info = None
    fallback_si_db = np.inf
    fallback_violation = np.inf

    for idx in range(num_samples):
        if idx == 0:
            candidate = np.array(
                [env.r_dl_user, env.phi_dl_user, env.theta_dl_user, env.r_ul_user, env.phi_ul_user, env.theta_ul_user],
                dtype=np.float64,
            )
        else:
            candidate = rng.uniform(lower, upper)
        _, violation, info = _objective(env, candidate, pso_config)
        if violation <= 0.0:
            if info["SI_dB"] < best_feasible_si_db:
                best_feasible_si_db = info["SI_dB"]
                best_feasible_info = info.copy()
        elif violation < fallback_violation or (np.isclose(violation, fallback_violation) and info["SI_dB"] < fallback_si_db):
            fallback_violation = violation
            fallback_si_db = info["SI_dB"]
            fallback_info = info.copy()

    if best_feasible_info is not None:
        selected_info = best_feasible_info
        selected_info["gain_violation_score"] = 0.0
        selected_info["selection_mode"] = "gain-feasible best SI"
        best_si_db = best_feasible_si_db
    else:
        selected_info = fallback_info
        selected_info["gain_violation_score"] = float(fallback_violation)
        selected_info["selection_mode"] = "lowest gain violation fallback"
        best_si_db = fallback_si_db

    return {
        "num_samples": int(num_samples),
        "best_si_db": float(best_si_db),
        "best_info": selected_info,
        "is_gain_feasible": bool(selected_info["gain_violation_score"] <= 0.0),
    }


def validate_pso_result(env, case, result, num_stability_runs=5, num_random_samples=1000, config=None):
    pso_config = DEFAULT_PSO_CONFIG.copy()
    if config:
        pso_config.update(config)

    env.reset_fixed(**case)
    focus = _focus_vector_from_result(result)
    objective_info = _info_from_focus_vector(env, focus)
    objective_score, objective_violation = _objective_from_info(env, objective_info, pso_config)
    recomputed = recompute_pso_metrics(env, case, result)
    perturbation = local_perturbation_check(env, case, result)
    stability = multi_run_pso_check(env, case, result["seed"], num_runs=num_stability_runs, config=pso_config)
    random_check = random_sanity_check(env, case, result["seed"] + 7919, num_samples=num_random_samples, config=pso_config)

    stored_mismatch = max(
        abs(result["best_si_db"] - recomputed["SI_dB"]),
        abs(result["dl_gain_loss_db"] - recomputed["DL_gain_loss_dB"]),
        abs(result["ul_gain_loss_db"] - recomputed["UL_gain_loss_dB"]),
    )
    objective_matches_report = bool(np.isclose(objective_info["SI_dB"], result["best_si_db"], atol=1e-9))
    feasible = bool(recomputed["DL_feasible"] and recomputed["UL_feasible"] and recomputed["total_violation"] <= 0.0)
    random_gap_db = float(random_check["best_si_db"] - result["best_si_db"])
    stable = not stability["is_unstable"]
    valid = bool(stored_mismatch < 1e-6 and objective_matches_report and feasible and stable)

    warnings = []
    if stored_mismatch >= 1e-6:
        warnings.append("stored PSO metrics do not match independent recomputation")
    if not objective_matches_report:
        warnings.append("PSO objective SI metric differs from reported SI")
    if not feasible:
        warnings.append("final PSO focus violates the 3 dB DL/UL gain rule")
    if perturbation["is_sharp_null"]:
        warnings.append("local perturbations indicate a sharp numerical null")
    if stability["is_unstable"]:
        warnings.append("multi-run PSO results are unstable")
    if random_gap_db > 30.0:
        warnings.append("PSO is far better than random sanity samples; inspect for one-point artifact")

    return {
        "stored": {
            "SI_dB": float(result["best_si_db"]),
            "DL_gain_ref_dB": float(result["dl_gain_before_db"]),
            "DL_gain_dB": float(result["dl_gain_after_db"]),
            "DL_gain_loss_dB": float(result["dl_gain_loss_db"]),
            "UL_gain_ref_dB": float(result["ul_gain_before_db"]),
            "UL_gain_dB": float(result["ul_gain_after_db"]),
            "UL_gain_loss_dB": float(result["ul_gain_loss_db"]),
            "tx_to_ul_gain_db": float(result["tx_to_ul_gain_db"]),
            "rx_to_dl_gain_db": float(result["rx_to_dl_gain_db"]),
        },
        "recomputed": recomputed,
        "objective": {
            "reported_si_db": float(result["best_si_db"]),
            "objective_si_db": float(objective_info["SI_dB"]),
            "objective_score": float(objective_score),
            "stored_objective_score": float(result["pso_objective_score"]),
            "violation": float(objective_violation),
            "objective_matches_report": objective_matches_report,
        },
        "constraint": {
            "DL_gain_loss_dB": float(recomputed["DL_gain_loss_dB"]),
            "UL_gain_loss_dB": float(recomputed["UL_gain_loss_dB"]),
            "DL_feasible": bool(recomputed["DL_feasible"]),
            "UL_feasible": bool(recomputed["UL_feasible"]),
            "total_violation": float(recomputed["total_violation"]),
        },
        "perturbation": perturbation,
        "stability": stability,
        "random_sanity": random_check,
        "stored_recompute_max_abs_diff": float(stored_mismatch),
        "random_gap_db": random_gap_db,
        "warnings": warnings,
        "appears_valid": valid and not warnings,
    }

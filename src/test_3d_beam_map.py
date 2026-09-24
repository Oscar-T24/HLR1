"""Standalone 3D beam-map test for a trained SAC near-field model.

This module is intentionally separate from train.py. It samples deterministic
3D DL/UL cases over a seed range, rolls out the saved actor checkpoint, reports
average SI/user-gain diagnostics, and saves a simple 3D geometry figure for the
first seed's users and RL/PSO focus beams.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from .config import CHECKPOINT_DIR, ENV_CONFIG, FIGURES_DIR, HFSS_FILE, SAC_CONFIG, TEST_3D_CONFIG
from .env import NearFieldBeamFocusingEnv
from .pso_optimizer import run_pso_focus_optimization, validate_pso_result
from .rl_initialized_pso import run_rl_initialized_pso
from .sac_agent import SACAgent
from .utils import load_hfss_channel, near_field_channel_to_user, near_field_conjugate_weights, sph_to_cart

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    si_ref_db = env.compute_SI_reference_dB()

    best_feasible_info = None
    best_feasible_si_db = np.inf
    fallback_info = None
    fallback_si_db = np.inf
    fallback_violation = np.inf
    final_info = None
    done = False

    while not done:
        action = agent.choose_action(obs, evaluate=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_info = info.copy()
        dl_violation = max(0.0, info["DL_gain_loss_dB"] - env.dl_gain_margin_db)
        ul_violation = max(0.0, info["UL_gain_loss_dB"] - env.ul_gain_margin_db)
        violation = env.lambda_dl_gain * dl_violation + env.lambda_ul_gain * ul_violation
        if violation <= 0.0:
            if info["SI_dB"] < best_feasible_si_db:
                best_feasible_si_db = info["SI_dB"]
                best_feasible_info = info.copy()
        elif violation < fallback_violation or (np.isclose(violation, fallback_violation) and info["SI_dB"] < fallback_si_db):
            fallback_violation = violation
            fallback_si_db = info["SI_dB"]
            fallback_info = info.copy()

    if final_info is None:
        raise RuntimeError("Policy rollout produced no environment steps.")
    if best_feasible_info is not None:
        best_feasible_info["selection_mode"] = "gain-feasible best SI"
        best_feasible_info["gain_violation_score"] = 0.0
        return si_ref_db, best_feasible_si_db, best_feasible_info, final_info
    fallback_info["selection_mode"] = "lowest gain violation fallback"
    fallback_info["gain_violation_score"] = float(fallback_violation)
    return si_ref_db, fallback_si_db, fallback_info, final_info


def evaluate_seed_case(env, agent, seed):
    rng = np.random.default_rng(seed)
    case = sample_3d_case(env, rng, seed)
    start = time.perf_counter()
    si_ref_db, best_si_db, best_info, final_info = rollout_best_focus(env, agent, case)
    runtime_s = time.perf_counter() - start
    return {
        "seed": int(seed),
        "case": case,
        "si_ref_db": float(si_ref_db),
        "best_si_db": float(best_si_db),
        "si_improvement_db": float(si_ref_db - best_si_db),
        "dl_gain_loss_db": float(best_info["DL_gain_loss_dB"]),
        "ul_gain_loss_db": float(best_info["UL_gain_loss_dB"]),
        "selection_mode": best_info.get("selection_mode", "best SI"),
        "gain_violation_score": float(best_info.get("gain_violation_score", 0.0)),
        "runtime_s": float(runtime_s),
        "best_info": best_info,
        "final_info": final_info,
    }


def evaluate_pso_seed_case(env, case):
    start = time.perf_counter()
    result = run_pso_focus_optimization(env, case, seed=case["seed"])
    result["runtime_s"] = float(time.perf_counter() - start)
    return result


def evaluate_rl_initialized_pso_seed_case(env, rl_result):
    start = time.perf_counter()
    result = run_rl_initialized_pso(env, rl_result["case"], rl_result["best_info"], seed=rl_result["seed"])
    result["runtime_s"] = float(time.perf_counter() - start)
    return result


def print_average_report(results, label="model"):
    si_improvements = np.asarray([item["si_improvement_db"] for item in results], dtype=np.float64)
    dl_gain_losses = np.asarray([item["dl_gain_loss_db"] for item in results], dtype=np.float64)
    ul_gain_losses = np.asarray([item["ul_gain_loss_db"] for item in results], dtype=np.float64)
    feasible_count = sum(item["gain_violation_score"] <= 0.0 for item in results)

    print(f"{len(results)}-seed {label} summary:")
    print(f"  Seeds evaluated:          {results[0]['seed']}..{results[-1]['seed']} ({len(results)} cases)")
    print(f"  Gain-feasible selections: {feasible_count}/{len(results)}")
    print()
    print("Average selected-state metrics:")
    print(f"  SI improvement:    mean {np.mean(si_improvements):7.2f} dB | std {np.std(si_improvements):6.2f} dB")
    print(f"  DL gain loss:      mean {np.mean(dl_gain_losses):7.2f} dB | std {np.std(dl_gain_losses):6.2f} dB")
    print(f"  UL gain loss:      mean {np.mean(ul_gain_losses):7.2f} dB | std {np.std(ul_gain_losses):6.2f} dB")
    print(f"  DL/UL margins:     {ENV_CONFIG['dl_gain_margin_db']:.1f} dB / {ENV_CONFIG['ul_gain_margin_db']:.1f} dB")
    print()


def summarize_results(results):
    si_improvements = np.asarray([item["si_improvement_db"] for item in results], dtype=np.float64)
    dl_gain_losses = np.asarray([item["dl_gain_loss_db"] for item in results], dtype=np.float64)
    ul_gain_losses = np.asarray([item["ul_gain_loss_db"] for item in results], dtype=np.float64)
    feasible_count = sum(item["gain_violation_score"] <= 0.0 for item in results)
    best_si_values = np.asarray([item["best_si_db"] for item in results], dtype=np.float64)
    runtimes = np.asarray([item.get("runtime_s", np.nan) for item in results], dtype=np.float64)
    return {
        "count": len(results),
        "feasible_count": feasible_count,
        "mean_si": float(np.mean(best_si_values)),
        "mean_si_improvement": float(np.mean(si_improvements)),
        "std_si_improvement": float(np.std(si_improvements)),
        "mean_dl_gain_loss": float(np.mean(dl_gain_losses)),
        "mean_ul_gain_loss": float(np.mean(ul_gain_losses)),
        "mean_runtime_ms": float(1000.0 * np.nanmean(runtimes)),
        "total_runtime_s": float(np.nansum(runtimes)),
    }


def print_compact_summary(results, pso_results, rlpso_results, selected_freq):
    rl = summarize_results(results)
    pso = summarize_results(pso_results)
    rlpso = summarize_results(rlpso_results)
    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Seeds: {results[0]['seed']}..{results[-1]['seed']} ({len(results)} cases)")
    print(f"Gain-loss margins: DL {ENV_CONFIG['dl_gain_margin_db']:.1f} dB / UL {ENV_CONFIG['ul_gain_margin_db']:.1f} dB")
    print()
    print("Summary")
    print("  Method        Feasible   Mean SI        Mean SI reduction   Mean DL/UL loss    Total time")
    print(
        f"  RL            {rl['feasible_count']:3d}/{rl['count']:<3d}   "
        f"{rl['mean_si']:11.2f} dB   "
        f"{rl['mean_si_improvement']:7.2f} +/- {rl['std_si_improvement']:<5.2f} dB   "
        f"{rl['mean_dl_gain_loss']:5.2f}/{rl['mean_ul_gain_loss']:<5.2f} dB   "
        f"{rl['total_runtime_s']:8.2f} s"
    )
    print(
        f"  PSO           {pso['feasible_count']:3d}/{pso['count']:<3d}   "
        f"{pso['mean_si']:11.2f} dB   "
        f"{pso['mean_si_improvement']:7.2f} +/- {pso['std_si_improvement']:<5.2f} dB   "
        f"{pso['mean_dl_gain_loss']:5.2f}/{pso['mean_ul_gain_loss']:<5.2f} dB   "
        f"{pso['total_runtime_s']:8.2f} s"
    )
    print(
        f"  RL-init PSO  {rlpso['feasible_count']:3d}/{rlpso['count']:<3d}   "
        f"{rlpso['mean_si']:11.2f} dB   "
        f"{rlpso['mean_si_improvement']:7.2f} +/- {rlpso['std_si_improvement']:<5.2f} dB   "
        f"{rlpso['mean_dl_gain_loss']:5.2f}/{rlpso['mean_ul_gain_loss']:<5.2f} dB   "
        f"{rlpso['total_runtime_s']:8.2f} s"
    )


def _compact_focus(focus):
    return f"Tx({focus['tx']['r']:.2f}m,{focus['tx']['phi']:.1f}deg,{focus['tx']['theta']:.1f}deg) Rx({focus['rx']['r']:.2f}m,{focus['rx']['phi']:.1f}deg,{focus['rx']['theta']:.1f}deg)"


def print_compact_case_comparison(
    case,
    rl_result,
    pso_result,
    rlpso_result,
    rl_tx_focus,
    rl_rx_focus,
    pso_tx_focus,
    pso_rx_focus,
    rlpso_tx_focus,
    rlpso_rx_focus,
    total_times,
):
    print()
    print(f"Plotted seed: {rl_result['seed']}")
    print(
        "  Users: "
        f"DL r={case['r_dl_user']:.2f}m phi={case['phi_dl_user']:.1f}deg theta={case['theta_dl_user']:.1f}deg | "
        f"UL r={case['r_ul_user']:.2f}m phi={case['phi_ul_user']:.1f}deg theta={case['theta_ul_user']:.1f}deg"
    )
    print("  Method       Ref SI     Best SI    SI reduction   DL/UL loss   Violation   Total time")
    print(
        f"  RL       {rl_result['si_ref_db']:8.2f}  {rl_result['best_si_db']:8.2f}  "
        f"{rl_result['si_improvement_db']:7.2f}   "
        f"{rl_result['dl_gain_loss_db']:5.2f}/{rl_result['ul_gain_loss_db']:<5.2f}   "
        f"{rl_result['gain_violation_score']:.2f}      "
        f"{total_times['rl']:.2f} s"
    )
    print(
        f"  PSO      {pso_result['si_ref_db']:8.2f}  {pso_result['best_si_db']:8.2f}  "
        f"{pso_result['si_improvement_db']:7.2f}   "
        f"{pso_result['dl_gain_loss_db']:5.2f}/{pso_result['ul_gain_loss_db']:<5.2f}   "
        f"{pso_result['gain_violation_score']:.2f}      "
        f"{total_times['pso']:.2f} s"
    )
    print(
        f"  RL+PSO   {rlpso_result['si_ref_db']:8.2f}  {rlpso_result['best_si_db']:8.2f}  "
        f"{rlpso_result['si_improvement_db']:7.2f}   "
        f"{rlpso_result['dl_gain_loss_db']:5.2f}/{rlpso_result['ul_gain_loss_db']:<5.2f}   "
        f"{rlpso_result['gain_violation_score']:.2f}      "
        f"{total_times['rlpso']:.2f} s"
    )
    print(
        "  RL focus:  "
        + _compact_focus({"tx": rl_tx_focus, "rx": rl_rx_focus})
    )
    print(
        "  PSO focus: "
        + _compact_focus({"tx": pso_tx_focus, "rx": pso_rx_focus})
    )
    print(
        "  RL+PSO focus: "
        + _compact_focus({"tx": rlpso_tx_focus, "rx": rlpso_rx_focus})
    )


def focus_from_info(info, side):
    return {
        "r": float(info[f"r_{side}_focus"]),
        "phi": float(info[f"phi_{side}_focus"]),
        "theta": float(info[f"theta_{side}_focus"]),
    }


def user_from_case(case, role):
    return {
        "r": float(case[f"r_{role}_user"]),
        "phi": float(case[f"phi_{role}_user"]),
        "theta": float(case[f"theta_{role}_user"]),
    }


def user_gain_db(pos_elem, focus, user_sph, fc):
    w = near_field_conjugate_weights(
        pos_elem,
        r=focus["r"],
        phi_deg=focus["phi"],
        theta_deg=focus["theta"],
        fc=fc,
    )
    h = near_field_channel_to_user(
        pos_elem,
        np.array([user_sph["r"], user_sph["theta"], user_sph["phi"]], dtype=np.float64),
        fc,
    )
    gain = np.abs(h @ w) ** 2
    return float(10.0 * np.log10(max(gain, 1e-15)))


def point_cart(point):
    return sph_to_cart(point["r"], point["theta"], point["phi"])


def spherical_angle_deg(point_a, point_b):
    cart_a = point_cart(point_a)
    cart_b = point_cart(point_b)
    norm_a = max(float(np.linalg.norm(cart_a)), 1e-12)
    norm_b = max(float(np.linalg.norm(cart_b)), 1e-12)
    cosine = np.dot(cart_a, cart_b) / (norm_a * norm_b)
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def estimate_3db_cone_aperture(pos_elem, focus, served_user, fc, margin_db, env, angular_steps):
    """Estimate the local main-lobe cone that stays within margin_db."""
    focus_gain_db = user_gain_db(pos_elem, focus, focus, fc)
    local_span_deg = max(18.0, 1.5 * env.bwfn_deg)
    phi_samples = np.linspace(
        max(env.phi_min, focus["phi"] - local_span_deg),
        min(env.phi_max, focus["phi"] + local_span_deg),
        max(37, angular_steps),
    )
    theta_samples = np.linspace(
        max(env.theta_min, focus["theta"] - local_span_deg),
        min(env.theta_max, focus["theta"] + local_span_deg),
        max(25, angular_steps // 2),
    )

    valid_angles = [0.0]
    for phi_value in phi_samples:
        for theta_value in theta_samples:
            sample = {"r": focus["r"], "phi": float(phi_value), "theta": float(theta_value)}
            gain_db = user_gain_db(pos_elem, focus, sample, fc)
            if focus_gain_db - gain_db <= margin_db:
                valid_angles.append(spherical_angle_deg(focus, sample))

    served_gain_db = user_gain_db(pos_elem, focus, served_user, fc)
    if focus_gain_db - served_gain_db <= margin_db:
        valid_angles.append(spherical_angle_deg(focus, served_user))

    return max(1.0, float(np.max(valid_angles)))


def beam_cone_surface(focus, aperture_deg=8.0, length=None, radial_steps=28, angular_steps=48):
    focus_cart = point_cart(focus)
    focus_length = max(float(np.linalg.norm(focus_cart)), 1e-9)
    length = focus_length if length is None else max(float(length), focus_length)
    axis = focus_cart / focus_length

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, helper)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, helper)
    basis_u /= max(np.linalg.norm(basis_u), 1e-12)
    basis_v = np.cross(axis, basis_u)

    t = np.linspace(0.0, length, radial_steps)
    angle = np.linspace(0.0, 2.0 * np.pi, angular_steps)
    cone_radius = t * np.tan(np.deg2rad(aperture_deg))

    center = t[:, None, None] * axis[None, None, :]
    ring = cone_radius[:, None, None] * (
        np.cos(angle)[None, :, None] * basis_u[None, None, :]
        + np.sin(angle)[None, :, None] * basis_v[None, None, :]
    )
    surface = center + ring
    return surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], focus_cart


def beam_3db_cone_surface(pos_elem, focus, served_user, fc, margin_db, env, resolution):
    aperture_deg = estimate_3db_cone_aperture(pos_elem, focus, served_user, fc, margin_db, env, resolution)
    focus_gain_db = user_gain_db(pos_elem, focus, focus, fc)
    served_gain_db = user_gain_db(pos_elem, focus, served_user, fc)
    served_user_cart = point_cart(served_user)
    cone_length = focus["r"]
    if focus_gain_db - served_gain_db <= margin_db:
        cone_length = max(cone_length, float(np.linalg.norm(served_user_cart)))
    x, y, z, focus_cart = beam_cone_surface(
        focus,
        aperture_deg=aperture_deg,
        length=cone_length,
        angular_steps=resolution,
    )
    return x, y, z, focus_cart, aperture_deg


def set_equal_axes(ax, points):
    points = np.asarray(points, dtype=np.float64)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.62 * max(np.max(maxs - mins), 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_3d_geometry(ax, env, tx_user, rx_user, tx_focus, rx_focus, resolution, title):
    tx_x, tx_y, tx_z, tx_focus_cart, tx_aperture = beam_3db_cone_surface(
        env.pos_tx,
        tx_focus,
        tx_user,
        env.fc,
        env.dl_gain_margin_db,
        env,
        resolution,
    )
    rx_x, rx_y, rx_z, rx_focus_cart, rx_aperture = beam_3db_cone_surface(
        env.pos_rx,
        rx_focus,
        rx_user,
        env.fc,
        env.ul_gain_margin_db,
        env,
        resolution,
    )

    ax.plot_surface(tx_x, tx_y, tx_z, color="#2f80ed", linewidth=0, antialiased=True, alpha=0.24, shade=True)
    ax.plot_surface(rx_x, rx_y, rx_z, color="#f2994a", linewidth=0, antialiased=True, alpha=0.24, shade=True)
    ax.plot([0.0, tx_focus_cart[0]], [0.0, tx_focus_cart[1]], [0.0, tx_focus_cart[2]], color="#2f80ed", linewidth=2.2)
    ax.plot([0.0, rx_focus_cart[0]], [0.0, rx_focus_cart[1]], [0.0, rx_focus_cart[2]], color="#f2994a", linewidth=2.2)

    tx_user_cart = point_cart(tx_user)
    rx_user_cart = point_cart(rx_user)
    ax.scatter([tx_user_cart[0]], [tx_user_cart[1]], [tx_user_cart[2]], c="red", s=130, marker="*", depthshade=False)
    ax.scatter([rx_user_cart[0]], [rx_user_cart[1]], [rx_user_cart[2]], c="cyan", s=90, marker="o", edgecolors="black", depthshade=False)
    ax.scatter([tx_focus_cart[0]], [tx_focus_cart[1]], [tx_focus_cart[2]], c="#2f80ed", s=80, marker="x", depthshade=False)
    ax.scatter([rx_focus_cart[0]], [rx_focus_cart[1]], [rx_focus_cart[2]], c="#f2994a", s=80, marker="x", depthshade=False)
    ax.scatter([0.0], [0.0], [0.0], c="black", s=35, marker="s", depthshade=False)

    ax.text(tx_user_cart[0], tx_user_cart[1], tx_user_cart[2], "  DL user", color="red")
    ax.text(rx_user_cart[0], rx_user_cart[1], rx_user_cart[2], "  UL user", color="teal")
    ax.text(tx_focus_cart[0], tx_focus_cart[1], tx_focus_cart[2], f"  Tx focus\n  3 dB cone {tx_aperture:.1f} deg", color="#2f80ed")
    ax.text(rx_focus_cart[0], rx_focus_cart[1], rx_focus_cart[2], f"  Rx focus\n  3 dB cone {rx_aperture:.1f} deg", color="#b65f00")

    set_equal_axes(ax, [np.zeros(3), tx_user_cart, rx_user_cart, tx_focus_cart, rx_focus_cart])
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("x distance [m]")
    ax.set_ylabel("y lateral [m]")
    ax.set_zlabel("z height [m]")
    ax.set_box_aspect((1, 1, 0.75))
    ax.view_init(elev=22, azim=-52)


def result_title(label, result, total_time_s):
    return f"{label}\nBest SI {result['best_si_db']:.2f} dB | Total time {total_time_s:.2f} s"


def print_case_report(
    env,
    case,
    si_ref_db,
    best_si_db,
    best_info,
    tx_user,
    rx_user,
    ref_tx_focus,
    ref_rx_focus,
    rl_tx_focus,
    rl_rx_focus,
    focus_heading="Model-selected best focus points",
    si_label="RL best SI with model",
    after_label="after model",
    cross_heading="Cross-user beam response after model",
):
    dl_gain_ref_db = user_gain_db(env.pos_tx, ref_tx_focus, tx_user, env.fc)
    dl_gain_rl_db = user_gain_db(env.pos_tx, rl_tx_focus, tx_user, env.fc)
    ul_gain_ref_db = user_gain_db(env.pos_rx, ref_rx_focus, rx_user, env.fc)
    ul_gain_rl_db = user_gain_db(env.pos_rx, rl_rx_focus, rx_user, env.fc)
    tx_to_ul_gain_rl_db = user_gain_db(env.pos_tx, rl_tx_focus, rx_user, env.fc)
    rx_to_dl_gain_rl_db = user_gain_db(env.pos_rx, rl_rx_focus, tx_user, env.fc)

    print("Random 3D two-user full-duplex test case:")
    print(f"  Tx/DL user: r={case['r_dl_user']:.2f} m, phi={case['phi_dl_user']:.2f} deg, theta={case['theta_dl_user']:.2f} deg")
    print(f"  Rx/UL user: r={case['r_ul_user']:.2f} m, phi={case['phi_ul_user']:.2f} deg, theta={case['theta_ul_user']:.2f} deg")
    print()
    print(focus_heading + ":")
    print(f"  Tx focus: r={rl_tx_focus['r']:.3f} m, phi={rl_tx_focus['phi']:.2f} deg, theta={rl_tx_focus['theta']:.2f} deg")
    print(f"  Rx focus: r={rl_rx_focus['r']:.3f} m, phi={rl_rx_focus['phi']:.2f} deg, theta={rl_rx_focus['theta']:.2f} deg")
    print(f"  Selection rule: {best_info.get('selection_mode', 'best SI')}")
    if best_info.get("gain_violation_score", 0.0) > 0.0:
        print(f"  Gain violation score: {best_info['gain_violation_score']:.2f}")
    print()
    print("Self-interference comparison:")
    print(f"  Reference SI without model: {si_ref_db:.2f} dB")
    print(f"  {si_label}:".ljust(31) + f"{best_si_db:.2f} dB")
    print(f"  SI improvement:             {si_ref_db - best_si_db:.2f} dB")
    print()
    print("User gain comparison:")
    print(f"  DL gain before model: {dl_gain_ref_db:.2f} dB")
    print(f"  DL gain {after_label}:".ljust(25) + f"{dl_gain_rl_db:.2f} dB")
    print(f"  DL gain change:       {dl_gain_rl_db - dl_gain_ref_db:.2f} dB")
    print(f"  UL gain before model: {ul_gain_ref_db:.2f} dB")
    print(f"  UL gain {after_label}:".ljust(25) + f"{ul_gain_rl_db:.2f} dB")
    print(f"  UL gain change:       {ul_gain_rl_db - ul_gain_ref_db:.2f} dB")
    print()
    print(cross_heading + ":")
    print(f"  Tx beam at Rx/UL user direction: {tx_to_ul_gain_rl_db:.2f} dB")
    print(f"  Rx beam at Tx/DL user direction: {rx_to_dl_gain_rl_db:.2f} dB")


def print_pso_conclusion(rl_result, pso_result):
    delta_db = pso_result["best_si_db"] - rl_result["best_si_db"]
    print()
    print("RL vs PSO SI conclusion:")
    print(f"  PSO SI - RL SI: {delta_db:.2f} dB")
    if abs(delta_db) <= 1.0:
        print("  PSO and RL are within about 1 dB; focus-location-only optimization is likely saturated for this case.")
    elif delta_db < 0.0:
        print(f"  PSO improves over RL by {-delta_db:.2f} dB on the plotted case.")
    else:
        print(f"  RL is better than PSO by {delta_db:.2f} dB on the plotted case.")


def print_metric_pair(label, stored, recomputed):
    print(f"  {label:<28} stored {stored:9.3f} dB | recomputed {recomputed:9.3f} dB | diff {stored - recomputed:+.3e} dB")


def print_focus_line(prefix, tx_focus, rx_focus):
    print(
        f"  {prefix} Tx(r={tx_focus['r']:.3f}, phi={tx_focus['phi']:.2f}, theta={tx_focus['theta']:.2f}) | "
        f"Rx(r={rx_focus['r']:.3f}, phi={rx_focus['phi']:.2f}, theta={rx_focus['theta']:.2f})"
    )


def print_pso_validation_report(validation):
    stored = validation["stored"]
    recomputed = validation["recomputed"]
    objective = validation["objective"]
    constraint = validation["constraint"]
    perturbation = validation["perturbation"]
    stability = validation["stability"]
    random_sanity = validation["random_sanity"]

    print()
    print("PSO validation/debug checks:")
    print("  Stored PSO metrics vs independent recomputation:")
    print_metric_pair("SI", stored["SI_dB"], recomputed["SI_dB"])
    print_metric_pair("DL gain before", stored["DL_gain_ref_dB"], recomputed["DL_gain_ref_dB"])
    print_metric_pair("DL gain after", stored["DL_gain_dB"], recomputed["DL_gain_dB"])
    print_metric_pair("DL gain loss", stored["DL_gain_loss_dB"], recomputed["DL_gain_loss_dB"])
    print_metric_pair("UL gain before", stored["UL_gain_ref_dB"], recomputed["UL_gain_ref_dB"])
    print_metric_pair("UL gain after", stored["UL_gain_dB"], recomputed["UL_gain_dB"])
    print_metric_pair("UL gain loss", stored["UL_gain_loss_dB"], recomputed["UL_gain_loss_dB"])
    print_metric_pair("Tx beam at UL user", stored["tx_to_ul_gain_db"], recomputed["tx_to_ul_gain_db"])
    print_metric_pair("Rx beam at DL user", stored["rx_to_dl_gain_db"], recomputed["rx_to_dl_gain_db"])
    print(f"  Max stored/recomputed abs diff: {validation['stored_recompute_max_abs_diff']:.3e} dB")
    print()
    print("  PSO objective metric check:")
    print(f"    Reported SI:             {objective['reported_si_db']:.3f} dB")
    print(f"    Objective SI term:       {objective['objective_si_db']:.3f} dB")
    print(f"    Objective score:         {objective['objective_score']:.3f}")
    print(f"    Stored objective score:  {objective['stored_objective_score']:.3f}")
    print(f"    Objective violation:     {objective['violation']:.3f}")
    print(f"    Objective/report match:  {objective['objective_matches_report']}")
    print()
    print("  3 dB gain constraint check:")
    print(f"    DL gain loss:       {constraint['DL_gain_loss_dB']:.3f} dB")
    print(f"    UL gain loss:       {constraint['UL_gain_loss_dB']:.3f} dB")
    print(f"    DL feasible:        {constraint['DL_feasible']}")
    print(f"    UL feasible:        {constraint['UL_feasible']}")
    print(f"    Total violation:    {constraint['total_violation']:.3f}")
    print()
    print("  Local perturbation check around PSO optimum:")
    print(f"    Base SI: {perturbation['base_si_db']:.3f} dB")
    for item in perturbation["perturbations"]:
        print(
            f"    {item['parameter']:<8} {item['delta']:+.3f}: "
            f"SI {item['SI_dB']:9.3f} dB | change {item['delta_SI_dB']:+8.3f} dB"
        )
    if perturbation["is_sharp_null"]:
        print("    WARNING: tiny focus perturbations produce a large SI jump; this may be a fragile sharp null.")
    print()
    print("  Multi-run PSO stability check:")
    for idx, item in enumerate(stability["runs"], start=1):
        print(f"    Run {idx}: seed {item['seed']} | best SI {item['best_si_db']:.3f} dB | feasible {item['is_gain_feasible']}")
        print_focus_line("       ", item["tx_focus"], item["rx_focus"])
    print(f"    Mean/std best SI: {stability['mean_best_si_db']:.3f} / {stability['std_best_si_db']:.3f} dB")
    if stability["is_unstable"]:
        print("    WARNING: PSO result is unstable across internal seeds.")
    print()
    print("  Random sanity check:")
    random_focus_tx = focus_from_info(random_sanity["best_info"], "tx")
    random_focus_rx = focus_from_info(random_sanity["best_info"], "rx")
    print(f"    Samples evaluated: {random_sanity['num_samples']}")
    print(f"    Best random SI:    {random_sanity['best_si_db']:.3f} dB | feasible {random_sanity['is_gain_feasible']}")
    print(f"    Random - PSO SI:   {validation['random_gap_db']:.3f} dB")
    print_focus_line("Random best", random_focus_tx, random_focus_rx)
    print()
    if validation["warnings"]:
        print("  Validation warnings:")
        for warning in validation["warnings"]:
            print(f"    - {warning}")
    if validation["appears_valid"]:
        print("PSO validation conclusion: PSO result appears valid and reproducible.")
    else:
        print("PSO validation conclusion: PSO result may be unstable/numerical; inspect before using as benchmark.")


def shared_legend_handles():
    return [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="red", markeredgecolor="red", markersize=12, label="DL user"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="cyan", markeredgecolor="black", markersize=8, label="UL user"),
        Line2D([0], [0], color="#2f80ed", linewidth=3, label="Tx 3 dB cone/focus"),
        Line2D([0], [0], color="#f2994a", linewidth=3, label="Rx 3 dB cone/focus"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="black", markeredgecolor="black", markersize=6, label="Array origin"),
    ]


def run(seed, num_seeds, resolution, output_dir, show, validate_pso, verbose):
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    os.makedirs(output_dir, exist_ok=True)
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    env = make_env(hsi, seed=seed)
    agent = load_agent()

    seed_values = np.arange(seed, seed + num_seeds, dtype=int)
    results = [evaluate_seed_case(env, agent, int(case_seed)) for case_seed in seed_values]
    pso_results = [evaluate_pso_seed_case(env, item["case"]) for item in results]
    rlpso_results = [evaluate_rl_initialized_pso_seed_case(env, item) for item in results]
    plot_result = results[0]
    plot_pso_result = pso_results[0]
    plot_rlpso_result = rlpso_results[0]
    total_times = {
        "rl": float(sum(item["runtime_s"] for item in results)),
        "pso": float(sum(item["runtime_s"] for item in pso_results)),
        "rlpso": float(sum(item["runtime_s"] for item in rlpso_results)),
    }

    case = plot_result["case"]
    si_ref_db = plot_result["si_ref_db"]
    best_si_db = plot_result["best_si_db"]
    best_info = plot_result["best_info"]

    tx_user = user_from_case(case, "dl")
    rx_user = user_from_case(case, "ul")
    ref_tx_focus = tx_user.copy()
    ref_rx_focus = rx_user.copy()
    rl_tx_focus = focus_from_info(best_info, "tx")
    rl_rx_focus = focus_from_info(best_info, "rx")
    pso_best_info = plot_pso_result["best_info"]
    pso_tx_focus = focus_from_info(pso_best_info, "tx")
    pso_rx_focus = focus_from_info(pso_best_info, "rx")
    rlpso_best_info = plot_rlpso_result["best_info"]
    rlpso_tx_focus = focus_from_info(rlpso_best_info, "tx")
    rlpso_rx_focus = focus_from_info(rlpso_best_info, "rx")

    print_compact_summary(results, pso_results, rlpso_results, selected_freq)
    print_compact_case_comparison(
        case,
        plot_result,
        plot_pso_result,
        plot_rlpso_result,
        rl_tx_focus,
        rl_rx_focus,
        pso_tx_focus,
        pso_rx_focus,
        rlpso_tx_focus,
        rlpso_rx_focus,
        total_times,
    )
    if verbose:
        print()
        print("Detailed plotted-case report")
        print_case_report(env, case, si_ref_db, best_si_db, best_info, tx_user, rx_user, ref_tx_focus, ref_rx_focus, rl_tx_focus, rl_rx_focus)
        print()
        print_case_report(
            env,
            case,
            plot_pso_result["si_ref_db"],
            plot_pso_result["best_si_db"],
            pso_best_info,
            tx_user,
            rx_user,
            ref_tx_focus,
            ref_rx_focus,
            pso_tx_focus,
            pso_rx_focus,
            focus_heading="PSO-selected best focus points",
            si_label="PSO best SI",
            after_label="after PSO",
            cross_heading="Cross-user beam response after PSO",
        )
    print_pso_conclusion(plot_result, plot_pso_result)
    if validate_pso:
        validation = validate_pso_result(env, case, plot_pso_result)
        print_pso_validation_report(validation)

    fig = plt.figure(figsize=(22.0, 8.0))
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.04, top=0.80, wspace=0.02)
    ax_rl = fig.add_subplot(1, 3, 1, projection="3d")
    ax_pso = fig.add_subplot(1, 3, 2, projection="3d")
    ax_rlpso = fig.add_subplot(1, 3, 3, projection="3d")
    plot_3d_geometry(ax_rl, env, tx_user, rx_user, rl_tx_focus, rl_rx_focus, max(24, resolution), result_title("RL Focus Geometry", plot_result, total_times["rl"]))
    plot_3d_geometry(ax_pso, env, tx_user, rx_user, pso_tx_focus, pso_rx_focus, max(24, resolution), result_title("PSO Focus Geometry", plot_pso_result, total_times["pso"]))
    plot_3d_geometry(ax_rlpso, env, tx_user, rx_user, rlpso_tx_focus, rlpso_rx_focus, max(24, resolution), result_title("RL-Initialized PSO Geometry", plot_rlpso_result, total_times["rlpso"]))
    fig.legend(
        handles=shared_legend_handles(),
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.945),
        fontsize=14,
        markerscale=1.25,
        handlelength=2.2,
        columnspacing=1.6,
    )
    fig.suptitle(
        f"3D RL vs PSO vs RL-Initialized PSO Beam Geometry | 3 dB gain-restricted cones | Original SI {plot_result['si_ref_db']:.2f} dB",
        y=0.99,
        fontsize=20,
        fontweight="bold",
    )

    output_path = Path(output_dir) / "learned_model_vs_pso_vs_rlinit_pso_3d.png"
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved 3D beam map: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone 3D beam-map test for the trained SAC model.")
    parser.add_argument(
        "--seed",
        type=int,
        default=5003,
        help="First random seed for the sampled 3D DL/UL test cases and the case shown in the figure.",
    )
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of consecutive seeds to evaluate.")
    parser.add_argument("--resolution", type=int, default=64, help="Smoothness of the simple geometric beam cones.")
    parser.add_argument("--output-dir", default=str(FIGURES_DIR), help="Directory where the 3D figure is saved.")
    parser.add_argument("--show", action="store_true", help="Display the Matplotlib window after saving.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed plotted-case diagnostics.")
    parser.add_argument("--validate-pso", action="store_true", help="Print expensive PSO validation/debug checks for the plotted case.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args.seed, args.num_seeds, args.resolution, args.output_dir, args.show, args.validate_pso, args.verbose)
    except FileNotFoundError as exc:
        raise SystemExit(
            "No trained checkpoint or HFSS file was found. Run `python -m src.train` first "
            "and confirm data/Tx_1x24_RP_HFSSDesign1.s24p exists."
        ) from exc

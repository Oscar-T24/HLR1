"""Export a MATLAB-compatible 2D RL-init PSO sanity-check case.

The MATLAB sanity script accepts range/phi focus variables and holds theta at
zero. This helper therefore samples a theta-disabled case, uses the trained RL
policy for the local initialization center, and runs RL-init PSO with theta
locked to zero so the printed values can be checked by the fixed MATLAB script.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .config import ENV_CONFIG, HFSS_FILE
from .plot_pso_convergence import load_agent, make_env, rollout_best_focus
from .plot_pso_convergence_theta_disabled import sample_theta_disabled_case
from .pso_optimizer import _info_from_focus_vector
from .rl_initialized_pso import run_rl_initialized_pso
from .utils import load_hfss_channel


def _matlab_value(value):
    return f"{float(value):.12g}"


def _focus_line(label, focus):
    return (
        f"{label}: r={focus['r']:.6f} m, "
        f"phi={focus['phi']:.6f} deg, theta={focus['theta']:.6f} deg"
    )


def _project_rl_info_to_theta_zero(env, rl_best_info):
    projected = rl_best_info.copy()
    projected["theta_tx_focus"] = 0.0
    projected["theta_rx_focus"] = 0.0
    focus_vector = np.array(
        [
            projected["r_tx_focus"],
            projected["phi_tx_focus"],
            0.0,
            projected["r_rx_focus"],
            projected["phi_rx_focus"],
            0.0,
        ],
        dtype=np.float64,
    )
    projected.update(_info_from_focus_vector(env, focus_vector))
    projected["selection_mode"] = "theta-zero projection of RL focus"
    return projected


def print_report(seed, selected_freq, case, rl_info_2d, result, runtime_s):
    tx_focus = result["best_tx_focus"]
    rx_focus = result["best_rx_focus"]

    print(f"Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"Seed: {seed}")
    print("Mode: MATLAB-compatible theta-disabled RL-init PSO")
    print()
    print("Theta-disabled users:")
    print(
        f"  DL user: r={case['r_dl_user']:.6f} m, "
        f"phi={case['phi_dl_user']:.6f} deg, theta={case['theta_dl_user']:.6f} deg"
    )
    print(
        f"  UL user: r={case['r_ul_user']:.6f} m, "
        f"phi={case['phi_ul_user']:.6f} deg, theta={case['theta_ul_user']:.6f} deg"
    )
    print()
    print("RL initialization center after theta-zero projection:")
    print(
        f"  Tx init: r={rl_info_2d['r_tx_focus']:.6f} m, "
        f"phi={rl_info_2d['phi_tx_focus']:.6f} deg, theta={rl_info_2d['theta_tx_focus']:.6f} deg"
    )
    print(
        f"  Rx init: r={rl_info_2d['r_rx_focus']:.6f} m, "
        f"phi={rl_info_2d['phi_rx_focus']:.6f} deg, theta={rl_info_2d['theta_rx_focus']:.6f} deg"
    )
    print()
    print("RL-init PSO 2D result:")
    print(f"  {_focus_line('Tx focus', tx_focus)}")
    print(f"  {_focus_line('Rx focus', rx_focus)}")
    print(f"  Ref SI:          {result['si_ref_db']:.6f} dB")
    print(f"  Best SI:         {result['best_si_db']:.6f} dB")
    print(f"  SI improvement:  {result['si_improvement_db']:.6f} dB")
    print(f"  DL gain loss:    {result['dl_gain_loss_db']:.6f} dB")
    print(f"  UL gain loss:    {result['ul_gain_loss_db']:.6f} dB")
    print(f"  Violation score: {result['gain_violation_score']:.6f}")
    print(f"  Runtime:         {runtime_s:.3f} s")
    print()
    print("MATLAB values to use without changing NF_FD_PSO_final_Distance_it.m logic:")
    print("```matlab")
    print(f"desired_r_i = {_matlab_value(case['r_dl_user'])};")
    print(f"desired_phi_i = {_matlab_value(case['phi_dl_user'])};")
    print(f"desired_r_j = {_matlab_value(case['r_ul_user'])};")
    print(f"desired_phi_j = {_matlab_value(case['phi_ul_user'])};")
    print()
    print(f"r_i_opt = {_matlab_value(tx_focus['r'])};")
    print(f"phi_i_opt = {_matlab_value(tx_focus['phi'])};")
    print(f"r_j_opt = {_matlab_value(rx_focus['r'])};")
    print(f"phi_j_opt = {_matlab_value(rx_focus['phi'])};")
    print()
    print("% MATLAB spherical vectors are [r, theta, phi]; theta is fixed at 0.")
    print("user_i_desired = [desired_r_i, 0, desired_phi_i];")
    print("user_j_desired = [desired_r_j, 0, desired_phi_j];")
    print("user_i_sph_opt = [r_i_opt, 0, phi_i_opt];")
    print("user_j_sph_opt = [r_j_opt, 0, phi_j_opt];")
    print("```")
    print()
    print("Expected Python metrics for the MATLAB recomputation:")
    print(f"  SI_dB_ref ~= {result['si_ref_db']:.6f}")
    print(f"  SI_dB_opt ~= {result['best_si_db']:.6f}")
    print(f"  DL gain loss ~= {result['dl_gain_loss_db']:.6f}")
    print(f"  UL gain loss ~= {result['ul_gain_loss_db']:.6f}")


def run(seed):
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    env = make_env(hsi, seed)
    agent = load_agent()
    case = sample_theta_disabled_case(env, np.random.default_rng(seed), seed)

    _, rl_best_info = rollout_best_focus(env, agent, case)
    rl_info_2d = _project_rl_info_to_theta_zero(env, rl_best_info)

    start = time.perf_counter()
    result = run_rl_initialized_pso(
        env,
        case,
        rl_info_2d,
        seed=seed,
        config={"radius_theta": 0.0},
    )
    runtime_s = time.perf_counter() - start
    print_report(seed, selected_freq, case, rl_info_2d, result, runtime_s)


def parse_args():
    parser = argparse.ArgumentParser(description="Export a MATLAB-compatible 2D RL-init PSO sanity-check case.")
    parser.add_argument("--seed", type=int, default=5000, help="Deterministic theta-disabled case seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args.seed)
    except FileNotFoundError as exc:
        raise SystemExit(
            "No trained checkpoint or HFSS file was found. Run `python -m src.train` first "
            "and confirm data/Tx_1x24_RP_HFSSDesign1.s24p exists."
        ) from exc

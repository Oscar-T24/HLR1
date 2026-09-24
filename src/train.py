"""Training entry point for the near-field SAC beamforming project."""

from __future__ import annotations

import os

import numpy as np

from .config import CHECKPOINT_DIR, ENV_CONFIG, FIGURES_DIR, FIXED_VALIDATION_CASES, HFSS_FILE, SAC_CONFIG, TRAIN_CONFIG
from .env import NearFieldBeamFocusingEnv
from .plotting import plot_training_diagnostics
from .sac_agent import SACAgent
from .utils import load_hfss_channel

import matplotlib.pyplot as plt


def make_env(hsi, seed=None):
    env_config = ENV_CONFIG.copy()
    if seed is not None:
        env_config["seed"] = seed
    return NearFieldBeamFocusingEnv(Hsi=hsi, **env_config)


def evaluate_random_search_baseline(env, n_samples=5000, seed=2026):
    rng = np.random.default_rng(seed)
    si_values = np.zeros(n_samples, dtype=np.float64)
    random_actions = rng.uniform(-1.0, 1.0, size=(n_samples, env.action_space.shape[0]))
    for i in range(n_samples):
        if env.direct_action_mode:
            r_tx = env._action_to_range(random_actions[i, 0])
            phi_tx = env._action_to_angle(random_actions[i, 1], env.phi_min, env.phi_max)
            theta_tx = env._action_to_angle(random_actions[i, 2], env.theta_min, env.theta_max)
            r_rx = env._action_to_range(random_actions[i, 3])
            phi_rx = env._action_to_angle(random_actions[i, 4], env.phi_min, env.phi_max)
            theta_rx = env._action_to_angle(random_actions[i, 5], env.theta_min, env.theta_max)
        else:
            r_tx = rng.uniform(env.r_min, env.r_max)
            phi_tx = rng.uniform(env.phi_min, env.phi_max)
            theta_tx = rng.uniform(env.theta_min, env.theta_max)
            r_rx = rng.uniform(env.r_min, env.r_max)
            phi_rx = rng.uniform(env.phi_min, env.phi_max)
            theta_rx = rng.uniform(env.theta_min, env.theta_max)
        si_values[i] = env.compute_SI_dB(r_tx, phi_tx, theta_tx, r_rx, phi_rx, theta_rx)

    plt.figure(figsize=(7, 4))
    plt.hist(si_values, bins=60, edgecolor="black", alpha=0.75)
    plt.xlabel("SI [dB]")
    plt.ylabel("Count")
    plt.title("Random Search SI Landscape")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "random_search_si_histogram.png", dpi=200)
    plt.close()
    print("Random-search SI statistics:")
    print(f"  samples: {n_samples}")
    print(f"  min:     {np.min(si_values):.2f} dB")
    print(f"  max:     {np.max(si_values):.2f} dB")
    print(f"  mean:    {np.mean(si_values):.2f} dB")
    print(f"  std:     {np.std(si_values):.2f} dB")
    return si_values, random_actions


def evaluate_policy_fixed_cases(agent, env, fixed_cases):
    best_si_values = []
    final_si_values = []
    for case in fixed_cases:
        obs, _ = env.reset_fixed(**case)
        done = False
        best_feasible_si = np.inf
        fallback_si = np.inf
        fallback_violation = np.inf
        final_si = np.nan
        while not done:
            action = agent.choose_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_si = info["SI_dB"]
            dl_violation = max(0.0, info["DL_gain_loss_dB"] - env.dl_gain_margin_db)
            ul_violation = max(0.0, info["UL_gain_loss_dB"] - env.ul_gain_margin_db)
            violation = env.lambda_dl_gain * dl_violation + env.lambda_ul_gain * ul_violation
            if violation <= 0.0:
                best_feasible_si = min(best_feasible_si, final_si)
            elif violation < fallback_violation or (np.isclose(violation, fallback_violation) and final_si < fallback_si):
                fallback_violation = violation
                fallback_si = final_si
        best_si_values.append(best_feasible_si if np.isfinite(best_feasible_si) else fallback_si + 1000.0)
        final_si_values.append(final_si)
    return float(np.mean(best_si_values)), float(np.mean(final_si_values)), np.asarray(best_si_values), np.asarray(final_si_values)


def get_training_episode_count():
    return int(TRAIN_CONFIG["n_episodes"])


def train():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    print("Loaded Hsi shape:", hsi.shape)
    print("Selected frequency:", selected_freq / 1e9, "GHz")

    env = make_env(hsi, seed=1234)
    validation_env = make_env(hsi, seed=9999)
    print("Training mode: normal SAC")
    print(
        "Random user range: "
        f"r=[{env.user_r_min:.2f}, {env.user_r_max:.2f}] m, "
        f"phi=[{env.user_phi_min:.1f}, {env.user_phi_max:.1f}] deg, "
        f"theta=[{env.user_theta_min:.1f}, {env.user_theta_max:.1f}] deg"
    )
    evaluate_random_search_baseline(env, TRAIN_CONFIG["n_random_samples"], seed=2026)

    agent_config = SAC_CONFIG.copy()
    agent_config["chkpt_dir"] = str(CHECKPOINT_DIR)
    agent = SACAgent(**agent_config)

    history = {key: [] for key in [
        "score_history", "score_per_step_history", "si_history", "final_si_history", "mean_si_history",
        "validation_best_si_history", "validation_final_si_history", "validation_episode_history",
        "penalty_history", "boundary_penalty_history", "saturation_history",
        "action_magnitude_history", "regret_history", "dl_gain_loss_history",
        "ul_gain_loss_history", "episode_action_mean", "episode_action_min",
        "episode_action_max", "episode_state_history",
        "episode_reset_mode_history",
    ]}
    warmup_rng = np.random.default_rng(TRAIN_CONFIG["warmup_seed"])
    total_steps = 0
    best_validation_si = np.inf
    n_train_episodes = get_training_episode_count()

    for ep in range(n_train_episodes):
        reset_mode = "near_user"
        obs, _ = env.reset()

        done = False
        score = 0.0
        best_si_ep = np.inf
        penalty_ep = boundary_penalty_ep = action_magnitude_ep = regret_ep = 0.0
        dl_gain_loss_ep = ul_gain_loss_ep = 0.0
        step_count = 0
        actions_ep = []
        si_ep = []

        while not done:
            if total_steps < TRAIN_CONFIG["initial_random_steps"]:
                action = np.clip(
                    warmup_rng.normal(0.0, TRAIN_CONFIG["warmup_action_std"], size=SAC_CONFIG["n_actions"]),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            else:
                action = agent.choose_action(obs)
            actions_ep.append(action.copy())
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, obs_, done)
            if total_steps >= TRAIN_CONFIG["initial_random_steps"]:
                for _ in range(TRAIN_CONFIG["updates_per_step"]):
                    agent.learn()
            total_steps += 1

            obs = obs_
            score += reward
            dl_gain_feasible = info.get("DL_gain_loss_dB", 0.0) <= env.dl_gain_margin_db
            ul_gain_feasible = info.get("UL_gain_loss_dB", 0.0) <= env.ul_gain_margin_db
            if dl_gain_feasible and ul_gain_feasible:
                best_si_ep = min(best_si_ep, info["SI_dB"])
            si_ep.append(info["SI_dB"])
            penalty_ep += info.get("focus_penalty", 0.0)
            boundary_penalty_ep += info.get("boundary_penalty", 0.0)
            action_magnitude_ep += info.get("action_magnitude", 0.0)
            regret_ep += info.get("regret_SI_dB", 0.0)
            dl_gain_loss_ep += max(0.0, info.get("DL_gain_loss_dB", 0.0))
            ul_gain_loss_ep += max(0.0, info.get("UL_gain_loss_dB", 0.0))
            step_count += 1

        actions_ep = np.asarray(actions_ep, dtype=np.float32)
        history["score_history"].append(score)
        history["score_per_step_history"].append(score / max(step_count, 1))
        history["si_history"].append(best_si_ep if np.isfinite(best_si_ep) else np.nan)
        history["final_si_history"].append(si_ep[-1])
        history["mean_si_history"].append(np.mean(si_ep))
        history["penalty_history"].append(penalty_ep / max(step_count, 1))
        history["boundary_penalty_history"].append(boundary_penalty_ep / max(step_count, 1))
        history["action_magnitude_history"].append(action_magnitude_ep / max(step_count, 1))
        history["regret_history"].append(regret_ep / max(step_count, 1))
        history["dl_gain_loss_history"].append(dl_gain_loss_ep / max(step_count, 1))
        history["ul_gain_loss_history"].append(ul_gain_loss_ep / max(step_count, 1))
        history["saturation_history"].append(np.mean(np.abs(actions_ep) > 0.95))
        history["episode_action_mean"].append(actions_ep.mean(axis=0))
        history["episode_action_min"].append(actions_ep.min(axis=0))
        history["episode_action_max"].append(actions_ep.max(axis=0))
        history["episode_state_history"].append({"r_dl": info["r_tx_focus"], "theta_dl": info["theta_tx_focus"], "r_ul": info["r_rx_focus"], "theta_ul": info["theta_rx_focus"]})
        history["episode_reset_mode_history"].append(reset_mode)

        if (ep + 1) % TRAIN_CONFIG["validation_every"] == 0:
            validation_best_si, validation_final_si, _, _ = evaluate_policy_fixed_cases(agent, validation_env, FIXED_VALIDATION_CASES)
            history["validation_episode_history"].append(ep + 1)
            history["validation_best_si_history"].append(validation_best_si)
            history["validation_final_si_history"].append(validation_final_si)
            if validation_best_si < best_validation_si:
                best_validation_si = validation_best_si
                agent.save_models()
                print(f"Saved best model at episode {ep + 1}: validation best SI = {best_validation_si:.2f} dB")

        if (ep + 1) % 50 == 0:
            recent_best_si = np.asarray(history["si_history"][-50:], dtype=np.float64)
            avg_best_si = np.nanmean(recent_best_si) if np.any(np.isfinite(recent_best_si)) else np.nan
            print(
                f"Episode {ep + 1:4d} | "
                f"Reset = {reset_mode:14s} | "
                f"Avg score = {np.mean(history['score_history'][-50:]):.2f} | "
                f"Avg best SI = {avg_best_si:.2f} dB | "
                f"Avg gain loss DL/UL = {np.mean(history['dl_gain_loss_history'][-50:]):.2f}/{np.mean(history['ul_gain_loss_history'][-50:]):.2f} dB | "
                f"Validation SI = {(history['validation_best_si_history'][-1] if history['validation_best_si_history'] else np.nan):.2f} dB"
            )

    for key in ["episode_action_mean", "episode_action_min", "episode_action_max"]:
        history[key] = np.asarray(history[key])
    for key, values in list(history.items()):
        if key not in {"episode_action_mean", "episode_action_min", "episode_action_max", "episode_state_history", "episode_reset_mode_history"}:
            history[key] = np.asarray(values)

    plot_training_diagnostics(
        history,
        str(FIGURES_DIR),
        ENV_CONFIG["direct_action_mode"],
        ENV_CONFIG["dl_gain_margin_db"],
        ENV_CONFIG["ul_gain_margin_db"],
    )
    print("Training finished.")
    return history


if __name__ == "__main__":
    train()

"""Evaluation entry point for a trained SAC near-field beamforming model."""

from __future__ import annotations

import os
import sys

import numpy as np

from .config import CHECKPOINT_DIR, ENV_CONFIG, FIGURES_DIR, FIXED_VALIDATION_CASES, HFSS_FILE, SAC_CONFIG
from .env import NearFieldBeamFocusingEnv
from .sac_agent import SACAgent
from .utils import load_hfss_channel


def evaluate(num_cases=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    hsi, selected_freq = load_hfss_channel(HFSS_FILE, target_freq=ENV_CONFIG["fc"])
    env = NearFieldBeamFocusingEnv(Hsi=hsi, **ENV_CONFIG)
    agent_config = SAC_CONFIG.copy()
    agent_config["chkpt_dir"] = str(CHECKPOINT_DIR)
    agent = SACAgent(**agent_config)
    try:
        agent.load_models()
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            "No compatible current checkpoint was found. Run `python -m src.train` "
            "to create checkpoints for the active 19-dimensional observation model."
        ) from exc

    cases = FIXED_VALIDATION_CASES if num_cases is None else FIXED_VALIDATION_CASES[:num_cases]
    rewards = []
    final_si_values = []
    best_si_values = []

    for case in cases:
        obs, _ = env.reset_fixed(**case)
        done = False
        episode_reward = 0.0
        best_si = np.inf
        final_si = np.nan
        while not done:
            action = agent.choose_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            final_si = info["SI_dB"]
            best_si = min(best_si, final_si)
        rewards.append(episode_reward)
        final_si_values.append(final_si)
        best_si_values.append(best_si)

    print("Evaluation summary:")
    print(f"  Selected frequency: {selected_freq / 1e9:.4f} GHz")
    print(f"  Cases:              {len(cases)}")
    print(f"  Average reward:     {np.mean(rewards):.2f}")
    print(f"  Average SI:         {np.mean(final_si_values):.2f} dB")
    print(f"  Best SI:            {np.min(best_si_values):.2f} dB")
    return rewards, final_si_values, best_si_values


if __name__ == "__main__":
    try:
        evaluate()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

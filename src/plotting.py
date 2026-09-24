"""Plotting utilities for SAC training diagnostics."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from .utils import moving_average


def plot_reward_curve(score_history, figures_path):
    episodes = np.arange(1, len(score_history) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, score_history, alpha=0.45, label="raw")
    for window in (20, 50):
        ma = moving_average(score_history, window)
        if ma is not None:
            plt.plot(np.arange(window, len(score_history) + 1), ma, label=f"MA{window}")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("SAC Training Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "reward_curve.png"), dpi=200)
    plt.close()


def plot_reward_per_step_curve(score_per_step_history, figures_path):
    episodes = np.arange(1, len(score_per_step_history) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, score_per_step_history, alpha=0.45, label="raw")
    for window in (20, 50):
        ma = moving_average(score_per_step_history, window)
        if ma is not None:
            plt.plot(np.arange(window, len(score_per_step_history) + 1), ma, label=f"MA{window}")
    plt.xlabel("Episode")
    plt.ylabel("Average reward per step")
    plt.title("SAC Training Reward Per Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "reward_per_step_curve.png"), dpi=200)
    plt.close()


def plot_si_curve(history, figures_path):
    episodes = np.arange(1, len(history["score_history"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, history["si_history"], alpha=0.35, label="best raw")
    plt.plot(episodes, history["final_si_history"], alpha=0.35, label="final raw")
    plt.plot(episodes, history["mean_si_history"], alpha=0.35, label="mean raw")
    for window in (20, 50):
        ma = moving_average(history["si_history"], window)
        if ma is not None:
            plt.plot(np.arange(window, len(history["si_history"]) + 1), ma, label=f"best MA{window}")
    if len(history["validation_best_si_history"]) > 0:
        plt.plot(history["validation_episode_history"], history["validation_best_si_history"], "o-", linewidth=2, label="validation best")
        plt.plot(history["validation_episode_history"], history["validation_final_si_history"], "s-", linewidth=2, label="validation final")
    plt.xlabel("Episode")
    plt.ylabel("Best SI per episode [dB]")
    plt.title("SAC Training SI Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "si_curve.png"), dpi=200)
    plt.close()


def plot_training_diagnostics(history, figures_path, direct_action_mode=False, dl_gain_margin_db=1.0, ul_gain_margin_db=None):
    if ul_gain_margin_db is None:
        ul_gain_margin_db = dl_gain_margin_db
    os.makedirs(figures_path, exist_ok=True)
    plot_reward_curve(history["score_history"], figures_path)
    if "score_per_step_history" in history:
        plot_reward_per_step_curve(history["score_per_step_history"], figures_path)
    plot_si_curve(history, figures_path)

    episodes = np.arange(1, len(history["score_history"]) + 1)
    simple_plots = [
        ("saturation_history", "SAC Action Saturation", "Saturation fraction", "action_saturation.png", "|action| > 0.95"),
        ("action_magnitude_history", "SAC Action Magnitude", "Mean squared action", "action_magnitude.png", "mean action^2"),
        ("regret_history", "SAC Within-Episode SI Regret", "Regret [dB]", "si_regret.png", "current SI - best SI"),
        ("boundary_penalty_history", "SAC Boundary Penalty", "Penalty", "boundary_penalty.png", "boundary penalty"),
    ]
    for key, title, ylabel, filename, label in simple_plots:
        plt.figure(figsize=(8, 3.5))
        plt.plot(episodes, history[key], label=label)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        if key == "saturation_history":
            plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, filename), dpi=200)
        plt.close()

    plt.figure(figsize=(8, 3.5))
    plt.plot(episodes, history["dl_gain_loss_history"], label="DL gain loss")
    plt.plot(episodes, history["ul_gain_loss_history"], label="UL gain loss")
    plt.axhline(dl_gain_margin_db, color="tab:blue", linewidth=0.8, linestyle="--", label="DL margin")
    plt.axhline(ul_gain_margin_db, color="tab:orange", linewidth=0.8, linestyle="--", label="UL margin")
    plt.xlabel("Episode")
    plt.ylabel("Gain loss [dB]")
    plt.title("SAC User Gain Preservation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "gain_loss_curve.png"), dpi=200)
    plt.close()

    action_names = ["r_tx", "phi_tx", "theta_tx", "r_rx", "phi_rx", "theta_rx"] if direct_action_mode else ["dr_tx", "dphi_tx", "dtheta_tx", "dr_rx", "dphi_rx", "dtheta_rx"]
    episode_action_mean = history["episode_action_mean"]
    episode_action_min = history["episode_action_min"]
    episode_action_max = history["episode_action_max"]
    action_plot_window = 20
    action_plot_stride = max(1, len(episodes) // 600)
    plot_episodes = episodes[::action_plot_stride]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    for idx, ax in enumerate(axes.ravel()):
        ax.plot(episodes, episode_action_mean[:, idx], alpha=0.15, label="mean raw")
        action_ma = moving_average(episode_action_mean[:, idx], action_plot_window)
        if action_ma is not None:
            ax.plot(np.arange(action_plot_window, len(episodes) + 1), action_ma, linewidth=1.6, label=f"mean MA{action_plot_window}")
        ax.fill_between(plot_episodes, episode_action_min[::action_plot_stride, idx], episode_action_max[::action_plot_stride, idx], alpha=0.08, label="min-max")
        ax.axhline(-1.0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(action_names[idx])
        ax.set_ylabel("Normalized action [-1, 1]")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Episode")
    axes[-1, 1].set_xlabel("Episode")
    axes[0, 0].legend(loc="best")
    fig.suptitle("SAC Action Evolution")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_path, "action_evolution.png"), dpi=200)
    plt.close(fig)

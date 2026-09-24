"""Documented project parameters for near-field RL/PSO beam optimization.

This module is the single source of truth for tunable experiment parameters.
Other modules may still import these values through ``src.config`` for
backward compatibility, but new parameter edits should be made here.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Repository root. All data/checkpoint/figure paths are derived from this path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input data, generated plots, and trained model checkpoint directories.
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Matplotlib cache/config directory kept inside the repository to avoid writing
# into user/global locations when scripts run in restricted environments.
MPLCONFIG_DIR = PROJECT_ROOT / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

# HFSS self-interference channel file and optional offline PSO search-book file.
HFSS_FILE = DATA_DIR / "Tx_1x24_RP_HFSSDesign1.s24p"
PSO_BOOK_FILE = DATA_DIR / "pso_search_book.npz"


# ---------------------------------------------------------------------------
# Physical and array parameters
# ---------------------------------------------------------------------------

# Carrier frequency used for the near-field channel and beamforming helpers.
FC = 3.5e9

# Uniform linear array element spacing in meters.
ELEMENT_SPACING = 0.08


# ---------------------------------------------------------------------------
# Environment parameters
# ---------------------------------------------------------------------------

ENV_CONFIG = {
    # RF/array setup.
    "fc": FC,
    "element_spacing": ELEMENT_SPACING,

    # Beamwidth-first-null estimate in degrees. The environment uses half of
    # this value as the minimum angular separation between sampled DL/UL users.
    "bwfn_deg": 12.0,

    # Maximum SAC environment steps per episode.
    "max_steps": 60,

    # Legal Tx/Rx focus search space.
    "r_min": 0.5,
    "r_max": 8.0,
    "phi_min": -90.0,
    "phi_max": 90.0,
    "theta_min": -30.0,
    "theta_max": 30.0,

    # Random user sampling space used by environment resets.
    "user_r_min": 1.5,
    "user_r_max": 3.0,
    "user_phi_min": -25.0,
    "user_phi_max": 25.0,
    "user_theta_min": -8.0,
    "user_theta_max": 8.0,

    # Maximum incremental focus movement per SAC action component.
    "max_delta_r": 0.15,
    "max_delta_phi": 0.75,
    "max_delta_theta": 1.00,

    # Initial random focus jitter around user locations at reset.
    "init_focus_jitter_r": 0.05,
    "init_focus_jitter_phi": 0.5,
    "init_focus_jitter_theta": 0.5,

    # Default deterministic user/focus geometry used before a reset override.
    "desired_r_dl": 2.0,
    "desired_r_ul": 2.0,
    "desired_theta_dl": 0.0,
    "desired_theta_ul": 0.0,

    # Reward shaping weights.
    "lambda_p": 0.2,
    "lambda_boundary": 1.0,
    "lambda_action": 0.08,
    "lambda_degrade": 0.35,
    "lambda_regret": 0.30,
    "lambda_si": 6.00,
    "lambda_gain": 1.20,
    "lambda_dl_gain": 1.00,
    "lambda_ul_gain": 4.00,

    # Gain-preservation constraints. The 3 dB law is represented by the DL/UL
    # gain-loss margins.
    "min_gain_margin_db": 3.0,
    "dl_gain_margin_db": 3.0,
    "ul_gain_margin_db": 3.0,

    # Terminal reward bonus for good final SI behavior.
    "terminal_si_bonus": 0.50,

    # If True, SAC actions map directly to absolute focus coordinates. If False,
    # SAC actions are interpreted as incremental focus updates.
    "direct_action_mode": False,

    # Active reward formulation and clipping bound.
    "reward_mode": "neg_si_gain_constraint",
    "reward_clip": 20.0,

    # Default environment RNG seed.
    "seed": 1234,
}


# ---------------------------------------------------------------------------
# SAC/RL agent parameters
# ---------------------------------------------------------------------------

SAC_CONFIG = {
    # Actor learning rate.
    "alpha": 5e-5,

    # Critic/value learning rate.
    "beta": 2e-4,

    # Observation dimension from the environment state vector.
    "input_dims": [19],

    # Action dimension: Tx/Rx range, azimuth, and elevation focus controls.
    "n_actions": 6,

    # Replay buffer size.
    "max_size": 100000,

    # Gradient-update minibatch size.
    "batch_size": 256,

    # Discount factor and soft target-network update coefficient.
    "gamma": 0.99,
    "tau": 0.005,

    # SAC reward scale and action normalization limit.
    "reward_scale": 1,
    "max_action": 1.0,

    # Hidden-layer widths for actor/critic/value networks.
    "layer1_size": 256,
    "layer2_size": 256,
}


# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------

TRAIN_CONFIG = {
    # Number of random baseline samples printed before SAC training.
    "n_random_samples": 5000,

    # Total training episodes.
    "n_episodes": 2000,

    # Run validation every N episodes.
    "validation_every": 5,

    # Number of initial environment steps using random/warmup actions.
    "initial_random_steps": 512,

    # Standard deviation for clipped Gaussian warmup actions.
    "warmup_action_std": 0.25,

    # SAC parameter updates per environment step after warmup.
    "updates_per_step": 1,

    # RNG seed for warmup action sampling.
    "warmup_seed": 4242,
}


# ---------------------------------------------------------------------------
# Full-space PSO parameters
# ---------------------------------------------------------------------------

DEFAULT_PSO_CONFIG = {
    # Number of particles in the global PSO swarm.
    "num_particles": 48,

    # Number of PSO update iterations.
    "num_iters": 90,

    # Standard PSO velocity coefficients.
    "inertia": 0.72,
    "cognitive": 1.45,
    "social": 1.45,

    # Penalty multiplier applied to DL/UL gain-margin violation.
    "penalty_db": 250.0,
}


# ---------------------------------------------------------------------------
# RL-initialized local PSO parameters
# ---------------------------------------------------------------------------

RL_INIT_PSO_CONFIG = {
    # Smaller swarm and iteration count for a fast local refinement stage.
    "num_particles": 24,
    "num_iters": 45,

    # Slightly lower PSO coefficients than the full-space baseline because the
    # swarm searches only a local box around the RL focus.
    "inertia": 0.70,
    "cognitive": 1.35,
    "social": 1.35,

    # Same constraint-violation penalty as the full PSO objective.
    "penalty_db": 250.0,

    # Local search half-width around the RL-selected focus. These values are
    # applied to both Tx and Rx focus vectors.
    "radius_r": 1.50,
    "radius_phi": 24.0,
    "radius_theta": 24.0,
}


# ---------------------------------------------------------------------------
# Offline PSO search-book parameters
# ---------------------------------------------------------------------------

PSO_BOOK_CONFIG = {
    # Seed range and default search budget used by build_pso_book.py.
    "seed_start": 3001,
    "seed_count": 100,
    "num_particles": 64,
    "num_iters": 120,

    # Objective penalty and PSO velocity settings used by the standalone book
    # builder. These are kept separate from DEFAULT_PSO_CONFIG because the book
    # builder is an offline data-generation utility.
    "penalty_db": 100.0,
    "initial_velocity_fraction": 0.05,
    "inertia": 0.72,
    "cognitive": 1.49,
    "social": 1.49,
    "velocity_clip_fraction": 0.20,
}


# ---------------------------------------------------------------------------
# 3D evaluation/sample-case parameters
# ---------------------------------------------------------------------------

TEST_3D_CONFIG = {
    # Candidate grids used by src.test_3d_beam_map when drawing deterministic
    # 3D DL/UL test cases from a seed.
    "dl_r_grid": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "ul_r_grid": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "phi_grid_min": -40.0,
    "phi_grid_max": 40.0,
    "phi_grid_step": 5.0,
    "theta_grid_min": -20.0,
    "theta_grid_max": 20.0,
    "theta_grid_step": 5.0,
}


# ---------------------------------------------------------------------------
# Theta-disabled 2D evaluation/sample-case parameters
# ---------------------------------------------------------------------------

THETA_DISABLED_TEST_CONFIG = {
    # Candidate grids used by src.plot_pso_convergence_theta_disabled when
    # drawing deterministic DL/UL test cases from a seed. Theta is held fixed
    # so the field of test varies only in range and phi.
    "dl_r_grid": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "ul_r_grid": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "phi_grid_min": -40.0,
    "phi_grid_max": 40.0,
    "phi_grid_step": 5.0,
    "theta_fixed": 0.0,
}


# ---------------------------------------------------------------------------
# Fixed validation cases
# ---------------------------------------------------------------------------

FIXED_VALIDATION_CASES = [
    dict(phi_dl_user=20.0, phi_ul_user=30.0, theta_dl_user=0.0, theta_ul_user=0.0, r_dl_user=2.0, r_ul_user=2.0, seed=3001),
    dict(phi_dl_user=10.0, phi_ul_user=28.0, theta_dl_user=0.0, theta_ul_user=0.0, r_dl_user=1.5, r_ul_user=3.0, seed=3002),
    dict(phi_dl_user=-25.0, phi_ul_user=-10.0, theta_dl_user=0.0, theta_ul_user=0.0, r_dl_user=3.0, r_ul_user=1.5, seed=3003),
    dict(phi_dl_user=-30.0, phi_ul_user=25.0, theta_dl_user=0.0, theta_ul_user=0.0, r_dl_user=2.5, r_ul_user=2.5, seed=3004),
]

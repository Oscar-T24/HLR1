"""Backward-compatible access to documented project parameters.

Parameter definitions live in :mod:`src.params`. Keep importing from this module
in older scripts if convenient, but edit parameter values in ``src/params.py``.
"""

from __future__ import annotations

from .params import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DEFAULT_PSO_CONFIG,
    ELEMENT_SPACING,
    ENV_CONFIG,
    FC,
    FIGURES_DIR,
    FIXED_VALIDATION_CASES,
    HFSS_FILE,
    MPLCONFIG_DIR,
    PROJECT_ROOT,
    PSO_BOOK_CONFIG,
    PSO_BOOK_FILE,
    RL_INIT_PSO_CONFIG,
    SAC_CONFIG,
    TEST_3D_CONFIG,
    THETA_DISABLED_TEST_CONFIG,
    TRAIN_CONFIG,
)

# SURE2026 Developer and Experiment Guide

This document is the starting point for developers working on the current
near-field full-duplex beam-optimization project. It explains the research
objective, active implementations, environment, configuration, commands, and
the boundary between current code and historical experiment artifacts.

## Research objective

The project optimizes the transmit (Tx) and receive (Rx) near-field focal
points of an interleaved 24-port full-duplex antenna array. The measured HFSS
self-interference channel is reduced while the downlink (DL) and uplink (UL)
user gains remain within their configured loss margins.

Each candidate solution is a six-value focus vector:

```text
[r_tx, phi_tx, theta_tx, r_rx, phi_rx, theta_rx]
```

The repository compares three approaches:

1. **SAC/RL:** a trained Soft Actor-Critic policy moves the Tx/Rx focal points.
2. **Full-space PSO:** particle swarm optimization searches the complete legal
   six-dimensional focus space.
3. **RL-initialized PSO:** SAC supplies a promising focus and a smaller PSO
   swarm refines it inside a local search box.

The primary reported comparison uses deterministic, theta-disabled cases
(theta fixed at zero) for seeds 5000 through 5049. The full environment and
optimizers still support all six focus coordinates.

## Project map

| Location | Purpose |
| --- | --- |
| `src/params.py` | Single source of truth for paths and experiment parameters |
| `src/config.py` | Backward-compatible re-export of `src.params` values |
| `src/env.py` | Gymnasium environment, SI/gain calculations, observations, actions, and rewards |
| `src/utils.py` | HFSS loading and near-field array/channel helpers |
| `src/networks.py` | SAC actor, twin critics, and value networks |
| `src/sac_agent.py` | Replay buffer, SAC updates, action selection, and checkpoint I/O |
| `src/train.py` | SAC training entry point |
| `src/evaluate.py` | SAC evaluation on fixed validation cases |
| `src/pso_optimizer.py` | Full-space constrained PSO baseline |
| `src/rl_initialized_pso.py` | Local PSO initialized from an SAC result |
| `src/plotting.py` | SAC training-diagnostic plots |
| `src/plot_pso_convergence_theta_disabled.py` | Current shared 50-case PSO convergence/reduction experiment |
| `src/plot_si_improvement_comparison.py` | Current shared-case SI-reduction comparison and table |
| `src/plot_si_distribution_paper_figures.py` | Current absolute-SI ECDF/distribution figures |
| `src/plot_convergence_seed_paper.py` | Current paper-style single-case and aggregate convergence figures |
| `src/test_3d_beam_map.py` | 3D SAC/PSO/RL-init-PSO comparison and visualization |
| `data/` | Required HFSS Touchstone input and optional generated PSO search book |
| `checkpoints/` | Trained SAC actor/critic/value model files |
| `figures/` | Generated experiment and training figures |
| `paper/` | Manuscript source and compiled paper |
| `overleaf/` | Self-contained copy of the manuscript for Overleaf upload |
| `poster/`, `wpr/`, `impact statement/` | Presentation and project-administration artifacts, not runtime code |
| `PROJECT_WORKFLOW_SLIDES.md` | Parameter tables, workflow, and reported-result narrative |
| `CHANGELOG.md` | Chronological record of implementation and documentation changes |

The normal data flow is:

```text
HFSS .s24p -> load_hfss_channel -> NearFieldBeamFocusingEnv
                                      |-> SAC training/evaluation
                                      |-> full-space PSO
                                      `-> SAC initializer -> local PSO
                                                   |
                                             result scripts -> figures/
```

## Environment

`NearFieldBeamFocusingEnv` extracts the odd/even interleaved Tx and Rx ports
from the 24-by-24 HFSS channel, producing a 12-by-12 self-interference channel.
It samples separated DL/UL user positions, initializes focal points near those
users, and ends an episode after `ENV_CONFIG["max_steps"]` steps. Gymnasium's
`terminated` flag remains false; the time limit is reported through
`truncated`.

### Action space

The action is a six-dimensional `Box(-1, 1)` ordered as Tx range, azimuth,
elevation, followed by Rx range, azimuth, elevation.

The active `direct_action_mode=False` interprets actions as incremental moves:

```text
delta = action * [max_delta_r, max_delta_phi, max_delta_theta] per side
```

The result is clipped to the legal focus bounds. Setting direct action mode to
true instead maps each normalized component directly over its complete bound.

### Observation space

The observation is a normalized 19-dimensional `Box(-1, 1)` in this order:

| Indices | Values |
| --- | --- |
| 0-2 | Tx focus: range, phi, theta |
| 3-5 | Rx focus: range, phi, theta |
| 6-8 | DL user: range, phi, theta |
| 9-11 | UL user: range, phi, theta |
| 12-16 | current SI, previous SI, SI step change, best SI, regret from best SI |
| 17-18 | DL and UL gain loss |

Ranges are mapped over the configured legal interval, angles are divided by
their maximum magnitude, SI terms are divided by 100 dB, and gain losses by
30 dB; every component is clipped to `[-1, 1]`.

### Active reward

The active mode is `neg_si_gain_constraint`. Define the SI advantage as:

```text
A_si = 0.70 * clip(SI_ref - SI_now, -20, 20)
     + 0.20 * clip(SI_prev - SI_now, -10, 10)
     + 0.10 * clip(SI_best - SI_now, -10, 10)
```

Gain loss beyond each 3 dB margin forms a weighted constraint penalty. A
gain-feasible step receives `4 + lambda_si * A_si - lambda_p * focus_penalty`.
An infeasible step is penalized for positive SI advantage, gain-margin
violation, and focus error. Both cases also subtract reduced boundary and
action-magnitude penalties, and the final reward is clipped to
`[-reward_clip, reward_clip]` (currently `[-20, 20]`).

`src/env.py` contains several alternative reward modes for controlled
experiments, but they are inactive unless `ENV_CONFIG["reward_mode"]` changes.
The exact active weights and margins are documented in `src/params.py` and
`PROJECT_WORKFLOW_SLIDES.md`.

## Current algorithms

### Soft Actor-Critic

`src/train.py` loads the 3.5 GHz HFSS channel, constructs the environment and
`SACAgent`, gathers transitions in a replay buffer, and performs SAC updates
after the configured warm-up. The agent uses a stochastic Gaussian actor,
twin critics, a value network, and a soft-updated target value network.
Validation uses the fixed cases in `FIXED_VALIDATION_CASES`; training saves the
five checkpoint files under `checkpoints/` and diagnostics under `figures/`.

During evaluation, `choose_action(..., evaluate=True)` uses the trained policy
without exploration sampling. Experiment scripts retain the best gain-feasible
SAC focus from an episode as the RL result and, when applicable, the initializer
for local PSO.

### Full-space PSO

`run_pso_focus_optimization()` in `src/pso_optimizer.py` searches all six focus
coordinates over the environment bounds. Particle zero starts at the two user
locations and all other particles are uniform random samples. The objective is:

```text
SI_dB + penalty_db * weighted_gain_margin_violation
```

The swarm updates velocity using inertia, cognitive, and social terms, clips
velocity to 25% of the global span, and clips positions to legal bounds. The
returned solution is the lowest-SI feasible candidate; if none is feasible, it
returns the lowest-violation fallback. Defaults are 48 particles and 90
iterations.

### RL-initialized PSO

`run_rl_initialized_pso()` in `src/rl_initialized_pso.py` converts the best SAC
result into a six-value center point. It restricts the swarm to the intersection
of the global bounds and a local box of `+/-1.5 m`, `+/-24 degrees` phi, and
`+/-24 degrees` theta around that center. Particle zero is the SAC solution;
the rest start uniformly in the local box. It uses the same constrained
objective and feasible-first result selection as full PSO. Defaults are 24
particles and 45 iterations.

## Parameters and configuration

Edit tunable values only in `src/params.py`:

| Group | Controls |
| --- | --- |
| Paths/constants | repository paths, HFSS input, PSO book, carrier frequency, spacing |
| `ENV_CONFIG` | focus/user bounds, episode length, action scale, reward and gain constraints |
| `SAC_CONFIG` | dimensions, learning rates, replay/batch size, discount, target update, networks |
| `TRAIN_CONFIG` | episode count, random baseline, validation cadence, warm-up, updates |
| `DEFAULT_PSO_CONFIG` | full-space swarm size, iterations, coefficients, penalty |
| `RL_INIT_PSO_CONFIG` | local swarm budget, coefficients, penalty, local radii |
| `PSO_BOOK_CONFIG` | offline search-book generation settings |
| `TEST_3D_CONFIG` | 3D sampled-case grids |
| `THETA_DISABLED_TEST_CONFIG` | current 2D evaluation grids and fixed theta |
| `FIXED_VALIDATION_CASES` | deterministic SAC validation cases |

`src/config.py` exists so older imports continue to work; it does not define
independent values. Repository rules require any parameter change to be
reflected in `PROJECT_WORKFLOW_SLIDES.md`, and every code change must be entered
in `CHANGELOG.md`.

## Setup and commands

Run commands from the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchaudio
```

The required input is `data/Tx_1x24_RP_HFSSDesign1.s24p`. Evaluation and most
comparison figures also require compatible checkpoints produced by training.

### Train and evaluate SAC

```bash
python -m src.train
python -m src.evaluate
```

### Generate current comparison figures

```bash
# Main 50-case theta-disabled convergence and reduction figures
python -m src.plot_pso_convergence_theta_disabled --seed 5000 --num-seeds 50

# Current box/point comparison and printed shared-results table
python -m src.plot_si_improvement_comparison --seed-start 5000 --num-seeds 50

# Paper ECDF/distribution outputs; use --median-only for the poster median ECDF
python -m src.plot_si_distribution_paper_figures --seed-start 5000 --num-seeds 50
python -m src.plot_si_distribution_paper_figures --seed-start 5000 --num-seeds 50 --median-only

# Paper single-seed and aggregate convergence panels
python -m src.plot_convergence_seed_paper --seed 5003 --seed-start 5000 --num-seeds 50

# Representative 3D comparison plus a multi-seed summary
python -m src.test_3d_beam_map --seed 5003 --num-seeds 100
```

All figure scripts accept `--help`; use it before overriding output filenames,
seed ranges, or case selections. Relative output paths normally resolve under
`figures/`.

### Supporting utilities

```bash
# Build the optional offline PSO search book
python -m src.build_pso_book

# Export one theta-disabled case for MATLAB comparison
python -m src.export_matlab_sanity_case --seed 5000
```

`NF_FD_PSO_final_Distance_it.m` is the MATLAB-side sanity-check implementation,
not the Python production path.

## Current versus historical files

Use these rules when deciding what to modify or cite:

- The active runtime implementation is the Python package under `src/`, with
  `src/params.py` as its parameter authority.
- The current reported 50-case theta-disabled results come from
  `plot_pso_convergence_theta_disabled.py`,
  `plot_si_improvement_comparison.py`,
  `plot_si_distribution_paper_figures.py`, and
  `plot_convergence_seed_paper.py`. The current narrative is in
  `PROJECT_WORKFLOW_SLIDES.md`, `paper/`, and `overleaf/`.
- `plot_pso_convergence.py` is the earlier general 3D Monte Carlo convergence
  pipeline. It remains runnable and useful for full 3D experiments, but its
  output is not the primary theta-disabled result currently reported.
- `plot_si_improvement_histogram.py` retains the older default 100-seed
  histogram workflow. Prefer the shared 50-case comparison and ECDF scripts
  for results intended to match the current paper/poster.
- Root-level `.pptx` files, dated weekly reports under `wpr/`, earlier poster
  decks under `poster/`, and existing PNGs are snapshots/artifacts. They do not
  define the implementation; regenerate results from `src/` after code or
  parameter changes.
- `paper/` is the manuscript source of record; `overleaf/` is a synchronized
  upload package and can lag if only one copy is edited.
- The README mentions `RL/RL_Nearfield.ipynb`, but no `RL/` directory is
  present in the current repository. Do not depend on that notebook.
- The compiled checkpoints represent the currently stored trained model, but
  they become incompatible if the observation/action/network dimensions
  change. Retrain rather than treating an incompatible checkpoint as history
  to reproduce.

When results appear inconsistent, inspect the generating script and
`src/params.py` first, then compare `PROJECT_WORKFLOW_SLIDES.md` and the latest
dated `CHANGELOG.md` entry. Do not infer the active configuration from an old
figure, slide deck, or filename alone.

# Handoff Note: DDPG and TD3 Applicability to SURE2026

## Purpose of this note

I wrote this note for the next student who continues the SURE2026 project. My
goal is to give you a brief introduction to Deep Deterministic Policy Gradient
(DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3), and to explain
how I think they could apply to the current near-field full-duplex
beam-optimization work.

I did **not** implement either algorithm, retrain models, tune hyperparameters,
generate figures, or compare their results with the current Soft Actor-Critic
(SAC) implementation. If you have enough time in a future project term, you can
use this note as a starting point for that work.

## Current project context

The system I worked with uses SAC to adjust a six-dimensional
continuous focus vector:

```text
[r_tx, phi_tx, theta_tx, r_rx, phi_rx, theta_rx]
```

The Gymnasium environment gives the agent six normalized actions in `[-1, 1]`.
In the active incremental-action mode, these actions move the Tx and Rx range,
azimuth, and elevation focal coordinates. The 19-dimensional observation
contains the Tx/Rx focus positions, DL/UL user positions, self-interference
(SI) history, and DL/UL gain losses.

The active reward encourages SI reduction while enforcing the configured 3 dB
DL/UL gain-loss margins. Based on this continuous state/action formulation, I
believe both DDPG and TD3 are technically compatible with the environment.

## DDPG summary

DDPG is an off-policy actor-critic algorithm for continuous control. Its actor
maps an observation directly to one deterministic action. A critic estimates
the action value of each observation-action pair, and the actor is updated to
choose actions that the critic predicts will have higher return.

The main components are:

- a deterministic actor and a single critic;
- target actor and target critic networks updated slowly for stability;
- a replay buffer for reusing past transitions; and
- externally added action noise during training to provide exploration.

DDPG is conceptually simpler than SAC, but it can be sensitive to
hyperparameters, exploration-noise design, and critic overestimation. Because
the policy is deterministic, inadequate exploration may cause it to settle on
a limited region of the non-smooth beam-focusing landscape.

### Potential use in this project

If you implement DDPG, you should be able to use the existing 19-value
observation and six-value normalized action interface without changing the
environment. Its deterministic policy would also provide a direct focus
adjustment during evaluation.

However, the project searches a difficult SI landscape with constraints and
potentially narrow useful regions. A deterministic policy driven by external
noise may explore this landscape less robustly than the current stochastic SAC
policy. For this reason, I would treat DDPG as a simple deterministic
actor-critic baseline rather than the first replacement to try for SAC.

## TD3 summary

TD3 extends DDPG to reduce value overestimation and unstable policy updates.
It retains a deterministic actor and replay buffer but adds three important
techniques:

1. **Twin critics:** two critics estimate the action value, and the smaller
   target estimate is used when computing the learning target.
2. **Delayed actor updates:** the actor and target networks update less often
   than the critics, allowing the value estimates more time to stabilize.
3. **Target-policy smoothing:** clipped noise is added to the target action so
   the critic is discouraged from exploiting narrow, erroneous value peaks.

These changes generally make TD3 a stronger and more stable deterministic
continuous-control method than basic DDPG, while retaining a relatively direct
training procedure.

### Potential use in this project

If you implement TD3, you can keep the current continuous observation and
action spaces. Its twin-critic structure also resembles the two critics already
used by SAC, so you may be able to reuse some network and replay-buffer ideas
from the existing implementation.

I think TD3 could be a meaningful alternative if you want a deterministic
learned initializer for RL-initialized PSO. Target-policy smoothing and clipped
double-Q learning could help reduce sensitivity to false value peaks in the SI
landscape. It would still require carefully selected exploration noise during
training and would not remove the need for the gain-constrained reward or the
local PSO refinement stage.

## Comparison with the current SAC approach

| Property | DDPG | TD3 | Current SAC |
| --- | --- | --- | --- |
| Policy during training | Deterministic plus external noise | Deterministic plus external noise | Stochastic policy |
| Critics | One | Two | Two |
| Exploration mechanism | Added action noise | Added action noise | Learned stochasticity and entropy objective |
| Protection from critic overestimation | Limited | Clipped twin-critic target | Twin-critic minimum target |
| Actor update schedule | Every update | Delayed | Every configured SAC update |
| Fit to six continuous actions | Yes | Yes | Yes, already implemented |
| Current repository status | Not implemented | Not implemented | Active implementation |

I recommend keeping SAC as the project baseline because it is already
implemented, trained, and connected to the PSO comparison pipeline. Its
stochastic policy naturally supports exploration, which is useful for
discovering candidate focus points before local PSO refinement.

If you only have time to try one deterministic alternative, I recommend TD3
before DDPG because it directly addresses several of DDPG's known stability and
overestimation weaknesses. DDPG is still useful as a simpler baseline and for
understanding the progression from a basic deterministic actor-critic method
to TD3.

## Work required for a future implementation

If you decide to implement either method, you will need at least:

- deterministic actor and target-actor networks;
- DDPG critic/target-critic or TD3 twin critic/target-critic networks;
- algorithm-specific learning updates and soft target updates;
- exploration-noise and, for TD3, target-smoothing-noise configuration;
- checkpoint naming and loading separate from the SAC checkpoints;
- integration with the existing training, evaluation, and RL-initialized PSO
  scripts; and
- a controlled comparison using identical seeds, environment parameters,
  training budgets, gain constraints, and PSO budgets.

For a fair comparison, I suggest reporting RL-only SI, gain feasibility,
RL-initialized PSO SI, convergence efficiency, runtime, and variability across
the same test cases. Keep the reward definition, case population, seeds,
training budget, and PSO budget identical across algorithms.

## Recommendation

I stopped at this conceptual review because the available project time was
better spent on the existing SAC and PSO pipeline. I have not made any claims
that DDPG or TD3 will outperform SAC; you would need dedicated training and a
controlled evaluation before drawing that conclusion.

If you have additional time, I suggest this order:

1. keep SAC as the established reference;
2. implement TD3 as the higher-priority deterministic alternative;
3. add DDPG if you also need a basic deterministic baseline; and
4. compare the methods under the same seeds, training budget, reward, gain
   constraints, and downstream PSO settings.

## Where I suggest you start

- Read `PROJECT_GUIDE.md` first for the complete developer and experiment guide.
- `src/env.py`: observation/action spaces, transitions, SI/gain metrics, reward
- `src/params.py`: environment, SAC, training, and PSO parameters
- `src/networks.py`: current SAC actor, critic, and value networks
- `src/sac_agent.py`: current replay buffer and SAC learning procedure
- `src/train.py`: current training loop
- `src/evaluate.py`: fixed-case SAC evaluation
- `src/rl_initialized_pso.py`: local PSO refinement from an RL-selected focus

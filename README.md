This repository contains : 
1. Feiyang's initial work for the PSO/SAC/RL-initialised PSO beamforming algorithm.
2. My contribution to the project (`Oscar HLR F2026`)

Main files of interest : 

`PSO_search.ipynb` builds on my previous implementation of vectorised PSO optimisation in a non-convex 3D surface, this time extended to 6D particles for the beamforming problem

`SAC_search.ipynb` implements Feiyang's SAC search. 

`SAC_only.ipynb` is my reproduction of the SAC search, and includes a baseline comparison with Feiyang's work. 

Other test notebooks worth looking at: 

`SAC_continuous_control_example.ipynb` implements the 1D continuous control stabilisation problem under stochastic disturbance with a dual-Q network SAC Agent (proof of concept)

Other test files can be found in the `test` directory, and are examples I implemented in my reading of `Reinforcement Learning` by Phil Winder [(Oreilly)](learning.oreilly.com/library/view/reinforcement-learning)

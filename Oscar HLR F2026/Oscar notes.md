Todolist for the week 3 deliverables
- [ ] Reproduce Full Space PSO search (PSO-only)
- [ ] Reproduce SAC optimization (SAC-only)

Subtasks : 
- [ ] Be familiar with the `Gymnasium` library for environments
- [ ] Look at `NearFieldBeamFocusingEnv` class 

Notes from Gymnasium : 

`class gymnasium.spaces.Box` : a `Box` is a continuous space used to describe valid observations / actions

For example to define the state variables of the `SlipperyCubeEnv` example we could define 

```python
state_space = gymnasium.spaces.Box(
                low=np.array([-x_max, -v_max], dtype=np.float32),
                high=np.array([x_max, v_max], dtype=np.float32),
                dtype=np.float32
            )
```

So it allows us to define a state space for the range of _valid_ positions and velocities as a vector $\vec{s} = \left( \begin{matrix} x \\ v \end{matrix} \right)$

The action Box is a six dimensional array for (r_Tx, azimuth_Tx, elevation_Tx,r_Rx,azimuth_Rx,elevation_Rx)


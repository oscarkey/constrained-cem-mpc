# Constrained CEM MPC

A model predictive control algorithm, using the cross-entropy optimisation method, which finds trajectories that satisfy polytopic state and action constraints.

Based on ['Constrained Cross-Entropy Method for Safe Reinforcement Learning'; Wen, Topcu.](https://papers.nips.cc/paper/7974-constrained-cross-entropy-method-for-safe-reinforcement-learning)

There are many tasks for which constrained MPC is useful. For example, if controlling an inverted pendulum we might specify that the pendulum never falls over. Alternatively, we might specify that a robotic vacuum cleaner remains within a safe area.

The algorithm uses PyTorch, so is suitable for any task where the model of the environment is implemented using PyTorch. For example, the model could be fixed dynamics known ahead of time, a deep learning model or a Gaussian process model (using [GPyTorch](https://gpytorch.ai)).

## How to use
1. Specify the dynamics and costs of the system, by implementing `DynamicsFunc`. These could be fixed, or a learnt model.
2. Specify constraints, by implementing `Constraint`. `ActionConstraint`, `StateConstraint` and `TerminalConstraint` are existing implementations.
3. Construct `ConstrainedCemMpc`
4. Call `get_actions(initial_state)` to compute an approximately optimal set of actions to take from an initial state, which satisfy the constraints.

## Demo code
To start, take a look at `demos/pendulum_demo.py`, a 2D pendulum demo using OpenAI Gym Pendulum-v0.

`demos/2d_path_demo.py` plots the trajectories in 2D, which is useful for debugging during development of the library.
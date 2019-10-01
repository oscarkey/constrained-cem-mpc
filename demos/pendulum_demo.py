"""A demo which uses the library to balance a pendulum.

Rather than swing up the pendulum, the pendulum starts in the upwards vertical position. The safety constraint is that
it must not fall more than a small angle from the vertical.

There is no reward for keeping the pendulum near vertical, just a requirement to stay in the safe region. Thus the
pendulum swings around within this region.

The demo uses the OpenAI Gym Pendulum-v0. The MPC algorithm has access to the exact dynamics of the pendulum, there is
no learning.
"""
import time
from typing import Tuple

import gym
import numpy as np
import torch
from torch import Tensor

from constrained_cem_mpc import ConstrainedCemMpc, ActionConstraint, box2torchpoly, TerminalConstraint, StateConstraint, \
    DynamicsFunc


class Dynamics(DynamicsFunc):
    """Exact dynamics of the pendulum.

    For this demo we assume we know the exact dynamics. These are taken from the Pendulum-v0 environment in OpenAI Gym.
    """

    def __call__(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        th = states[:, 0]
        thdot = states[:, 1]

        max_speed = 8
        max_torque = 2.
        g = 10.
        m = 1.
        l = 1.
        dt = .05

        u = actions.clamp(-max_torque, max_torque)[:, 0]

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = newthdot.clamp(-max_speed, max_speed)

        objective_cost = torch.zeros_like(th)

        return torch.stack((newth, newthdot), dim=1), objective_cost


def observation_to_state(observation: Tuple[float, float, float]) -> Tensor:
    """Converts the Gym observation into a state, (angle, angular velocity)."""
    th = np.arccos(observation[0])
    thdot = observation[2]
    return torch.tensor([th, thdot], dtype=torch.double)


def reset_env(env):
    """Resets the pendulum in the safe area."""
    env.reset()
    env.env.state = env.np_random.uniform(low=[-0.1, -0.5], high=[0.1, 0.5])
    env.env.last_u = None
    return env.env._get_obs()


def main():
    torch.set_default_dtype(torch.double)

    env = gym.make('Pendulum-v0')

    constraints = [ActionConstraint(box2torchpoly([[-2.0, 2.0]])),  #
                   TerminalConstraint(box2torchpoly([[0, 1], [-0.8, 0.8]])),  #
                   StateConstraint(box2torchpoly([[0, 1], [-8, 8]]))]
    mpc = ConstrainedCemMpc(dynamics_func=Dynamics(), constraints=constraints, state_dimen=2, action_dimen=1,
                            time_horizon=10, num_rollouts=50, num_elites=10, num_iterations=8)

    observation = reset_env(env)
    for i in range(400):
        env.render()
        state = observation_to_state(observation)
        optimisation_start = time.time()
        actions, _ = mpc.get_actions(state)
        optimisation_end = time.time()
        print('time optimising:', optimisation_end - optimisation_start)

        # Sometimes the optimisation process may fail to find a safe action sequence, in which case we do nothing.
        if actions is None:
            action = torch.tensor([0])
            print('taking default action: ', action)
        else:
            action = actions[0]
            print('taking mpc action: ', action)

        observation, reward, done, info = env.step(action.numpy())

    env.close()


if __name__ == '__main__':
    main()

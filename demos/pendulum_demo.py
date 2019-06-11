import time
from typing import Tuple

import gym
import numpy as np
import torch
from torch import Tensor

from constrained_cem_mpc import ConstrainedCemMpc, ActionConstraint, box2torchpoly, TerminalConstraint, StateConstraint, \
    DynamicsFunc


class Dynamics(DynamicsFunc):
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


def observation_to_state(observation: (float, float, float)) -> Tensor:
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
        time_start = time.time()
        actions, _ = mpc.get_actions(state)
        time_end = time.time()
        print('time in solver:', time_end - time_start)

        if actions is None:
            action = torch.tensor([0])
            print('default action', action)
        else:
            action = actions[0]
            print('mpc action', action)

        observation, reward, done, info = env.step(action.numpy())

    env.close()


if __name__ == '__main__':
    main()
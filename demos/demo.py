from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from constrained_cem_mpc import ConstrainedCemMpc, TerminalConstraint, StateConstraint, ActionConstraint, box2torchpoly, \
    Rollouts, DynamicsFunc
from utils import assert_shape

state_dimen = 2
action_dimen = 2
x_max = 10
y_max = 10

objective_poly = box2torchpoly([[1, 2], [8, 9]])
safe_area = box2torchpoly([[0, 10], [0, 10]])
terminal_constraint = box2torchpoly([[8, 9], [8, 9]])


def dynamics(s, a):
    assert_shape(s, (state_dimen,))
    assert_shape(a, (action_dimen,))

    return s + a


class Dynamics(DynamicsFunc):
    def __call__(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        assert states.size(1) == state_dimen
        assert actions.size(1) == action_dimen

        # TODO: re-enable objective cost.
        return states + actions, torch.zeros(states.size(0))


def check_intersect(t, c):
    assert t.shape[1] == state_dimen
    for i in range(t.shape[0]):
        if t[i].numpy() in c:
            return True
    return False


def plot_trajectories(ts: Tensor, axes=None):
    should_show = False
    if axes is None:
        axes = plt.axes()
        should_show = True

    axes.set_xticks([0, x_max])
    axes.set_yticks([0, y_max])

    safe_area.plot(ax=axes, color='lightgrey')
    objective_poly.plot(ax=axes, color='palegreen')
    terminal_constraint.plot(ax=axes, color='deepskyblue')

    for t in ts:
        xs = [s[0].item() for s in t]
        ys = [s[1].item() for s in t]
        axes.plot(xs, ys)

    if should_show:
        plt.show()


def mean_objective_cost(rollouts: Rollouts):
    return rollouts.objective_costs.mean().item()


def mean_constraint_cost(rollouts: Rollouts):
    return rollouts.constraint_costs.mean().item()


def plot_costs(rollouts_by_time: [Rollouts]):
    avg_objective_costs = [mean_objective_cost(rollouts) for rollouts in rollouts_by_time]
    avg_constraint_costs = [mean_constraint_cost(rollouts) for rollouts in rollouts_by_time]
    xs = range(1, len(rollouts_by_time) + 1)
    plt.plot(xs, avg_objective_costs, label='objective cost')
    plt.plot(xs, avg_constraint_costs, label='constraint cost')
    plt.plot([0, 50], [0, 0], color='black')
    plt.legend()
    plt.show()


def main():
    torch.set_default_dtype(torch.float64)

    constraints = [TerminalConstraint(terminal_constraint),  #
                   StateConstraint(safe_area),  #
                   ActionConstraint(box2torchpoly([[-1, 1], [-1, 1]]))]
    mpc = ConstrainedCemMpc(Dynamics(), constraints, state_dimen, action_dimen, time_horizon=20, num_rollouts=400,
                            num_elites=30, num_iterations=50)
    rollouts_by_time = mpc.optimize_trajectories(torch.tensor([0.5, 0.5]))

    plot_trajectories(rollouts_by_time[-1].trajectories[0:10])
    plot_costs(rollouts_by_time)


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from constrained_cem_mpc import ConstrainedCemMpc, TerminalConstraint, StateConstraint, ActionConstraint, box2torchpoly, \
    Rollout
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


def objective_cost(trajectory, _):
    return F.pairwise_distance(objective_poly.chebXc.unsqueeze(0), trajectory[10].unsqueeze(0))


def check_intersect(t, c):
    assert t.shape[1] == state_dimen
    for i in range(t.shape[0]):
        if t[i].numpy() in c:
            return True
    return False


def plot_trajectories(ts, axes=None):
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


def mean_objective_cost(rollouts: [Rollout]):
    return sum([rollout.objective_cost for rollout in rollouts]) / float(len(rollouts))


def mean_constraint_cost(rollouts: [Rollout]):
    return sum([rollout.constraint_cost for rollout in rollouts]) / float(len(rollouts))


def plot_costs(rollouts_by_time: [[Rollout]]):
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
    mpc = ConstrainedCemMpc(dynamics, objective_cost, constraints, state_dimen, action_dimen, time_horizon=20,
                            num_rollouts=400, num_elites=30, num_iterations=50, num_workers=2)
    rollouts_by_time = mpc.optimize_trajectories(torch.tensor([0.5, 0.5]))

    plot_trajectories([rollout.trajectory for rollout in rollouts_by_time[-1][0:10]])
    plot_costs(rollouts_by_time)


if __name__ == '__main__':
    main()

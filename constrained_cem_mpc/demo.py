import matplotlib.pyplot as plt
import torch

from constrained_cem_mpc import ConstrainedCemMpc, TerminalConstraint, ObstaclesConstraint, ActionConstraint, \
    box2torchpoly
from utils import assert_shape

state_dimen = 2
action_dimen = 2
x_max = 10
y_max = 10

# obstacle_constraints = [polytope.box2poly([[2, 7], [2, 7]]),  #
#                         polytope.box2poly([[-1, 0], [0, y_max]]),  #
#                         polytope.box2poly([[x_max, x_max + 1], [0, y_max]]),  #
#                         polytope.box2poly([[0, x_max], [-1, 0]]),  #
#                         polytope.box2poly([[0, x_max], [y_max, y_max + 1]])]
obstacle_constraints = [box2torchpoly([[2, 7], [2, 7]])]
terminal_constraint = box2torchpoly([[7, 8], [7, 8]])


def dynamics(s, a):
    assert_shape(s, (state_dimen,))
    assert_shape(a, (action_dimen,))

    return s + a


def objective_cost(t):
    return 0


def check_intersect(t, c):
    assert t.shape[1] == state_dimen
    for i in range(t.shape[0]):
        if t[i].numpy() in c:
            return True
    return False


def plot_trajs(ts, axes=None):
    should_show = False
    if axes is None:
        axes = plt.axes()
        should_show = True

    axes.set_xticks([0, x_max])
    axes.set_yticks([0, y_max])

    for obstacle_constraint in obstacle_constraints:
        obstacle_constraint.plot(ax=axes)
    terminal_constraint.plot(ax=axes)

    for t in ts:
        xs = [s[0].item() for s in t]
        ys = [s[1].item() for s in t]
        axes.plot(xs, ys)

    if should_show:
        plt.show()


def main():
    torch.set_default_dtype(torch.float64)

    constraints = [TerminalConstraint(terminal_constraint),  #
                   ObstaclesConstraint(obstacle_constraints),  #
                   ActionConstraint(box2torchpoly([[-1, 1], [-1, 1]]))]
    mpc = ConstrainedCemMpc(dynamics, objective_cost, constraints, state_dimen, action_dimen, plot_trajs,
                            time_horizon=15, num_rollouts=100, num_elites=10, num_iterations=200)
    ts_by_time = mpc.find_trajectory(torch.tensor([0.5, 0.5]))

    # for t in range(0, len(ts_by_time), 10):
    #     plot_trajs(ts_by_time[t][0:10])
    plot_trajs(ts_by_time[-1][0:10])


if __name__ == '__main__':
    main()

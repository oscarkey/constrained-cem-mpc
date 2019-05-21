import matplotlib.pyplot as plt
import numpy as np
from polytope import polytope

from constrained_cem_mpc import ConstrainedCemMpc
from utils import assert_shape

state_dimen = 2
action_dimen = 2
x_max = 10
y_max = 10

obstacle_constraints = [polytope.box2poly([[4, 5], [4, 5]])]
terminal_constraint = polytope.box2poly([[7, 8], [7, 8]])


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


def constraint_cost(t):
    cost = 0

    # Work out how to implement cost for obstacles.
    for c in obstacle_constraints:
        if check_intersect(t, c):
            cost += 1

    if t[-1] not in terminal_constraint:
        cost += np.linalg.norm(terminal_constraint.chebXc - t[-1].numpy())

    # Actions?

    return cost


def plot_trajs(ts, axes=None):
    should_show = False
    if axes is None:
        axes = plt.axes()
        should_show = True

    axes.set_xticks([0, x_max])
    axes.set_yticks([0, y_max])

    obstacle_constraints[0].plot(ax=axes)
    terminal_constraint.plot(ax=axes)

    for t in ts:
        xs = [s[0].item() for s in t]
        ys = [s[1].item() for s in t]
        axes.plot(xs, ys)

    if should_show:
        plt.show()


def main():
    mpc = ConstrainedCemMpc(dynamics, objective_cost, [constraint_cost], state_dimen, action_dimen, plot_trajs,
                            time_horizon=15, num_rollouts=100, num_elites=10, num_iterations=100)
    mpc.find_trajectory()


if __name__ == '__main__':
    main()

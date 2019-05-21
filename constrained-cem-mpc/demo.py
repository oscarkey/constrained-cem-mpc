from polytope import polytope

state_dimen = 2
action_dimen = 2
x_max = 10
y_max = 10

obstacle_constraints = [polytope.box2poly([[4, 5], [4, 5]])]
safe_constraint = polytope.box2poly([[7, 8], [7, 8]])

def dynamics(s, a):
    assert s.shape == (state_dimen,)
    assert a.shape == (action_dimen,)

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

    if t[-1] not in safe_constraint:
        # cost += np.linalg.norm(safe_constraint.chebXc - t[-1].numpy())
        cost += 100

    # Actions?

    return cost

def plot_trajs(axes, ts):
    axes.set_xticks([0, x_max])
    axes.set_yticks([0, y_max])

    obstacle_constraints[0].plot(ax=axes)
    safe_constraint.plot(ax=axes)

    for t in ts:
        xs = [s[0].item() for s in t]
        ys = [s[1].item() for s in t]
        axes.plot(xs, ys)

def main():
    pass

if __name__ == '__main__':
    main()
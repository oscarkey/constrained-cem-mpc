import torch

_T = 15
_R = 3
_E = 10
_N = 1


class ConstrainedCemMpc:

    def __init__(self, dynamics_func, objective_func, constraint_funcs, state_dimen: int, action_dimen: int, plot_func):
        self._dynamics_func = dynamics_func
        self._objective_func = objective_func
        self._constraint_funcs = constraint_funcs
        self._state_dimen = state_dimen
        self._action_dimen = action_dimen
        self._plot_func = plot_func

    def _sample_trajectory(self, means, stds):
        actions = torch.distributions.Normal(means, stds).sample()

        trajectory = [torch.zeros((self._state_dimen,), dtype=torch.float)]
        for a in actions:
            trajectory.append(self._dynamics_func(trajectory[-1], a))

        return torch.stack(trajectory), actions

    def _compute_constraint_cost(self, trajectory):
        return sum([constraint_func(trajectory) for constraint_func in self._constraint_funcs])

    def find_trajectory(self):
        means = torch.zeros((_R, _T, self._action_dimen))
        stds = torch.ones((_R, _T, self._action_dimen))
        ts_by_time = []
        for i in range(_N):
            ts = [self._sample_trajectory(mean, std) for mean, std in zip(means, stds)]
            costs = [(aes, self._compute_constraint_cost(t)) for (t, aes) in ts]

            costs.sort(key=lambda x: x[1])
            elites = [aes for aes, _ in costs[:_E]]
            elite_aes = torch.stack(elites)

            means = elite_aes.mean(dim=0)
            stds = elite_aes.std(dim=0)

            ts_by_time.append([x[0] for x in ts])
            print([x[1] for x in costs])
            self._plot_func([x[1] for x in ts])

# axes = plt.axes()  # plot_trajs(axes, ts_by_time[499][0:20])  # plt.show()

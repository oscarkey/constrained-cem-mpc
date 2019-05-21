import torch

from utils import assert_shape


class ConstrainedCemMpc:

    def __init__(self, dynamics_func, objective_func, constraint_funcs, state_dimen: int, action_dimen: int, plot_func,
                 time_horizon: int, num_rollouts: int, num_elites: int, num_iterations: int):
        self._dynamics_func = dynamics_func
        self._objective_func = objective_func
        self._constraint_funcs = constraint_funcs
        self._state_dimen = state_dimen
        self._action_dimen = action_dimen
        self._plot_func = plot_func
        self._time_horizon = time_horizon
        self._num_rollouts = num_rollouts
        self._num_elites = num_elites
        self._num_iterations = num_iterations

    def _sample_trajectory(self, initial_state, means, stds):
        assert_shape(initial_state, (self._state_dimen,))
        assert_shape(means, (self._time_horizon, self._action_dimen))
        assert_shape(stds, (self._time_horizon, self._action_dimen))

        actions = torch.distributions.Normal(means, stds).sample()
        assert_shape(actions, (self._time_horizon, self._action_dimen))

        trajectory = [initial_state]
        for a in actions:
            trajectory.append(self._dynamics_func(trajectory[-1], a))

        trajectory_tensor = torch.stack(trajectory)
        # One more state than _T because of the initial state.
        assert_shape(trajectory_tensor, (self._time_horizon + 1, self._state_dimen))

        return trajectory_tensor, actions

    def _compute_constraint_cost(self, trajectory):
        return sum([constraint_func(trajectory) for constraint_func in self._constraint_funcs])

    def find_trajectory(self, initial_state):
        means = torch.zeros((self._time_horizon, self._action_dimen))
        stds = torch.ones((self._time_horizon, self._action_dimen))
        ts_by_time = []
        for i in range(self._num_iterations):
            ts = [self._sample_trajectory(initial_state, means, stds) for _ in range(self._num_rollouts)]
            costs = [(aes, self._compute_constraint_cost(t)) for (t, aes) in ts]

            costs.sort(key=lambda x: x[1])
            elites = [aes for aes, _ in costs[:self._num_elites]]
            elite_aes = torch.stack(elites)

            means = elite_aes.mean(dim=0)
            stds = elite_aes.std(dim=0)

            ts_by_time.append([x[0] for x in ts])

        return ts_by_time

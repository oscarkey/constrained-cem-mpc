from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from polytope import Polytope, polytope

from utils import assert_shape


class TorchPolytope:
    """Represents a polytope in n-dimensional space, with some limited operations.

    Uses Torch tensors, and thus is hopefully a faster replacement for the polytope package.
    """

    def __init__(self, polytope: Polytope):
        self._A = torch.tensor(polytope.A)
        self._b = torch.tensor(polytope.b)
        self.chebXc = torch.tensor(polytope.chebXc)

    def __contains__(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), 'Expected tensor, got {x}'
        return len((torch.matmul(self._A, x) - self._b > 0).nonzero()) == 0

    def plot(self, *args, **kwargs):
        Polytope(self._A.numpy(), self._b.numpy()).plot(*args, **kwargs)


def box2torchpoly(box: [[float]]) -> TorchPolytope:
    """Similar to polytope.box2poly(), but returns a TorchPolytope."""
    return TorchPolytope(polytope.box2poly(box))


class Constraint(ABC):
    """Represents a constraint that the trajectory must satisfy to be valid."""

    @abstractmethod
    def __call__(self, trajectory, actions) -> float:
        """Returns the (>=0) cost of the given trajectory wrt the constraints this function represents."""
        pass


class TerminalConstraint(Constraint):
    """A terminal polytope that the trajectory must finish inside to be valid (i.e. a safe area).

    If the trajectory is not valid then the cost is the Euclidian distance to the center of the constraint.
    """

    def __init__(self, safe_area: TorchPolytope) -> None:
        super().__init__()
        self._safe_area = safe_area

    def __call__(self, trajectory, actions) -> float:
        if trajectory[-1] not in self._safe_area:
            return F.pairwise_distance(self._safe_area.chebXc.unsqueeze(0), trajectory[-1].unsqueeze(0))
        else:
            return 0


class ObstaclesConstraint(Constraint):
    """Represents obstacles that the trajectory must avoid (i.e. constraint on all states)."""

    def __init__(self, obstacles: [TorchPolytope]) -> None:
        super().__init__()
        self._obstacles = obstacles

    def __call__(self, trajectory, actions) -> float:
        # NB: only checks the states at each timestep, not the "lines" between them.
        # I'm not quite sure what we need for the final implementation.

        cost = 0
        for state in trajectory:
            for obstacle in self._obstacles:
                if state in obstacle:
                    cost += 3

        return cost


class ActionConstraint(Constraint):
    """Constrains actions to lie within some polytope."""

    def __init__(self, safe_region: TorchPolytope, penalty: float = 3):
        super().__init__()
        self._self_region = safe_region
        self._penalty = penalty

    def __call__(self, trajectory, actions) -> float:
        cost = 0
        for action in actions:
            if action not in self._self_region:
                cost += self._penalty
        return cost


class ConstrainedCemMpc:

    def __init__(self, dynamics_func, objective_func, constraints: [Constraint], state_dimen: int, action_dimen: int,
                 plot_func, time_horizon: int, num_rollouts: int, num_elites: int, num_iterations: int):
        self._dynamics_func = dynamics_func
        self._objective_func = objective_func
        self._constraints = constraints
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

    def _compute_constraint_cost(self, trajectory, actions):
        return sum([constraint_func(trajectory, actions) for constraint_func in self._constraints])

    def find_trajectory(self, initial_state):
        means = torch.zeros((self._time_horizon, self._action_dimen))
        stds = torch.ones((self._time_horizon, self._action_dimen))
        ts_by_time = []
        for i in range(self._num_iterations):
            ts = [self._sample_trajectory(initial_state, means, stds) for _ in range(self._num_rollouts)]
            costs = [(aes, self._compute_constraint_cost(t, aes)) for (t, aes) in ts]

            costs.sort(key=lambda x: x[1])
            elites = [aes for aes, _ in costs[:self._num_elites]]
            elite_aes = torch.stack(elites)

            means = elite_aes.mean(dim=0)
            stds = elite_aes.std(dim=0)

            ts_by_time.append([x[0] for x in ts])

        # TODO: Check that trajectory satisfies all constraints.
        # Need to do some intelligent combination of iterating and restarting until objective function is good
        # and constraints are satisfied.

        return ts_by_time

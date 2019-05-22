from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union

import torch
import torch.nn.functional as F
from polytope import Polytope, polytope
from torch import Tensor
from torch.multiprocessing import Pool

from utils import assert_shape


class TorchPolytope:
    """Represents a polytope in n-dimensional space, with some limited operations.

    Uses Torch tensors, and thus is hopefully a faster replacement for the polytope package.
    """

    def __init__(self, polytope: Polytope):
        self._A = torch.tensor(polytope.A)
        self._b = torch.tensor(polytope.b)
        self.dim = polytope.dim
        self.chebXc = torch.tensor(polytope.chebXc)

    def __contains__(self, x: Tensor):
        assert_shape(x, (self.dim,))
        return len((torch.matmul(self._A, x) - self._b > 0).nonzero()) == 0

    def contains_points(self, xs: Tensor) -> int:
        """Returns the number of the given points which are inside the polytope

        :param xs [N x dim]
        """
        assert len(xs.shape) == 2 and xs.shape[1] == self.dim
        num_points = xs.shape[0]
        B = self._b.repeat(num_points, 1).transpose(0, 1)
        res = torch.matmul(self._A, xs.transpose(0, 1)) - B
        indices_of_positives = (res > 0).nonzero()
        columns_of_positives = indices_of_positives[:, 1]
        points_outside = len(columns_of_positives.unique())
        return num_points - points_outside

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


class StateConstraint(Constraint):
    """Represents the safe area that the trajectory must remain inside."""

    def __init__(self, safe_area: TorchPolytope) -> None:
        super().__init__()
        self._safe_area = safe_area

    def __call__(self, trajectory, actions) -> float:
        safe_points = self._safe_area.contains_points(trajectory)
        return 3 * (trajectory.shape[0] - safe_points)


class ActionConstraint(Constraint):
    """Constrains actions to lie within some polytope."""

    def __init__(self, safe_region: TorchPolytope, penalty: float = 3):
        super().__init__()
        self._self_region = safe_region
        self._penalty = penalty

    def __call__(self, trajectory, actions) -> float:
        num_safe_actions = self._self_region.contains_points(actions)
        return self._penalty * (actions.shape[0] - num_safe_actions)


Rollout = namedtuple('Rollout', 'trajectory actions objective_cost constraint_cost')


class RolloutFunction:
    """Computes rollouts (trajectory, action sequence, cost) given an initial state and parameters.

    This class and all of its members are passed between processes, so should be kept lightweight.
    """

    def __init__(self, dynamics_func, objective_func, constraints: [Constraint], state_dimen: int, action_dimen: int,
                 time_horizon: int):
        self._dynamics_func = dynamics_func
        self._objective_func = objective_func
        self._constraints = constraints
        self._state_dimen = state_dimen
        self._action_dimen = action_dimen
        self._time_horizon = time_horizon

    def perform_rollout(self, args: (Tensor, Tensor, Tensor)) -> Rollout:
        """Samples a trajectory, and returns the trajectory and the cost.

        :return (sequence of states, sequence of actions, cost)
        """
        initial_state, means, stds = args
        trajectory, actions = self._sample_trajectory(initial_state, means, stds)

        objective_cost = self._objective_func(trajectory, actions)
        constraint_cost = self._compute_constraint_cost(trajectory, actions)

        return Rollout(trajectory, actions, objective_cost, constraint_cost)

    def _sample_trajectory(self, initial_state: Tensor, means: Tensor, stds: Tensor) -> (Tensor, Tensor):
        """Randomly samples T actions and computes the trajectory.

        :return (sequence of states, sequence of actions)
        """
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

    def _compute_constraint_cost(self, trajectory, actions) -> float:
        return sum([constraint(trajectory, actions) for constraint in self._constraints])


class ConstrainedCemMpc:
    """Performs MPC to compute the next action, while remaining within a set of constraints.

    Uses the cross-entropy method to optimise the trajectories, while including constraints as additional optimisation
    objectives.

    This method is based on 'Constrained Cross-Entropy Method for Safe Reinforcement Learning', Wen, Topcu.
    """

    def __init__(self, dynamics_func, objective_func, constraints: [Constraint], state_dimen: int, action_dimen: int,
                 time_horizon: int, num_rollouts: int, num_elites: int, num_iterations: int, num_workers: int = 0,
                 rollout_function: RolloutFunction = None):
        """Creates a new instance.

        :param num_workers If >0, we will spawn worker processes to compute the rollouts. Otherwise, we will compute the
        rollouts in the current process and thread.
        :param rollout_function Only set this in unit tests, normally it we be created automatically.
        """
        self._action_dimen = action_dimen
        self._time_horizon = time_horizon
        self._num_rollouts = num_rollouts
        self._num_elites = num_elites
        self._num_iterations = num_iterations
        self._process_pool = Pool(num_workers) if num_workers > 0 else None

        if rollout_function is None:
            rollout_function = RolloutFunction(dynamics_func, objective_func, constraints, state_dimen, action_dimen,
                                               time_horizon)
        self._rollout_function = rollout_function

    def optimize_trajectories(self, initial_state: Tensor) -> [[Rollout]]:
        """Performs stochastic rollouts and optimises them using CEM, subject to the constraints.

        :return A list of lists where each inner list is all the rollouts from a single optimisation step. The final
        step is last in the outer list.
        """
        means = torch.zeros((self._time_horizon, self._action_dimen))
        stds = torch.ones((self._time_horizon, self._action_dimen))
        rollouts_by_time = []
        for i in range(self._num_iterations):
            rollouts = self._perform_rollouts(initial_state, means, stds)
            elite_rollouts = self._select_elites(rollouts)
            elite_actions = torch.stack([rollout.actions for rollout in elite_rollouts])

            means = elite_actions.mean(dim=0)
            stds = elite_actions.std(dim=0)

            rollouts_by_time.append(rollouts)

        return rollouts_by_time

    def _perform_rollouts(self, initial_state, means, stds) -> [Rollout]:
        if self._process_pool is not None:
            return self._process_pool.map(self._rollout_function.perform_rollout,
                                          [(initial_state, means, stds) for _ in range(self._num_rollouts)])
        else:
            return [self._rollout_function.perform_rollout((initial_state, means, stds)) for _ in
                    range(self._num_rollouts)]

    def _select_elites(self, rollouts: [Rollout]) -> [Rollout]:
        """Returns a list of the elite rollouts.

        If there are sufficient rollouts which satisfy the constraints, return these sorted by objective cost.
        Otherwise, return rollouts sorted by constraint cost.
        """
        feasible = [rollout for rollout in rollouts if rollout.constraint_cost == 0]
        if len(feasible) >= self._num_elites:
            return sorted(feasible, key=lambda rollout: rollout.objective_cost)[:self._num_elites]
        else:
            return sorted(rollouts, key=lambda x: x.constraint_cost)[:self._num_elites]

    def get_action(self, state: Tensor) -> Union[Tensor, None]:
        """Computes and returns the approximately optimal action to take from the given state, if we can find one.

        The action is guaranteed to be safe wrt to the constraints.

        :return the action, or None if we didn't find a safe action
        """
        # TODO: Add retries.
        rollouts = self.optimize_trajectories(state)[-1]
        feasible_rollouts = [rollout for rollout in rollouts if rollout.constraint_cost == 0]
        if len(feasible_rollouts) > 0:
            best_rollout = sorted(feasible_rollouts, key=lambda rollout: rollout.objective_cost)[0]
            return best_rollout.actions[0]
        else:
            return None

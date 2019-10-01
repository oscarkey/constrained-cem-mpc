"""An MPC implementation which supports polytopic constraints for trajectories, using the CEM optimisation method."""
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, List, Tuple

import torch
import torch.nn.functional as F
from polytope import Polytope, polytope
from torch import Tensor
from torch.distributions import Normal

from constrained_cem_mpc.utils import assert_shape


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

        :param xs: [N x dim]
        """
        assert len(xs.shape) == 2 and xs.shape[1] == self.dim, f'Wanted (N, {self.dim}) got {xs.shape}'
        num_points = xs.shape[0]
        B = self._b.repeat(num_points, 1).transpose(0, 1)
        res = torch.matmul(self._A, xs.transpose(0, 1)) - B
        indices_of_positives = (res > 0).nonzero()
        columns_of_positives = indices_of_positives[:, 1]
        points_outside = len(columns_of_positives.unique())
        return num_points - points_outside

    def plot(self, *args, **kwargs):
        Polytope(self._A.numpy(), self._b.numpy()).plot(*args, **kwargs)

    def to(self, device):
        """Moves the tensors representing the polytope to the given device.

        :param device: same as device argument of torch.Tensor.to()
        """
        self._A = self._A.to(device)
        self._b = self._b.to(device)
        self.chebXc = self.chebXc.to(device)
        return self


def box2torchpoly(box: List[List[float]]) -> TorchPolytope:
    """Constructs a TorchPolytope from a box.

    Similar to polytope.box2poly(), but returns a TorchPolytope.

    :param box: [n x 2] where n is the number of dimensions, the boundaries of the box
    """
    return TorchPolytope(polytope.box2poly(box))


class Constraint(ABC):
    """Represents a constraint that the trajectory must satisfy to be valid."""

    @abstractmethod
    def __call__(self, trajectory: Tensor, actions: Tensor) -> float:
        """Returns the (>=0) cost of the given trajectory wrt the constraints this function represents.

        :param trajectory: [T x state dimen]
        :param actions: [T x action dimen]
        """
        pass


class TerminalConstraint(Constraint):
    """A terminal polytope that the trajectory must finish inside to be valid (i.e. a safe area).

    If the trajectory is not valid then the cost is the Euclidian distance to the center of the constraint.
    """

    def __init__(self, safe_area: TorchPolytope) -> None:
        super().__init__()
        self._safe_area = safe_area

    def __call__(self, trajectory: Tensor, actions: Tensor) -> float:
        if trajectory[-1] not in self._safe_area:
            return F.pairwise_distance(self._safe_area.chebXc.unsqueeze(0), trajectory[-1].unsqueeze(0))
        else:
            return 0


class StateConstraint(Constraint):
    """Constrains the trajectory states to lie within a polytope."""

    def __init__(self, safe_area: TorchPolytope) -> None:
        super().__init__()
        self._safe_area = safe_area

    def __call__(self, trajectory: Tensor, actions: Tensor) -> float:
        safe_points = self._safe_area.contains_points(trajectory)
        return 3 * (trajectory.shape[0] - safe_points)


class ActionConstraint(Constraint):
    """Constrains the trajectory actions to lie within a polytope."""

    def __init__(self, safe_region: TorchPolytope, penalty: float = 3):
        super().__init__()
        self._self_region = safe_region
        self._penalty = penalty

    def __call__(self, trajectory: Tensor, actions: Tensor) -> float:
        num_safe_actions = self._self_region.contains_points(actions)
        return self._penalty * (actions.shape[0] - num_safe_actions)


class DynamicsFunc(ABC):
    """The user should implement this to specify the dynamics of the system, and the objective function."""

    @abstractmethod
    def __call__(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the next state and cost of the action.

        :param states: [N x state dimen] the current state, where N is the batch dimension
        :param actions: [N x action dimen] the action to perform, where N is the batch dimension
        :returns:
            next states [N x state dimen],
            costs [N x 0]
        """
        pass


Rollouts = namedtuple('Rollouts', 'trajectories actions objective_costs constraint_costs')


class RolloutFunction:
    """Computes rollouts (trajectory, action sequence, cost) given an initial state and parameters.

    This logic is in a separate class to allow multithreaded rollouts, though this is not currently implemented.
    """

    def __init__(self, dynamics: DynamicsFunc, constraints: List[Constraint], state_dimen: int, action_dimen: int,
                 time_horizon: int, num_rollouts: int):
        self._dynamics = dynamics
        self._constraints = constraints
        self._state_dimen = state_dimen
        self._action_dimen = action_dimen
        self._time_horizon = time_horizon
        self._num_rollouts = num_rollouts

    def perform_rollouts(self, args: Tuple[Tensor, Tensor, Tensor]) -> Rollouts:
        """Samples a trajectory, and returns the trajectory and the cost.

        :param args: (initial_state [state_dimen], action means, action stds)
        :returns: (sequence of states, sequence of actions, cost)
        """
        initial_state, means, stds = args
        initial_states = initial_state.repeat((self._num_rollouts, 1))
        trajectories, actions, objective_costs = self._sample_trajectory(initial_states, means, stds)
        constraint_costs = self._compute_constraint_costs(trajectories, actions)

        return Rollouts(trajectories, actions, objective_costs, constraint_costs)

    def _sample_trajectory(self, initial_states: Tensor, means: Tensor, stds: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Randomly samples T actions and computes the trajectory.

        :returns: (sequence of states, sequence of actions, costs)
        """
        assert_shape(initial_states, (self._num_rollouts, self._state_dimen))
        assert_shape(means, (self._time_horizon, self._action_dimen))
        assert_shape(stds, (self._time_horizon, self._action_dimen))

        actions = Normal(means, stds).sample(sample_shape=(self._num_rollouts,))
        assert_shape(actions, (self._num_rollouts, self._time_horizon, self._action_dimen))

        # One more state than the time horizon because of the initial state.
        trajectories = torch.empty((self._num_rollouts, self._time_horizon + 1, self._state_dimen),
                                   device=initial_states.device)
        trajectories[:, 0, :] = initial_states
        objective_costs = torch.zeros((self._num_rollouts,), device=initial_states.device)
        for t in range(self._time_horizon):
            next_states, costs = self._dynamics(trajectories[:, t, :], actions[:, t, :])

            assert_shape(next_states, (self._num_rollouts, self._state_dimen))
            assert_shape(costs, (self._num_rollouts,))

            trajectories[:, t + 1, :] = next_states
            # TODO: worry about underflow.
            objective_costs += costs

        return trajectories, actions, objective_costs

    def _compute_constraint_costs(self, trajectories: Tensor, actions: Tensor) -> Tensor:
        # TODO: Batch computation of constraint costs.
        costs = torch.empty((self._num_rollouts), device=trajectories.device)
        for i in range(self._num_rollouts):
            costs[i] = sum([constraint(trajectories[i], actions[i]) for constraint in self._constraints])
        return costs


class ConstrainedCemMpc:
    """An MPC implementation which supports polytopic constraints for trajectories, using cross-entropy optimisation.

    This method is based on 'Constrained Cross-Entropy Method for Safe Reinforcement Learning'; Wen, Topcu. It includes
    the constraints as additional optimisation objectives, which must be satisfied before the trajectory cost is
    optimised.
    """

    def __init__(self, dynamics_func: DynamicsFunc, constraints: List[Constraint], state_dimen: int, action_dimen: int,
                 time_horizon: int, num_rollouts: int, num_elites: int, num_iterations: int,
                 rollout_function: RolloutFunction = None):
        """Creates a new instance.

        :param constraints: list of constraints which any trajectory must satisfy, or [] if there are no constraints
        :param state_dimen: number of dimensions of the state
        :param action_dimen: number of dimensions of the actions
        :param time_horizon: T, number of time steps into the future that the algorithm plans
        :param num_rollouts: number of trajectories that the algorithm samples each optimisation iteration
        :param num_elites: number of trajectories, i.e. the best m, which the algorithm refits the distribution to
        :param num_iterations: number of iterations of CEM
        :param rollout_function: only set in unit tests
        """
        self._action_dimen = action_dimen
        self._time_horizon = time_horizon
        self._num_rollouts = num_rollouts
        self._num_elites = num_elites
        self._num_iterations = num_iterations

        if rollout_function is None:
            rollout_function = RolloutFunction(dynamics_func, constraints, state_dimen, action_dimen, time_horizon,
                                               num_rollouts)
        self._rollout_function = rollout_function

    def optimize_trajectories(self, initial_state: Tensor) -> List[Rollouts]:
        """Performs stochastic rollouts and optimises them using CEM, subject to the constraints.

        The trajectories this function returns are not guaranteed to be safe. Thus, normally, do not call this method
        directly. Instead, call get_actions().

        :returns: A list of rollouts from each CEM iteration. The final step is last.
        """
        means = torch.zeros((self._time_horizon, self._action_dimen), device=initial_state.device)
        stds = torch.ones((self._time_horizon, self._action_dimen), device=initial_state.device)
        rollouts_by_iteration = []
        for i in range(self._num_iterations):
            rollouts = self._rollout_function.perform_rollouts((initial_state, means, stds))
            elite_rollouts = self._select_elites(rollouts)

            means = elite_rollouts.actions.mean(dim=0)
            stds = elite_rollouts.actions.std(dim=0)

            rollouts_by_iteration.append(rollouts)

        return rollouts_by_iteration

    def _select_elites(self, rollouts: Rollouts) -> Rollouts:
        """Returns the elite rollouts.

        If there are sufficient rollouts which satisfy the constraints, return these sorted by objective cost.
        Otherwise, return rollouts sorted by constraint cost.
        """
        feasible_ids = (rollouts.constraint_costs == 0).nonzero()
        if feasible_ids.size(0) >= self._num_elites:
            _, sorted_ids_of_feasible_ids = rollouts.objective_costs[feasible_ids].squeeze().sort()
            elites_ids = feasible_ids[sorted_ids_of_feasible_ids[0:self._num_elites]].squeeze()
            return Rollouts(rollouts.trajectories[elites_ids], rollouts.actions[elites_ids],
                            rollouts.objective_costs[elites_ids], rollouts.constraint_costs[elites_ids])
        else:
            _, sorted_ids = rollouts.constraint_costs.sort()
            elites_ids = sorted_ids[0:self._num_elites]
            return Rollouts(rollouts.trajectories[elites_ids], rollouts.actions[elites_ids],
                            rollouts.objective_costs[elites_ids], rollouts.constraint_costs[elites_ids])

    def get_actions(self, state: Tensor) -> Tuple[Union[Tensor, None], List[Rollouts]]:
        """Computes the approximately optimal actions to take from the given state.

        The sequence of actions is guaranteed to be safe wrt to the constraints.

        :param state: [state dimen], the initial state to plan from
        :returns: (the actions [N x action dimen] or None if we didn't find a safe sequence of actions,
                   rollouts by iteration as returned by optimize_trajectories())
        """
        rollouts_by_iteration = self.optimize_trajectories(state)

        # Use the rollouts from the final optimisation step.
        rollouts = rollouts_by_iteration[-1]

        feasible_ids = (rollouts.constraint_costs == 0).nonzero().squeeze(0)
        if feasible_ids.size(0) > 0:
            _, sorted_ids_of_feasible_ids = rollouts.objective_costs[feasible_ids].sort()
            best_rollout_id = feasible_ids[sorted_ids_of_feasible_ids[0]].item()
            return rollouts.actions[best_rollout_id], rollouts_by_iteration
        else:
            return None, rollouts_by_iteration

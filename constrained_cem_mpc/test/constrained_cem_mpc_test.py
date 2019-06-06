from typing import Tuple

import numpy as np
import torch
from polytope import polytope
from torch import Tensor

from constrained_cem_mpc import TerminalConstraint, StateConstraint, ActionConstraint, TorchPolytope, box2torchpoly, \
    Constraint, ConstrainedCemMpc, Rollout, DynamicsFunc
from constrained_cem_mpc.constrained_cem_mpc import RolloutFunction


def setup_module():
    torch.set_default_dtype(torch.double)


class TestTerminalConstraint:
    def test__misses_constraint__returns_l2_distance(self):
        constraint = box2torchpoly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        # A trajectory that passes through the terminal constraint but does not end in it.
        trajectory = torch.tensor([[0, 0], [10.5, 10.5], [0, 0]], dtype=torch.double)

        cost = func(trajectory, actions=torch.zeros((1,)))

        distance = np.sqrt(np.square(10.5) + np.square(10.5))
        assert np.isclose(cost, distance)

    def test__hits_constraint__returns_zero(self):
        constraint = box2torchpoly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        trajectory = torch.tensor([[0, 0], [0, 1], [10.5, 10.5]], dtype=torch.double)

        cost = func(trajectory, actions=torch.zeros((1,)))

        assert cost == 0


class TestStateConstraint:
    def test__inside_safe_area__returns_zero(self):
        safe_area = box2torchpoly([[0, 10], [0, 10]])
        func = StateConstraint(safe_area)
        trajectory = torch.tensor([[0, 0], [0.5, 0.5], [9, 10]], dtype=torch.double)

        cost = func(trajectory, actions=torch.zeros((1,)))

        assert cost == 0

    def test__states_in_unsafe_area__returns_high_cost(self):
        safe_area = box2torchpoly([[0, 10], [0, 10]])
        func = StateConstraint(safe_area)
        # Have one point outside the safe area.
        trajectory = torch.tensor([[0, 0], [11, 11], [2.5, 2.5]], dtype=torch.double)

        cost = func(trajectory, actions=torch.zeros((1,)))

        assert cost >= 0


class TestActionConstraint:
    def test__all_actions_safe__returns_zero(self):
        safe_area = box2torchpoly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.tensor([[0, 0], [0.5, 0.2], [-0.9, -0.99]], dtype=torch.double)

        cost = func(trajectory=torch.zeros((1,)), actions=actions)

        assert cost == 0

    def test__one_action_unsafe__returns_single_penalty(self):
        safe_area = box2torchpoly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.tensor([[0, 0], [1.1, 0.0], [-0.9, -0.99]], dtype=torch.double)

        cost = func(trajectory=torch.zeros((1,)), actions=actions)

        assert cost == 5

    def test__two_actions_unsafe__returns_double_penalty(self):
        safe_area = box2torchpoly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.tensor([[0, 0], [-2, -3], [0.5, 1.2]], dtype=torch.double)

        cost = func(trajectory=torch.zeros((1,)), actions=actions)

        assert cost == 10


class TestTorchPolytope:
    def test__contains__does_not_contain__returns_False(self):
        torch_polytope = box2torchpoly([[0, 1], [0, 1]])
        assert torch.tensor([-1, -1], dtype=torch.float64) not in torch_polytope

    def test__contains__does_contain__returns_true(self):
        torch_polytope = box2torchpoly([[0, 1], [0, 1]])
        assert torch.tensor([0.5, 0.5], dtype=torch.float64) in torch_polytope

    def test__contains_points__contains_no_points__returns_correct_count(self):
        torch_polytope = box2torchpoly([[0, 10], [0, 10]])
        points = torch.tensor([[11, 15], [-1, 0]], dtype=torch.double)

        count = torch_polytope.contains_points(points)

        assert count == 0

    def test__contains_points__contains_some_points__returns_correct_count(self):
        torch_polytope = box2torchpoly([[0, 10], [0, 10]])
        points = torch.tensor([[1, 1], [1, 2], [5, 6], [11, 11], [-10, -2]], dtype=torch.double)

        count = torch_polytope.contains_points(points)

        assert count == 3

    def test__chebXc__is_tensor(self):
        torch_polytope = box2torchpoly([[0, 1], [0, 1]])
        assert isinstance(torch_polytope.chebXc, torch.Tensor)

    def test__chebXc__equal_to_input_polytope(self):
        np_polytope = polytope.box2poly([[0, 1], [0, 1]])
        torch_polytope = TorchPolytope(np_polytope)

        assert np.array_equal(torch_polytope.chebXc.numpy(), np_polytope.chebXc)

    def test__dim__equal_to_input_polytype(self):
        np_polytope = polytope.box2poly([[0, 1], [0, 1]])
        torch_polytope = TorchPolytope(np_polytope)

        assert torch_polytope.dim == np_polytope.dim


STATE_DIMEN = 2
ACTION_DIMEN = 2
TIME_HORIZON = 3


class FakeConstraint(Constraint):
    def __call__(self, trajectory, actions) -> float:
        if trajectory[-1][0].item() == 1 and trajectory[-1][1].item() == 1:
            return 5.0
        else:
            return 0.0


class TestRolloutFunction:
    class BasicDynamics(DynamicsFunc):
        def __init__(self, objective_cost: float):
            self._objective_cost = objective_cost

        def __call__(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
            return state + action, torch.tensor([self._objective_cost])

    def _set_up(self):
        func = RolloutFunction(self.BasicDynamics(objective_cost=0.0), [], STATE_DIMEN, ACTION_DIMEN, TIME_HORIZON)
        initial_state = torch.tensor([0, 0], dtype=torch.double)
        means = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.double)
        stds = torch.zeros((TIME_HORIZON, ACTION_DIMEN), dtype=torch.double)

        return func, initial_state, means, stds

    def test__perform_rollout__trajectory_correct(self):
        func, initial_state, means, stds = self._set_up()

        rollout = func.perform_rollout((initial_state, means, stds))
        trajectory = rollout.trajectory

        assert trajectory.shape == (TIME_HORIZON + 1, STATE_DIMEN)

        assert trajectory[0][0] == 0
        assert trajectory[0][1] == 0

        assert trajectory[1][0] == 1
        assert trajectory[1][1] == 0

        assert trajectory[2][0] == 1
        assert trajectory[2][1] == 1

        assert trajectory[3][0] == 1
        assert trajectory[3][1] == 1

    def test__perform_rollout__actions_correct(self):
        func, initial_state, means, stds = self._set_up()

        rollout = func.perform_rollout((initial_state, means, stds))
        actions = rollout.actions

        assert actions.shape == (TIME_HORIZON, ACTION_DIMEN)

        assert actions[0][0] == 1
        assert actions[0][1] == 0

        assert actions[1][0] == 0
        assert actions[1][1] == 1

        assert actions[2][0] == 0
        assert actions[2][1] == 0

    def test__perform_rollout__objective_cost_correct(self):
        _, initial_state, means, stds = self._set_up()
        func = RolloutFunction(self.BasicDynamics(objective_cost=15.0), [FakeConstraint()], STATE_DIMEN, ACTION_DIMEN,
                               TIME_HORIZON)

        rollout = func.perform_rollout((initial_state, means, stds))

        # We take three actions.
        assert rollout.objective_cost == 15 * 3

    def test__perform_rollout__constraint_cost_correct(self):
        _, initial_state, means, stds = self._set_up()
        func = RolloutFunction(self.BasicDynamics(objective_cost=0.0), [FakeConstraint()], STATE_DIMEN, ACTION_DIMEN,
                               TIME_HORIZON)

        rollout = func.perform_rollout((initial_state, means, stds))

        assert rollout.constraint_cost == 5


class TestConstrainedCemMpc:
    def test__optimize_trajectories__rollouts_fail_constraints__one_step_opt_correct(self, mocker):
        # Set objective cost inverse to constraint cost, to check it orders by constraint cost.
        rollout1 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[10.0, 0.0]]), 8, 3)
        rollout2 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[1.0, 0.0]]), 9, 2)
        rollout3 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[3.0, 0.0]]), 10, 1)
        rolloutx = Rollout(torch.tensor([[0, 0], [100, 0]]), torch.tensor([[100.0, 0.0]]), 20, 20)

        rollout_function = mocker.Mock()
        rollout_function.perform_rollout.side_effect = [rollout1, rollout2, rollout3, rolloutx, rolloutx, rolloutx]

        mpc = ConstrainedCemMpc(dynamics_func=None, objective_func=None, constraints=[], state_dimen=2, action_dimen=2,
                                time_horizon=1, num_rollouts=3, num_elites=2, num_iterations=2, num_workers=0,
                                rollout_function=rollout_function)

        mpc.optimize_trajectories(torch.tensor([0, 0]))

        args = [call.args for call in rollout_function.perform_rollout.call_args_list]
        # We expect it to take mean and std of the 2nd and 3rd rollouts, and use these on the next step.
        # The next step should be the 4th call to the function.
        initial_state, means, stds = rollout_function.perform_rollout.call_args_list[4][0][0]
        assert means[0][0].item() == 2
        assert means[0][1].item() == 0

        # ddof=1 because Pytorch using Bessel correction but numpy does not.
        # https://github.com/pytorch/pytorch/issues/1082
        assert np.isclose(stds[0][0].item(), np.std([1.0, 3.0], ddof=1))

        assert stds[0][1] == 0

    def test__optimize_trajectories__rollouts_pass_constraints__one_step_opt_correct(self, mocker):
        # Set constraint cost to 0 as all rollouts pass constraints
        rollout1 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[10.0, 0.0]]), 8, 0)
        rollout2 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[1.0, 0.0]]), 10, 0)
        rollout3 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[2.0, 0.0]]), 9, 0)
        rolloutx = Rollout(torch.tensor([[0, 0], [100, 0]]), torch.tensor([[100.0, 0.0]]), 20, 0)

        rollout_function = mocker.Mock()
        rollout_function.perform_rollout.side_effect = [rollout1, rollout2, rollout3, rolloutx, rolloutx, rolloutx]

        mpc = ConstrainedCemMpc(dynamics_func=None, objective_func=None, constraints=[], state_dimen=2, action_dimen=2,
                                time_horizon=1, num_rollouts=3, num_elites=2, num_iterations=2, num_workers=0,
                                rollout_function=rollout_function)

        mpc.optimize_trajectories(torch.tensor([0, 0]))

        args = [call.args for call in rollout_function.perform_rollout.call_args_list]
        # We expect it to take mean and std of the 1st and 3rd rollouts, and use these on the next step.
        # The next step should be the 4th call to the function.
        initial_state, means, stds = rollout_function.perform_rollout.call_args_list[4][0][0]
        assert means[0][0].item() == 6
        assert means[0][1].item() == 0

        # ddof=1 because Pytorch using Bessel correction but numpy does not.
        # https://github.com/pytorch/pytorch/issues/1082
        assert np.isclose(stds[0][0].item(), np.std([10.0, 2.0], ddof=1))

        assert stds[0][1] == 0

    def test__get_actions__no_feasible_rollout__returns_none(self, mocker):
        # All non-zero constraint costs so no rollout is feasible
        rollout1 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0.0, 0.0]]), 0, 3)
        rollout2 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0.0, 0.0]]), 0, 2)
        rollout3 = Rollout(torch.tensor([[0, 0], [0, 0]]), torch.tensor([[0.0, 0.0]]), 0, 1)

        rollout_function = mocker.Mock()
        rollout_function.perform_rollout.side_effect = [rollout1, rollout2, rollout3]

        mpc = ConstrainedCemMpc(dynamics_func=None, objective_func=None, constraints=[], state_dimen=2, action_dimen=2,
                                time_horizon=1, num_rollouts=3, num_elites=2, num_iterations=1, num_workers=0,
                                rollout_function=rollout_function)

        actions, _ = mpc.get_actions(torch.tensor([0.0, 0.0]))

        assert actions is None

    def test__get_actions__feasible_rollouts__returns_actions_from_best(self, mocker):
        rollout1 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0.0, 0.0]]), 0, 3)
        rollout2 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0.0, 0.0]]), 0, 2)
        rollout3 = Rollout(torch.tensor([[0, 0], [0, 0]]), torch.tensor([[0.0, 0.0]]), 0, 1)
        rollout4 = Rollout(torch.tensor([[0, 0], [1, 0], [1, 0]]), torch.tensor([[0.0, 0.0], [0.0, 0.0]]), 1, 3)
        rollout5 = Rollout(torch.tensor([[0, 0], [1, 0], [1, 0]]), torch.tensor([[1.0, 1.5], [3.0, 2.5]]), 5, 0)
        rollout6 = Rollout(torch.tensor([[0, 0], [0, 0], [1, 0]]), torch.tensor([[0.0, 0.0], [0.0, 0.0]]), 10, 0)

        rollout_function = mocker.Mock()
        rollout_function.perform_rollout.side_effect = [rollout1, rollout2, rollout3, rollout4, rollout5, rollout6]

        mpc = ConstrainedCemMpc(dynamics_func=None, objective_func=None, constraints=[], state_dimen=2, action_dimen=2,
                                time_horizon=1, num_rollouts=3, num_elites=2, num_iterations=2, num_workers=0,
                                rollout_function=rollout_function)

        action, _ = mpc.get_actions(torch.tensor([0.0, 0.0]))

        assert action[0][0] == 1.0
        assert action[0][1] == 1.5
        assert action[1][0] == 3.0
        assert action[1][1] == 2.5

    def test__get_actions__returns_rollouts_by_time(self, mocker):
        rollout1 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0.0, 0.0]]), 0, 3)
        rollout2 = Rollout(torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0.0, 0.0]]), 0, 2)
        rollout3 = Rollout(torch.tensor([[0, 0], [0, 0]]), torch.tensor([[0.0, 0.0]]), 0, 1)

        rollout_function = mocker.Mock()
        rollout_function.perform_rollout.side_effect = [rollout1, rollout2, rollout3]

        mpc = ConstrainedCemMpc(dynamics_func=None, objective_func=None, constraints=[], state_dimen=2, action_dimen=2,
                                time_horizon=1, num_rollouts=3, num_elites=2, num_iterations=1, num_workers=0,
                                rollout_function=rollout_function)

        _, rollouts_by_time = mpc.get_actions(torch.tensor([0.0, 0.0]))

        assert len(rollouts_by_time) == 1
        assert rollouts_by_time[0][0] == rollout1
        assert rollouts_by_time[0][1] == rollout2
        assert rollouts_by_time[0][2] == rollout3

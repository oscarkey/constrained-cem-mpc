import numpy as np
import torch
from polytope import polytope

from constrained_cem_mpc import TerminalConstraint, ObstaclesConstraint, ActionConstraint, TorchPolytope, box2torchpoly, \
    RolloutFunction, Constraint


class TestTerminalConstraint:
    def test__misses_constraint__returns_l2_distance(self):
        constraint = box2torchpoly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        # A trajectory that passes through the terminal constraint but does not end in it.
        trajectory = torch.tensor([[0, 0], [10.5, 10.5], [0, 0]], dtype=torch.double)

        cost = func(trajectory, actions=None)

        distance = np.sqrt(np.square(10.5) + np.square(10.5))
        assert np.isclose(cost, distance)

    def test__hits_constraint__returns_zero(self):
        constraint = box2torchpoly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        trajectory = torch.tensor([[0, 0], [0, 1], [10.5, 10.5]], dtype=torch.double)

        cost = func(trajectory, actions=None)

        assert cost == 0


class TestObstaclesConstraint:
    def test__misses_obstacle__returns_zero(self):
        obstacle1 = box2torchpoly([[1, 2], [1, 2]])
        obstacle2 = box2torchpoly([[2, 3], [2, 3]])
        func = ObstaclesConstraint([obstacle1, obstacle2])
        trajectory = torch.tensor([[0, 0], [0.5, 0.5]], dtype=torch.double)

        cost = func(trajectory, actions=None)

        assert cost == 0

    def test__hits_obstacle__returns_high_cost(self):
        obstacle1 = box2torchpoly([[1, 2], [1, 2]])
        obstacle2 = box2torchpoly([[2, 3], [2, 3]])
        func = ObstaclesConstraint([obstacle1, obstacle2])
        # Miss the first obstacle, hit the second.
        trajectory = torch.tensor([[0, 0], [2.5, 0], [2.5, 2.5]], dtype=torch.double)

        cost = func(trajectory, actions=None)

        assert cost >= 0


class TestActionConstraint:
    def test__all_actions_safe__returns_zero(self):
        safe_area = box2torchpoly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.tensor([[0, 0], [0.5, 0.2], [-0.9, -0.99]], dtype=torch.double)

        cost = func(trajectory=None, actions=actions)

        assert cost == 0

    def test__one_action_unsafe__returns_single_penalty(self):
        safe_area = box2torchpoly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.tensor([[0, 0], [1.1, 0.0], [-0.9, -0.99]], dtype=torch.double)

        cost = func(trajectory=None, actions=actions)

        assert cost == 5

    def test__two_actions_unsafe__returns_double_penalty(self):
        safe_area = box2torchpoly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.tensor([[0, 0], [-2, -3], [0.5, 1.2]], dtype=torch.double)

        cost = func(trajectory=None, actions=actions)

        assert cost == 10


class TestTorchPolytope:
    def test__contains__does_not_contain__returns_False(self):
        torch_polytope = box2torchpoly([[0, 1], [0, 1]])
        assert torch.tensor([-1, -1], dtype=torch.float64) not in torch_polytope

    def test__contains__does_contain__returns_True(self):
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


class TestConstraint(Constraint):
    def __call__(self, trajectory, actions) -> float:
        if trajectory[-1][0].item() == 1 and trajectory[-1][1].item() == 1:
            return 5.0
        else:
            return 0.0


class TestRolloutFunction:

    @staticmethod
    def _dynamics_function(state, action):
        return state + action

    def _set_up(self):
        func = RolloutFunction(self._dynamics_function, [], STATE_DIMEN, ACTION_DIMEN, TIME_HORIZON)
        initial_state = torch.tensor([0, 0], dtype=torch.double)
        means = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.double)
        stds = torch.zeros((TIME_HORIZON, ACTION_DIMEN), dtype=torch.double)

        return func, initial_state, means, stds

    def test__perform_rollout__trajectory_correct(self):
        func, initial_state, means, stds = self._set_up()

        trajectory, _, _ = func.perform_rollout((initial_state, means, stds))

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

        _, actions, _ = func.perform_rollout((initial_state, means, stds))

        assert actions.shape == (TIME_HORIZON, ACTION_DIMEN)

        assert actions[0][0] == 1
        assert actions[0][1] == 0

        assert actions[1][0] == 0
        assert actions[1][1] == 1

        assert actions[2][0] == 0
        assert actions[2][1] == 0

    def test__perform_rollout__cost_correct(self):
        _, initial_state, means, stds = self._set_up()
        func = RolloutFunction(self._dynamics_function, [TestConstraint()], STATE_DIMEN, ACTION_DIMEN, TIME_HORIZON)

        _, _, cost = func.perform_rollout((initial_state, means, stds))

        assert cost == 5

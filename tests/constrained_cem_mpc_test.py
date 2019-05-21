import numpy as np
import torch
from polytope import polytope

from constrained_cem_mpc import TerminalConstraint, ObstaclesConstraint, ActionConstraint


class TestTerminalConstraint:
    def test__misses_constraint__returns_l2_distance(self):
        constraint = polytope.box2poly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        # A trajectory that passes through the terminal constraint but does not end in it.
        trajectory = torch.tensor([[0, 0], [10.5, 10.5], [0, 0]])

        cost = func(trajectory, actions=None)

        distance = np.sqrt(np.square(10.5) + np.square(10.5))
        assert cost == distance

    def test__hits_constraint__returns_zero(self):
        constraint = polytope.box2poly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        trajectory = torch.tensor([[0, 0], [0, 1], [10.5, 10.5]])

        cost = func(trajectory, actions=None)

        assert cost == 0


class TestObstaclesConstraint:
    def test__misses_obstacle__returns_zero(self):
        obstacle1 = polytope.box2poly([[1, 2], [1, 2]])
        obstacle2 = polytope.box2poly([[2, 3], [2, 3]])
        func = ObstaclesConstraint([obstacle1, obstacle2])
        trajectory = torch.Tensor([[0, 0], [0.5, 0.5]])

        cost = func(trajectory, actions=None)

        assert cost == 0

    def test__hits_obstacle__returns_high_cost(self):
        obstacle1 = polytope.box2poly([[1, 2], [1, 2]])
        obstacle2 = polytope.box2poly([[2, 3], [2, 3]])
        func = ObstaclesConstraint([obstacle1, obstacle2])
        # Miss the first obstacle, hit the second.
        trajectory = torch.Tensor([[0, 0], [2.5, 0], [2.5, 2.5]])

        cost = func(trajectory, actions=None)

        assert cost >= 0


class TestActionConstraint:
    def test__all_actions_safe__returns_zero(self):
        safe_area = polytope.box2poly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.Tensor([[0, 0], [0.5, 0.2], [-0.9, -0.99]])

        cost = func(trajectory=None, actions=actions)

        assert cost == 0

    def test__one_action_unsafe__returns_single_penalty(self):
        safe_area = polytope.box2poly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.Tensor([[0, 0], [1.1, 0.0], [-0.9, -0.99]])

        cost = func(trajectory=None, actions=actions)

        assert cost == 5

    def test__two_actions_unsafe__returns_double_penalty(self):
        safe_area = polytope.box2poly([[-1, 1], [-1, 1]])
        func = ActionConstraint(safe_area, penalty=5)
        actions = torch.Tensor([[0, 0], [-2, -3], [0.5, 1.2]])

        cost = func(trajectory=None, actions=actions)

        assert cost == 10

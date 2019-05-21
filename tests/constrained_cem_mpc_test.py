import numpy as np
import torch
from polytope import polytope

from constrained_cem_mpc import TerminalConstraint


class TestTerminalConstraint():
    def test__misses_constraint__returns_l2_distance(self):
        constraint = polytope.box2poly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        # A trajectory that passes through the terminal constraint but does not end in it.
        trajectory = torch.tensor([[0, 0], [10.5, 10.5], [0, 0]])

        cost = func(trajectory)

        distance = np.sqrt(np.square(10.5) + np.square(10.5))
        assert cost == distance

    def test__hits_constraint__returns_zero(self):
        constraint = polytope.box2poly([[10, 11], [10, 11]])
        func = TerminalConstraint(constraint)
        trajectory = torch.tensor([[0, 0], [0, 1], [10.5, 10.5]])

        cost = func(trajectory)

        assert cost == 0

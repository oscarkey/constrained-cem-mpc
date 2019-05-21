import torch

import demo
import numpy as np


class TestDemo(object):
    def test_constraint_func_terminal_miss_returns_high_cost(self):
        # A trajectory that passes through the terminal constraint but does not end in it.
        constraint_center = demo.terminal_constraint.chebXc
        trajectory = torch.tensor([[0, 0], constraint_center, [0, 0]])

        cost = demo.constraint_cost(trajectory)

        distance = np.sqrt(np.square(constraint_center[0]) + np.square(constraint_center[1]))
        assert cost == distance

    def test_constraint_func_terminal_hit_returns_zero(self):
        constraint_center = demo.terminal_constraint.chebXc
        trajectory = torch.tensor([[0, 0], [0, 1], constraint_center])

        cost = demo.constraint_cost(trajectory)

        assert cost == 0

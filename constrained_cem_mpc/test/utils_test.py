import pytest
import torch

import utils


def test_assert_shape_correct_does_nothing():
    utils.assert_shape(torch.zeros((10, 20, 30)), (10, 20, 30))


def test_assert_shape_incorrect_throws():
    with pytest.raises(ValueError):
        utils.assert_shape(torch.zeros((10, 20, 30)), (10, 20, 31))
from typing import Tuple

import torch


def assert_shape(x: torch.tensor, shape: Tuple) -> None:
    if x.shape != shape:
        raise ValueError(f'Wanted shape {shape}, got {x.shape}')

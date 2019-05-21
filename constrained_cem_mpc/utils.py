import torch


def assert_shape(x: torch.tensor, shape: tuple) -> bool:
    if x.shape != shape:
        raise ValueError(f'Wanted shape {shape}, got {x.shape}')
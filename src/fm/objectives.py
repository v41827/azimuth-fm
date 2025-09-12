import torch

def linear_path_xt(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    return (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1

def target_velocity_const(x0: torch.Tensor, x1: torch.Tensor):
    return x1 - x0
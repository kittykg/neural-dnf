import torch
from torch import Tensor


def mutex_tanh(x: Tensor, c: float | None) -> Tensor:
    """
    y_i = MutexTanh(d)_i = 2 * \\frac{e^{d_i}}{\sum_{j}e^{d_j}} - 1
    y_i \in (-1, 1)
    """
    if c is None:
        exp_sum = torch.exp(x).sum(dim=-1, keepdim=True)
        return 2 * torch.exp(x) / exp_sum - 1

    exp_sum = torch.exp(x + c).sum(dim=-1, keepdim=True)
    return (2 * torch.exp(x + c) / exp_sum) - 1

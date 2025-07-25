from enum import Enum
from typing import Callable

import torch
from torch import nn, Tensor

from neural_dnf.common import mutex_tanh

WEIGHT_INIT_TYPES = ["normal", "x_normal", "x_uniform", "orthogonal", "uniform"]


class SemiSymbolicLayerType(Enum):
    """
    Semi-symbolic layer, implemented based on pix2rule
    y = f(x @ W.T + beta), beta = delta * (max(abs(W)) - sum(abs(W)))
    Where f is tanh and delta is in range [-1, 1]
    delta = 1: conjunction
    delta = -1: disjunction
    """

    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"


class BaseSemiSymbolic(nn.Module):
    layer_type: SemiSymbolicLayerType

    in_features: int
    out_features: int

    weights: nn.Parameter
    delta: float
    weight_init_type: str

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str,
    ) -> None:
        super().__init__()

        self.layer_type = layer_type

        self.in_features = in_features  # P
        self.out_features = out_features  # Q

        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )

        assert (
            weight_init_type in WEIGHT_INIT_TYPES
        ), f"Invalid weight_init_type: {weight_init_type}"
        self.weight_init_type = weight_init_type
        layer_init_fn = self._get_layer_init_fn()
        layer_init_fn(self.weights)
        self.delta = delta

    def _get_layer_init_fn(self) -> Callable[[nn.Parameter], Tensor]:
        if self.weight_init_type == "normal":
            return lambda x: nn.init.normal_(x, mean=0.0, std=0.1)
        elif self.weight_init_type == "x_normal":
            return lambda x: nn.init.xavier_normal_(
                x, gain=nn.init.calculate_gain("tanh")
            )
        elif self.weight_init_type == "x_uniform":
            return lambda x: nn.init.xavier_uniform_(
                x, gain=nn.init.calculate_gain("tanh")
            )
        elif self.weight_init_type == "orthogonal":
            return lambda x: nn.init.orthogonal_(
                x, gain=nn.init.calculate_gain("tanh")  # type: ignore
            )
        else:
            return lambda x: nn.init.uniform_(x, a=-6, b=6)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"in_features={self.in_features}",
                f"out_features={self.out_features}",
                f"layer_type={self.layer_type},"
                f"current_delta={self.delta:.2f}",
            ]
        )

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P

        bias = self.bias_calculation()
        # bias: Q

        out = input @ self.weights.T
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum

    def bias_calculation(self) -> Tensor:
        abs_weight = torch.abs(self.weights)
        # abs_weight: Q x P
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        # max_abs_w: Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: Q
        if self.layer_type == SemiSymbolicLayerType.CONJUNCTION:
            bias = max_abs_w - sum_abs_w
        else:
            bias = sum_abs_w - max_abs_w
        # bias: Q
        return bias


class SemiSymbolic(BaseSemiSymbolic):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str = "normal",
    ) -> None:
        super(SemiSymbolic, self).__init__(
            in_features, out_features, layer_type, delta, weight_init_type
        )


class SemiSymbolicMutexTanh(BaseSemiSymbolic):
    """
    Normal SemiSymbolic layer but with mutex-tanh activation.
    """

    mutex_tanh_constant: float | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str = "normal",
        mutex_tanh_constant: float | None = None,
    ) -> None:
        super().__init__(
            in_features, out_features, layer_type, delta, weight_init_type
        )
        self.mutex_tanh_constant = mutex_tanh_constant

    def forward(self, input: Tensor) -> Tensor:
        out = super().forward(input)
        return mutex_tanh(out, self.mutex_tanh_constant)

    def get_raw_output(self, input: Tensor) -> Tensor:
        return super().forward(input)

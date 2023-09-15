from typing import Callable, Literal, Optional, Union, List, Tuple
from torch.nn import Parameter, Linear
from torch import Tensor
import torch.nn.functional as F
import torch

class LinearIdentity(Linear):
    def __init__(
        self,
        in_features: int,
        freeze_params: bool,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__(in_features, in_features, bias, device, dtype)
        if not freeze_params:
            self.register_buffer("weight_mask", torch.eye(in_features))
        self.weight = Parameter(
            torch.ones((in_features, in_features)), requires_grad=not freeze_params
        )
        if bias:
            self.bias = Parameter(
                torch.zeros(in_features), requires_grad=not freeze_params
            )

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight * self.weight_mask, self.bias)
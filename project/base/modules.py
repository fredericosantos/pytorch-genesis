from typing import List, Optional, Union
from torch import BoolTensor, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Flatten, Identity, Sequential, Parameter, LogSoftmax, Softmax, InstanceNorm1d, BatchNorm1d
from modules.hyperparams_options import *
from modules.base.linear_sparse import LinearSparse

import modules.param_init as param_init


class Layer(Module):
    def __init__(
        self,
        weight_init: Optional[WEIGHT_INITS],
        weight_init_hparams: dict,
        bias_init: Optional[BIAS_INITS],
        bias_init_hparams: dict,
    ):
        super().__init__()
        self.pooling = None
        self.reshape = None
        self.regularization = None
        self.activation = None
        self.w_init_fn_ = param_init.get_weight_init_fn_(weight_init)
        self.w_init_hparams = weight_init_hparams
        self.b_init_fn_ = param_init.get_bias_init_fn(bias_init)
        self.b_init_hparams = bias_init_hparams
        self.out_features: int
        self.out_channels: int
        self.out_width: int
        self.out_height: int

    def forward(self, input: Tensor, identity: Optional[Tensor] = None) -> Tensor:
        if self.pooling is not None:
            input = self.pooling(input)
        if self.reshape is not None:
            input = self.reshape(input)
        input = self.main(input)
        if self.regularization is not None:
            input = self.regularization(input)
        if identity is not None:
            input += identity
        if self.activation is not None:
            input = self.activation(input)
        return input

    def freeze(self):
        self.requires_grad_(False)

    def unfreeze(self):
        self.requires_grad_(True)

    def reset_parameters(self):
        for module in ["pooling", "reshape", "main"]:
            m = getattr(self, module)
            if m is not None:
                if hasattr(m, "weight") and m.weight is not None:
                    if self.w_init_fn_ is not None:
                        self.w_init_fn_(m.weight, **self.w_init_hparams)
                    else:
                        m.reset_parameters()
                if hasattr(m, "bias") and m.bias is not None:
                    if self.b_init_fn_ is not None:
                        self.b_init_fn_(m.bias, **self.b_init_hparams)
                    else:
                        m.reset_parameters()


class OutputLinear(LinearSparse):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        one_class_only: bool,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            sparse=one_class_only,
            device=device,
            dtype=dtype,
        )
        if one_class_only:
            # Only one class is allowed to receive inputs.
            output_idx = torch.randint(0, 10, (1,), device=device).repeat((1, self.in_features))
            weight_mask = torch.zeros_like(self.weight, device=device)
            weight_mask.scatter_(0, output_idx, 1)
            self.register_buffer("weight_mask", weight_mask)
            self._register_bias_mask()

        self._update_sparsity()


class Reshape(Module):
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(-1, self.channels, self.height, self.width)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, height={self.height}, width={self.width}"
    

class MaxNorm1d(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x / x.abs().max(1, keepdim=True).values


class Norm1d(Module):
    def __init__(self, p: int = 1):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=self.p, dim=1)


class MeanNorm1d(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x - x.mean(1, keepdim=True)

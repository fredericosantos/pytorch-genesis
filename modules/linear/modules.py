from typing import Optional, Literal
import torch as torch

# Model building
from torch import Tensor
from modules.base.linear_sparse import LinearSparse
from modules.hyperparams_options import *


class MainLinear(LinearSparse):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        sparse: bool,
        sparsity: float,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, sparse=sparse, device=device, dtype=dtype,
        )
        if sparse:
            weight_mask = torch.zeros_like(self.weight)
            self.register_buffer("weight_mask", weight_mask)
            self.add_connections(sparsity, device)
            self._register_bias_mask()
        else:
            self._update_sparsity()


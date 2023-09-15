from typing import Optional
import torch as torch

# Model building
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.nn import functional as F


class LinearSparse(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        sparse: bool,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        self.sparse = sparse
        self._bias = bias
        self.dtype = dtype

        super().__init__(in_features, out_features, bias, device, dtype)
        self._update_sparsity()

    def forward(self, input: Tensor) -> Tensor:
        weight = (
            (self.weight * self.weight_mask)
            if hasattr(self, "weight_mask")
            else self.weight
        )
        bias = (
            None
            if not self._bias
            else (self.bias * self.bias_mask)
            if hasattr(self, "bias_mask")
            else self.bias
        )
        return F.linear(
            input=input,
            weight=weight,
            bias=bias,
        )

    def add_connections(
        self, sparsity: float = 0.0, device: Optional[str] = None
    ) -> None:
        n_avail_cxts = self.available_connections.sum()
        total_cxts = n_avail_cxts + self.weight_mask.data.sum()
        n_cxts = self._calculate_num_cxts(sparsity, n_avail_cxts, total_cxts)
        cxts_counter = 0
        weight_mask_og = self.weight_mask.clone()
        while cxts_counter < n_cxts:
            missing_cxts = n_cxts - cxts_counter
            neurons_prob = torch.ones_like(self.available_nodes) / len(
                self.available_nodes
            )
            idx = torch.multinomial(neurons_prob, missing_cxts, replacement=True)
            chosen_neurons, cxts_per_neuron = torch.unique(
                self.available_nodes[idx], return_counts=True
            )

            # This ensures that each neuron has at least one connection
            if cxts_counter == 0 and self.sparsity == 1:
                cxts_per_neuron = torch.ones_like(self.available_nodes)
                chosen_neurons = self.available_nodes

            for i, n in enumerate(chosen_neurons):
                avail_cxts_n = self.available_connections[n].sum()
                if avail_cxts_n <= cxts_per_neuron[i]:
                    self.weight_mask.data[n, self.available_connections[n]] = 1
                    cxts_counter += avail_cxts_n
                    self.cxt_probs[n, self.available_connections[n]] = False
                else:
                    choices = torch.multinomial(self.cxt_probs[n], cxts_per_neuron[i])
                    self.weight_mask.data[n, choices] = 1
                    self.cxt_probs[n, choices] = 0
                    self.cxt_probs[n] = self.cxt_probs[n] / self.cxt_probs[n].sum()
                    cxts_counter += cxts_per_neuron[i]
            self._register_weight_mask(self.weight_mask)
        mask = (self.weight_mask - weight_mask_og) == 1
        self._register_weight_mask(mask)

    def remove_connections(
        self, sparsity: float = 1.0, device: Optional[str] = None
    ) -> None:
        """
        Remove connections on the module.
        Set `sparsity` to `1.0` to remove all connections from module
        except one connection per neuron.
        Args:
            sparsity (float): The sparsity of the module.
        """

        n_avail_cxts = self.weight_mask.data.sum()
        total_cxts = n_avail_cxts + self.available_connections.sum()
        density = 1 - sparsity
        n_cxts = self._calculate_num_cxts(density, n_avail_cxts, total_cxts)
        idx = (self.weight_mask.data != 0).nonzero()
        idx = idx[torch.randperm(len(idx), device=device)]
        cxt_counter = 0
        for row, col in idx:
            if self.weight_mask.data[row].sum() > 1:
                self.weight_mask.data[row][col] = 0
                cxt_counter += 1
            if cxt_counter == n_cxts:
                break
        self._update_sparsity()

    def _update_sparsity(self):
        ifw = torch.ones_like(self.weight)
        if hasattr(self, "weight_mask"):
            ifw *= self.weight_mask.data == 0

        self.cxt_probs = torch.divide(
            ifw,
            ifw.sum(1, keepdim=True),
            out=ifw,
        ).nan_to_num_(0.0)

        self.available_connections = self.cxt_probs != 0
        self.available_nodes = self.available_connections.sum(1).nonzero().flatten()

        if self.sparse:
            total_cxts = torch.numel(self.weight) * self.out_features
            self.register_buffer(
                "sparsity", self.available_connections.sum() / total_cxts
            )
        else:
            self.register_buffer("sparsity", torch.tensor(0.0))

    def _register_weight(self, weight: Tensor, requires_grad: bool = True):
        self.register_parameter(
            "weight", Parameter(weight, requires_grad=requires_grad)
        )
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def _register_weight_mask(self, mask: Tensor):
        self.register_buffer("weight_mask", mask)
        self._register_bias_mask()
        self._update_sparsity()

    def _register_bias(self, bias: Tensor, requires_grad: bool = True):
        if self._bias:
            delattr(self, "bias")
            with torch.no_grad():
                self.register_parameter(
                    "bias", Parameter(bias, requires_grad=requires_grad)
                )

    def _register_bias_mask(self):
        if self._bias and hasattr(self, "weight_mask"):
            self.register_buffer("bias_mask", self.weight_mask.sum(1).clip(max=1))

    def _calculate_num_cxts(
        self,
        sparsity: float,
        n_avail_cxts: int,
        total_cxts: int,
    ):
        assert 0 <= sparsity <= 1, f"{sparsity = } must be between 0 and 1"
        density = 1 - sparsity
        n_cxts = min(n_avail_cxts, max(1, int(total_cxts * density)))
        return n_cxts

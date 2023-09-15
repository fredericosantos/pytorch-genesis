from typing import Literal, List, Optional

from torch import Tensor
from torch.nn import (
    Module,
    Identity,
    LogSoftmax,
    Softmax,
)


from modules.hyperparams_options import ACTIVATIONS, REGULARIZATIONS
import modules.activation_functions as activations
import modules.regularization_functions as regularizations


class MergeBlock(Module):
    def __init__(
        self,
        num_classes: int,
        regularization: Optional[REGULARIZATIONS],
        type_softmax: Optional[Literal["softmax", "logsoftmax"]],
        activation_fn: Optional[ACTIVATIONS],
        activation_fn_hparams: Optional[dict],
    ):
        super().__init__()
        if regularization is not None:
            reg_hparams = regularizations.get_hparams(
                fn=regularization,
                block_type="linear",
                num_features=num_classes,
            )
            reg = regularizations.get_module(
                fn=regularization, block_type="linear", **reg_hparams
            )
        else:
            reg = Identity()
        self.add_module("regularization", reg)
        self.add = Add()
        self.activation = activations.get_module(activation_fn, activation_fn_hparams)
        if type_softmax is None:
            self.softmax = Identity()
        elif type_softmax == "softmax":
            self.softmax = Softmax(dim=1)
        elif type_softmax == "logsoftmax":
            self.softmax = LogSoftmax(dim=1)

    def forward(self, blocks_y_hat: List[Tensor]) -> Tensor:
        y_hat = blocks_y_hat[0]
        if len(blocks_y_hat) != 1:
            for i in range(1, len(blocks_y_hat)):
                y_hat = self.softmax(y_hat)
                y_hat = self.add(y_hat, self.softmax(blocks_y_hat[i]))
                y_hat = self.regularization(y_hat)
        y_hat = self.activation(y_hat)
        return y_hat


class Add(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        return x_0 + x_1

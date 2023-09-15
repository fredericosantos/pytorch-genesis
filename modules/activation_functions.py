from typing import (
    Callable,
    Optional,
)
from torch import Tensor
from torch.nn import Module, ReLU, Tanh, Sigmoid, LeakyReLU, GELU, Identity
from modules.hyperparams_options import ACTIVATIONS


def get_module(fn: Optional[ACTIVATIONS], hparams: Optional[dict] = None) -> Callable:
    if fn is None:
        return Identity()
    elif hparams is None:
        hparams = get_hparams(fn)

    if fn.lower() == "ReLU".lower():
        return ReLU(**hparams)
    elif fn.lower() == "GELU".lower():
        return GELU()
    elif fn.lower() == "Sigmoid".lower():
        return Sigmoid()
    elif fn.lower() == "Tanh".lower():
        return Tanh()
    elif fn.lower() == "LeakyReLU".lower():
        return LeakyReLU(**hparams)
    else:
        raise NotImplementedError(
            f"""
            Activation function {fn} not implemented.
            """
        )


def get_hparams(fn: Optional[ACTIVATIONS]) -> dict:
    if fn is None:
        return {}
    elif fn.lower() == "ReLU".lower():
        inplace: bool = True
        return dict(inplace=inplace)
    elif fn.lower() == "GELU".lower():
        return {}
    elif fn.lower() == "Sigmoid".lower():
        return {}
    elif fn.lower() == "Tanh".lower():
        return {}
    elif fn.lower() == "LeakyReLU".lower():
        negative_slope: float = 0.01
        inplace: bool = True
        return dict(negative_slope=negative_slope, inplace=inplace)
    else:
        raise NotImplementedError(
            f"""
            Activation function {fn} not implemented.
            """
        )


class AoN(Module):
    r"""Applies All-or-Nothing (AoN) activation function.

    All-or-Nothing (AoN) is an activation function with the following formula:


    .. math::
        \text{AoN}(x) = \begin{cases}
            \alpha & \text{if } x > 0 \\
            0 & \text{if } x <= 0
        \end{cases}

    :math:`\alpha` is a parameter that controls the output range. By default
    :math:`\alpha=0.45`.

    Args:
        alpha (float): the :math:`\alpha` value for the activation function. 
        Default: 0.45

        inplace (bool, optional): can optionally do the operation in-place. 
        Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    """
    __constants__ = ["alpha", "inplace"]

    alpha: float
    inplace: bool

    def __init__(
        self,
        alpha: float = 0.45,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.inplace = inplace
        assert self.alpha > 0, "alpha must be positive"

    def forward(self, input: Tensor) -> Tensor:
        input[input > 0] = self.alpha
        input[input <= 0] = 0
        return input

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"alpha={self.alpha}" + inplace_str

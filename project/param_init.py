import math
from typing import Any, Callable, Literal, Optional, Union, List, Tuple

# Model building
import torch
from torch import Tensor
import torch.nn as nn
import warnings
from modules.hyperparams_options import *

from pytorch_lightning import LightningModule


def get_weight_init_fn_(init_fn: Optional[WEIGHT_INITS]) -> Callable:
    if init_fn is None:
        return
    elif init_fn.lower() == "xavier_uniform":
        return nn.init.xavier_uniform_
    elif init_fn.lower() == "xavier_normal":
        return nn.init.xavier_normal_
    elif init_fn.lower() == "kaiming_uniform":
        return nn.init.kaiming_uniform_
    elif init_fn.lower() == "kaiming_normal":
        return nn.init.kaiming_normal_
    elif init_fn.lower() == "uniform":
        return nn.init.uniform_
    elif init_fn.lower() == "normal":
        return nn.init.normal_
    else:
        raise NotImplementedError(
            f"""
            Initialization function {init_fn} not implemented.
            """
        )


def get_weight_init_hparams(init: Optional[WEIGHT_INITS], activation_fn: Optional[ACTIVATIONS]):
    if init is None:
        return {}
    elif init.lower() in ["kaiming_normal", "kaiming_uniform"]:
        if activation_fn is None:
            hparams = dict()
        elif activation_fn.lower() == "ReLU".lower():
            hparams = dict(mode="fan_in", nonlinearity="relu",)
        elif activation_fn.lower() == "LeakyReLU".lower():
            hparams = dict(mode="fan_in", nonlinearity="leaky_relu",)
        elif activation_fn.lower() == "GELU".lower():
            hparams = dict(mode="fan_in", nonlinearity="selu",)
        elif activation_fn.lower() == "relu1".lower():
            hparams = dict(mode="fan_in", nonlinearity="tanh",)
        elif activation_fn.lower() == "tanh".lower():
            hparams = dict(mode="fan_in", nonlinearity="tanh",)
        elif activation_fn.lower() == "sigmoid".lower():
            hparams = dict(mode="fan_in", nonlinearity="sigmoid",)
        else:
            raise NotImplementedError(
                f"""
                Initialization function {init} not implemented for activation function {activation_fn}.
                """
            )
    elif init.lower() in ["xavier_normal", "xavier_uniform"]:
        if activation_fn is None:
            hparams = dict(gain=_calculate_gain(None))
        elif activation_fn.lower() == "ReLU".lower():
            hparams = dict(gain=_calculate_gain("relu"))
        elif activation_fn.lower() == "LeakyReLU".lower():
            hparams = dict(gain=_calculate_gain("leaky_relu"))
        elif activation_fn.lower() == "GELU".lower():
            hparams = dict(gain=_calculate_gain("gelu"))
        elif activation_fn.lower() == "relu1".lower():
            hparams = dict(gain=_calculate_gain("tanh"))
        elif activation_fn.lower() == "tanh".lower():
            hparams = dict(gain=_calculate_gain("tanh"))
        elif activation_fn.lower() == "sigmoid".lower():
            hparams = dict(gain=_calculate_gain("sigmoid"))
        else:
            raise NotImplementedError(
                f"""
                Initialization function {init} not implemented for activation function {activation_fn}.
                """
            )
    else:
        raise NotImplementedError(
            f"""
            Initialization function {init} not implemented.
            """
        )
    return hparams


def get_bias_init_fn(init_fn: Optional[BIAS_INITS]) -> Callable:
    if init_fn is None:
        return 
    if init_fn.lower() == "zeros":
        return nn.init.zeros_
    else:
        raise NotImplementedError(
            f"""
            Initialization function {init_fn} not implemented.
            """
        )


def get_bias_init_hparams(init: Optional[BIAS_INITS]):
    if init is None:
        return {}
    if init.lower() == "zeros":
        hparams = {}
    else:
        raise NotImplementedError(
            f"""
            Initialization function {init} not implemented.
            """
        )
    return hparams


def xs_randn(model: LightningModule, input: Optional[Tensor] = None) -> Tensor:
    model.hook_xs = True
    with torch.no_grad():
        input = torch.randn_like(model.example_input_array, device=model.device) if input is None else input
        model(input)
        inputs = model.hook_xs
        del model.hook_xs
    return inputs


# NOT USED
def fs_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: Optional[NONLINEARITIES] = "relu",
):
    """Forward Stabilizer normal initialization
    Args:
        tensor (Tensor): tensor to be initialized
        a (float): If ``'nonlinearity'`` is ``'leaky_relu'``, the negative slope 
        of the rectifier used after this layer. If ``'nonlinearity'`` is ``'relu'``,
        it represents the stabilizing value to add to the gain.
        nonlinearity (str): nonlinearity function to be applied after this layer. Default ``'relu'``.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def fs_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: Optional[NONLINEARITIES] = "relu",
):
    """Forward Stabilizer uniform initialization
    Args:
        tensor (Tensor): tensor to be initialized
        a (float): If ``'nonlinearity'`` is ``'leaky_relu'``, the negative slope 
        of the rectifier used after this layer. If ``'nonlinearity'`` is ``'relu'``,
        it represents the stabilizing value to add to the gain.
        nonlinearity (str): nonlinearity function to be applied after this layer. Default ``'relu'``.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def fs_uniform2_(
    tensor: Tensor, a: float = 0.0,
):
    """Forward Stabilizer uniform initialization
    Args:
        tensor (Tensor): tensor to be initialized
        a (float): If ``'nonlinearity'`` is ``'leaky_relu'``, the negative slope 
        of the rectifier used after this layer. If ``'nonlinearity'`` is ``'relu'``,
        it represents the stabilizing value to add to the gain.
        nonlinearity (str): nonlinearity function to be applied after this layer. Default ``'relu'``.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = a * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * (std)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def calculate_bound(
    tensor: Tensor, a: float, mode: Literal["fan_in", "fan_out"], nonlinearity: NONLINEARITIES = "relu",
):
    """Calculate the bound for uniform initialization
    Args:
        tensor (Tensor): tensor to be initialized
        a (float): If ``'nonlinearity'`` is ``'leaky_relu'``, the negative slope 
        of the rectifier used after this layer. If ``'nonlinearity'`` is ``'relu'``,
        it represents the stabilizing value to add to the gain.
        nonlinearity (str): nonlinearity function to be applied after this layer. Default ``'relu'``.
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return bound


def _calculate_gain(
    nonlinearity: Optional[NONLINEARITIES], param=None,
):
    if nonlinearity == "sigmoid" or nonlinearity is None:
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        param = param or 0
        return math.sqrt(2.0) + param
    elif nonlinearity == "leaky_relu":
        param = 0.01
        if not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    elif nonlinearity == "gelu":
        param = param or 0.01
        if not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

from typing import Union, List, Tuple
import torch as torch
import random
from numpy.random import choice

# Model building
from torch import FloatTensor, Tensor
import torch.nn as nn
from torch.nn import Module, Parameter, Linear, init
from torch.nn import functional as F


def conv_kernel(
    i: int, radius: float, in_features: int, device=None, experiment_sq: int = 1
):
    """Convolution kernel for a single neuron"""
    width = int(torch.sqrt(torch.tensor(in_features, device=device)).item())
    z = torch.zeros((width, width), dtype=torch.float32, device=device)
    idx = torch.nonzero(z + 1)[i]
    ci, cj = idx
    I, J = torch.meshgrid(
        torch.arange(width, device=device), torch.arange(width, device=device),
    )
    dist = torch.sqrt((I - ci) ** 2 + (J - cj) ** 2)
    z[torch.where(dist <= radius)] = (radius + 1 - dist)[
        torch.where(dist <= radius)
    ] / (radius + 1)
    z = z ** experiment_sq
    return z.flatten()


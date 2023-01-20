from typing import Any, Literal, Optional, Union
import warnings
import torch
from torch import Tensor
from torch.nn import MSELoss, Module, NLLLoss, CrossEntropyLoss
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F
import torch.nn as nn
from modules.hyperparams_options import *


def get_loss_fn(loss_type: LOSSES, hparams: dict = {}) -> Module:
    if loss_type == "cross_entropy":
        return CrossEntropyLoss(**hparams)
    elif loss_type == "rmse":
        return RSMELoss(**hparams)
    elif loss_type == "mse":
        return MSELoss(**hparams)
    elif loss_type == "mae":
        return nn.L1Loss(**hparams)
    elif loss_type == "relu1_mse":
        return MSELoss(**hparams)
    elif loss_type.lower() == "nllloss":
        return NLLLoss(**hparams)
    elif loss_type == "polyloss":
        return PolyLoss(**hparams)
    elif loss_type == "focalloss":
        return FocalLoss(**hparams)
    elif loss_type == "synthloss":
        return SynthLoss(**hparams)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class ReLU1_MSELoss(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(torch.clamp(input, min=0, max=1), target)


class RSMELoss(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse = super().forward(input, target)
        return torch.sqrt(mse)


class MSELossClamp(MSELoss):
    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            # when prediction of 0 is correct
            condition0 = target == 0
            condition2 = input < 0
            # when it predicts that it is NOT the class
            condition1 = input < 0
            input[condition0 & condition2 & condition1] = input[condition0 & condition2] / (self.num_classes - 1)
            mask = torch.ones_like(input)
            mask[condition0 & condition2 & condition1] = 0
            self.register_buffer("mask", mask)
            # mask[(target == 0) & (input < 0)] /= 10

        input = input * self.mask
        self.input = input.detach().clone()
        return super().forward(input, target)


class MSELossDivClass(MSELoss):
    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            # when the prediction is not for the target class
            # and the prediction is not the class
            condition0 = (target == 0) & (input < 0)
            condition1 = input < 0
            # when it predicts that it is NOT the class
            mask = torch.ones_like(input)
            mask[condition0] = mask[condition0] / (self.num_classes - 1)
            self.register_buffer("mask", mask)

        input = input * self.mask
        self.register_buffer("_input", input)
        return super().forward(input, target)


class preCEL(_Loss):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            # when the prediction is not for the target class
            # and the prediction is not the class
            condition0 = target == 0
            condition1 = input < 0
            # when it predicts that it is NOT the class
            mask = torch.ones_like(input)
            mask[condition0 & condition1] = mask[condition0 & condition1] / (self.num_classes - 1)
            self.register_buffer("mask", mask)

        input = input * self.mask
        return input


class PolyLoss(CrossEntropyLoss):
    def __init__(
        self,
        epsilon: float = 1,
        weight: Optional[Tensor] = None,
        size_average: Optional[Any] = None,
        ignore_index: int = -100,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
        label_smoothing: float = 0,
    ):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        self.reduction = reduction
        self.epsilon = epsilon
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        ce = super().forward(outputs, targets)
        pt = F.one_hot(targets, num_classes=outputs.size()[1]) * self.softmax(outputs)
        if self.reduction == "mean":
            return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()
        elif self.reduction == "sum":
            return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).sum()
        else:
            (ce + self.epsilon * (1.0 - pt.sum(dim=1)))


class FocalLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__(weight=alpha, reduction=reduction)
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = super().forward(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class SynthLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, alpha=None, a=1, b=1, c=1, gamma=2, epsilon=0, g=1, f=1, reduction="mean"):
        super().__init__(weight=alpha, reduction=reduction)
        self.a = a
        self.b = b
        self.c = c
        self.gamma = gamma
        self.epsilon = epsilon
        self.g = g
        self.f = f

    def forward(self, input, target):
        ce_loss = self.a * (super().forward(input, target) ** self.b)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.c - pt) ** self.gamma
        poly_loss = self.epsilon * ((self.g - pt) ** self.f)
        synth_loss = focal_loss * ce_loss + poly_loss
        return synth_loss

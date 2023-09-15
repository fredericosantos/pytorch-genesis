from typing import Callable, Optional, Literal, get_args
from torch.nn import (
    Identity,
    LayerNorm,
    BatchNorm1d,
    BatchNorm2d,
    InstanceNorm1d,
    InstanceNorm2d,
    Dropout,
    Dropout2d,
    GroupNorm,
)
from modules.hyperparams_options import (
    REGULARIZATIONS,
    BLOCK_TYPES,
    CONV_TYPES,
    LINEAR_TYPES,
)

DYNAMIC_NORM: Literal = Literal["BatchNorm", "InstanceNorm"]


def adjust_fn_to_block_type(fn: DYNAMIC_NORM, block_type: BLOCK_TYPES) -> dict:
    if block_type in get_args(CONV_TYPES):
        return fn + "2d"
    elif block_type in get_args(LINEAR_TYPES):
        return fn + "1d"


def get_module(
    fn: Optional[REGULARIZATIONS] = None,
    block_type: Optional[BLOCK_TYPES] = None,
    **kwargs,
) -> Callable:
    if fn is None or fn == "None":
        return Identity()
    elif fn == "LayerNorm":
        return LayerNorm(**kwargs)
    elif fn == "GroupNorm":
        if block_type in get_args(CONV_TYPES):
            return GroupNorm(**kwargs)
        elif block_type in get_args(LINEAR_TYPES):
            raise ValueError(
                f"""GroupNorm cannot be implemented for {block_type} blocks."""
            )
    else:
        fn = adjust_fn_to_block_type(fn, block_type)
        if fn == "BatchNorm1d":
            return BatchNorm1d(**kwargs)
        elif fn == "BatchNorm2d":
            return BatchNorm2d(**kwargs)
        elif fn == "InstanceNorm1d":
            return InstanceNorm1d(**kwargs)
        elif fn == "InstanceNorm2d":
            return InstanceNorm2d(**kwargs)
        elif fn == "Dropout1d":
            return Dropout(**kwargs)
        elif fn == "Dropout2d":
            return Dropout2d(**kwargs)
        else:
            raise NotImplementedError(
                f"""Regularization function {fn} not implemented."""
            )


def get_hparams(
    fn: Optional[REGULARIZATIONS],
    block_type: Optional[BLOCK_TYPES] = None,
    num_features: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> dict:
    if fn is None:
        return {}
    factory_kwargs = dict(device=device, dtype=dtype)
    if fn == "LayerNorm":
        if block_type in get_args(CONV_TYPES):
            return dict(
                normalized_shape=[num_features, height, width], **factory_kwargs
            )
        elif block_type in get_args(LINEAR_TYPES):
            return dict(normalized_shape=[num_features], **factory_kwargs)
    if fn == "GroupNorm":
        if block_type in get_args(CONV_TYPES):
            return dict(
                num_groups=num_features // 2,
                num_channels=num_features,
                **factory_kwargs,
            )
        elif block_type in get_args(LINEAR_TYPES):
            raise ValueError(
                f"""GroupNorm cannot be implemented for {block_type} blocks."""
            )
    elif fn == "Dropout":
        return dict(p=0.2)
    elif fn == "BatchNorm" or fn == "InstanceNorm":
        return dict(num_features=num_features, **factory_kwargs)
    else:
        raise NotImplementedError(f"""Regularization function {fn} not implemented.""")

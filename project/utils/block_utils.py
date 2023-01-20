from typing import Literal, Optional, Tuple, Union, get_args
import random

OUTPUT_DIMENSIONS_TYPES = Literal["random", "same", "half"]
CHANNEL_DIMENSIONS_TYPES = Literal["same", "double", "half"]
KERNEL_DIMENSIONS_TYPES = Literal["equal", "random"]


def pair(t: Union[int, tuple]) -> tuple:
    return t if isinstance(t, tuple) else (t, t)


def random_layer_hparams(in_channels: int, in_height: int, in_width: int, keep_input_dims: bool) -> tuple:
    """
    Randomly selects values for kernel, padding and stride hyperparameters of the 
    convolutional module.
    """
    output_dims_type = "same" if keep_input_dims else select_output_dimensions()
    out_channels = select_out_channels(in_channels, output_dims_type)
    kernel_size = select_kernel_size(in_channels, in_height, in_width, output_dims_type)
    padding = select_padding(kernel_size)
    stride = select_stride(kernel_size, output_dims_type)
    return (
        out_channels,
        kernel_size,
        stride,
        padding,
    )


def select_output_dimensions() -> str:
    """
    Selects the output dimensions (W, H). Either random output dimensions or output dimensions 
    that are a multiple of the input dimensions.
    Returns `random`, `same`, or `half`.
    """
    options = list(get_args(OUTPUT_DIMENSIONS_TYPES))
    return random.choice(options)


def select_out_channels(in_channels: int, output_dims_type: OUTPUT_DIMENSIONS_TYPES) -> int:
    """
    Selects the output channels (C) of the convolutional layer."""
    # TODO: fix this
    # min_channels = max(32, in_channels)
    min_c = 8
    max_c = 512
    min_channels = max(min_c, in_channels)
    max_channels = min(max_c, min_channels * 2)
    if output_dims_type == "same":
        choices = [min_channels // 2, min_channels // 4, min_channels]
        choice = random.choice(choices)
    elif output_dims_type == "half":
        choice = max_channels
    elif output_dims_type == "random":
        choices = [min_channels // 2, min_channels // 4, max_channels, max_channels * 2]
        choice = random.choice(choices)
    lower_bound = max(min_c, choice)
    upper_bound = min(max_c, lower_bound)
    return upper_bound
    # return random.choice([min_channels, min(512, min_channels * 4), max(1, min_channels // 2)])


def select_out_features(in_features: int) -> int:
    out_features = max(32, in_features // 2)
    return out_features


# TODO: update kernel size to be mindful of whether its "equal" or "random" when restricted by in_height and in_width
def select_kernel_size(
    in_channels: int, in_height: int, in_width: int, output_dims_type: OUTPUT_DIMENSIONS_TYPES
) -> Tuple[int, int]:
    if in_channels >= 512:
        sizes = [3]
    elif in_channels >= 256:
        sizes = [3]
    elif in_channels >= 128:
        sizes = [3, 5]
    elif in_channels >= 1:
        sizes = [3, 5, 7]
    if output_dims_type == "same" or output_dims_type == "random":
        sizes += [1]
    options = list(get_args(KERNEL_DIMENSIONS_TYPES))
    kernel_dims_type = random.choice(options)
    if kernel_dims_type == "equal":
        dim = random.choice(sizes)
        kernel_size = (dim, dim)
    else:
        kernel_size = (random.choice(sizes), random.choice(sizes))
    return kernel_size


def select_padding(kernel_size: Union[int, Tuple[int, int]],) -> Tuple[int, int]:
    kernel_size = pair(kernel_size)
    padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    return padding


def select_stride(padding: Union[int, Tuple[int, int]], output_dims_type: OUTPUT_DIMENSIONS_TYPES,) -> Tuple[int, int]:
    padding = pair(padding)
    if output_dims_type == "same":
        stride = (1, 1)
    elif output_dims_type == "half":
        stride = (2, 2)
    elif output_dims_type == "random":
        if padding[0] != padding[1]:
            stride = (random.randint(1, padding[0] + 1), random.randint(1, padding[1] + 1))
        else:
            i = random.randint(1, padding[0])
            stride = (i, i)
    else:
        raise ValueError(f"Invalid output dimensions type: {output_dims_type}.")
    return stride


def get_output_dims(
    in_height: int,
    in_width: int,
    kernel_size: Union[tuple, int],
    stride: Union[tuple, int],
    padding: Union[tuple, int],
    dilation: Union[tuple, int],
) -> Tuple[int, int]:
    """
        Calculates the output dimensions of the convolutional layer.
        """
    kernel_size = pair(kernel_size)
    stride = pair(stride)
    padding = pair(padding)
    dilation = pair(dilation)

    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return out_height, out_width


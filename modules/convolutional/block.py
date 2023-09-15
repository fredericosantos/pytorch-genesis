from typing import List, Tuple, Union, Optional
from torch import Tensor
import torch

# Model building
from torch.nn import Conv2d, Identity, Module

from modules.base.block import BaseBlock
from modules.base.modules import Reshape, Concatenate, Layer
import modules.utils.block_utils as block_utils
from modules.hyperparams_options import (
    WEIGHT_INITS,
    BIAS_INITS,
    REGULARIZATIONS,
    ACTIVATIONS,
    PADDING_MODES,
)


class ConvBlock(BaseBlock):
    def __init__(
        self,
        # connection hparams
        block_branch: bool,
        connection_index: Tuple[int, int],
        freeze_evolved: bool,
        # block hparams
        keep_input_dims: bool,
        residual_connection: bool,
        dense_connection: bool,
        in_reshape: bool,
        in_pooling: bool,
        pooling_module: Optional[Module],
        input_dims: Tuple[int],
        # layer hparams
        layers_bias: bool,
        layers_weight_init: Optional[WEIGHT_INITS],
        layers_bias_init: Optional[BIAS_INITS],
        layers_regularization: Optional[REGULARIZATIONS],
        layers_regularization_hparams: dict,
        layers_activation_fn: Optional[ACTIVATIONS],
        layers_activation_fn_hparams: dict,
        # output parameters
        num_outputs: int,
        output_bias: bool,
        output_class_specialization: bool,
        output_regularization: Optional[REGULARIZATIONS],
        output_regularization_hparams: Optional[dict],
        output_weight_init: Optional[WEIGHT_INITS],
        output_bias_init: Optional[BIAS_INITS],
        device: Optional[str],
        dtype: Optional[str],
    ):
        super().__init__(
            block_type="conv",
            block_branch=block_branch,
            freeze_evolved=freeze_evolved,
            num_outputs=num_outputs,
            in_pooling=in_pooling,  # Changed from False
            in_reshape=in_reshape,
            input_dims=input_dims,
            connection_index=connection_index,
            layers_bias=layers_bias,
            layers_weight_init=layers_weight_init,
            layers_bias_init=layers_bias_init,
            layers_regularization=layers_regularization,
            layers_regularization_hparams=layers_regularization_hparams,
            layers_activation_fn=layers_activation_fn,
            layers_activation_fn_hparams=layers_activation_fn_hparams,
            output_bias=output_bias,
            output_regularization=output_regularization,
            output_regularization_hparams=output_regularization_hparams,
            output_class_specialization=output_class_specialization,
            output_weight_init=output_weight_init,
            output_bias_init=output_bias_init,
            output_pooling=True,
            output_flatten=True,
            device=device,
            dtype=dtype,
        )
        self.keep_input_dims = keep_input_dims
        self.residual_connection = residual_connection
        self.pooling_module = pooling_module
        self.dense_connection = dense_connection

    def _forward_layers(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x_residual = self.downsample(x) if hasattr(self, "downsample") else None
        dense_layer_shift = 1 if self.dense_connection else 0

        x_initial = x if self.dense_connection else None
        layers_output = []
        for n in range(self.num_layers):
            layer: Layer = self.get_submodule(f"layer_{n}")
            x = layer.forward(
                x,
                x_residual if n == self.num_layers - 1 - dense_layer_shift else None,
                x_initial if n == self.num_layers - 1 else None,
            )
            layers_output.append(x)
        return x, layers_output

    def add_layer(
        self,
        # module parameters
        in_channels: int,
        in_height: int,
        in_width: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple],
        padding: Union[int, tuple],
        dilation: Union[int, tuple],
        groups: int,
        bias: bool,
        padding_mode: PADDING_MODES,
        # layer parameters
        weight_init: Optional[WEIGHT_INITS] = None,
        bias_init: Optional[BIAS_INITS] = None,
        regularization: Optional[REGULARIZATIONS] = "BatchNorm",
        regularization_hparams: dict = {},
        activation_fn: Optional[ACTIVATIONS] = "ReLU",
        activation_fn_hparams: Optional[dict] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        stride = block_utils.pair(stride)
        main_module = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        reshape_module = (
            Reshape(in_channels, in_height, in_width)
            if self.pooling_module is None
            else None
        )
        if self.residual_connection:
            self._add_downsample_layer(
                in_channels,
                in_height,
                in_width,
                out_channels,
                stride,
                weight_init,
                bias_init,
                regularization,
                regularization_hparams,
                device,
                dtype,
            )
        if self.in_pooling and self.num_layers == 0:
            in_height, in_width = in_height // 2, in_width // 2
        self._add_layer(
            main_module=main_module,
            name=f"layer_{self.num_layers}",
            out_features=out_channels,
            pooling_module=self.pooling_module,
            reshape_module=reshape_module,
            regularization=regularization,
            regularization_hparams=regularization_hparams,
            activation_fn=activation_fn,
            activation_fn_hparams=activation_fn_hparams,
            weight_init=weight_init,
            bias_init=bias_init,
            device=device,
            dtype=dtype,
        )
        self._add_output_layer(in_features=out_channels, device=device, dtype=dtype)

        layer = self.last_layer()
        layer.in_height, layer.in_width = in_height, in_width
        layer.in_channels = in_channels
        layer.out_height, layer.out_width = block_utils.get_output_dims(
            in_height, in_width, kernel_size, stride, padding, dilation
        )
        layer.out_channels = out_channels

    # TODO finish implementation of identity layer
    def add_identity_layer(
        self,
        # module parameters
        in_channels: int,
        in_height: int,
        in_width: int,
        out_channels: int,
        stride: Union[int, tuple],
        # layer parameters
        weight_init: Optional[WEIGHT_INITS] = None,
        bias_init: Optional[BIAS_INITS] = None,
        regularization: Optional[REGULARIZATIONS] = "BatchNorm",
        regularization_hparams: dict = {},
        activation_fn: Optional[ACTIVATIONS] = "ReLU",
        activation_fn_hparams: Optional[dict] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        stride = block_utils.pair(stride)

        reshape_module = (
            Reshape(in_channels, in_height, in_width)
            if self.pooling_module is None
            else None
        )
        if self.residual_connection:
            self._add_downsample_layer(
                in_channels,
                in_height,
                in_width,
                out_channels,
                stride,
                weight_init,
                bias_init,
                regularization,
                regularization_hparams,
                device,
                dtype,
            )
        if self.in_pooling and self.num_layers == 0:
            in_height, in_width = in_height // 2, in_width // 2
        self._add_layer(
            main_module=Identity(),
            name=f"layer_{self.num_layers}",
            out_features=out_channels,
            pooling_module=self.pooling_module,
            reshape_module=reshape_module,
            regularization=regularization,
            regularization_hparams=regularization_hparams,
            activation_fn=activation_fn,
            activation_fn_hparams=activation_fn_hparams,
            weight_init=weight_init,
            bias_init=bias_init,
            device=device,
            dtype=dtype,
        )
        self._add_output_layer(in_features=out_channels, device=device, dtype=dtype)
        layer = self.last_layer()
        # TODO: adjust this to the pooling module as the dims may be different
        layer.in_height, layer.in_width = in_height, in_width
        layer.out_height, layer.out_width = in_height, in_width
        layer.out_channels = out_channels

    def _add_concatenate_layer(
        self,
        # layer parameters
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        input_channels, input_height, input_width = self.input_dims
        in_channels = self.last_layer().out_channels
        in_height = input_height
        in_width = input_width
        out_channels = in_channels + input_channels
        assert input_height == in_height and input_width == in_width, (
            f"Input dimensions (height, width) = ({input_height}, {input_width})"
            f"must match the last layer's output dimensions = ({in_height}, {in_width})"
        )

        self._add_layer(
            main_module=Concatenate(dim=1),
            name=f"layer_{self.num_layers}",
            out_features=out_channels,
            pooling_module=None,
            reshape_module=None,
            regularization=None,
            regularization_hparams=None,
            activation_fn=None,
            activation_fn_hparams=None,
            weight_init=None,
            bias_init=None,
            device=device,
            dtype=dtype,
        )
        self._add_output_layer(in_features=out_channels, device=device, dtype=dtype)
        layer: Layer = self.last_layer()
        layer.in_height, layer.in_width = in_height, in_width
        layer.in_channels = in_channels
        layer.out_channels = out_channels
        layer.out_height, layer.out_width = in_height, in_width

    def _add_downsample_layer(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_channels: int,
        stride: Tuple[int, int],
        weight_init: Optional[WEIGHT_INITS],
        bias_init: Optional[BIAS_INITS],
        regularization: Optional[REGULARIZATIONS],
        regularization_hparams: dict,
        device,
        dtype,
    ):
        if self.in_pooling and self.num_layers == 0:
            stride = (stride[0] * 2, stride[1] * 2)
        if hasattr(self, "downsample"):
            if stride[0] != 1 or stride[1] != 1 or in_channels != out_channels:
                stride_ds = block_utils.pair(self.downsample.main.stride)
                stride_ds = (stride_ds[0] * stride[0], stride_ds[1] * stride[1])
                in_channels_ds = self.downsample.in_channels
                main_module = Conv2d(
                    in_channels=in_channels_ds,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride_ds,
                    padding=0,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                self.downsample.add_module("main", main_module)
                self._add_regularization_module(
                    self.downsample,
                    self.block_type,
                    out_channels,
                    regularization,
                    regularization_hparams,
                    device,
                    dtype,
                )
        else:
            if stride[0] == 1 and stride[1] == 1 and in_channels == out_channels:
                main_module = Identity()
                main_module.stride = stride
                regularization = None
            else:
                main_module = Conv2d(
                    in_channels,
                    out_channels,
                    (1, 1),
                    stride,
                    0,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
            self._make_layer("downsample", None, weight_init, bias_init)
            self._add_layer_modules(
                main_module=main_module,
                name="downsample",
                out_features=out_channels,
                pooling_module=None,
                reshape_module=Reshape(in_channels, in_height, in_width),
                regularization=regularization,
                regularization_hparams=regularization_hparams,
                activation_fn=None,
                activation_fn_hparams=None,
                device=device,
                dtype=dtype,
            )
            self.downsample.in_channels = in_channels
            self.downsample.in_height = in_height
            self.downsample.in_width = in_width

        out_height_ds, out_width_ds = block_utils.get_output_dims(
            self.downsample.in_height,
            self.downsample.in_width,
            (1, 1),
            self.downsample.main.stride,
            0,
            0,
        )
        self.downsample.out_channels = out_channels
        self.downsample.out_height = out_height_ds
        self.downsample.out_width = out_width_ds

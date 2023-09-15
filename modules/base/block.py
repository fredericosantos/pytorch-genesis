from typing import List, Literal, Tuple, Optional
import torch as torch
from re import match

from torch import Tensor
from torch.nn import Module, Flatten, AdaptiveAvgPool2d

from torchmetrics import MetricCollection

from modules.base.modules import Layer
from modules.hyperparams_options import (
    BLOCK_TYPES,
    WEIGHT_INITS,
    BIAS_INITS,
    REGULARIZATIONS,
    ACTIVATIONS,
)
import modules.activation_functions as activations
import modules.regularization_functions as regularizations
import modules.param_init as param_init
from modules.base.modules import OutputLinear


class BaseBlock(Module):
    def __init__(
        self,
        # block parameters
        block_type: BLOCK_TYPES,
        block_branch: bool,
        connection_index: Tuple[int, int],
        freeze_evolved: bool,
        in_pooling: bool,
        in_reshape: bool,
        input_dims: Tuple[int],
        # main layer parameters
        layers_bias: bool,
        layers_weight_init: WEIGHT_INITS,
        layers_bias_init: BIAS_INITS,
        layers_regularization: Optional[REGULARIZATIONS],
        layers_regularization_hparams: dict,
        layers_activation_fn: Optional[ACTIVATIONS],
        layers_activation_fn_hparams: dict,
        # output parameters
        num_outputs: int,
        output_bias: bool,
        output_class_specialization: bool,
        output_pooling: bool,
        output_flatten: bool,
        output_regularization: Optional[REGULARIZATIONS],
        output_regularization_hparams: Optional[dict],
        output_weight_init: Optional[WEIGHT_INITS],
        output_bias_init: Optional[BIAS_INITS],
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        # block parameters
        self.block_type = block_type
        self.block_branch = block_branch
        self.num_outputs = num_outputs
        self.connection_index = connection_index
        self.freeze_evolved = freeze_evolved
        self.in_pooling = in_pooling
        self.in_reshape = in_reshape
        self.input_dims = input_dims

        # main layer parameters
        self.layers_bias = layers_bias
        self.layers_weight_init = layers_weight_init
        self.layers_bias_init = layers_bias_init
        self.regularization = layers_regularization
        self.regularization_hparams = layers_regularization_hparams
        self.activation_fn = layers_activation_fn
        self.activation_fn_hparams = layers_activation_fn_hparams

        # output parameters
        self.output_bias = output_bias
        self.output_class_specialization = output_class_specialization
        self.output_pooling = output_pooling
        self.output_flatten = output_flatten
        self.output_regularization = output_regularization
        self.output_regularization_hparams = output_regularization_hparams
        self.output_weight_init = output_weight_init
        self.output_bias_init = output_bias_init

        # setup
        self.is_frozen: bool = False
        self.num_outgoing_connections: int = 0
        self.connected_to_merge_block: bool = True
        self.num_layers: int = 0
        self.device = device
        self.dtype = dtype

        # Metrics
        self.train_metrics: MetricCollection
        self.valid_metrics: MetricCollection
        self.test_metrics: MetricCollection

        # Layers
        self.layer_out: Layer
        self.downsample: Layer

    def _forward_layers(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        layers_output = []
        for n in range(self.num_layers):
            layer: Layer = self.get_submodule(f"layer_{n}")
            x = layer.forward(x)
            layers_output.append(x)
        return x, layers_output

    def _forward_layer_out(self, x: Tensor) -> Tensor:
        return self.layer_out(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x, layers_output = self._forward_layers(x)
        y_hat = self._forward_layer_out(x)
        return y_hat, layers_output

    def reset_parameters(self):
        for name, module in self.named_children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _add_layer(
        self,
        main_module: Module,
        name: str,
        out_features: int,
        pooling_module: Optional[Module],
        reshape_module: Optional[Module],
        regularization: Optional[REGULARIZATIONS],
        regularization_hparams: Optional[dict],
        activation_fn: Optional[ACTIVATIONS],
        activation_fn_hparams: Optional[dict],
        weight_init: Optional[WEIGHT_INITS],
        bias_init: Optional[BIAS_INITS],
        device: Optional[str],
        dtype: Optional[str],
    ):
        self._make_layer(name, activation_fn, weight_init, bias_init)
        self._add_layer_modules(
            main_module=main_module,
            name=name,
            out_features=out_features,
            pooling_module=pooling_module,
            reshape_module=reshape_module,
            regularization=regularization,
            regularization_hparams=regularization_hparams,
            activation_fn=activation_fn,
            activation_fn_hparams=activation_fn_hparams,
            device=device,
            dtype=dtype,
        )
        # if name is layer_n, where n is any number, then increment num_layers
        if match(r"layer_\d+", name):
            self.num_layers += 1
        # Output Layer
        # self._add_output_layer(out_features, device, dtype)

    def _make_layer(
        self,
        name: str,
        activation_fn: Optional[ACTIVATIONS],
        weight_init: Optional[WEIGHT_INITS],
        bias_init: Optional[BIAS_INITS],
    ):
        # delete layer_out if it exists
        if hasattr(self, "layer_out"):
            delattr(self, "layer_out")
        # Layer
        weight_init_hparams = param_init.get_weight_init_hparams(
            weight_init, activation_fn
        )
        bias_init_hparams = param_init.get_bias_init_hparams(bias_init)
        self.add_module(
            name,
            Layer(
                weight_init=weight_init,
                weight_init_hparams=weight_init_hparams,
                bias_init=bias_init,
                bias_init_hparams=bias_init_hparams,
            ),
        )

    def _add_layer_modules(
        self,
        main_module: Module,
        name: str,
        out_features: int,
        pooling_module: Optional[Module],
        reshape_module: Optional[Module],
        regularization: Optional[REGULARIZATIONS],
        regularization_hparams: Optional[dict],
        activation_fn: Optional[ACTIVATIONS],
        activation_fn_hparams: Optional[dict],
        device: Optional[str],
        dtype: Optional[str],
    ):
        layer: Layer = self.get_submodule(name)

        # Main Pooling
        if self.in_pooling and self.num_layers == 0:
            self._add_pooling_module(layer, pooling_module)

        # Main Reshape (Flatten/Reshape)
        if self.in_reshape and self.num_layers == 0:
            self._add_reshape_module(layer, reshape_module)

        # Main module
        layer.add_module("main", main_module)

        # Regularization
        self._add_regularization_module(
            layer,
            self.block_type,
            out_features,
            regularization,
            regularization_hparams,
            device,
            dtype,
        )

        # Activation
        self._add_activation_fn_module(layer, activation_fn, activation_fn_hparams)
        layer.reset_parameters()

    def _add_output_layer(
        self, in_features: int, device: Optional[str], dtype: Optional[str]
    ):
        self._make_layer(
            "layer_out", None, self.output_weight_init, self.output_bias_init
        )
        # Output Pooling
        if self.output_pooling:
            self._add_pooling_module(self.layer_out)

        # Output Flatten
        if self.output_flatten:
            self._add_reshape_module(self.layer_out, Flatten())

        # Output Linear
        self.layer_out.add_module(
            "main",
            OutputLinear(
                in_features=in_features,
                out_features=self.num_outputs,
                bias=self.output_bias,
                one_class_only=self.output_class_specialization,
                device=device,
                dtype=dtype,
            ),
        )
        self._add_regularization_module(
            self.layer_out,
            "linear",
            self.num_outputs,
            self.output_regularization,
            self.output_regularization_hparams,
            device,
            dtype,
        )
        # TODO: Add activation function at output layer (after thesis delivery)
        self.layer_out.reset_parameters()

    def _add_pooling_module(
        self, layer: Layer, pooling_module: Optional[Module] = None
    ):
        del layer.pooling
        layer.add_module("pooling", pooling_module or AdaptiveAvgPool2d((1, 1)))

    def _add_reshape_module(self, layer: Layer, reshape_module: Module):
        del layer.reshape
        layer.add_module("reshape", reshape_module)

    def _add_regularization_module(
        self,
        layer: Layer,
        layer_type: BLOCK_TYPES,
        out_features: int,
        regularization: Optional[REGULARIZATIONS],
        regularization_hparams: Optional[dict],
        device,
        dtype,
    ):
        if regularization is None:
            reg = regularizations.get_module(fn=regularization)

        elif regularization.lower() == "instancenorm":
                if (
                    hasattr(layer, "out_width")
                    and layer.out_width == 1
                    and layer.out_height == 1
                ):
                    # No need to add instance normalization for (n, c, 1, 1)
                    #  output of convolutions
                    return

        hparams = regularizations.get_hparams(
            fn=regularization,
            block_type=layer_type,
            num_features=out_features,
            height=layer.out_height if hasattr(layer, "out_height") else None,
            width=layer.out_width if hasattr(layer, "out_width") else None,
            device=device,
            dtype=dtype,
        )
        if isinstance(hparams, dict):
            hparams |= (
                regularization_hparams if regularization_hparams is not None else {}
            )
        else:
            hparams = (
                regularization_hparams if regularization_hparams is not None else {}
            )
        reg = regularizations.get_module(
            fn=regularization, block_type=layer_type, **hparams
        )
        del layer.regularization
        layer.add_module("regularization", reg)

    def _add_activation_fn_module(
        self,
        layer: Layer,
        activation_fn: ACTIVATIONS,
        activation_fn_hparams: Optional[dict],
    ):
        act_fn = activations.get_module(activation_fn, activation_fn_hparams)
        del layer.activation
        layer.add_module("activation", act_fn)

    def add_connections_out(self, sparsity: float, device: Optional[str] = None):
        if self.layer_out.main.sparsity > 0:
            self.layer_out.main.add_connections(sparsity)
            if self.freeze_evolved:
                self.freeze(freeze_output_layer=False)

    def remove_connections_out(self, sparsity: float, device: Optional[str] = None):
        """
        Removes connection(s) from layers to the output layer.
        """
        if self.layer_out.main.sparsity < 1:
            self.layer_out.main.remove_connections(sparsity)

    def freeze(self, freeze_output_layer: bool = True):
        self.is_frozen = True
        self.requires_grad_(False)
        if not freeze_output_layer:
            self.layer_out.unfreeze()

    def unfreeze(self):
        self.is_frozen = False
        self.requires_grad_(True)

    def last_layer(self) -> Layer:
        return getattr(self, f"layer_{self.num_layers - 1}")

    def connect_to_output_layer(self):
        if self.num_outgoing_connections == 0:
            self.connected_to_merge_block = True

    def disconnect_from_output_layer(self):
        if self.num_outgoing_connections > 0:
            self.connected_to_merge_block = False

from copy import deepcopy
import random
import numpy as np
import torch as torch
from typing import Dict, List, Tuple, Optional, Literal, Union, get_args

# Model building
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import AvgPool2d

from torch.nn.modules.loss import _Loss

import torch.nn.functional as F

from torch.optim import RAdam, Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR
from modules.base.block import BaseBlock
import torchmetrics as tm
from torchmetrics import MetricCollection, Metric
from modules.base.modules import Layer
from modules.base.merge_block import MergeBlock

# Library
from modules.linear.block import LinearBlock
from modules.convolutional.block import ConvBlock
import modules.loss_functions as loss_functions
from modules.identity import LinearIdentity
import modules.utils.block_utils as block_utils
from modules.hyperparams_options import (
    CONV_TYPES,
    LINEAR_TYPES,
    PADDING_MODES,
    REGULARIZATIONS,
    WEIGHT_INITS,
    BIAS_INITS,
    ACTIVATIONS,
    LOSSES,
    OPTIMIZERS,
    LR_SCHEDULERS,
    OUTPUT_NORMS,
    BLOCK_TYPES,
)
from modules.optimizers import Lion


class EvoModel(LightningModule):
    def __init__(
        self,
        data_shape: Tuple[int],
        num_classes: int,
        # Block hyperparameters
        block_branch: bool = True,
        conv2conv_only: bool = True,
        only_last_layer_cxts: bool = False,
        connection_weights: Union[Dict[BLOCK_TYPES, list], list] = [1, 1],
        input_connection_probability: float = 0.0,
        # Output Layer hyperparameters
        freeze_output_layers: bool = True,
        output_bias: bool = False,
        output_weight_init: Optional[WEIGHT_INITS] = None,
        output_bias_init: Optional[BIAS_INITS] = None,
        output_activation_fn: Optional[ACTIVATIONS] = None,
        output_activation_fn_hparams: Optional[dict] = None,
        output_regularization: Optional[OUTPUT_NORMS] = None,
        output_regularization_hparams: Optional[dict] = None,
        output_merged_regularization: Optional[OUTPUT_NORMS] = None,
        # Loss Function
        loss_fn: LOSSES = "cross_entropy",
        loss_fn_hparams: dict = {},
        type_softmax: Optional[Literal["softmax", "logsoftmax"]] = None,
        residual_loss: bool = False,
        # Optimizers
        lr: float = 0.05,
        optimizer: OPTIMIZERS = "lion",
        optimizer_hparams: dict = {},
        lr_scheduler: Optional[LR_SCHEDULERS] = None,
        lr_scheduler_hparams: dict = {},
        # Mutation hyperparameters
        freeze_evolved: bool = True,
        # Metrics
        metrics: List[Metric] = [tm.Accuracy()],
        # Experimental Hyperparameters
        transforms_layer: bool = False,
    ):
        super().__init__()
        metrics = [metric.to(self.device) for metric in metrics]
        self.save_hyperparameters()
        # multiply each element of data_shape tuple
        input_features = torch.prod(torch.tensor(data_shape)).item()

        # This activates manual optimization of the model if set to False
        self.automatic_optimization = True

        # Transforms Layer
        if transforms_layer:
            self.transforms_layer = LinearIdentity(input_features, False, bias=True)

        self.example_input_array = torch.ones(64, *data_shape).float()

        # Loss functions
        self.loss: _Loss = loss_functions.get_loss_fn(
            loss_type=loss_fn, hparams=loss_fn_hparams
        )

        # Metrics
        phases = ["train", "valid", "test"]
        for t in phases:
            setattr(self, f"{t}_metrics", MetricCollection(metrics, prefix=f"{t}/"))
        # Setup of model parameters
        self.num_blocks = 0
        self.solo_step = True if self.hparams.residual_loss else False

        self.merge_block = MergeBlock(
            num_classes,
            output_merged_regularization,
            type_softmax,
            output_activation_fn,
            output_activation_fn_hparams,
        )

    def forward(self, x):
        if self.hparams.transforms_layer:
            x = self.transforms_layer(x)
        inputs = [[x]]
        y_hat = x
        blocks_y_hat = []
        all_blocks_y_hat = []

        if self.num_blocks > 0:
            for i in range(self.num_blocks):
                block: BaseBlock = self.get_submodule(f"block_{i}")
                block_idx, submod_idx = block.connection_index
                y_hat_output_layer, output_layers = block(inputs[block_idx][submod_idx])
                inputs.append(output_layers)
                if block.connected_to_merge_block:
                    blocks_y_hat.append(y_hat_output_layer)
                all_blocks_y_hat.append(y_hat_output_layer)
            y_hat = self.merge_block(blocks_y_hat)

        if hasattr(self, "hook_xs"):
            self.hook_xs = deepcopy(inputs)
        return y_hat, all_blocks_y_hat

    def _step(
        self,
        batch,
        type_step: Literal["train", "valid", "test"],
        solo_step: bool = False,
    ):
        x, y = batch
        y_hat, blocks_y_hat = self(x)
        if self.hparams.loss_fn in ["rmse", "mse", "mae"]:
            y = F.one_hot(y, num_classes=self.hparams.num_classes).float()

        # Loss
        y_ = y
        for i, block_y in enumerate(blocks_y_hat):
            loss = self.loss(block_y, y_)
            self.log(f"{type_step}/loss_block_{i}", loss, on_step=False, on_epoch=True)
            if self.hparams.residual_loss:
                y_hat_previous = self.merge_block(blocks_y_hat[: i + 1])
                y_ = self._calc_residual(y_hat_previous, y)

        model_loss = self.loss(y_hat, y)
        if not self.hparams.residual_loss and not solo_step:
            loss = model_loss

        self.log(f"{type_step}/loss", model_loss, on_step=True, on_epoch=True)

        # Metrics
        # Block metrics
        for i, block_y in enumerate(blocks_y_hat):
            block = self.get_submodule(f"block_{i}")
            if self.hparams.residual_loss:
                preds = torch.argmax(self.merge_block(blocks_y_hat[: i + 1]), dim=1)
            else:
                preds = torch.argmax(block_y, dim=1)
            self.preds = preds
            block_metrics_output = block._modules[f"{type_step}_metrics"](
                preds, batch[1]
            )
            self.log_dict(block_metrics_output, on_step=False, on_epoch=True)
        # Model metrics
        preds = torch.argmax(y_hat, dim=1)
        metrics_output = self._modules[f"{type_step}_metrics"](preds, batch[1])
        self.log_dict(metrics_output, prog_bar=False, on_step=False, on_epoch=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", self.solo_step)

    # ** Validation + Testing
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    # ** Optimizers
    def configure_optimizers(self):
        optim_conf = {}
        self.hparams.optimizer_hparams["lr"] = self.hparams.lr
        optimizer_hparams = dict(
            params=self.parameters(), **self.hparams.optimizer_hparams
        )
        if self.hparams.optimizer.lower() == "sgd":
            optim_conf["optimizer"] = SGD(**optimizer_hparams)
        elif self.hparams.optimizer.lower() == "radam":
            optim_conf["optimizer"] = RAdam(**optimizer_hparams)
        elif self.hparams.optimizer.lower() == "adam":
            optim_conf["optimizer"] = Adam(**optimizer_hparams)
        elif self.hparams.optimizer.lower() == "lion":
            optim_conf["optimizer"] = Lion(**optimizer_hparams)
        else:
            raise ValueError(f"Unknown optimizer {self.hparams.optimizer}")
        if self.hparams.lr_scheduler is None:
            return optim_conf
        if self.hparams.lr_scheduler.lower() == "onecycle":
            optim_conf["lr_scheduler"] = dict(
                scheduler=OneCycleLR(
                    optim_conf["optimizer"], **self.hparams.lr_scheduler_hparams
                ),
                interval="step",
            )
        return optim_conf

    # *** Neuroevolution
    # * Blocks
    def _post_add_block(self, device: Optional[str]):
        self.num_blocks += 1
        block = self.last_block()
        for type_step in ["train", "valid", "test"]:
            block._modules[f"{type_step}_metrics"] = MetricCollection(
                deepcopy(self.hparams.metrics),
                prefix=f"{type_step}/",
                postfix=f"_block_{self.num_blocks-1}",
            ).to(device)
        self._connect_block(block)

    def add_block_linear(
        self,
        # module parameters
        out_features: int,
        sparsity: float = 0.0,
        layers_sparse: bool = False,
        # block parameters
        connection_index: Optional[Tuple[int, int]] = None,
        block_branch: bool = False,
        # main layers parameters
        layers_bias: bool = True,
        layers_weight_init: Optional[WEIGHT_INITS] = None,
        layers_bias_init: Optional[BIAS_INITS] = None,
        regularization: REGULARIZATIONS = "Dropout",
        regularization_hparams: Optional[dict] = None,
        activation_fn: ACTIVATIONS = "ReLU",
        activation_fn_hparams: Optional[dict] = None,
        # output layer parameters
        output_class_specialization: bool = False,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        block_type: str = "linear"
        device = device or self.device

        self.freeze_last_block()
        cxt_idx = self._select_connection_index(
            block_type, connection_index, block_branch=block_branch
        )
        input_dims = self._get_input_dims(cxt_idx)
        block_branch = self.check_block_branch(block_branch, cxt_idx[0])
        input_type = self.get_input_type(cxt_idx)
        in_pooling = True if input_type in get_args(CONV_TYPES) else False
        in_flatten = True if input_type in get_args(CONV_TYPES) else False

        self.add_module(
            f"block_{self.num_blocks}",
            LinearBlock(
                block_branch=block_branch,
                in_pooling=in_pooling,
                in_flatten=in_flatten,
                input_dims=input_dims,
                layers_sparse=layers_sparse,
                freeze_evolved=self.hparams.freeze_evolved,
                num_outputs=self.hparams.num_classes,
                connection_index=cxt_idx,
                layers_bias=layers_bias,
                layers_weight_init=layers_weight_init,
                layers_bias_init=layers_bias_init,
                layers_regularization=regularization,
                layers_regularization_hparams=regularization_hparams,
                layers_activation_fn=activation_fn,
                layers_activation_fn_hparams=activation_fn_hparams,
                output_class_specialization=output_class_specialization,
                output_regularization=self.hparams.output_regularization,
                output_regularization_hparams=self.hparams.output_regularization_hparams,
                output_bias=self.hparams.output_bias,
                output_weight_init=self.hparams.output_weight_init,
                output_bias_init=self.hparams.output_bias_init,
                device=device,
                dtype=dtype,
            ),
        )

        self._post_add_block(device)
        self.add_layer_linear(
            out_features=out_features,
            sparsity=sparsity,
            device=device,
        )

    # TODO [#LOW]: Implement groups + padding_mode
    def add_block_conv(
        self,
        # module parameters
        out_channels: Optional[int] = None,
        kernel_size: Optional[Union[int, tuple]] = None,
        stride: Optional[Union[int, tuple]] = None,
        padding: Optional[Union[int, tuple]] = None,
        dilation: Optional[Union[int, tuple]] = 1,
        groups: int = 1,
        padding_mode: PADDING_MODES = "zeros",
        # block parameters
        connection_index: Optional[Tuple[int, int]] = None,
        block_branch: bool = False,
        keep_input_dims: bool = False,
        residual_connection: bool = False,
        dense_connection: bool = False,
        in_pooling: bool = False,
        # main layers parameters
        main_identity: bool = False,
        layers_bias: bool = False,
        layers_weight_init: Optional[WEIGHT_INITS] = None,
        layers_bias_init: Optional[BIAS_INITS] = None,
        regularization: Optional[REGULARIZATIONS] = "BatchNorm",
        regularization_hparams: Optional[dict] = None,
        activation_fn: Optional[ACTIVATIONS] = "ReLU",
        activation_fn_hparams: Optional[dict] = None,
        # output layer parameters
        output_class_specialization: bool = False,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        block_type: str = "conv"
        device = device or self.device

        self.freeze_last_block()
        cxt_idx = self._select_connection_index(
            block_type, connection_index, block_branch=block_branch
        )
        block_branch = self.check_block_branch(block_branch, cxt_idx[0])
        cxt_type = self.get_input_type(cxt_idx)
        input_dims = self._get_input_dims(cxt_idx)
        in_reshape = True if cxt_type in get_args(LINEAR_TYPES) else False

        self.add_module(
            f"block_{self.num_blocks}",
            ConvBlock(
                block_branch=block_branch,
                keep_input_dims=keep_input_dims,
                residual_connection=residual_connection,
                dense_connection=dense_connection,
                in_reshape=in_reshape,
                in_pooling=in_pooling,
                pooling_module=AvgPool2d(2, 2) if in_pooling else None,
                input_dims=input_dims,
                freeze_evolved=self.hparams.freeze_evolved,
                num_outputs=self.hparams.num_classes,
                connection_index=cxt_idx,
                layers_bias=layers_bias,
                layers_weight_init=layers_weight_init,
                layers_bias_init=layers_bias_init,
                layers_regularization=regularization,
                layers_regularization_hparams=regularization_hparams,
                layers_activation_fn=activation_fn,
                layers_activation_fn_hparams=activation_fn_hparams,
                output_class_specialization=output_class_specialization,
                output_regularization=self.hparams.output_regularization,
                output_regularization_hparams=self.hparams.output_regularization_hparams,
                output_bias=self.hparams.output_bias,
                output_weight_init=self.hparams.output_weight_init,
                output_bias_init=self.hparams.output_bias_init,
                device=device,
                dtype=dtype,
            ),
        )
        self._post_add_block(device)
        self.add_layer_conv(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            main_identity=main_identity,
            device=device,
        )

    def check_block_branch(self, block_branch: bool, block_idx: int):
        if block_idx != self.num_blocks:
            block_branch = False
        return block_branch

    def freeze_last_block(self):
        if self.num_blocks != 0 and self.hparams.freeze_evolved:
            block = self.last_block()
            block.freeze(self.hparams.freeze_output_layers)
            if block.block_branch:
                self._freeze_block_path(block.connection_index)

    def unfreeze_last_block(self):
        if self.num_blocks != 0 and self.hparams.freeze_evolved:
            block = self.last_block()
            block.unfreeze()
            if block.block_branch:
                self._unfreeze_block_path(block.connection_index)

    def remove_last_block(self) -> None:
        if self.num_blocks > 0:
            block = self.last_block()
            self.disconnect_block(block)
            delattr(self, f"block_{self.num_blocks-1}")
            self.num_blocks -= 1
            self.unfreeze_last_block()

    def last_block(self) -> Union[ConvBlock, LinearBlock]:
        """
        Returns the last block in the model.
        """
        if self.num_blocks > 0:
            return self.get_submodule(f"block_{self.num_blocks-1}")

    # * Layers
    def add_layer_linear(
        self,
        out_features: Optional[int],
        sparsity: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        device = device or self.device
        module_type = "linear"
        self._assert_block_type(module_type)
        block: LinearBlock = self.last_block()
        cxt_idx = self._get_connection_index()
        dims = self._get_input_dims(cxt_idx, block.in_pooling)
        in_features = np.prod(dims)
        out_features = out_features or block_utils.select_out_features(in_features)

        block.add_layer(
            in_features=in_features,
            out_features=out_features,
            sparsity=sparsity,
            bias=block.layers_bias,
            sparse=block.layers_sparse,
            # params common to all module_types
            weight_init=block.layers_weight_init,
            bias_init=block.layers_bias_init,
            regularization=block.regularization,
            regularization_hparams=block.regularization_hparams,
            activation_fn=block.activation_fn,
            activation_fn_hparams=block.activation_fn_hparams,
            dtype=block.dtype,
            device=device,
        )

    # TODO: Implement groups + padding mode
    def add_layer_conv(
        self,
        out_channels: Optional[int] = None,
        kernel_size: Optional[Union[int, tuple]] = None,
        stride: Optional[Union[int, tuple]] = None,
        padding: Optional[Union[int, tuple]] = None,
        dilation: Optional[Union[int, tuple]] = 1,
        groups: int = 1,
        padding_mode: PADDING_MODES = "zeros",
        main_identity: bool = False,
        main_bias: bool = False,
        main_weight_init: Optional[WEIGHT_INITS] = None,
        main_bias_init: Optional[BIAS_INITS] = None,
        regularization: Optional[REGULARIZATIONS] = None,
        regularization_hparams: Optional[dict] = None,
        activation_fn: Optional[ACTIVATIONS] = None,
        activation_fn_hparams: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        device = device or self.device
        self._assert_block_type("conv")
        block: ConvBlock = self.last_block()
        cxt_idx = self._get_connection_index()
        dims = self._get_input_dims(cxt_idx)
        if len(dims) == 3:
            in_channels, in_height, in_width = dims
        elif len(dims) == 1:
            # calculate in_channels, in_height, in_width
            for d in [16, 8, 4, 2]:
                if dims[0] // (d**2) >= 4:
                    in_channels = int(dims[0] // (d**2))
                    in_height = d
                    in_width = d
                    break
        (
            out_channels_,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
        ) = block_utils.random_layer_hparams(in_channels, block.keep_input_dims)
        out_channels = out_channels_ if out_channels is None else out_channels
        kernel_size = kernel_size_ if kernel_size is None else kernel_size
        stride = stride_ if stride is None else stride
        padding = padding_ if padding is None else padding
        dilation = dilation_ if dilation is None else dilation

        if main_identity:
            block.add_identity_layer(
                in_channels=in_channels,
                in_height=in_height,
                in_width=in_width,
                out_channels=out_channels,
                stride=stride,
                weight_init=main_weight_init or block.layers_weight_init,
                bias_init=main_bias_init or block.layers_bias_init,
                regularization=regularization or block.regularization,
                regularization_hparams=regularization_hparams
                or block.regularization_hparams,
                activation_fn=activation_fn or block.activation_fn,
                activation_fn_hparams=activation_fn_hparams
                or block.activation_fn_hparams,
                device=device,
                dtype=block.dtype,
            )
        else:
            block.add_layer(
                in_channels=in_channels,
                in_height=in_height,
                in_width=in_width,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                bias=main_bias or block.layers_bias,
                # params common to all module_types
                weight_init=main_weight_init or block.layers_weight_init,
                bias_init=main_bias_init or block.layers_bias_init,
                regularization=regularization or block.regularization,
                regularization_hparams=regularization_hparams
                or block.regularization_hparams,
                activation_fn=activation_fn or block.activation_fn,
                activation_fn_hparams=activation_fn_hparams
                or block.activation_fn_hparams,
                device=device,
                dtype=block.dtype,
            )

    # ** Connections
    # * Layer connections
    def add_connections_linear(
        self,
        sparsity: float,
        device: Optional[str] = None,
    ) -> None:
        """Adds connection(s) from neurons on the expansion layer
        to random neuron(s) in the concatenated layer of all previous neurons
        with weighted probability based on layer distance.
        """
        device = device or self.device
        self._assert_block_type("linear")
        block = self.last_block()
        block.add_connections_layer(sparsity, device)

    def remove_connections_linear(
        self, sparsity: float, device: Optional[str] = None
    ) -> None:
        device = device or self.device
        self._assert_block_type("linear")
        block = self.last_block()
        block.remove_connections_layer(sparsity, device)

    # * Output connections
    def add_connections_out(
        self,
        sparsity: float,
        device: Optional[str] = None,
    ) -> None:
        """
        Adds connection(s) from the expansion to the aggregation layer.
        When this method is called, no more connections are allowed
        to be added to the expansion layer.
        This is to maintain the requirement for unimodal error landscape.
        """
        device = device or self.device
        block = self.last_block()
        block.add_connections_out(sparsity, device)

    def remove_connections_out(
        self, sparsity: float, device: Optional[str] = None
    ) -> None:
        device = device or self.device
        block = self.last_block()
        block.remove_connections_out(sparsity, device)

    # ** Parameters
    def reset_parameters(self) -> None:
        """
        Resets parameters of all blocks given the layer type. If no layer type is given,
        resets all parameters of all layers in all blocks.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters") and not isinstance(self, type(self)):
                module.reset_parameters()
        # for i in range(self.num_blocks):
        #     block: BaseBlock = self.get_submodule(f"block_{i}")
        #     block.reset_parameters()

    # ** Auxiliary functions
    def _get_connection_weights(self, block_type: BLOCK_TYPES) -> Tensor:
        lw = self.hparams.connection_weights
        if isinstance(lw, dict) and block_type in lw:
            lw = lw[block_type]
        lw = lw[: self.num_blocks + 1]
        lw = lw + [lw[-1]] * (self.num_blocks + 1 - len(lw))
        if block_type == "conv" and self.hparams.conv2conv_only:
            for i, n in enumerate(range(self.num_blocks - 1, -1, -1)):
                block = self.get_submodule(f"block_{n}")
                if block.block_type == "linear":
                    lw.insert(i, 0)
        lw = lw[: self.num_blocks + 1][::-1]
        return torch.tensor(lw)

    def _assert_block_type(self, block_type: BLOCK_TYPES) -> None:
        block = self.last_block()
        assert block.block_type == block_type, f"Last block is not of type {block_type}"

    # ** Transforms functions
    def freeze_transforms(self) -> None:
        """
        Freezes the transforms layer.
        """
        if self.hparams.transforms_layer:
            self.transforms_layer.requires_grad_(False)

    def _get_input_dims(
        self, connection_index: Tuple[int, int], pooling: bool = False
    ) -> Tuple[int, int, int]:
        block_idx, layer_idx = connection_index
        if block_idx == 0:
            dims = (
                (self.hparams.data_shape[0],)
                if pooling
                else tuple(i for i in self.hparams.data_shape)
            )
        else:
            # blocks start at index 1, layers start at index 0
            block: Union[ConvBlock, LinearBlock] = self.get_submodule(
                f"block_{block_idx-1}"
            )
            layer: Layer = block.get_submodule(f"layer_{layer_idx}")
            if hasattr(layer, "out_channels"):
                if pooling:
                    dims = (layer.out_channels,)
                else:
                    dims = (layer.out_channels, layer.out_height, layer.out_width)
            elif hasattr(layer, "out_features"):
                dims = (layer.out_features,)
            else:
                raise ValueError(
                    f"{layer} from block {block_idx-1} does not have "
                    "out_channels or out_features"
                )
        return dims

    def _get_connection_index(self, block_num: Optional[int] = None) -> Tuple[int, int]:
        """Returns either the connection index of the last block or the block
        specified by block_num. If block_num is None, returns the connection
        index of the last block. If block_num is specified, returns the
        connection index of the specified block."""
        if self.num_blocks == 0:
            return 0, 0
        if block_num is None:
            block = self.last_block()
            cxt_idx = (
                block.connection_index
                if block.num_layers == 0
                else (self.num_blocks, block.num_layers - 1)
            )
        else:
            block = self.get_submodule(f"block_{block_num}")
            cxt_idx = block.connection_index
        return cxt_idx

    def _adjust_input_connection_probability(self, cxt_probs: Tensor) -> Tensor:
        if self.hparams.input_connection_probability > 0:
            prob_adjustment = 1 - self.hparams.input_connection_probability
            cxt_probs *= prob_adjustment
            cxt_probs[0] = self.hparams.input_connection_probability
            cxt_probs /= cxt_probs.sum()

    def _select_connection_index(
        self,
        block_type: BLOCK_TYPES,
        connection_index: Optional[tuple],
        counter: int = 0,
        block_branch: bool = False,
    ) -> Tuple[int]:
        connection_weights = self._get_connection_weights(block_type)
        if connection_index is None:
            custom_cxt_idx = False
            blocks_probs = connection_weights / connection_weights.sum()
            self._adjust_input_connection_probability(blocks_probs)
            block_idx = torch.multinomial(blocks_probs, 1).item()
            if block_idx == 0:
                return 0, 0
            else:
                block: Union[ConvBlock, LinearBlock] = self.get_submodule(
                    f"block_{block_idx-1}"
                )
                block_branch = self.check_block_branch(block_branch, block_idx)
                if block.connected_to_merge_block and block_branch:
                    is_last_layer = True
                else:
                    is_last_layer = (
                        random.choice([True, False])
                        if not self.hparams.only_last_layer_cxts
                        else True
                    )
                if block.num_layers == 1:
                    layer_idx = 0
                elif is_last_layer:
                    layer_idx = block.num_layers - 1
                else:
                    layer_idx = random.randint(0, block.num_layers - 2)
        else:
            custom_cxt_idx = True
            block_idx, layer_idx = connection_index
            if block_idx == 0:
                return block_idx, 0
        connected_block = self.get_submodule(f"block_{block_idx-1}")
        connected_layer: Layer = connected_block.get_submodule(f"layer_{layer_idx}")
        # make sure that we do not connect to layer
        # with out_width == 1 and out_height == 1
        if isinstance(self.last_block(), ConvBlock) and isinstance(
            connected_block, ConvBlock
        ):
            if connected_layer.out_width == 1 and connected_layer.out_height == 1:
                if custom_cxt_idx:
                    raise ValueError(
                        "Cannot connect conv layer to conv layer with "
                        "out_width == 1 and out_height == 1"
                    )
                elif counter == (tries := 100):
                    print(
                        f"Could not find a valid connection after {tries} "
                        "tries, connecting to input"
                    )
                    return 0, 0
                return self._select_connection_index(
                    block_type, connection_index, counter + 1, block_branch
                )
        return block_idx, layer_idx

    def get_input_type(self, cxt_idx: Tuple[int, int]) -> BLOCK_TYPES:
        if cxt_idx[0] == 0:
            if len(self.hparams.data_shape) > 1:
                return "conv"
            else:
                return "linear"
        return self.get_submodule(f"block_{cxt_idx[0]-1}").block_type

    def _connect_block(self, block: BaseBlock):
        if block.block_branch:
            cxt_idx = block.connection_index
            if cxt_idx != (0, 0):
                connected_block: BaseBlock = self.get_submodule(f"block_{cxt_idx[0]-1}")
                connected_block.num_outgoing_connections += 1
                connected_block.disconnect_from_output_layer()
                self._unfreeze_block_path(cxt_idx)

    def disconnect_block(self, block: BaseBlock):
        if block.block_branch:
            cxt_idx = block.connection_index
            if cxt_idx != (0, 0):
                connected_block: BaseBlock = self.get_submodule(f"block_{cxt_idx[0]-1}")
                connected_block.num_outgoing_connections -= 1
                connected_block.connect_to_output_layer()
                self._freeze_block_path(cxt_idx)

    def _freeze_block_path(self, connection_index: Tuple[int, int]) -> None:
        block_idx = connection_index[0]
        if block_idx != 0 and self.hparams.freeze_evolved:
            block: BaseBlock = self.get_submodule(f"block_{block_idx-1}")
            block.freeze(self.hparams.freeze_output_layers)
            self._freeze_block_path(block.connection_index)

    def _unfreeze_block_path(self, connection_index: Tuple[int, int]) -> None:
        block_idx = connection_index[0]
        if block_idx != 0 and self.hparams.freeze_evolved:
            block: BaseBlock = self.get_submodule(f"block_{block_idx-1}")
            if block.num_outgoing_connections == 1:
                block.unfreeze()
                self._unfreeze_block_path(block.connection_index)

    # * RESIDUAL LOSS
    def _clamp_y(self, y_hat: Tensor, y: Tensor):
        y_inverse = y.clone().fill_(1) - y
        y_hat_mask = y_hat * y_inverse
        y_hat_mask = y_hat_mask.clamp(0, 1)
        return y_hat_mask * y_inverse + y

    def _clamp_y_2(self, y_hat: Tensor, y: Tensor):
        y_negatives = y == 0
        y_neg_clip = (y_negatives * y_hat).clamp_min(0)
        y_positives = y == 1
        y_pos_clip = (y_positives * y_hat).clamp_max(1)
        return y_neg_clip + y_pos_clip

    def _calc_residual(self, y_hat: Tensor, y: Tensor):
        # y_hat = self._clamp_y_2(y_hat, y)
        return y - y_hat

    # * PRINT UTILS
    def print_connection_stats(self) -> None:
        for name, module in self.named_modules():
            try:
                print(
                    f"{name.upper()}\t\t[{module._get_name()}\tN_CXTS = "
                    f"{module.num_connections}\tCXT = {module.connection_index[0]}]"
                )
            except AttributeError:
                pass

    def print_grad_states(self, only_frozen: bool = False) -> Tuple[str, bool]:
        for name, param in self.named_parameters():
            if param.requires_grad and not only_frozen:
                print(f"{name}: {param.requires_grad}")

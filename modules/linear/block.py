from typing import Optional, Tuple

# Model building
from torch.nn import Flatten
from modules.linear.modules import MainLinear
from modules.base.block import BaseBlock
from modules.hyperparams_options import (
    WEIGHT_INITS,
    BIAS_INITS,
    REGULARIZATIONS,
    ACTIVATIONS,
)


class LinearBlock(BaseBlock):
    def __init__(
        self,
        # connection hparams
        block_branch: bool,
        connection_index: Tuple[int, int],
        freeze_evolved: bool,
        # block hparams
        layers_sparse: bool,
        in_pooling: bool,
        in_flatten: bool,
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
        """Creates a new block.

        Args:
            in_features (int): For the first block it is the
                number of input features. For subsequent blocks,
                it is the number of features of the concatenated
                layer.
            layers_sparse (bool): Whether the expansion layer's connections are sparse.
            output_sparse (bool): Whether the aggregation layer is sparse.
                This setting should be ``True`` only for experiments as it disables
                the option of adding connections to the expansion layer.
        """
        super().__init__(
            block_type="linear",
            block_branch=block_branch,
            freeze_evolved=freeze_evolved,
            num_outputs=num_outputs,
            in_pooling=in_pooling,
            input_dims=input_dims,
            in_reshape=in_flatten,
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
            output_pooling=False,
            output_flatten=False,
            output_weight_init=output_weight_init,
            output_bias_init=output_bias_init,
            device=device,
            dtype=dtype,
        )
        self.layers_sparse = layers_sparse

    def add_layer(
        self,
        # module parameters
        in_features: int,
        out_features: int,
        sparse: bool,
        sparsity: float,
        bias: bool,
        # layer parameters
        weight_init: Optional[WEIGHT_INITS] = None,
        bias_init: Optional[BIAS_INITS] = None,
        regularization: REGULARIZATIONS = "Dropout",
        regularization_hparams: dict = {},
        activation_fn: ACTIVATIONS = "ReLU",
        activation_fn_hparams: dict = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        main_module = MainLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            sparse=sparse,
            sparsity=sparsity,
            device=device,
            dtype=dtype,
        )

        self._add_layer(
            main_module=main_module,
            name=f"layer_{self.num_layers}",
            out_features=out_features,
            weight_init=weight_init,
            bias_init=bias_init,
            pooling_module=None,
            reshape_module=Flatten(),
            regularization=regularization,
            regularization_hparams=regularization_hparams,
            activation_fn=activation_fn,
            activation_fn_hparams=activation_fn_hparams,
            device=device,
            dtype=dtype,
        )
        self._add_output_layer(in_features=out_features, device=device, dtype=dtype)
        layer = self.last_layer()
        layer.out_features = out_features

    # * Connection methods
    def add_connections_layer(
        self,
        sparsity: float,
        device: Optional[str] = None,
    ):
        """Adds connection(s) on random neurons in the last layer
        with weighted probability based on layer distance.
        """
        layer = self.get_submodule(f"layer_{self.num_layers - 1}")
        if not self.is_frozen and layer.main.sparsity > 0:
            layer.main.add_connections(sparsity, device)

    def remove_connections_layer(self, sparsity: float, device: Optional[str] = None):
        """Randomly removes connections on the last layer."""
        layer = self.get_submodule(f"layer_{self.num_layers - 1}")
        if not self.is_frozen:
            layer.main.remove_connections(sparsity, device)

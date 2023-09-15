from typing import List
from modules.model import EvoModel


def _add_dense_layer(
    model: EvoModel, features: int, growth_rate: int = 32, bn_size: int = 4
):
    # Identity Layer
    model.add_block_conv(
        connection_index=model._get_connection_index(),
        block_branch=True,
        out_channels=features,
        dense_connection=True,
        main_identity=True,
        regularization=None,
        activation_fn=None,
    )
    # bn + relu (bottleneck)
    model.add_layer_conv(
        out_channels=features,
        main_identity=True,
        regularization="BatchNorm",
        activation_fn="ReLU",
    )
    # conv1x1 (bottleneck)
    model.add_layer_conv(
        out_channels=bn_size * growth_rate,
        kernel_size=1,
        stride=1,
        padding=0,
        main_identity=False,
        regularization=None,
        activation_fn=None,
    )
    # bn + relu (main)
    model.add_layer_conv(
        out_channels=bn_size * growth_rate,
        main_identity=True,
        regularization="BatchNorm",
        activation_fn="ReLU",
    )
    # conv3x3 (main)
    model.add_layer_conv(
        out_channels=growth_rate,
        kernel_size=3,
        stride=1,
        padding=1,
        main_identity=False,
        regularization=None,
        activation_fn=None,
    )
    # Concatenate
    model.last_block()._add_concatenate_layer()


def _add_dense_block(
    model: EvoModel,
    num_layers: int,
    num_input_features: int,
    growth_rate: int = 32,
    bn_size: int = 4,
):
    for i in range(num_layers):
        _add_dense_layer(
            model,
            features=num_input_features + i * growth_rate,
            growth_rate=growth_rate,
            bn_size=bn_size,
        )


def _add_transition_block(model: EvoModel, num_input_features: int):
    # Identity Layer
    model.add_block_conv(
        connection_index=model._get_connection_index(),
        block_branch=True,
        out_channels=num_input_features,
        dense_connection=False,
        main_identity=True,
        regularization=None,
        activation_fn=None,
    )
    # bn + relu (transition)
    model.add_layer_conv(
        out_channels=num_input_features,
        main_identity=True,
        regularization="BatchNorm",
        activation_fn="ReLU",
    )
    # conv1x1 (transition)
    model.add_layer_conv(
        out_channels=num_input_features // 2,
        kernel_size=1,
        stride=1,
        padding=0,
        main_identity=False,
        regularization=None,
        activation_fn=None,
    )
    # avgpool2x2 (transition)
    model.add_block_conv(
        connection_index=model._get_connection_index(),
        block_branch=True,
        out_channels=num_input_features // 2,
        dense_connection=False,
        main_identity=True,
        regularization=None,
        activation_fn=None,
        in_pooling=True,
    )


# DenseNet
def _densenet(
    model: EvoModel,
    block_config: List[int] = [6, 12, 24, 16],
    num_init_features: int = 64,
    growth_rate: int = 32,
    bn_size: int = 4,
):
    model.add_block_conv(
        out_channels=num_init_features,
        connection_index=(0, 0),
        kernel_size=1,
        stride=1,
        padding=0,
        block_branch=True,
    )

    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        _add_dense_block(
            model,
            num_layers=num_layers,
            num_input_features=num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
        )
        num_features += num_layers * growth_rate
        if i != len(block_config) - 1:
            # Transition layer
            _add_transition_block(model, num_input_features=num_features)
            num_features = num_features // 2

    model.add_block_conv(
        num_features,
        main_identity=True,
        regularization="BatchNorm",
        activation_fn=None,
        connection_index=model._get_connection_index(),
        block_branch=True,
    )

def densenet121(model: EvoModel):
    _densenet(model, [6, 12, 24, 16], 64, 32, 4)
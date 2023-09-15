from typing import List
from modules.model import EvoModel


def _basicblock(model, channels, stride):
    # Convolution Block
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=stride,
        padding=(1, 1),
        dilation=(1, 1),
        residual_connection=True,
    )

    # Convolution Layer
    model.add_layer_conv(
        channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )


def resnet18(model: EvoModel, num_input_features: int = 64):
    _resnet(model, [2, 2, 2, 2], num_input_features)


def resnet34(model: EvoModel, num_input_features: int = 64):
    _resnet(model, [3, 4, 6, 3], num_input_features)

def _resnet(model: EvoModel, block_config: List[int], num_input_features: int):
    features = num_input_features
    model.add_block_conv(
        features,
        connection_index=(0, 0),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        residual_connection=False,
    )

    # Convolutional Blocks
    for i, num_blocks in enumerate(block_config):
        for j in range(num_blocks):
            # Use stride 2 for the first block except for the first
            stride = 2 if i > 0 and j == 0 else 1
            _basicblock(model, features, stride=stride)
        features *= 2  # Double the number of channels for each new block
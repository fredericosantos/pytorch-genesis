from typing import Literal, List, Optional

# for noobity sake, the literal lists must be updated in
# every single file to be compatible with vscode IDE

ACTIVATIONS = Literal["ReLU", "GELU", "LeakyReLU"] # , "Sigmoid", "Tanh"]
NONLINEARITIES: Literal = Literal["relu", "gelu", "leaky_relu", "selu", "sigmoid", "tanh"]

REGULARIZATIONS = Literal["LayerNorm", "BatchNorm", "InstanceNorm", "Dropout"]
OUTPUT_NORMS = Optional[Literal["InstanceNorm"]]

POOLING = Literal["max", "avg"]

WEIGHT_INITS = Literal[
    "kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform",
]
BIAS_INITS = Literal["zeros"]
PADDING_MODES = Literal["zeros", "reflect", "replicate", "circular"]
OPTIMIZERS = Literal["adam", "radam", "sgd"]
LR_SCHEDULERS = Literal["onecycle", "cosine"]
LOSSES = Literal[
    "cross_entropy",
    "focalloss",
    "synthloss",
    "mse",
    "rmse",
    "mse_divclass",
    "relu1_mse",
    "preCEL",
    "mae",
    "nllloss",
    "polyloss",
]
BLOCK_TYPES = Literal["linear", "conv"]
CONV_TYPES = Literal["conv"]
LINEAR_TYPES = Literal["linear"]
LAYER_TYPES = Literal["layer", "output"]
MUTATION_TYPES = Literal["parent", "elite"]
MERGE_TYPES = Literal["sum", "linear"]

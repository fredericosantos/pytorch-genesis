from typing import List, Literal
from pytorch_lightning import Trainer
from modules.evolution.mutation import MutationTrainer
from modules.model import EvoModel
from modules.architectures import densenet, resnet


def evolve_resnet18_old(
    model,
    mutation_trainer: MutationTrainer,
    trainer,
    mutation_epochs: int = 5,
    lr_finder: bool = False,
    clear_notebook_output: bool = True,
    plot_lr_finder: bool = False,
    min_lr: float = 0.05,
    max_lr: float = 0.5,
):
    LOOPS = 1
    AUTO_SCALE_BATCH_SIZE = False
    MIN_BATCH_SIZE = 64

    channels = 64
    model.add_block_conv(
        channels,
        connection_index=(0, 0),
        block_branch=model.hparams.block_branch,
        kernel_size=(7, 7),
        stride=(1, 1),
        padding=(3, 3),
        residual_connection=False,
    )

    mutation_trainer.fit(
        trainer,
        model,
        generations=LOOPS,
        mutation_epochs=mutation_epochs,
        clear_notebook_output=clear_notebook_output,
        plot=plot_lr_finder,
        auto_scale_batch_size=AUTO_SCALE_BATCH_SIZE,
        min_batch_size=MIN_BATCH_SIZE,
        lr_find=lr_finder,
        min_lr=min_lr,
        max_lr=max_lr,
        update_check_val_every_n_epoch=False,
        enable_mutation_progress_bar=False,
    )
    strides = [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
    kernel_sizes = [3] * len(strides)
    paddings = [1] * len(strides)

    for i, s in enumerate(strides):
        if i % 2 == 0:
            model.add_block_conv(
                channels,
                connection_index=model._get_connection_index(),
                block_branch=model.hparams.block_branch,
                kernel_size=kernel_sizes[i],
                stride=s,
                padding=paddings[i],
                residual_connection=True,
            )
        else:
            model.add_layer_conv(
                channels, kernel_size=kernel_sizes[i], stride=s, padding=paddings[i]
            )
        mutation_trainer.fit(
            trainer,
            model,
            generations=LOOPS,
            mutation_epochs=mutation_epochs,
            clear_notebook_output=clear_notebook_output,
            plot=plot_lr_finder,
            auto_scale_batch_size=AUTO_SCALE_BATCH_SIZE,
            min_batch_size=MIN_BATCH_SIZE,
            lr_find=lr_finder,
            min_lr=min_lr,
            max_lr=max_lr,
            update_check_val_every_n_epoch=1,
            enable_mutation_progress_bar=False,
        )
        if i % 4 == 3:
            channels *= 2

    # FC Layer is in the last conv block


def evolve_output_norm(
    model: EvoModel,
    mutation_trainer: MutationTrainer,
    trainer: Trainer,
    mutation_epochs: int,
    fit_mutations: bool,
    enable_mutation_progress_bar: bool,
    progress_bar_position: int = 0,
    lr_finder: bool = False,
    update_check_val_every_n_epoch: bool = False,
    clear_notebook_output: bool = True,
    plot_lr_finder: bool = False,
    min_lr: float = 0.05,
    max_lr: float = 0.5,
    n_blocks: int = 3,
    n_layers: int = 4,
):
    n_layers = max(n_layers, 4)
    LOOPS = 1
    AUTO_SCALE_BATCH_SIZE = False
    MIN_BATCH_SIZE = 64
    for _ in range(n_blocks):
        channels = 64
        model.add_block_conv(
            channels,
            connection_index=(0, 0),
            kernel_size=3,
            stride=2,
            padding=2,
            dilation=1,
            residual_connection=False,
        )
        for _, c in zip(range(n_layers), [channels * 2, channels * 4, channels * 8]):
            model.add_layer_conv(c, kernel_size=3, stride=2, padding=2, dilation=1)
        if fit_mutations:
            mutation_trainer.fit(
                trainer,
                model,
                generations=LOOPS,
                mutation_epochs=mutation_epochs,
                clear_notebook_output=clear_notebook_output,
                plot=plot_lr_finder,
                auto_scale_batch_size=AUTO_SCALE_BATCH_SIZE,
                min_batch_size=MIN_BATCH_SIZE,
                lr_find=lr_finder,
                min_lr=min_lr,
                max_lr=max_lr,
                update_check_val_every_n_epoch=update_check_val_every_n_epoch,
                enable_mutation_progress_bar=enable_mutation_progress_bar,
                progress_bar_position=progress_bar_position,
            )
    if not fit_mutations:
        mutation_trainer.fit(
            trainer,
            model,
            generations=LOOPS,
            mutation_epochs=mutation_epochs,
            clear_notebook_output=clear_notebook_output,
            plot=plot_lr_finder,
            auto_scale_batch_size=AUTO_SCALE_BATCH_SIZE,
            min_batch_size=MIN_BATCH_SIZE,
            lr_find=lr_finder,
            min_lr=min_lr,
            max_lr=max_lr,
            update_check_val_every_n_epoch=update_check_val_every_n_epoch,
            enable_mutation_progress_bar=enable_mutation_progress_bar,
            progress_bar_position=progress_bar_position,
        )


def evolve_resnet(
    model,
    resnet_type: Literal["resnet18", "resnet34"],
    mutation_trainer: MutationTrainer,
    trainer,
    mutation_epochs: int,
    num_input_channels: int = 64,
    lr_finder: bool = False,
    clear_notebook_output: bool = True,
    plot_lr_finder: bool = False,
    min_lr: float = 0.05,
    max_lr: float = 0.5,
):
    LOOPS = 1

    if resnet_type == "resnet18":
        block_config = [2, 2, 2, 2]
    elif resnet_type == "resnet34":
        block_config = [3, 4, 6, 3]
    else:
        raise ValueError(f"Unknown resnet type {resnet_type}")

    trainer_params = dict(
        generations=LOOPS,
        mutation_epochs=mutation_epochs,
        clear_notebook_output=clear_notebook_output,
        plot=plot_lr_finder,
        lr_find=lr_finder,
        min_lr=min_lr,
        max_lr=max_lr,
        update_check_val_every_n_epoch=False,
        enable_mutation_progress_bar=False,
    )

    num_channels = num_input_channels
    model.add_block_conv(
        num_channels,
        connection_index=(0, 0),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        residual_connection=False,
    )

    mutation_trainer.fit(trainer, model, **trainer_params)
    for i, num_blocks in enumerate(block_config):
        for j in range(num_blocks):
            # Use stride 2 for the first block except for the first
            stride = 2 if i > 0 and j == 0 else 1
            resnet._basicblock(model, num_channels, stride=stride)
            mutation_trainer.fit(trainer, model, **trainer_params)
        num_channels *= 2  # Double the number of channels for each new block


def evolve_densenet(
    model,
    densenet_type: Literal["densenet121"],
    mutation_trainer: MutationTrainer,
    trainer,
    mutation_epochs: int,
    num_input_channels: int = 64,
    growth_rate: int = 32,
    bn_size: int = 4,
    lr_finder: bool = False,
    clear_notebook_output: bool = True,
    plot_lr_finder: bool = False,
    min_lr: float = 0.05,
    max_lr: float = 0.5,
):
    LOOPS = 1

    if densenet_type == "densenet121":
        block_config = [6, 12, 24, 16]
    else:
        raise NotImplementedError(f"Unknown densenet type {densenet_type}")

    trainer_params = dict(
        generations=LOOPS,
        mutation_epochs=mutation_epochs,
        clear_notebook_output=clear_notebook_output,
        plot=plot_lr_finder,
        lr_find=lr_finder,
        min_lr=min_lr,
        max_lr=max_lr,
        update_check_val_every_n_epoch=False,
        enable_mutation_progress_bar=False,
    )

    model.add_block_conv(
        out_channels=num_input_channels,
        connection_index=(0, 0),
        kernel_size=1,
        stride=1,
        padding=0,
        block_branch=True,
    )

    num_features = num_input_channels
    mutation_trainer.fit(trainer, model, **trainer_params)

    for i, num_layers in enumerate(block_config):
        densenet._add_dense_block(
            model,
            num_layers=num_layers,
            num_input_features=num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
        )
        num_features += num_layers * growth_rate
        if i != len(block_config) - 1:
            # Transition layer
            densenet._add_transition_block(model, num_input_features=num_features)
            num_features = num_features // 2
        mutation_trainer.fit(trainer, model, **trainer_params)

    model.add_block_conv(
        num_features,
        main_identity=True,
        regularization="BatchNorm",
        activation_fn=None,
        connection_index=model._get_connection_index(),
        block_branch=True,
    )
    mutation_trainer.fit(trainer, model, **trainer_params)
    

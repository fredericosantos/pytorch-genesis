from pytorch_lightning import Trainer
from modules.evolution.mutation import MutationTrainer
from modules.model import EvoModel


# RESNET18
def resnet18(model: EvoModel):
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

    # Layer 1
    # BasicBlock0
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # BasicBlock1
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Layer 2
    channels *= 2
    # BasicBlock0
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # BasicBlock1
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Layer 3
    channels *= 2
    # BasicBlock0
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # BasicBlock1
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Layer 4
    channels *= 2
    # BasicBlock0
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # BasicBlock1
    model.add_block_conv(
        channels,
        connection_index=model._get_connection_index(),
        block_branch=model.hparams.block_branch,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        residual_connection=True,
    )
    model.add_layer_conv(channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # FC Layer is in the last conv block


def evolve_resnet18(
    model,
    mutation_trainer: MutationTrainer,
    trainer,
    mutation_epochs: int = 3,
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
        update_check_val_every_n_epoch=1,
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
            model.add_layer_conv(channels, kernel_size=kernel_sizes[i], stride=s, padding=paddings[i])
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
            channels, connection_index=(0, 0), kernel_size=3, stride=2, padding=2, residual_connection=False,
        )
        for _, c in zip(range(n_layers), [channels * 2, channels * 4, channels * 8]):
            model.add_layer_conv(c, kernel_size=3, stride=2, padding=2)
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


import random
import torch as torch
from copy import deepcopy
from typing import Any, Callable, List, Tuple, Optional, Literal, Dict, Union, get_args
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error

import logging

import numpy as np
from modules.base.block import BaseBlock

import modules.utils.block_utils as block_utils
from modules.model import EvoModel
import modules.regularization_functions as reg_fn
from modules.hyperparams_options import *
from tqdm import tqdm, trange
from rich.progress import track
from IPython.display import clear_output


class MutationCallback(Callback):
    def __init__(
        self,
        datamodule: LightningDataModule,
        block_types: BLOCK_TYPES = ["linear", "conv"],
        num_epochs: int = 1,
        mutation_options: Optional[Union[list, dict]] = None,
        get_mutation_hparams: Callable = None,
        fitness_threshold: float = 1.0,
        population: int = 1,
        first_population: int = 1,
        verbose: bool = False,
        evalutation_method: Literal["train_loss", "valid_loss", "train_acc", "valid_acc"] = "valid_acc",
        max_num_layers: int = 5,
        min_num_layers: int = 1,
        max_channels: int = 512,
        min_channels: int = 32,
    ):
        self.datamodule = datamodule
        self.mutation_options = mutation_options
        self.get_mutation_hparams = get_mutation_hparams
        self.block_types = block_types
        self.num_epochs = num_epochs
        self.fitness_threshold = fitness_threshold
        self._verbose = verbose
        self._pop = population
        self.pop = first_population
        self.first_pop = first_population
        self.is_first_gen = True
        self.evalutation_method = evalutation_method
        self.max_num_layers = max_num_layers
        self.min_num_layers = min_num_layers
        self.max_channels = max_channels
        self.min_channels = min_channels
        self.counters = {arg: 0 for arg in ["mutation", "elite", "total_mutations", "initial_search"]}
        self.checkpoint = {}
        self.mutation = {}

    def on_train_epoch_end(self, trainer: Trainer, model: EvoModel) -> None:
        self.counters["mutation"] += 1
        if self.counters["mutation"] == self.num_epochs:
            self._evaluate_mutation(trainer, model)

    def save_checkpoint(self, trainer: Trainer, model: EvoModel, cp_type: MUTATION_TYPES) -> None:
        """
        Saves the current block and number of blocks.
        """
        block = model.last_block() if model.num_blocks > 0 else None
        cpkt = {
            "block": deepcopy(block),
            "num_blocks": deepcopy(model.num_blocks),
        }
        self.checkpoint[cp_type] = cpkt

        if model.num_blocks > 0:
            trainer.save_checkpoint(f"mutation_checkpoint_{cp_type}.ckpt", True)

    def load_checkpoint(self, trainer: Trainer, model: EvoModel, cp_type: MUTATION_TYPES) -> None:
        ckpt = self.checkpoint[cp_type]
        while model.num_blocks >= ckpt["num_blocks"]:
            model.remove_last_block()
            if model.num_blocks == 0:
                break
        model.freeze_last_block()
        if ckpt["block"] is not None:
            setattr(model, f"block_{ckpt['num_blocks']-1}", deepcopy(ckpt["block"]))
            model._connect_block(ckpt["block"])
        model.num_blocks = deepcopy(ckpt["num_blocks"])
        if model.num_blocks > 0:
            checkpoint = torch.load(f"mutation_checkpoint_{cp_type}.ckpt")
            model.load_state_dict(checkpoint["state_dict"])

    def apply_mutation(self, trainer: Trainer, model: EvoModel) -> None:
        if self.counters["elite"] == 0:
            self.save_checkpoint(trainer, model, "parent")
            self.save_checkpoint(trainer, model, "elite")
        mutation_options = self._get_mutations(model, True)
        initial_mutation = random.choice(mutation_options)
        mutation_hparams = self._get_mutation_hparams(model, initial_mutation)
        mutation_to_call = getattr(model, f"{initial_mutation}")
        mutation_to_call(**mutation_hparams)

        self._save_mutation_hparams(model, True)

        num_layers = random.choice(list(range(self.max_num_layers)))
        layers_added = 0
        for _ in range(num_layers):
            mutation_options = self._get_mutations(model, False)
            mutation = random.choice(mutation_options)
            layer_hparams = self._get_mutation_hparams(model, mutation)
            mutation_to_call = getattr(model, f"{mutation}")
            mutation_to_call(**layer_hparams)
            self._save_mutation_hparams(model, False)
            layers_added += 1
            if model.last_block().block_type in get_args(CONV_TYPES):
                l = model.last_block().last_layer()
                if l.out_width == 1 and l.out_height == 1:
                    break

        self.counters["total_mutations"] += 1
        self.counters["elite"] += 1
        if self.is_first_gen:
            self.counters["initial_search"] += 1
        self.configure_optimizers(trainer, model)
        if self._verbose:
            print(f"MUTATION: {initial_mutation.upper()} WITH {num_layers} LAYERS ~ [{model.num_blocks} BLOCKS]")

    def _evaluate_mutation(self, trainer: Trainer, model: EvoModel) -> None:
        improved = False

        train_acc = trainer.logged_metrics["train/Accuracy"]
        valid_acc = trainer.logged_metrics["valid/Accuracy"]
        valid_loss = trainer.logged_metrics["valid/loss_epoch"]
        train_loss = trainer.logged_metrics["train/loss_epoch"]

        if hasattr(self, "metrics"):
            train_elite_acc = self.metrics["elite"]["train_Accuracy"]
            train_elite_loss = self.metrics["elite"]["train_loss_epoch"]
            valid_elite_acc = self.metrics["elite"]["valid_Accuracy"]
            valid_elite_loss = self.metrics["elite"]["valid_loss_epoch"]

            if self.evalutation_method == "train_loss":
                condition = train_loss * self.fitness_threshold < train_elite_loss
            elif self.evalutation_method == "train_acc":
                condition = train_acc > train_elite_acc * self.fitness_threshold
            elif self.evalutation_method == "valid_loss":
                condition = valid_loss * self.fitness_threshold < valid_elite_loss
            elif self.evalutation_method == "valid_acc":
                condition = valid_acc > valid_elite_acc * self.fitness_threshold
        else:
            condition = True

        if self._verbose:
            print(f"[EVALUATING MUTATION] [ELITE: {self.counters['elite']} / {self.pop}]")
            if hasattr(self, "metrics"):
                print(f"ACCURACY {valid_elite_acc:.2%} -> {valid_acc:.2%}")
                print(f"LOSS {valid_elite_loss:.4} -> {valid_loss:.4}")

        if condition:
            improved = True
            self._save_metrics(trainer, "elite")
            self.save_checkpoint(trainer, model, "elite")

        self._log_mutation_hparams(trainer, model, improved, valid_acc)

        if self.counters["elite"] >= self.pop:
            reset_elite_counters = True
            self.load_checkpoint(trainer, model, "elite")
            self._save_metrics(trainer, "parent")
            self.save_checkpoint(trainer, model, "parent")
        else:
            reset_elite_counters = False
            self.load_checkpoint(trainer, model, "parent")

        self._reset_counters(reset_elite_counters)

        if self._verbose:
            print(f"{self.mutation}")

    def _get_mutations(self, model: EvoModel, is_new_block: bool = True) -> List[str]:
        if is_new_block:
            mutations = [f"add_block_{t}" for t in self.block_types]
        else:
            block = model.last_block()
            mutations = [f"add_layer_{block.block_type}"]

        if isinstance(self.mutation_options, dict):
            for mutation in mutations:
                if mutation not in self.mutation_options.keys():
                    mutations.remove(mutation)
        if isinstance(self.mutation_options, list):
            for mutation in mutations:
                if mutation not in self.mutation_options:
                    mutations.remove(mutation)
        return mutations

    def _get_mutation_hparams(self, model: EvoModel, mutation: str):
        if self.get_mutation_hparams is not None:
            return self.get_mutation_hparams(model, mutation)
        rnd_sparsity = random.choice(np.linspace(0.05, 1.0, 20))
        linear_nodes = [64, 128, 256, 512, 1024]
        add_block_common = dict(
            block_branch=random.choice([True, False]) if model.hparams.block_branch else False,
            # added possibility of having class specialization but only 20% of the time
            output_class_specialization=random.choice([True, False, False, False, False]), 
            layers_weight_init=random.choice(list(get_args(WEIGHT_INITS))),
            layers_bias_init=random.choice(list(get_args(BIAS_INITS))),
            activation_fn=random.choice(list(get_args(ACTIVATIONS))),
            activation_fn_hparams=None,
        )
        add_layer_common = dict()
        mutation_options = {
            "add_block_linear": dict(
                out_features=random.choice(linear_nodes),
                sparsity=rnd_sparsity,
                layers_bias=random.choice([True, False]),
                layers_sparse=random.choice([True, False]),
                regularization=random.choice(list(get_args(REGULARIZATIONS)) + [None]),
                regularization_hparams=None,
            )
            | add_block_common,
            "add_block_conv": dict(
                layers_bias=False,
                residual_connection=(res_cxt := random.choice([True, False])),
                keep_input_dims=random.choice([True, False]) if res_cxt else False,
                regularization="BatchNorm",
                regularization_hparams=None,
            )
            | add_block_common,
            "add_layer_linear": dict(out_features=None, sparsity=rnd_sparsity,) | add_layer_common,
            "add_layer_conv": {} | add_layer_common,
        }
        return mutation_options[mutation]

    def _save_mutation_hparams(self, model: EvoModel, new_block: bool = False):
        b = model.last_block()
        if new_block:
            block_hparams = dict(
                block_type=b.block_type,
                block_branch=b.block_branch,
                connection_index=b.connection_index,
                output_class_specialization=b.output_class_specialization,
                output_bias=b.output_bias,
                layers_weight_init=b.layers_weight_init,
                layers_bias_init=b.layers_bias_init,
                activation_fn=b.activation_fn,
                activation_fn_hparams=b.activation_fn_hparams,
                regularization=b.regularization,
                regularization_hparams=b.regularization_hparams,
            )
            if b.block_type == "linear":
                block_hparams |= dict(layers_bias=b.layers_bias, layers_sparse=b.layers_sparse,)
            elif b.block_type == "conv":
                block_hparams |= dict(
                    layers_bias=b.layers_bias,
                    residual_connection=b.residual_connection,
                    keep_input_dims=b.keep_input_dims,
                )
            else:
                raise NotImplementedError(f"block type {b.block_type} not implemented.")
            self.mutation["improved"] = False
            self.mutation["block_hparams"] = block_hparams
            self.mutation["layers_hparams"] = dict()

        if b.block_type == "linear":
            layer_hparams = dict(out_features=b.last_layer().main.out_features, sparsity=b.last_layer().main.sparsity)
        elif b.block_type == "conv":
            layer_hparams = dict(
                out_channels=b.last_layer().main.out_channels,
                kernel_size=b.last_layer().main.kernel_size,
                stride=b.last_layer().main.stride,
                padding=b.last_layer().main.padding,
            )
        self.mutation["layers_hparams"][b.num_layers - 1] = layer_hparams
        self.mutation["num_layers"] = b.num_layers

    def _log_mutation_hparams(self, trainer: Trainer, model: EvoModel, improved: bool, valid_acc: float):
        self.mutation["improved"] = improved
        if model.logger is not None:
            tb = model.logger.experiment
            tb.add_text("mutation", str(self.mutation), trainer.global_step)
            if improved:
                trainer.logger.log_graph(model)

    # TODO: remove this
    def _log_tb_total_params(self, trainer: Trainer, model: EvoModel):
        if trainer.logger is not None:
            m_s = ModelSummary(model)
            model.log("parameters_stats/total", m_s.total_parameters)
            model.log("parameters_stats/trainable", m_s.trainable_parameters)
            model.log("parameters_stats/trainable_ratio", m_s.trainable_parameters / m_s.total_parameters)

    def _reset_counters(self, elite: bool = True):
        self.counters["mutation"] = 0
        if elite:
            self.counters["elite"] = 0
        if self.counters["initial_search"] == self.first_pop and self.is_first_gen:
            self.is_first_gen = False
            self.pop = self._pop

    def _save_metrics(self, trainer: Trainer, key: MUTATION_TYPES):
        """If given `elite` as key, saves the `logged_metrics` to the `metrics['elite']` dict.
        If given `parent` as key, stores the current `metrics['elite']` under `metrics['parent']`
        """

        if not hasattr(self, "metrics"):
            self.metrics = {arg: {} for arg in ["parent", "elite"]}
        if key == "elite":
            for t in ["train", "valid"]:
                for j in ["loss_epoch", "Accuracy"]:
                    self.metrics[key][f"{t}_{j}"] = trainer.logged_metrics[f"{t}/{j}"]
        elif key == "parent":
            self.metrics["parent"] = deepcopy(self.metrics["elite"])

    def _load_metrics(self, key: MUTATION_TYPES):
        if key == "elite":
            return self.metrics["elite"]
        elif key == "parent":
            return self.metrics["parent"]
        else:
            raise ValueError(f"key {key} not recognized.")

    def configure_optimizers(self, trainer: Trainer, model: EvoModel, lr: Optional[float] = None):
        if lr is not None:
            model.hparams.lr = lr
        model.configure_optimizers()
        trainer.strategy.setup_optimizers(trainer)

    # TODO: Implement resnet block mutation
    def _add_resnet_block(self, model: EvoModel):
        model.add_block_conv(
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), residual_connection=True,
        )
        model.add_layer_conv(
            model.last_block().last_layer().out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )


class MutationTrainer:
    def __init__(self, datamodule: LightningDataModule, batch_size: int):
        self.datamodule = datamodule
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")

    def fit(
        self,
        trainer: Trainer,
        model: EvoModel,
        generations: int,
        mutation_epochs: int,
        mutate: bool = True,
        reset_parameters: bool = False,
        update_check_val_every_n_epoch: bool = True,
        clear_notebook_output: bool = False,
        auto_scale_batch_size: bool = False,
        min_batch_size: int = 128,
        lr_find: bool = True,
        min_lr: float = 0.005,
        max_lr: float = 0.5,
        lr_find_num_training: int = 100,
        fitness_threshold: Optional[float] = None,
        plot: bool = True,
        enable_mutation_progress_bar: bool = True,
        progress_bar_position: int = 0
    ) -> None:
        mutation = False
        for i, callback in enumerate(trainer.callbacks):
            if callback.__class__.__name__ == "MutationCallback":
                mut_cb: MutationCallback = trainer.callbacks[i]
                if fitness_threshold is not None:
                    mut_cb.fitness_threshold = fitness_threshold
                mutation = True
        if not mutate:
            mutation = False
        is_first_gen: bool = mut_cb.is_first_gen if mutation else False
        total_mutations = (
            (generations) * mut_cb._pop + (mut_cb.first_pop - mut_cb._pop if is_first_gen else 0)
            if mutation
            else generations
        )
        for i in trange(total_mutations, desc="Evolving",disable=not enable_mutation_progress_bar, position=progress_bar_position, leave=False):
            if mutation:
                mut_cb.apply_mutation(trainer, model)
            if reset_parameters:
                model.reset_parameters()
            if auto_scale_batch_size:
                trainer_extra = Trainer(gpus="auto", enable_progress_bar=False, progress_bar_refresh_rate=0)
                trainer_extra.tuner.scale_batch_size(model, datamodule=self.datamodule, init_val=min_batch_size)
                del trainer_extra
                garbage_collection_cuda()
            if lr_find:
                trainer_extra = Trainer(gpus="auto", enable_progress_bar=False, progress_bar_refresh_rate=0)
                lr_finder = trainer_extra.tuner.lr_find(
                    model, datamodule=self.datamodule, min_lr=min_lr, max_lr=max_lr, num_training=lr_find_num_training,
                )
                if plot:
                    lr_finder.plot(True, True)
                model.hparams.lr = lr_finder.suggestion()
                del trainer_extra
                garbage_collection_cuda()
                # TODO: empirically prove that lr * 2 is best
                model.hparams.lr_scheduler_hparams["max_lr"] = model.hparams.lr * 1
                if enable_mutation_progress_bar:
                    print(f"Max Learning Rate: {model.hparams.lr_scheduler_hparams['max_lr']}")
            if trainer.current_epoch != 0:
                trainer.fit_loop.max_epochs += mutation_epochs
            else:
                trainer.fit_loop.max_epochs = mutation_epochs
            model.hparams.lr_scheduler_hparams["epochs"] = mutation_epochs
            model.hparams.lr_scheduler_hparams["steps_per_epoch"] = len(self.datamodule.train_dataloader())
            if update_check_val_every_n_epoch:
                trainer.check_val_every_n_epoch = mutation_epochs
            if mutation:
                mut_cb.num_epochs = mutation_epochs
            model.configure_optimizers()
            trainer.fit(model, datamodule=self.datamodule)
            if clear_notebook_output:
                clear_output()


class ParameterMonitor(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, model: EvoModel) -> None:
        if hasattr(model, "logger"):
            with torch.no_grad():
                y_hat, all_blocks_y_hat = model(torch.ones_like(model.example_input_array, device=model.device))

            for i in range(model.num_blocks):
                block: BaseBlock = model.get_submodule(f"block_{i}")
                if not block.is_frozen:
                    model.logger.experiment.add_histogram(
                        f"outputs/block_{i}", all_blocks_y_hat[i], global_step=model.global_step
                    )
            model.logger.experiment.add_histogram(f"outputs/model", y_hat, global_step=model.global_step)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    model.logger.experiment.add_histogram(f"parameters/{name}", param, global_step=model.global_step)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_tb_total_params(model=pl_module)

    def _log_tb_total_params(self, model: EvoModel):
        m_s = ModelSummary(model)
        model.log("parameters_stats/total", m_s.total_parameters)
        model.log("parameters_stats/trainable", m_s.trainable_parameters)
        model.log("parameters_stats/trainable_ratio", m_s.trainable_parameters / m_s.total_parameters)

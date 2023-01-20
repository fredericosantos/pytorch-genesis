from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning import LightningModule, Trainer


def create_graph(model: LightningModule, trainer: Trainer, log_generations: bool = False) -> None:
    if log_generations:
        if hasattr(trainer, "n_generations"):
            trainer.n_generations += 1
        else:
            trainer.n_generations = 0
        generations = "_" + str(trainer.n_generations)
    else:
        generations = ""
    with SummaryWriter(log_dir=trainer.logger.log_dir + generations) as w:
        w.add_graph(model, model.example_input_array.to(model.device))



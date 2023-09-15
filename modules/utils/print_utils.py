from typing import Tuple
from pytorch_lightning import LightningModule


def print_connection_stats(model: LightningModule) -> None:
    for name, module in model.named_modules():
        try:
            print(
                f"{name.upper()}\t\t[{module._get_name()}\tN_CXTS = {module.num_connections}\tCXT = {module.connection_index[0]}]"
            )
        except:
            ...


def print_grad_states(model: LightningModule) -> Tuple[str, bool]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

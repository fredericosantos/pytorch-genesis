import numpy as np

# PyTorch
from typing import Literal
import torch as torch
import torchvision

# Lightning Bolts
from pl_bolts.datamodules import (
    MNISTDataModule,
    FashionMNISTDataModule,
    CIFAR10DataModule,
)
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, emnist_normalization


def get_datamodule(
    dataset_name: Literal["MNIST", "FashionMNIST", "CIFAR10"],
    data_dir: str = ".",
    batch_size: int = 256,
    val_split: float = 0.1,
    num_workers: int = 20,
):
    if dataset_name == "MNIST":
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.ToTensor(),
                emnist_normalization("mnist"),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), emnist_normalization("mnist"),]
        )

        dm = MNISTDataModule(
            data_dir=data_dir,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )

    elif dataset_name == "FashionMNIST":
        ...
    elif dataset_name == "CIFAR10":
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), cifar10_normalization(),])

        dm = CIFAR10DataModule(
            data_dir=data_dir,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    dm.prepare_data()
    dm.setup(stage="fit")
    return dm

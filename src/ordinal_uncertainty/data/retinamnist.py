"""Native-resolution RetinaMNIST loader for the canonical experiments."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def retinamnist_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    download: bool,
    image_size: int,
) -> tuple[dict[str, DataLoader], dict[str, object]]:
    """Build official RetinaMNIST loaders without changing native 28x28 inputs."""
    if image_size != 28:
        raise ValueError("canonical RetinaMNIST experiments require native image_size=28")
    try:
        from medmnist import RetinaMNIST, INFO
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("medmnist is required; install the project dependencies") from error

    transform = transforms.ToTensor()
    datasets = {
        split: RetinaMNIST(split=split, root=str(data_root), download=download, transform=transform)
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    info = INFO["retinamnist"]
    return loaders, {"num_classes": len(info["label"]), "task": info["task"], "image_size": image_size}

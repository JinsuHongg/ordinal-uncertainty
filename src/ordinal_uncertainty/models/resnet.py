"""Small-image ResNet18 classifier used by Experiment 0."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def make_resnet18(num_classes: int) -> nn.Module:
    """Build an unpretrained ResNet18 adapted to 64px RetinaMNIST inputs."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

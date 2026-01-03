from typing import Literal

import torch.nn as nn

from .resnet152 import ResNet152Binary
from .simple_cnn import SimpleCNN
from .vgg16 import VGG16Binary


def get_model(
    model_name: Literal["simplecnn", "vgg16", "resnet152"],
    freeze_features: bool = False,
    num_classes: int = 1,
) -> nn.Module:
    if model_name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "vgg16":
        return VGG16Binary(num_classes=num_classes, freeze_features=freeze_features)
    elif model_name == "resnet152":
        return ResNet152Binary(num_classes=num_classes, freeze_features=freeze_features)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

import torch
import torch.nn as nn
from torchvision import models


class ResNet152Binary(nn.Module):
    def __init__(self, num_classes: int = 1, freeze_features: bool = False) -> None:
        super().__init__()
        self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

        # Freeze feature extractor parameters if specified
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

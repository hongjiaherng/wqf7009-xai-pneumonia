import torch
import torch.nn as nn
from torchvision import models


class VGG16Binary(nn.Module):
    def __init__(self, num_classes: int = 1, freeze_features: bool = False) -> None:
        super().__init__()
        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)

        # Freeze feature extractor parameters if specified
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Modify the classifier for binary classification
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

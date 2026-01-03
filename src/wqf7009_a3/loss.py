import torch
from torchvision.datasets import ImageFolder


def _get_pos_weight(train_dataset: ImageFolder) -> torch.Tensor:
    labels = [label for _, label in train_dataset.samples]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    return pos_weight


def get_weighted_bce_loss(train_dataset: ImageFolder):
    pos_weight = _get_pos_weight(train_dataset)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

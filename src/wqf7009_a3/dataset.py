from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_SIZE = 256
IMG_SIZE = 224


transform = transforms.Compose(
    [
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def get_dataloader(
    data_dir: Path,
    split: Literal["train", "val", "test"],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform=transform,
) -> DataLoader:
    dataset = datasets.ImageFolder(root=data_dir / split, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader

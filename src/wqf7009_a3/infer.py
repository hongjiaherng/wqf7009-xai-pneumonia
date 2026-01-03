import torch
import torch.nn as nn


def predict(images: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Make predictions on a batch of images.

    Args:
        images (torch.Tensor): Batch of images, shape (B, C, H, W).
        model (nn.Module): Trained model for inference.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of (predictions, probabilities).

    """
    model.eval()

    with torch.no_grad():
        outputs = model(images)  # (B, 1)
        probs = torch.sigmoid(outputs).squeeze()  # (B)
        preds = (probs >= 0.5).long()  # (B)

    return preds, probs

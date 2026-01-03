import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from tqdm.auto import tqdm

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def visualize_gradcam(
    image: torch.Tensor,
    label: int,
    output: int,
    conf: float,
    image_id: int,
    idx_to_classname: dict,
    gradcam: GradCAM,
    save_path: str | None = None,
) -> None:
    """
    Visualize Grad-CAM heatmaps for a given image.

    Parameters
    ----------
    image: torch.Tensor
        The input image tensor. (C, H, W)
    label: int
        The true label of the image.
    output: int
        The predicted label of the image.
    conf: float
        The confidence of the prediction.
    image_id: int
        The ID of the image.
    idx_to_classname: dict
        A mapping from index to class name.
    gradcam: GradCAM
        The Grad-CAM object.
    save_path: str or None
        If provided, saves the figure to this path. If None, does not save.

    """

    input_tensor = image.unsqueeze(0)  # (1, C, H, W)
    targets = [BinaryClassifierOutputTarget(output)]

    grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)  # (1, H, W)
    grayscale_cam = grayscale_cam[0, :]  # (H, W)

    original_image = image.permute(1, 2, 0).detach().cpu().numpy()
    original_image = original_image * np.array(IMAGENET_STD).reshape(
        1, 1, 3
    ) + np.array(IMAGENET_MEAN).reshape(1, 1, 3)  # Denormalize
    original_image = np.clip(original_image, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(
        f"Original Image (ID: {image_id})\nTrue: {idx_to_classname[label]}; Pred: {idx_to_classname[output]}\nConf: {conf:.4f}"
    )
    axes[0].axis("off")

    # Grad-CAM heatmap
    im1 = axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap\nRed: High Importance; Blue: Low Importance")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    # Overlay
    viz = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    axes[2].imshow(viz)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")


class RISE(nn.Module):
    """
    Randomized Input Sampling for Explanation (RISE).

    Generates explanations by randomly masking input images and weighting
    the masks by their corresponding model predictions.

    Reference: Petsiuk et al. "RISE: Randomized Input Sampling for Explanation
    of Black-box Models" (BMVC 2018)
    """

    def __init__(
        self, model: nn.Module, input_size: tuple[int, int], batch_size: int = 100
    ):
        """
        Initialize the RISE explainer.

        Parameters
        ----------
        model: nn.Module
            The pre-trained model to explain.
        input_size: tuple[int, int]
            The size of the input images (height, width), e.g., (224, 224).
        batch_size: int
            The batch size for inference to prevent OOM. Defaults to 100.
        """
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.masks: torch.Tensor | None = None  # Shape: (N, 1, H, W)
        self.N = 0
        self.p1 = 0.0  # Probability of 1 in the mask
        self.device = next(model.parameters()).device

    def generate_masks(self, N: int, s: int, p1: float) -> None:
        """
        Generate random binary masks for RISE.

        Parameters
        ----------
        N: int
            Number of masks to generate.
        s: int
            Grid size (e.g., 8 means an 8x8 grid).
        p1: float
            Probability of masking (occlusion density).
        """
        self.N = N
        self.p1 = p1

        # 1. Calculate dimensions
        H, W = self.input_size
        cell_h = int(np.ceil(H / s))
        cell_w = int(np.ceil(W / s))
        up_size_h = (s + 1) * cell_h
        up_size_w = (s + 1) * cell_w

        # 2. Generate low-resolution random grid
        # grid shape: (N, 1, s, s)
        grid = (torch.rand(N, 1, s, s) < p1).float().to(self.device)

        # 3. Upsample masks
        # We use PyTorch interpolate instead of skimage for speed and GPU support
        self.masks = torch.empty((N, 1, H, W), device=self.device)

        with torch.no_grad():
            for i in tqdm(range(0, N, self.batch_size), desc="Generating RISE masks"):
                batch_end = min(i + self.batch_size, N)
                batch_grid = grid[i:batch_end]

                # Upsample to larger than image
                upsampled = F.interpolate(
                    batch_grid,
                    size=(up_size_h, up_size_w),
                    mode="bilinear",
                    align_corners=False,
                )

                # Random cropping to (H, W) for shift invariance
                # We do this for every mask in the batch
                for j in range(upsampled.shape[0]):
                    # Random shift indices
                    x = np.random.randint(0, cell_h)
                    y = np.random.randint(0, cell_w)

                    self.masks[i + j] = upsampled[j, :, x : x + H, y : y + W]

        print(f"Generated {N} masks of size {self.input_size}")

    def forward(self, x: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """
        Apply RISE to input images to generate saliency maps.

        Parameters
        ----------
        x: torch.Tensor
            Input image tensor of shape (C, H, W) or (B, C, H, W).
        target_class: int
            1 for positive class (Pneumonia), 0 for negative class (Normal).

        Returns
        -------
        torch.Tensor
            Saliency map tensor of shape (1, H, W) or (B, 1, H, W).

        """
        if self.masks is None:
            raise ValueError(
                "Masks have not been generated. Call generate_masks() first."
            )

        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)

        return (
            self._forward_single(x, target_class)
            if x.size(0) == 1
            else self._forward_batch(x, target_class)
        )

    def _forward_single(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Optimized forward pass for single image input.

        Parameters
        ----------
        x: torch.Tensor
            Input image tensor of shape (1, C, H, W).
        Returns
        -------
        torch.Tensor
            Saliency map tensor of shape (1, H, W).
        """
        N = self.N
        _, _, H, W = x.shape

        # Broadcasting: (1, C, H, W) * (N, 1, H, W) -> (N, C, H, W)
        stack = torch.mul(self.masks, x)  # (N, C, H, W)

        p_list: list[torch.Tensor] = []

        for i in tqdm(range(0, N, self.batch_size), desc="Computing RISE saliency"):
            with torch.no_grad():
                batch_imgs = stack[i : min(i + self.batch_size, N)]
                outputs = self.model(batch_imgs)  # (batch_size, 1)
                probs = self._get_prob(outputs, target_class)  # (batch_size, 1)
                p_list.append(probs)

        p = torch.cat(p_list, dim=0)  # (N, 1)
        n_classes = p.size(1)  # 1 for binary

        # Weighted sum: (1, N) @ (N, H*W) -> (1, H*W)
        sal = torch.matmul(p.transpose(0, 1), self.masks.view(N, -1))
        sal = sal.view((n_classes, H, W))  # (1, H, W)
        sal = sal / N / self.p1  # (1, H, W)
        return sal

    def _forward_batch(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Forward pass for batch image input.

        Parameters
        ----------
        x: torch.Tensor
            Input image tensor of shape (B, C, H, W).
        Returns
        -------
        torch.Tensor
            Saliency map tensor of shape (B, 1, H, W).
        """
        N = self.N
        B, C, H, W = x.shape

        # Total inferences required: N * B
        total_inferences = N * B
        p_list: list[torch.Tensor] = []

        for i in tqdm(
            range(0, total_inferences, self.batch_size), desc="Computing RISE saliency"
        ):
            end_idx = min(i + self.batch_size, total_inferences)

            # 1. Calculate indices on the fly
            indices = torch.arange(i, end_idx, device=self.device)
            mask_indices = torch.div(indices, B, rounding_mode="floor")  # (B,)
            image_indices = indices % B  # (B,)

            # 2. Construct only this mini-batch
            batch_masks = self.masks[mask_indices]  # (B, 1, H, W)
            batch_imgs = x[image_indices]  # (B, C, H, W)
            masked_inputs = batch_masks * batch_imgs  # (B, C, H, W)

            # 3. Model inference
            with torch.no_grad():
                outputs = self.model(masked_inputs)  # (batch_size, 1)
                probs = self._get_prob(outputs, target_class)  # (batch_size, 1)
                p_list.append(probs)

        p = torch.cat(p_list, dim=0)  # (N*B, 1)
        n_classes = p.size(1)  # 1 for binary
        p = p.view(N, B, n_classes)  # (N, B, 1)

        # Weighted sum: (B, 1, N) @ (N, H*W) -> (B, 1, H*W)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, -1))
        sal = sal.view((B, n_classes, H, W))  # (B, 1, H, W)
        sal = sal / N / self.p1  # (B, 1, H, W)
        return sal

    def _get_prob(self, outputs: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Get the probability of the target class from model outputs.

        Parameters
        ----------
        outputs: torch.Tensor
            Model outputs of shape (batch_size, 1).
        target_class: int
            1 for positive class (Pneumonia), 0 for negative class (Normal).

        Returns
        -------
        torch.Tensor
            Probability tensor of shape (batch_size, 1).
        """
        probs = torch.sigmoid(outputs)  # (batch_size, 1)
        if target_class == 1:
            return probs  # Positive class
        else:
            return 1 - probs  # Negative class


def visualize_rise(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    label: int,
    output: int,
    conf: float,
    image_id: int,
    idx_to_classname: dict,
    save_path: str | None = None,
) -> None:
    """
    Visualize RISE saliency maps for a given image.

    Parameters
    ----------
    image: torch.Tensor
        The input image tensor. (C, H, W)
    saliency_map: torch.Tensor
        The saliency map tensor. (1, H, W)
    label: int
        The true label of the image.
    output: int
        The predicted label of the image.
    conf: float
        The confidence of the prediction.
    image_id: int
        The ID of the image.
    idx_to_classname: dict
        A mapping from index to class name.
    save_path: str or None
        If provided, saves the figure to this path. If None, does not save.

    """

    original_image = image.permute(1, 2, 0).detach().cpu().numpy()
    original_image = original_image * np.array(IMAGENET_STD).reshape(
        1, 1, 3
    ) + np.array(IMAGENET_MEAN).reshape(1, 1, 3)  # Denormalize
    original_image = np.clip(original_image, 0, 1)

    saliency_map = saliency_map.squeeze().detach().cpu().numpy()  # (H, W)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(
        f"Original Image (ID: {image_id})\nTrue: {idx_to_classname[label]}; Pred: {idx_to_classname[output]}\nConf: {conf:.4f}"
    )
    axes[0].axis("off")

    # RISE saliency map
    im1 = axes[1].imshow(saliency_map, cmap="jet")
    axes[1].set_title("RISE Saliency Map\nRed: High Importance; Blue: Low Importance")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(saliency_map, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

import argparse
import sys
from pathlib import Path

import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Relative imports from your package
from .dataset import get_dataloader, transform
from .models import get_model

console = Console()


def get_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chest X-Ray Models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["simplecnn", "vgg16", "resnet152"],
        help="Model architecture used for training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth model checkpoint file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/chest_xray", help="Path to dataset root"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )

    return parser.parse_args()


def print_metrics_table(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    conf_matrix,
    classes: list,
) -> None:
    """Prints a rich table with detailed metrics and confusion matrix."""

    # 1. Main Metrics Table
    metrics_table = Table(title="Detailed Evaluation Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan", justify="right")
    metrics_table.add_column("Value", style="bold green", justify="left")

    metrics_table.add_row("Accuracy", f"{accuracy:.4f}")
    metrics_table.add_row("Precision", f"{precision:.4f}")
    metrics_table.add_row("Recall (Sensitivity)", f"{recall:.4f}")
    metrics_table.add_row("F1 Score", f"{f1:.4f}")

    console.print(metrics_table)

    # 2. Confusion Matrix Table
    cm_table = Table(title="Confusion Matrix", box=box.HEAVY_EDGE)
    cm_table.add_column("Actual \\ Predicted", style="white dim", justify="right")

    for cls in classes:
        cm_table.add_column(f"Pred {cls}", justify="center")

    for i, row_label in enumerate(classes):
        row_values = [str(x) for x in conf_matrix[i]]
        # Highlight diagonal (correct predictions) in green
        formatted_values = []
        for j, val in enumerate(row_values):
            if i == j:
                formatted_values.append(f"[bold green]{val}[/]")
            elif int(val) > 0:
                formatted_values.append(f"[bold red]{val}[/]")  # Errors in red
            else:
                formatted_values.append(val)

        cm_table.add_row(f"Actual {row_label}", *formatted_values)

    console.print(cm_table)


def main() -> None:
    args = get_eval_args()

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)

    # Validation
    if not checkpoint_path.exists():
        console.print(
            f"[bold red]Error:[/][white] Checkpoint not found at: {checkpoint_path}[/]"
        )
        sys.exit(1)
    if not data_dir.exists():
        console.print(
            f"[bold red]Error:[/][white] Data dir not found at: {data_dir}[/]"
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [bold yellow]{device}[/]")

    # 1. Load Data
    with console.status(f"[bold green]Loading {args.split} dataset...[/]"):
        dataloader = get_dataloader(
            data_dir,
            args.split,
            args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=device.type == "cuda",
            transform=transform,
        )

    # Get Class names (assuming ImageFolder structure)
    if isinstance(dataloader.dataset, ImageFolder):
        class_names = dataloader.dataset.classes
    else:
        class_names = ["Class 0", "Class 1"]

    # 2. Load Model Architecture
    # We pass freeze=False here because we are loading trained weights anyway.
    # The architecture structure is what matters.
    model = get_model(args.model, freeze_features=False).to(device)

    # 3. Load Weights
    try:
        console.print(f"Loading weights from: [blue]{checkpoint_path.name}[/]")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        console.print(f"[bold red]Error loading weights:[/] {e}")
        sys.exit(1)

    model.eval()

    # 4. Inference Loop
    all_preds = []
    all_labels = []

    console.print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)

            # Apply sigmoid
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # 6. Display Results
    console.print(
        Panel(
            f"Evaluation Config: Model=[cyan]{args.model}[/] | Split=[cyan]{args.split}[/]",
            style="bold blue",
        )
    )
    print_metrics_table(acc, prec, rec, f1, cm, class_names)


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

from .dataset import get_dataloader, transform
from .loss import get_weighted_bce_loss
from .models import get_model
from .trainer import evaluate, train_one_epoch

# Initialize Rich Console globally for consistent printing
console = Console()

PROJECT_DIR = Path(os.getcwd())
LOG_DIR = PROJECT_DIR / "runs"
MODELS_DIR = PROJECT_DIR / "models"


def _print_config_table(
    run_name: str,
    device: torch.device,
    model: nn.Module,
    freeze: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    data_dir: Path,
    num_workers: int,
    log_dir: Path,
    project_dir: Path,
) -> None:
    """
    Prints a beautiful configuration table using Rich.
    """
    table = Table(title=f"Training Configuration: {run_name}", box=box.ROUNDED)

    # Add columns
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Calculate params nicely with commas (e.g., 1,200,000)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Add Rows
    table.add_row("Device", str(device))
    table.add_row("Model Architecture", model._get_name())
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    table.add_row("Backbone Frozen", "Yes" if freeze else "No")
    table.add_row("Epochs", str(epochs))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Learning Rate", f"{lr:.1e}")  # Scientific notation for small LR
    table.add_row("Data Directory", data_dir.as_posix())
    table.add_row("Number of Workers", str(num_workers))

    # Handle path relativity safely
    try:
        tb_path = (log_dir / run_name).relative_to(project_dir).as_posix()
    except ValueError:
        tb_path = (log_dir / run_name).as_posix()

    table.add_row("TensorBoard Logs", tb_path)

    console.print(table)


def _print_results_table(
    run_name: str,
    train_metrics: tuple[float, float, float],
    val_metrics: tuple[float, float, float],
    test_metrics: tuple[float, float, float],
    total_time: float,
) -> None:
    """
    Prints a comprehensive comparison table for the best model.
    """
    # Unpack metrics (Loss, Acc, F1)
    t_loss, t_acc, t_f1 = train_metrics
    v_loss, v_acc, v_f1 = val_metrics
    te_loss, te_acc, te_f1 = test_metrics

    # Create Table
    table = Table(
        title=f"Final Performance Summary: {run_name}",
        box=box.HEAVY_EDGE,
        caption=f"Total Training Time: {str(timedelta(seconds=int(total_time)))}",
        caption_style="white dim",
    )

    # Add Columns
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Train", style="magenta", justify="center")
    table.add_column("Validation", style="yellow", justify="center")
    table.add_column("Test (Final)", style="bold green", justify="center")

    # Add Rows
    table.add_row("Loss", f"{t_loss:.4f}", f"{v_loss:.4f}", f"{te_loss:.4f}")
    table.add_row("Accuracy", f"{t_acc:.4f}", f"{v_acc:.4f}", f"{te_acc:.4f}")
    table.add_row("F1 Score", f"{t_f1:.4f}", f"{v_f1:.4f}", f"{te_f1:.4f}")

    console.print(table)
    console.print(
        Panel(f"Training Run Complete: [bold white]{run_name}[/]", style="bold green")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Chest X-Ray Classification Models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["simplecnn", "vgg16", "resnet152"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze feature extractor (transfer learning)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for dataloaders"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/chest_xray", help="Path to dataset root"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of DataLoader workers"
    )

    args = parser.parse_args()

    # --- Type Handling & Validation ---
    model_name: str = args.model
    freeze: bool = args.freeze
    epochs: int = args.epochs
    batch_size: int = args.batch_size
    data_dir: Path = Path(args.data_dir)
    lr: float = args.lr
    num_workers: int = args.num_workers

    # Robust validation with user feedback
    if epochs <= 0:
        console.print("[bold red]Error:[/][white] Epochs must be positive.[/]")
        sys.exit(1)
    if batch_size <= 0:
        console.print("[bold red]Error:[/][white] Batch size must be positive.[/]")
        sys.exit(1)
    if lr <= 0:
        console.print("[bold red]Error:[/][white] Learning rate must be positive.[/]")
        sys.exit(1)
    if not data_dir.exists():
        console.print(
            f"[bold red]Error:[/][white] Data directory not found at: {data_dir}[/]"
        )
        sys.exit(1)
    if num_workers < 0:
        console.print(
            "[bold red]Error:[/][white] Number of workers cannot be negative.[/]"
        )
        sys.exit(1)

    LOG_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # Detect Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Setup Data ---
    # Using console.status for a nice spinner while loading data
    with console.status("[bold green]Loading datasets...[/]"):
        train_loader = get_dataloader(
            data_dir,
            "train",
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            transform=transform,
        )
        val_loader = get_dataloader(
            data_dir,
            "val",
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            transform=transform,
        )
        test_loader = get_dataloader(
            data_dir,
            "test",
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            transform=transform,
        )

    # --- 2. Setup Model ---
    model = get_model(model_name, freeze, num_classes=1).to(device)

    # --- 3. Setup Logging ---
    freeze_status = "frozen" if freeze else "unfrozen"
    if model_name == "simplecnn":
        freeze_status = "baseline"

    run_name = f"{model_name}_{freeze_status}_{epochs}ep_{int(time.time())}"
    writer = SummaryWriter(log_dir=LOG_DIR / run_name)

    # --- 4. Optimizer & Loss ---
    # Validate dataset type for type checking
    if isinstance(train_loader.dataset, ImageFolder):
        criterion = get_weighted_bce_loss(train_dataset=train_loader.dataset).to(device)
    else:
        # Fallback or error if dataset isn't ImageFolder (optional safety)
        raise TypeError("Dataset must be ImageFolder to calculate weighted loss")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Print Configuration Table
    _print_config_table(
        run_name,
        device,
        model,
        freeze,
        epochs,
        batch_size,
        lr,
        data_dir,
        num_workers,
        LOG_DIR,
        PROJECT_DIR,
    )

    # --- Training Loop ---
    best_f1 = 0.0
    start_time = time.time()

    try:
        for epoch in range(epochs):
            t_loss, t_acc, t_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            v_loss, v_acc, v_f1 = evaluate(model, val_loader, criterion, device)

            # Print concise epoch stats
            console.print(
                f"[bold]Ep [{epoch + 1}/{epochs}][/] | "
                f"Train: loss={t_loss:.4f} acc={t_acc:.4f} f1={t_f1:.4f} | "
                f"Val: loss={v_loss:.4f} acc={v_acc:.4f} [bold blue]f1={v_f1:.4f}[/]"
            )

            # Log to TensorBoard
            writer.add_scalars("Loss", {"Train": t_loss, "Val": v_loss}, epoch)
            writer.add_scalars("F1", {"Train": t_f1, "Val": v_f1}, epoch)
            writer.add_scalars("Accuracy", {"Train": t_acc, "Val": v_acc}, epoch)

            # Save Best Model
            if v_f1 > best_f1:
                best_f1 = v_f1
                torch.save(model.state_dict(), MODELS_DIR / f"{run_name}_best.pth")
                console.print(
                    f"  [green]-> New best model saved (F1: {best_f1:.4f})[/]"
                )

    except KeyboardInterrupt:
        console.print(
            "\n[bold red]Training interrupted by user. Proceeding to evaluation...[/]"
        )
    end_time = time.time()
    total_time = end_time - start_time

    writer.close()

    # --- Test Evaluation ---
    console.print("\n[bold yellow] Loading best model for Test evaluation...[/]")
    best_model_path = MODELS_DIR / f"{run_name}_best.pth"

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        _print_results_table(
            run_name, train_metrics, val_metrics, test_metrics, total_time
        )

    else:
        console.print(
            "[bold red]Error:[/][white] No best model file found. Training might have failed.[/]"
        )


if __name__ == "__main__":
    main()

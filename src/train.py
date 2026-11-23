"""Training pipeline for ST-GCN, Adaptive ST-GCN, and baseline models."""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.default import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    OUTPUTS_DIR,
    PLOTS_DIR,
)
from src.dataset import get_dataloaders
from src.graph_builder import get_graph_data
from src.models.stgcn import STGCN
from src.models.adaptive_stgcn import AdaptiveSTGCN
from src.models.baselines import MultivariateLSTM
from src.visualize import plot_loss_curves


def get_device() -> torch.device:
    """Select CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cheb_polys: torch.Tensor = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        if cheb_polys is not None:
            preds = model(x, cheb_polys)
        else:
            preds = model(x)

        loss = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(dataloader.dataset)


def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    cheb_polys: torch.Tensor = None,
) -> float:
    """Evaluate loss on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if cheb_polys is not None:
                preds = model(x, cheb_polys)
            else:
                preds = model(x)

            loss = criterion(preds, y)
            total_loss += loss.item() * len(x)

    return total_loss / len(dataloader.dataset)


def train_model(
    model_name: str = "stgcn",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> Dict:
    """End-to-end model training routine with Early Stopping."""
    device = get_device()
    train_ld, val_ld, test_ld, scaler = get_dataloaders(batch_size=batch_size)

    # Graph components for ST-GCN variants
    cheb_polys = None
    if model_name in ["stgcn", "adaptive_stgcn"]:
        _, _, cheb_polys = get_graph_data()
        cheb_polys = cheb_polys.to(device)

    # Instantiate model
    if model_name == "stgcn":
        model = STGCN()
    elif model_name == "adaptive_stgcn":
        model = AdaptiveSTGCN()
    elif model_name == "lstm":
        model = MultivariateLSTM()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    ckpt_dir = OUTPUTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{model_name}_best.pt"

    best_val_loss = float("inf")
    patience_counter = 0
    train_history = []
    val_history = []

    print(f"\n=======================================================")
    print(f"Starting Training: {model_name.upper()} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"=======================================================")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_ld, optimizer, criterion, device, cheb_polys)
        val_loss = evaluate_epoch(model, val_ld, criterion, device, cheb_polys)
        scheduler.step(val_loss)

        train_history.append(train_loss)
        val_history.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.6f}")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "model_name": model_name,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break

    # Save loss plot
    loss_plot_path = PLOTS_DIR / f"loss_curve_{model_name}.png"
    plot_loss_curves(train_history, val_history, model_name=model_name.upper(), save_path=loss_plot_path)
    print(f"\nTraining finished! Best Val Loss: {best_val_loss:.5f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Loss curves saved to: {loss_plot_path}")

    return {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "checkpoint_path": best_ckpt_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stgcn", choices=["stgcn", "adaptive_stgcn", "lstm"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

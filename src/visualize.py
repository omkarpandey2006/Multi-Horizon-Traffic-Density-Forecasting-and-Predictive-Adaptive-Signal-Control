"""Visualization utilities for EDA, graph inspection, and forecast evaluation."""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from configs.default import PLOTS_DIR, NUM_CAMERAS, FEATURES


def set_plot_style():
    """Apply consistent, professional styling for all project plots."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11


def plot_day_timeseries(
    day_tensor: np.ndarray,
    date_str: str,
    save_path: Optional[Path] = None,
):
    """Plot QueueDensity and StopDensity across all 6 cameras for an entire 15-hour day.

    Args:
        day_tensor: shape (T, 6, 2)
        date_str: date label e.g. '2020-10-01'
    """
    set_plot_style()
    T = day_tensor.shape[0]
    time_minutes = np.arange(T) / 60.0 + 7.0  # Approx starting at 07:00 AM

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for cam_idx in range(NUM_CAMERAS):
        ax = axes[cam_idx]
        q_density = day_tensor[:, cam_idx, 0]
        s_density = day_tensor[:, cam_idx, 1]

        ax.plot(time_minutes, q_density, label="QueueDensity", color="#1f77b4", alpha=0.85, linewidth=1.2)
        ax.plot(time_minutes, s_density, label="StopDensity", color="#d62728", alpha=0.85, linewidth=1.2)
        ax.set_title(f"Camera {cam_idx + 1}")
        ax.set_ylim(-0.02, 1.0)
        ax.grid(True, linestyle="--", alpha=0.6)
        if cam_idx in [0, 2, 4]:
            ax.set_ylabel("Density Ratio")
        if cam_idx in [4, 5]:
            ax.set_xlabel("Time of Day (Hour, IST)")
        if cam_idx == 0:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle(f"1-Minute Traffic Density Timeline — {date_str} (6 CCTV Cameras)", fontsize=14, y=1.02)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    title: str = "Inter-Camera Pearson Correlation (QueueDensity)",
    save_path: Optional[Path] = None,
):
    """Plot pairwise camera correlation matrix heatmap."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [f"Cam {i+1}" for i in range(NUM_CAMERAS)]

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title(title, pad=12)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_min: int,
    feature_name: str = "QueueDensity",
    num_samples: int = 300,
    save_path: Optional[Path] = None,
):
    """Plot actual vs predicted curves across cameras for a given forecast horizon.

    Args:
        y_true: (N_samples, 6)
        y_pred: (N_samples, 6)
    """
    set_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    timesteps = np.arange(min(len(y_true), num_samples))

    for cam_idx in range(NUM_CAMERAS):
        ax = axes[cam_idx]
        ax.plot(timesteps, y_true[:len(timesteps), cam_idx], label="Ground Truth", color="black", linewidth=1.3)
        ax.plot(timesteps, y_pred[:len(timesteps), cam_idx], label=f"ST-GCN Pred (+{horizon_min}m)", color="#e65100", linestyle="--", linewidth=1.2)
        ax.set_title(f"Camera {cam_idx + 1}")
        ax.grid(True, linestyle="--", alpha=0.5)
        if cam_idx in [0, 2, 4]:
            ax.set_ylabel(f"{feature_name}")
        if cam_idx in [4, 5]:
            ax.set_xlabel("Test Timestep (Minutes)")
        if cam_idx == 0:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle(f"Forecast Comparison vs Ground Truth — Horizon: +{horizon_min} min ({feature_name})", fontsize=14, y=1.02)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "ST-GCN",
    save_path: Optional[Path] = None,
):
    """Plot training and validation loss progression over epochs."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train MSE Loss", color="#1f77b4", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val MSE Loss", color="#ff7f0e", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (Scaled)")
    ax.set_title(f"Training & Validation Loss — {model_name}")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

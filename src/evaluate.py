"""Comprehensive evaluation framework for ST-GCN, baselines, and hybrid models.

Reports metrics (MAE, RMSE, R^2) on the raw, unscaled scale:
- Overall performance
- Per-horizon (+5 min, +15 min, +30 min)
- Per-feature (QueueDensity, StopDensity)
- Per-camera (Cameras 1 to 6)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs.default import (
    HORIZONS,
    NUM_CAMERAS,
    FEATURES,
    OUTPUTS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
)
from src.dataset import get_dataloaders
from src.graph_builder import get_graph_data
from src.models.stgcn import STGCN
from src.models.adaptive_stgcn import AdaptiveSTGCN
from src.models.baselines import PersistenceModel, MultivariateLSTM
from src.visualize import plot_predictions_vs_actual


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE (%), and R2 between arrays of any matching shape."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2 variance explained
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    # Masked MAPE: evaluate on observations where true density >= 0.02 to avoid near-zero inflation
    mask = y_true >= 0.02
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = float(np.mean(np.abs(y_true - y_pred) / (y_true + 1e-4)) * 100.0)
        
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def get_model_predictions(
    model_name: str,
    test_loader,
    scaler,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate raw unscaled predictions and ground truth for test partition.

    Returns:
        all_preds_raw: (N_samples, Horizons, Nodes, Features)
        all_targets_raw: (N_samples, Horizons, Nodes, Features)
    """
    all_preds = []
    all_targets = []

    if model_name == "persistence":
        model = PersistenceModel(horizons=HORIZONS)
        for x, y in test_loader:
            preds = model.predict(x)
            all_preds.append(preds.numpy())
            all_targets.append(y.numpy())

    elif model_name == "seasonal_naive":
        from src.models.baselines import SeasonalNaiveModel
        from src.preprocessing import load_processed_dataset
        model = SeasonalNaiveModel(horizons=HORIZONS)
        ds = load_processed_dataset()
        num_train = int(len(ds["tensors"]) * 0.70)
        model.fit(ds["tensors"][:num_train])
        for x, y in test_loader:
            preds = model.predict(x)
            all_preds.append(preds.numpy())
            all_targets.append(y.numpy())

    else:
        # Load checkpoint
        ckpt_path = OUTPUTS_DIR / "checkpoints" / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint for {model_name} not found at {ckpt_path}. Train the model first.")

        cheb_polys = None
        if model_name in ["stgcn", "adaptive_stgcn"]:
            _, _, cheb_polys = get_graph_data()
            cheb_polys = cheb_polys.to(device)

        if model_name == "stgcn":
            model = STGCN().to(device)
        elif model_name == "adaptive_stgcn":
            model = AdaptiveSTGCN().to(device)
        elif model_name == "lstm":
            model = MultivariateLSTM().to(device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                if cheb_polys is not None:
                    preds = model(x, cheb_polys)
                else:
                    preds = model(x)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.numpy())

    preds_scaled = np.concatenate(all_preds, axis=0)      # (N, H, Nodes, Features)
    targets_scaled = np.concatenate(all_targets, axis=0)  # (N, H, Nodes, Features)

    # Invert scaling to obtain true physical density metrics
    # Scaler shape: (1, 1, 1, Features)
    preds_raw = scaler.inverse_transform(preds_scaled)
    targets_raw = scaler.inverse_transform(targets_scaled)

    # Physically clip density bounds [0.0, 1.0]
    preds_raw = np.clip(preds_raw, 0.0, 1.0)
    # Enforce physical definition: StopDensity <= QueueDensity
    preds_raw[:, :, :, 1] = np.minimum(preds_raw[:, :, :, 1], preds_raw[:, :, :, 0])

    return preds_raw, targets_raw


def evaluate_all(models: List[str] = ["persistence", "stgcn"]) -> pd.DataFrame:
    """Evaluate and compare specified models on held-out test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, scaler = get_dataloaders(batch_size=32)

    rows = []

    for m_name in models:
        try:
            preds, targets = get_model_predictions(m_name, test_loader, scaler, device)
        except FileNotFoundError as e:
            print(f"Skipping {m_name}: {e}")
            continue

        # 1. Overall
        overall_m = compute_metrics(targets, preds)
        rows.append({
            "Model": m_name.upper(),
            "Scope": "Overall",
            "Horizon": "All",
            "Feature": "All",
            "MAE": round(overall_m["MAE"], 4),
            "RMSE": round(overall_m["RMSE"], 4),
            "MAPE": round(overall_m["MAPE"], 2),
            "R2": round(overall_m["R2"], 4),
        })

        # 2. Per Horizon
        for h_idx, h_min in enumerate(HORIZONS):
            h_metrics = compute_metrics(targets[:, h_idx], preds[:, h_idx])
            rows.append({
                "Model": m_name.upper(),
                "Scope": "Horizon",
                "Horizon": f"+{h_min}m",
                "Feature": "All",
                "MAE": round(h_metrics["MAE"], 4),
                "RMSE": round(h_metrics["RMSE"], 4),
                "MAPE": round(h_metrics["MAPE"], 2),
                "R2": round(h_metrics["R2"], 4),
            })

        # 3. Per Feature
        for f_idx, f_name in enumerate(FEATURES):
            f_metrics = compute_metrics(targets[:, :, :, f_idx], preds[:, :, :, f_idx])
            rows.append({
                "Model": m_name.upper(),
                "Scope": "Feature",
                "Horizon": "All",
                "Feature": f_name,
                "MAE": round(f_metrics["MAE"], 4),
                "RMSE": round(f_metrics["RMSE"], 4),
                "MAPE": round(f_metrics["MAPE"], 2),
                "R2": round(f_metrics["R2"], 4),
            })

            # Save sample forecast vs actual curves for the primary ST-GCN model
            if m_name == "stgcn":
                for h_idx, h_min in enumerate(HORIZONS):
                    fig_path = PLOTS_DIR / f"forecast_comparison_{f_name}_h{h_min}m.png"
                    plot_predictions_vs_actual(
                        y_true=targets[:, h_idx, :, f_idx],
                        y_pred=preds[:, h_idx, :, f_idx],
                        horizon_min=h_min,
                        feature_name=f_name,
                        num_samples=250,
                        save_path=fig_path,
                    )

    df_results = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / "model_comparison_metrics.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n=========================================================================")
    print(f"Evaluation Complete! Saved metrics to: {out_csv}")
    print(f"=========================================================================")
    print(df_results[df_results["Scope"] == "Overall"][["Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    return df_results


if __name__ == "__main__":
    evaluate_all(models=["persistence"])

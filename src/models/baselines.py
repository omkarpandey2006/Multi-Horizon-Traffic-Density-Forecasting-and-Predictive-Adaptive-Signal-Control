"""Baseline models for multi-horizon traffic density forecasting.

Includes:
1. PersistenceModel (Last observation forward)
2. SeasonalNaiveModel (Historical average by time-of-day)
3. MultivariateLSTM (Multi-camera deep temporal baseline without graph convolutions)
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn

from configs.default import NUM_CAMERAS, NUM_FEATURES, HORIZONS


class PersistenceModel:
    """Predicts future density as the most recently observed value in the input sequence.

    Input shape: (Batch, T, Nodes, Features)
    Output shape: (Batch, len(Horizons), Nodes, Features)
    """

    def __init__(self, horizons: List[int] = HORIZONS):
        self.num_horizons = len(horizons)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, N, F)
        # Take the last time step x[:, -1, :, :] -> shape (B, N, F)
        last_obs = x[:, -1, :, :]
        # Replicate across all prediction horizons -> (B, num_horizons, N, F)
        preds = last_obs.unsqueeze(1).repeat(1, self.num_horizons, 1, 1)
        return preds


class SeasonalNaiveModel:
    """Predicts future density using the average historical profile for that specific minute-of-day.

    Fit on training partition: computes mean (T_day, N, F).
    """

    def __init__(self, horizons: List[int] = HORIZONS):
        self.horizons = horizons
        self.historical_mean = None
        self.overall_mean = None

    def fit(self, train_day_tensors: List[np.ndarray]):
        min_len = min(t.shape[0] for t in train_day_tensors)
        clipped = np.stack([t[:min_len] for t in train_day_tensors], axis=0)
        self.historical_mean = np.mean(clipped, axis=0)  # (min_len, N, F)
        self.overall_mean = np.mean(self.historical_mean, axis=0, keepdims=True) # (1, N, F)

    def predict(self, x: torch.Tensor, current_minute_indices: Optional[List[int]] = None) -> torch.Tensor:
        B, T, N, F = x.shape
        if current_minute_indices is not None and self.historical_mean is not None:
            preds = np.zeros((B, len(self.horizons), N, F), dtype=np.float32)
            for b_idx, start_m in enumerate(current_minute_indices):
                for h_idx, h in enumerate(self.horizons):
                    target_m = (start_m + h) % len(self.historical_mean)
                    preds[b_idx, h_idx] = self.historical_mean[target_m]
            return torch.from_numpy(preds).float()
        elif self.overall_mean is not None:
            # Replicate overall historical mean across batch and horizons
            mean_tensor = torch.from_numpy(self.overall_mean).float() # (1, N, F)
            preds = mean_tensor.unsqueeze(0).repeat(B, len(self.horizons), 1, 1) # (B, H, N, F)
            return preds
        else:
            # Fallback to mean of input sequence x
            mean_x = torch.mean(x, dim=1, keepdim=True) # (B, 1, N, F)
            preds = mean_x.repeat(1, len(self.horizons), 1, 1)
            return preds


class MultivariateLSTM(nn.Module):
    """Multivariate LSTM forecasting model.

    Flattens spatial camera nodes into features: (Batch, T, N * F).
    Processes with multi-layer LSTM and projects to multi-horizon predictions:
    (Batch, len(Horizons), N, F).
    """

    def __init__(
        self,
        num_nodes: int = NUM_CAMERAS,
        in_features: int = NUM_FEATURES,
        num_horizons: int = len(HORIZONS),
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.num_horizons = num_horizons
        input_dim = num_nodes * in_features

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_horizons * num_nodes * in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, T, Nodes, Features)
        Returns:
            out: (Batch, num_horizons, Nodes, Features)
        """
        B, T, N, F = x.shape
        # Flatten nodes and features
        x_flat = x.view(B, T, N * F)
        lstm_out, _ = self.lstm(x_flat)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim)
        out_flat = self.fc(last_hidden)   # (B, H * N * F)
        out = out_flat.view(B, self.num_horizons, N, F)
        return out

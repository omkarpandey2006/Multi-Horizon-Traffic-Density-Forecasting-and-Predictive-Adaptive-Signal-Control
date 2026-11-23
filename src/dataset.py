"""PyTorch Dataset and DataLoader with sliding windows for ST-GCN forecasting.

Key Rules:
1. Chronological splitting by day (Train: 70%, Val: 15%, Test: 15%).
2. Standard scaling fit strictly on the training partition.
3. Sliding windows are extracted strictly WITHIN each day; no windows cross overnight gaps.
4. Input window: 60 minutes -> Targets at multi-horizons: t+5, t+15, t+30 minutes.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.default import (
    INPUT_WINDOW,
    HORIZONS,
    MAX_HORIZON,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    NUM_CAMERAS,
    FEATURES,
    DATA_PROCESSED_DIR,
)
from src.preprocessing import load_processed_dataset


class StandardScaler:
    """Z-score normalizer fit per-feature (or per-node-feature) on training data."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Fit scaler on 3D array (Total_T, NUM_CAMERAS, NUM_FEATURES)."""
        # Compute mean and std over time and cameras, keeping feature dimension (1, 1, F)
        self.mean = np.mean(data, axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = np.std(data, axis=(0, 1), keepdims=True).astype(np.float32)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-5, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean

    def to_torch(self, device: torch.device = torch.device("cpu")):
        return torch.from_numpy(self.mean).to(device), torch.from_numpy(self.std).to(device)


class TrafficDataset(Dataset):
    """PyTorch Dataset yielding (X, Y) window pairs:

    X shape: (INPUT_WINDOW, NUM_CAMERAS, NUM_FEATURES)  e.g. (60, 6, 2)
    Y shape: (len(HORIZONS), NUM_CAMERAS, NUM_FEATURES) e.g. (3, 6, 2)
    """

    def __init__(self, x_samples: np.ndarray, y_samples: np.ndarray):
        self.x = torch.from_numpy(x_samples).float()
        self.y = torch.from_numpy(y_samples).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def generate_daywise_sliding_windows(
    day_tensors: List[np.ndarray],
    input_window: int = INPUT_WINDOW,
    horizons: List[int] = HORIZONS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sliding windows strictly within each day tensor."""
    max_h = max(horizons)
    all_x = []
    all_y = []

    for tensor in day_tensors:
        T = tensor.shape[0]
        # Number of possible valid windows in this single day
        num_windows = T - input_window - max_h + 1
        if num_windows <= 0:
            continue

        for i in range(num_windows):
            # Input window: [i, i + input_window)
            x_window = tensor[i : i + input_window]  # (60, 6, 2)

            # Targets at specified future offsets: t + h
            # Where t is the last timestep of the input window (i + input_window - 1)
            target_steps = [i + input_window - 1 + h for h in horizons]
            y_window = tensor[target_steps]  # (3, 6, 2)

            all_x.append(x_window)
            all_y.append(y_window)

    x_arr = np.stack(all_x, axis=0) if all_x else np.empty((0, input_window, NUM_CAMERAS, 2))
    y_arr = np.stack(all_y, axis=0) if all_y else np.empty((0, len(horizons), NUM_CAMERAS, 2))
    return x_arr, y_arr


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    input_window: int = INPUT_WINDOW,
    horizons: List[int] = HORIZONS,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """Prepare Train, Validation, and Test DataLoaders with strict chronological day partitioning.

    Split:
        Train: first 70% days (~28 days)
        Val: next 15% days (~6 days)
        Test: final 15% days (~6 days)
    """
    dataset = load_processed_dataset()
    tensors = dataset["tensors"]
    num_days = len(tensors)

    num_train = int(num_days * TRAIN_RATIO)
    num_val = int(num_days * VAL_RATIO)

    train_days = tensors[:num_train]
    val_days = tensors[num_train : num_train + num_val]
    test_days = tensors[num_train + num_val :]

    print(f"Data partitioning: {len(train_days)} train days, {len(val_days)} val days, {len(test_days)} test days.")

    # Fit scaler on training days only
    train_concat = np.concatenate(train_days, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_concat)

    # Scale each day
    scaled_train_days = [scaler.transform(t) for t in train_days]
    scaled_val_days = [scaler.transform(t) for t in val_days]
    scaled_test_days = [scaler.transform(t) for t in test_days]

    # Generate sliding windows
    x_train, y_train = generate_daywise_sliding_windows(scaled_train_days, input_window, horizons)
    x_val, y_val = generate_daywise_sliding_windows(scaled_val_days, input_window, horizons)
    x_test, y_test = generate_daywise_sliding_windows(scaled_test_days, input_window, horizons)

    print(f"Window samples extracted: Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    train_loader = DataLoader(
        TrafficDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True,
    )
    val_loader = DataLoader(
        TrafficDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TrafficDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    train_ld, val_ld, test_ld, sc = get_dataloaders()
    for batch_x, batch_y in train_ld:
        print(f"Sample Batch X shape: {batch_x.shape} (Batch, T, Nodes, Features)")
        print(f"Sample Batch Y shape: {batch_y.shape} (Batch, Horizons, Nodes, Features)")
        break

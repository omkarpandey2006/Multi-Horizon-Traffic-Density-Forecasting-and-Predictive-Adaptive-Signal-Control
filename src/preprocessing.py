"""Data engineering and preprocessing pipeline for Delhi Traffic Density dataset.

Steps:
1. Audit and load raw 1-second records for 6 cameras from CSV files.
2. Resample 1-second observations to 1-minute intervals.
3. Compute observation counts and handle intra-day missing intervals.
4. Construct synchronized day-wise tensors X in R^{T x N x F}.
"""

import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from configs.default import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    NUM_CAMERAS,
    FEATURES,
    RESAMPLE_FREQ,
)


def get_raw_csv_files(data_dir: Path = DATA_RAW_DIR) -> List[Path]:
    """Retrieve and chronologically sort all daily CSV files."""
    files = list(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

    # Determine start date for sorting
    def get_file_start_epoch(filepath: Path) -> int:
        df_head = pd.read_csv(filepath, nrows=2)
        return int(df_head["EpochTime"].iloc[0])

    files_sorted = sorted(files, key=get_file_start_epoch)
    return files_sorted


def audit_single_file(filepath: Path) -> Dict:
    """Inspect a raw CSV file to check schema, row counts, ranges, and validity."""
    df = pd.read_csv(filepath)
    epoch_min = df["EpochTime"].min()
    epoch_max = df["EpochTime"].max()
    start_dt = datetime.fromtimestamp(epoch_min)
    end_dt = datetime.fromtimestamp(epoch_max)

    # Check density relationships: StopDensity <= QueueDensity
    inversions = {}
    for c in range(1, NUM_CAMERAS + 1):
        q_col = f"QueueDensity{c}"
        s_col = f"StopDensity{c}"
        if q_col in df.columns and s_col in df.columns:
            # allow tiny numerical tolerance 1e-4
            inv_count = int((df[s_col] > df[q_col] + 1e-4).sum())
            inversions[f"cam_{c}"] = inv_count

    return {
        "filename": filepath.name,
        "total_records": len(df),
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_hours": (epoch_max - epoch_min) / 3600.0,
        "stop_greater_than_queue_count": inversions,
        "null_count": int(df.isnull().sum().sum()),
    }


def audit_dataset(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Run an audit over all raw daily files and return a summary DataFrame."""
    files = get_raw_csv_files(data_dir)
    print(f"Auditing {len(files)} daily CSV files...")
    results = [audit_single_file(f) for f in files]
    df_audit = pd.DataFrame(results)
    return df_audit


def resample_daily_csv(
    filepath: Path,
    max_gap_to_interpolate: int = 5,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, str]:
    """Resample a single daily CSV from 1-second to 1-minute intervals.

    Returns:
        tensor: shape (T, NUM_CAMERAS, NUM_FEATURES) containing resampled density values
        obs_count: shape (T, NUM_CAMERAS) with count of observations per minute
        time_index: pandas DatetimeIndex of the 1-minute intervals
        date_str: string representation of the date (YYYY-MM-DD)
    """
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["EpochTime"], unit="s")
    df = df.set_index("datetime").sort_index()

    date_str = df.index[0].strftime("%Y-%m-%d")

    # Regular 1-minute frequency grid from 07:00:00 to 22:00:00 or observed start to end
    start_time = df.index.min().floor("min")
    end_time = df.index.max().ceil("min")
    full_minute_index = pd.date_range(start_time, end_time, freq=RESAMPLE_FREQ)

    # 1. Compute 1-minute mean for density metrics
    df_mean = df.resample(RESAMPLE_FREQ).mean()
    # 2. Compute observation count per minute (expect up to 60)
    df_count = df.resample(RESAMPLE_FREQ)["EpochTime"].count()

    # Reindex to full uniform minute grid
    df_mean = df_mean.reindex(full_minute_index)
    df_count = df_count.reindex(full_minute_index).fillna(0)

    # Handle intra-day missing data policy:
    # Interpolate short gaps (<= max_gap_to_interpolate minutes) linearly
    # Any larger missing intervals remain NaN or are filled forward with caution
    density_cols = []
    for c in range(1, NUM_CAMERAS + 1):
        density_cols.extend([f"QueueDensity{c}", f"StopDensity{c}"])

    df_mean[density_cols] = df_mean[density_cols].interpolate(
        method="linear", limit=max_gap_to_interpolate, limit_direction="both"
    )
    # Remaining NaNs if any at edges
    df_mean[density_cols] = df_mean[density_cols].bfill().ffill()

    # Ensure valid density bounds [0.0, 1.0] and StopDensity <= QueueDensity
    for c in range(1, NUM_CAMERAS + 1):
        q_col = f"QueueDensity{c}"
        s_col = f"StopDensity{c}"
        df_mean[q_col] = df_mean[q_col].clip(lower=0.0, upper=1.0)
        df_mean[s_col] = df_mean[s_col].clip(lower=0.0, upper=1.0)
        # Enforce physical definition: StopDensity cannot exceed QueueDensity
        df_mean[s_col] = np.minimum(df_mean[s_col], df_mean[q_col])

    # Construct array of shape (T, NUM_CAMERAS, 2)
    T = len(full_minute_index)
    tensor = np.zeros((T, NUM_CAMERAS, len(FEATURES)), dtype=np.float32)
    obs_counts = np.zeros((T, NUM_CAMERAS), dtype=np.int32)

    for i in range(NUM_CAMERAS):
        cam_id = i + 1
        tensor[:, i, 0] = df_mean[f"QueueDensity{cam_id}"].values
        tensor[:, i, 1] = df_mean[f"StopDensity{cam_id}"].values
        obs_counts[:, i] = df_count.values

    return tensor, obs_counts, full_minute_index, date_str


def process_and_save_dataset(
    data_dir: Path = DATA_RAW_DIR,
    out_dir: Path = DATA_PROCESSED_DIR,
) -> Dict[str, Path]:
    """Process all daily CSV files into 1-minute synchronized tensors and save as .npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = get_raw_csv_files(data_dir)
    print(f"Processing {len(files)} daily files into 1-minute tensors...")

    day_tensors = []
    day_counts = []
    day_dates = []
    day_lengths = []

    for idx, f in enumerate(files):
        tensor, counts, time_index, date_str = resample_daily_csv(f)
        day_tensors.append(tensor)
        day_counts.append(counts)
        day_dates.append(date_str)
        day_lengths.append(len(tensor))
        if (idx + 1) % 10 == 0 or idx == len(files) - 1:
            print(f"  Processed {idx + 1}/{len(files)}: {f.name} -> {date_str} (T={len(tensor)} min)")

    # Save dictionary of daily tensors
    dataset_file = out_dir / "delhi_traffic_resampled_1min.npz"
    save_dict = {
        "num_days": len(day_tensors),
        "dates": np.array(day_dates),
        "lengths": np.array(day_lengths),
    }
    for idx, (t, c) in enumerate(zip(day_tensors, day_counts)):
        save_dict[f"tensor_{idx}"] = t
        save_dict[f"counts_{idx}"] = c

    np.savez_compressed(dataset_file, **save_dict)
    print(f"\nSuccessfully saved resampled dataset to: {dataset_file}")

    return {"dataset_file": dataset_file}


def load_processed_dataset(filepath: Optional[Path] = None) -> Dict:
    """Load the preprocessed day-wise tensors from .npz file."""
    if filepath is None:
        filepath = DATA_PROCESSED_DIR / "delhi_traffic_resampled_1min.npz"
    if not filepath.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {filepath}. Run process_and_save_dataset first.")

    data = np.load(filepath, allow_pickle=True)
    num_days = int(data["num_days"])
    dates = list(data["dates"])
    lengths = list(data["lengths"])

    day_tensors = [data[f"tensor_{i}"] for i in range(num_days)]
    day_counts = [data[f"counts_{i}"] for i in range(num_days)]

    return {
        "num_days": num_days,
        "dates": dates,
        "lengths": lengths,
        "tensors": day_tensors,
        "counts": day_counts,
    }


if __name__ == "__main__":
    audit_summary = audit_dataset()
    print("\nDataset Audit Summary (first 5 days):")
    print(audit_summary[["filename", "start_time", "end_time", "total_records", "duration_hours"]].head())
    process_and_save_dataset()

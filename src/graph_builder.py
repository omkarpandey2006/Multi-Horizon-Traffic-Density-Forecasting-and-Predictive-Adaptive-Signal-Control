"""Graph construction and Chebyshev polynomial Laplacian calculations for ST-GCN.

Per Yu et al. (IJCAI 2018):
1. Compute pairwise Pearson correlation matrix from training timestamps.
2. Form thresholded adjacency matrix A with self-connections.
3. Compute normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}.
4. Scale Laplacian: L_tilde = (2 / lambda_max) * L - I.
5. Compute Chebyshev polynomials T_0(L_tilde), ..., T_{K-1}(L_tilde).
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp
import torch

from configs.default import (
    NUM_CAMERAS,
    CORR_THRESHOLD,
    CHEB_K,
    TRAIN_RATIO,
    DATA_PROCESSED_DIR,
)
from src.preprocessing import load_processed_dataset


def compute_training_correlation_matrix(
    tensors: list,
    train_ratio: float = TRAIN_RATIO,
    feature_idx: int = 0,
) -> np.ndarray:
    """Compute pairwise Pearson correlation matrix across cameras using training days only.

    Args:
        tensors: list of daily tensors, each of shape (T_d, 6, 2)
        train_ratio: proportion of days for training partition (first 70%)
        feature_idx: feature used for spatial relationship (0 = QueueDensity)
    """
    num_train_days = int(len(tensors) * train_ratio)
    train_tensors = tensors[:num_train_days]

    # Concatenate training timesteps along time dimension: shape (Total_train_T, 6)
    train_series = np.concatenate([t[:, :, feature_idx] for t in train_tensors], axis=0)

    # Compute 6x6 Pearson correlation
    corr_matrix = np.corrcoef(train_series, rowvar=False)
    # Replace NaNs if any with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix


def build_adjacency_matrix(
    corr_matrix: np.ndarray,
    threshold: float = CORR_THRESHOLD,
) -> np.ndarray:
    """Construct binary or weighted adjacency matrix with self-loops.

    Args:
        corr_matrix: (6, 6) pairwise correlation matrix
        threshold: minimum correlation required to set an edge between camera nodes
    """
    N = corr_matrix.shape[0]
    # Keep edges exceeding threshold
    adj = np.where(corr_matrix >= threshold, corr_matrix, 0.0)

    # Enforce symmetry: A = max(A, A^T)
    adj = np.maximum(adj, adj.T)

    # Add self-loops (diagonal = 1.0)
    np.fill_diagonal(adj, 1.0)
    return adj


def compute_scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute scaled graph Laplacian L_tilde = (2 / lambda_max) * L - I.

    L = I - D^{-1/2} A D^{-1/2}
    """
    N = adj.shape[0]
    # Degree matrix
    degree = np.sum(adj, axis=1)
    # D^{-1/2}
    d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    pos_mask = degree > 0
    d_inv_sqrt[pos_mask] = 1.0 / np.sqrt(degree[pos_mask])
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    L = np.eye(N) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # Find largest eigenvalue lambda_max
    eigvals = np.linalg.eigvalsh(L)
    lambda_max = eigvals.max()

    if lambda_max < 1e-4:
        lambda_max = 2.0

    # Rescale to [-1, 1] for Chebyshev polynomial stability
    L_tilde = (2.0 / lambda_max) * L - np.eye(N)
    return L_tilde.astype(np.float32)


def compute_chebyshev_polynomials(
    L_tilde: np.ndarray,
    K: int = CHEB_K,
) -> torch.Tensor:
    """Compute Chebyshev polynomials up to order K: T_0, T_1, ..., T_{K-1}.

    Recurrence:
        T_0(x) = I
        T_1(x) = x
        T_k(x) = 2 * x * T_{k-1}(x) - T_{k-2}(x)

    Returns:
        torch.Tensor of shape (K, N, N)
    """
    N = L_tilde.shape[0]
    cheb_polys = []

    # T_0 = I
    T_0 = np.eye(N, dtype=np.float32)
    cheb_polys.append(T_0)

    if K > 1:
        # T_1 = L_tilde
        T_1 = L_tilde.copy()
        cheb_polys.append(T_1)

    for k in range(2, K):
        T_k = 2.0 * L_tilde @ cheb_polys[-1] - cheb_polys[-2]
        cheb_polys.append(T_k)

    # Stack along first dimension -> shape (K, N, N)
    cheb_tensor = torch.from_numpy(np.stack(cheb_polys, axis=0)).float()
    return cheb_tensor


def get_graph_data(
    threshold: float = CORR_THRESHOLD,
    K: int = CHEB_K,
    save_dir: Optional[Path] = DATA_PROCESSED_DIR,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """End-to-end pipeline to compute correlation, adjacency, and Chebyshev polynomials.

    Returns:
        corr_matrix: (6, 6)
        adj_matrix: (6, 6)
        cheb_tensor: (K, 6, 6) PyTorch tensor
    """
    dataset = load_processed_dataset()
    corr_matrix = compute_training_correlation_matrix(dataset["tensors"])
    adj_matrix = build_adjacency_matrix(corr_matrix, threshold=threshold)
    L_tilde = compute_scaled_laplacian(adj_matrix)
    cheb_tensor = compute_chebyshev_polynomials(L_tilde, K=K)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "correlation_matrix.npy", corr_matrix)
        np.save(save_dir / "adjacency_matrix.npy", adj_matrix)
        torch.save(cheb_tensor, save_dir / "chebyshev_polynomials.pt")

    return corr_matrix, adj_matrix, cheb_tensor


if __name__ == "__main__":
    corr, adj, cheb = get_graph_data()
    print("Computed Inter-Camera Pearson Correlation Matrix:")
    print(np.round(corr, 3))
    print(f"\nAdjacency Matrix (threshold={CORR_THRESHOLD}):")
    print(np.round(adj, 3))
    print(f"\nChebyshev Polynomials Tensor Shape: {cheb.shape}")

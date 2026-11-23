"""Spatio-Temporal Graph Convolutional Networks (ST-GCN).

Faithfully implements the architecture proposed by Yu, Yin, and Zhu (IJCAI 2018):
"Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"

Key components:
1. Temporal Gated Conv (1D Conv + Gated Linear Unit GLU: P * sigmoid(Q))
2. Chebyshev Spatial Graph Convolution (Order K=3 polynomial filter)
3. Spatio-Temporal Convolutional Block (Sandwich: Temporal -> Graph -> Temporal)
4. Multi-Horizon Forecast Head
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.default import NUM_CAMERAS, NUM_FEATURES, HORIZONS, CHEB_K, DROPOUT


class TemporalGatedConv(nn.Module):
    """Temporal Gated Convolution using 1D convolution and Gated Linear Unit (GLU).

    Takes input tensor: (Batch, C_in, Nodes, Time)
    Applies 1D conv along Time dimension, mapping C_in -> 2 * C_out.
    Splits into P and Q, outputs P * sigmoid(Q) with length (Time - Kt + 1).
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dropout: float = DROPOUT):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size

        # 2D Conv where height is 1 (per node) and width is time kernel
        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=2 * c_out,
            kernel_size=(1, kernel_size),
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (Batch, C_in, Nodes, Time)
        Returns:
            out: shape (Batch, C_out, Nodes, Time - kernel_size + 1)
        """
        x_conv = self.conv(x)  # (B, 2 * C_out, N, T - Kt + 1)
        p, q = torch.split(x_conv, self.c_out, dim=1)
        out = p * torch.sigmoid(q)
        out = self.dropout(out)
        return out


class ChebyshevGraphConv(nn.Module):
    """Spectral Graph Convolution using Chebyshev polynomial approximation.

    Formula:
        y = sum_{k=0}^{K-1} T_k(L_tilde) * X * Theta_k
    """

    def __init__(self, c_in: int, c_out: int, K: int = CHEB_K):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.K = K

        # Learnable filter parameters Theta: shape (K, c_in, c_out)
        self.weight = nn.Parameter(torch.FloatTensor(K, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot uniform initialization
        stdv = 1.0 / math.sqrt(self.c_in * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, cheb_polys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (Batch, C_in, Nodes, Time)
            cheb_polys: shape (K, Nodes, Nodes) precomputed Chebyshev polynomials
        Returns:
            out: shape (Batch, C_out, Nodes, Time)
        """
        B, C_in, N, T = x.shape
        # Permute x to (K, B, T, N, C_in) for batch graph multiplication
        # Let's compute term-by-term cleanly:
        # x_perm: (B, T, N, C_in)
        x_perm = x.permute(0, 3, 2, 1)

        out = torch.zeros(B, T, N, self.c_out, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            # T_k: (N, N)
            T_k = cheb_polys[k]  # (N, N)
            # Spatial aggregation using batched BLAS matmul: (N, N) @ (B, T, N, C_in) -> (B, T, N, C_in)
            x_k = torch.matmul(T_k, x_perm)
            # Channel projection: (B, T, N, C_in) @ Theta_k (C_in, C_out) -> (B, T, N, C_out)
            out = out + torch.matmul(x_k, self.weight[k])

        out = out + self.bias
        out = F.relu(out)
        # Permute back to standard format: (B, C_out, N, T)
        return out.permute(0, 3, 2, 1)


class STConvBlock(nn.Module):
    """Spatio-Temporal Convolutional Block (ST-Conv Block).

    Sandwich architecture:
        Temporal Gated Conv (Kt)
        -> Chebyshev Graph Conv
        -> Temporal Gated Conv (Kt)
        + Residual connection & LayerNorm
    """

    def __init__(
        self,
        c_in: int,
        c_mid: int,
        c_out: int,
        num_nodes: int = NUM_CAMERAS,
        kt: int = 3,
        K: int = CHEB_K,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.kt = kt
        self.total_time_drop = 2 * (kt - 1)  # Two temporal conv layers

        # 1. First Temporal Gated Conv
        self.temp_conv1 = TemporalGatedConv(c_in, c_mid, kernel_size=kt, dropout=dropout)

        # 2. Chebyshev Graph Conv
        self.graph_conv = ChebyshevGraphConv(c_mid, c_mid, K=K)

        # 3. Second Temporal Gated Conv
        self.temp_conv2 = TemporalGatedConv(c_mid, c_out, kernel_size=kt, dropout=dropout)

        # 4. Residual projection if channels differ
        if c_in != c_out:
            self.res_conv = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.res_conv = None

        self.layer_norm = nn.LayerNorm([c_out, num_nodes])

    def forward(self, x: torch.Tensor, cheb_polys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, C_in, Nodes, Time)
            cheb_polys: (K, Nodes, Nodes)
        Returns:
            out: (Batch, C_out, Nodes, Time - 2 * (Kt - 1))
        """
        # Residual branch: trim time to match output length
        if self.res_conv is not None:
            res = self.res_conv(x[:, :, :, self.total_time_drop :])
        else:
            res = x[:, :, :, self.total_time_drop :]

        # Main branch
        h1 = self.temp_conv1(x)                   # Time drops by (Kt - 1)
        h2 = self.graph_conv(h1, cheb_polys)      # Time unchanged
        h3 = self.temp_conv2(h2)                  # Time drops by (Kt - 1)

        # Add residual
        out = h3 + res

        # LayerNorm across (Channels, Nodes) for each batch and timestep
        # out shape: (B, C_out, N, T') -> permute to (B, T', C_out, N)
        out = out.permute(0, 3, 1, 2)
        out = self.layer_norm(out)
        out = out.permute(0, 2, 3, 1)  # (B, C_out, N, T')

        return out


class STGCN(nn.Module):
    """Paper-faithful Spatio-Temporal Graph Convolutional Network (ST-GCN).

    Structure:
        Input: (Batch, Time=60, Nodes=6, Features=2)
        -> ST-Conv Block 1 (Channels: 2 -> 32 -> 64, Time: 60 -> 56)
        -> ST-Conv Block 2 (Channels: 64 -> 32 -> 128, Time: 56 -> 52)
        -> Output Layer:
             Temporal Conv (Kernel = remaining Time 52) -> collapses time to 1
             Linear Multi-Horizon Head -> (Batch, len(Horizons), Nodes=6, Features=2)
    """

    def __init__(
        self,
        num_nodes: int = NUM_CAMERAS,
        in_features: int = NUM_FEATURES,
        horizons: List[int] = HORIZONS,
        input_window: int = 60,
        kt: int = 3,
        cheb_k: int = CHEB_K,
        channels: List[List[int]] = None,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizons = horizons
        self.num_horizons = len(horizons)

        if channels is None:
            channels = [[in_features, 32, 64], [64, 32, 128]]

        # Block 1
        self.block1 = STConvBlock(
            c_in=channels[0][0],
            c_mid=channels[0][1],
            c_out=channels[0][2],
            num_nodes=num_nodes,
            kt=kt,
            K=cheb_k,
            dropout=dropout,
        )

        # Block 2
        self.block2 = STConvBlock(
            c_in=channels[1][0],
            c_mid=channels[1][1],
            c_out=channels[1][2],
            num_nodes=num_nodes,
            kt=kt,
            K=cheb_k,
            dropout=dropout,
        )

        # Time reduction after 2 blocks
        # Each block drops 2 * (kt - 1) timesteps = 4 timesteps (for kt=3)
        # Total time drop = 8 timesteps -> remaining T = input_window - 8 (e.g. 60 - 8 = 52)
        remaining_t = input_window - 4 * (kt - 1)
        self.remaining_t = remaining_t

        # Output projection: collapse temporal dimension
        self.output_temp_conv = nn.Conv2d(
            in_channels=channels[1][2],
            out_channels=128,
            kernel_size=(1, remaining_t),
        )
        self.fc_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.num_horizons * in_features),
        )

    def forward(self, x: torch.Tensor, cheb_polys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Time, Nodes, Features)  e.g. (32, 60, 6, 2)
            cheb_polys: (K, Nodes, Nodes) precomputed Chebyshev polynomials
        Returns:
            out: (Batch, num_horizons, Nodes, Features) e.g. (32, 3, 6, 2)
        """
        B, T, N, F = x.shape

        # Re-arrange to (Batch, Features, Nodes, Time) for 2D Conv
        x_in = x.permute(0, 3, 2, 1)

        # Pass through ST-Conv Blocks
        out1 = self.block1(x_in, cheb_polys)
        out2 = self.block2(out1, cheb_polys)

        # Output temporal collapse: (B, C, N, T_rem) -> (B, 128, N, 1)
        out_collapsed = self.output_temp_conv(out2)

        # Permute to (B, N, 128)
        out_collapsed = out_collapsed.squeeze(-1).permute(0, 2, 1)

        # Multi-horizon projection: (B, N, num_horizons * F)
        preds_flat = self.fc_head(out_collapsed)

        # Reshape to (Batch, num_horizons, Nodes, Features)
        preds = preds_flat.view(B, N, self.num_horizons, F).permute(0, 2, 1, 3)
        return preds

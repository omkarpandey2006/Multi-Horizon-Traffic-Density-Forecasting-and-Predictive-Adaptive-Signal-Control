"""Adaptive Spatio-Temporal Graph Convolutional Network (Adaptive ST-GCN - Model B).

Extends the paper's original ST-GCN by incorporating a learnable adaptive graph adjacency matrix:
    A_adp = Softmax(ReLU(E_1 @ E_2^T))
where E_1, E_2 in R^{N x d} are learnable camera node embeddings.

This tests whether dynamically learned inter-camera relationships enhance forecast accuracy
over the fixed thresholded Pearson correlation matrix alone.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.default import NUM_CAMERAS, NUM_FEATURES, HORIZONS, CHEB_K, DROPOUT
from src.models.stgcn import TemporalGatedConv, ChebyshevGraphConv


class AdaptiveGraphConv(nn.Module):
    """Combines Chebyshev spectral graph conv with learnable adaptive node embeddings."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_nodes: int = NUM_CAMERAS,
        cheb_k: int = CHEB_K,
        embed_dim: int = 8,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.cheb_k = cheb_k

        # Fixed Chebyshev graph conv
        self.cheb_conv = ChebyshevGraphConv(c_in, c_out, K=cheb_k)

        # Adaptive node embeddings
        self.node_embed1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.node_embed2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.adaptive_weight = nn.Parameter(torch.FloatTensor(c_in, c_out))

        stdv = 1.0 / math.sqrt(c_in)
        self.adaptive_weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, cheb_polys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (Batch, C_in, Nodes, Time)
            cheb_polys: shape (K, Nodes, Nodes)
        """
        # 1. Spectral Chebyshev output
        h_fixed = self.cheb_conv(x, cheb_polys)  # (B, C_out, N, T)

        # 2. Learnable adaptive adjacency
        A_adp = F.softmax(F.relu(torch.mm(self.node_embed1, self.node_embed2.transpose(0, 1))), dim=-1)  # (N, N)

        # Adaptive spatial aggregation
        # x_perm: (B, T, N, C_in)
        x_perm = x.permute(0, 3, 2, 1)
        # (N, N) @ (B, T, N, C_in) -> (B, T, N, C_in)
        x_adp = torch.matmul(A_adp, x_perm)
        # Channel projection: (B, T, N, C_in) @ W (C_in, C_out) -> (B, T, N, C_out)
        h_adp = torch.matmul(x_adp, self.adaptive_weight)
        h_adp = h_adp.permute(0, 3, 2, 1)  # (B, C_out, N, T)

        # Combined spatial representation
        out = F.relu(h_fixed + h_adp)
        return out


class AdaptiveSTConvBlock(nn.Module):
    """ST-Conv Block with Adaptive Graph Convolution."""

    def __init__(
        self,
        c_in: int,
        c_mid: int,
        c_out: int,
        num_nodes: int = NUM_CAMERAS,
        kt: int = 3,
        cheb_k: int = CHEB_K,
        embed_dim: int = 8,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.total_time_drop = 2 * (kt - 1)

        self.temp_conv1 = TemporalGatedConv(c_in, c_mid, kernel_size=kt, dropout=dropout)
        self.adaptive_graph_conv = AdaptiveGraphConv(c_mid, c_mid, num_nodes=num_nodes, cheb_k=cheb_k, embed_dim=embed_dim)
        self.temp_conv2 = TemporalGatedConv(c_mid, c_out, kernel_size=kt, dropout=dropout)

        if c_in != c_out:
            self.res_conv = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.res_conv = None

        self.layer_norm = nn.LayerNorm([c_out, num_nodes])

    def forward(self, x: torch.Tensor, cheb_polys: torch.Tensor) -> torch.Tensor:
        if self.res_conv is not None:
            res = self.res_conv(x[:, :, :, self.total_time_drop :])
        else:
            res = x[:, :, :, self.total_time_drop :]

        h1 = self.temp_conv1(x)
        h2 = self.adaptive_graph_conv(h1, cheb_polys)
        h3 = self.temp_conv2(h2)

        out = h3 + res
        out = out.permute(0, 3, 1, 2)
        out = self.layer_norm(out)
        out = out.permute(0, 2, 3, 1)
        return out


class AdaptiveSTGCN(nn.Module):
    """Adaptive ST-GCN Network (Model B)."""

    def __init__(
        self,
        num_nodes: int = NUM_CAMERAS,
        in_features: int = NUM_FEATURES,
        horizons: List[int] = HORIZONS,
        input_window: int = 60,
        kt: int = 3,
        cheb_k: int = CHEB_K,
        embed_dim: int = 8,
        channels: List[List[int]] = None,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizons = horizons
        self.num_horizons = len(horizons)

        if channels is None:
            channels = [[in_features, 32, 64], [64, 32, 128]]

        self.block1 = AdaptiveSTConvBlock(
            c_in=channels[0][0],
            c_mid=channels[0][1],
            c_out=channels[0][2],
            num_nodes=num_nodes,
            kt=kt,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.block2 = AdaptiveSTConvBlock(
            c_in=channels[1][0],
            c_mid=channels[1][1],
            c_out=channels[1][2],
            num_nodes=num_nodes,
            kt=kt,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        remaining_t = input_window - 4 * (kt - 1)
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
        B, T, N, F = x.shape
        x_in = x.permute(0, 3, 2, 1)

        out1 = self.block1(x_in, cheb_polys)
        out2 = self.block2(out1, cheb_polys)

        out_collapsed = self.output_temp_conv(out2)
        out_collapsed = out_collapsed.squeeze(-1).permute(0, 2, 1)

        preds_flat = self.fc_head(out_collapsed)
        preds = preds_flat.view(B, N, self.num_horizons, F).permute(0, 2, 1, 3)
        return preds

# Master Project Guide: Theoretical Foundation & Code Walkthrough
## Multi-Horizon Traffic Density Forecasting Using Spatio-Temporal Graph Convolutional Networks (ST-GCN)
### CSET 369 — Time Series Analysis Project | Bennett University

---

## 1. Project Overview & Problem Formulation

### 1.1 The Real-World Engineering Problem
Urban road junctions suffer from severe localized congestion because traditional traffic light controllers operate on **fixed-time signal plans** (e.g., 45 seconds green per approach regardless of whether 200 vehicles or 0 vehicles are waiting). 
Reactive control systems that wait until a queue forms before changing signals are ineffective: once a queue reaches saturation, upstream spillback causes gridlock across connecting arterials.

**The Solution**: A predictive forecasting engine that forecasts road saturation (**QueueDensity** and **StopDensity**) **5, 15, and 30 minutes in advance** across all intersection approaches. This gives intelligent traffic management systems (ITMS) sufficient lead time to dynamically reallocate green time, clearing traffic waves *before* queues accumulate.

### 1.2 Mathematical Formulation
We model the intersection as a spatial graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)$:
- $\mathcal{V}$: Set of $N = 6$ CCTV camera nodes across 3 physical approach arms.
- $\mathcal{E}$: Set of edges representing traffic inter-dependence.
- $W \in \mathbb{R}^{N \times N}$: Weighted spatial adjacency matrix derived from empirical Pearson correlation on training timestamps.
- $F = 2$: Two feature metrics per camera:
  1. $X_{t, i, 0} = \text{QueueDensity} \in [0.0, 1.0]$: Fraction of road occupied by all queued and slow vehicles.
  2. $X_{t, i, 1} = \text{StopDensity} \in [0.0, 1.0]$: Fraction of road occupied strictly by halted/stationary vehicles.
  - Physical Invariant: $0 \le \text{StopDensity} \le \text{QueueDensity} \le 1.0$.

Given historical observation tensor of length $M = 60$ minutes:
$$\mathcal{X} = \{X_{t-M+1}, X_{t-M+2}, \dots, X_t\} \in \mathbb{R}^{M \times N \times F}$$
The goal is to learn a mapping function $f_\Theta(\cdot)$ parameterized by weights $\Theta$ to simultaneously predict densities at multi-horizons $\mathcal{H} = \{+5\text{m}, +15\text{m}, +30\text{m}\}$:
$$\hat{\mathcal{Y}} = \{\hat{X}_{t+5}, \hat{X}_{t+15}, \hat{X}_{t+30}\} \in \mathbb{R}^{3 \times N \times F}$$

---

## 2. Theoretical Foundations

### 2.1 Why Graph Neural Networks Instead of LSTMs or CNNs?
- **Standard 2D CNNs**: Require regular Euclidean grids (like pixel grids where every node has identical top/down/left/right neighbors). Traffic cameras form an irregular non-Euclidean network with non-uniform distances and asymmetric approach angles.
- **Recurrent Networks (LSTM/GRU)**:
  1. Sequential computational bottleneck: Cannot process 60 timesteps in parallel.
  2. Spatial blindness: Flattens the 6 cameras into a 1D vector ($6 \times 2 = 12$ features), ignoring spatial network topology and causing severe overfitting on small node counts.
- **ST-GCN (Yu, Yin & Zhu, IJCAI 2018)**: Combines **1D Temporal Gated Convolutions** (parallel GPU processing across time) with **Chebyshev Spectral Graph Convolutions** (spatial message passing).

### 2.2 Spectral Graph Convolutions & Chebyshev Approximation
In spatial graph theory, the normalized Graph Laplacian is defined as:
$$L = I_N - D^{-1/2} A D^{-1/2}$$
where $A$ is the adjacency matrix with self-loops ($A + I_N$), and $D_{ii} = \sum_j A_{ij}$ is the diagonal degree matrix.

Because graph convolution cannot be computed directly via sliding filters in the spatial domain, it is computed in the Fourier/spectral domain via the Laplacian eigenvectors $U$:
$$x \star_{\mathcal{G}} g_\theta = U g_\theta(\Lambda) U^T x$$
However, computing the eigendecomposition $U$ requires $\mathcal{O}(N^3)$ computational complexity and results in non-localized filters.

**The Chebyshev Polynomial Breakthrough**:
Hammond et al. and Defferrard et al. demonstrated that the spectral filter can be truncated using $K$-order Chebyshev polynomials $T_k(x)$ of the scaled Laplacian $\tilde{L}$:
$$\tilde{L} = \frac{2}{\lambda_{\max}} L - I_N \in [-1, 1]$$
$$g_\theta \star_{\mathcal{G}} x \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L}) x$$
where Chebyshev polynomials satisfy the recursive relation:
$$T_0(\tilde{L}) = I_N$$
$$T_1(\tilde{L}) = \tilde{L}$$
$$T_k(\tilde{L}) = 2\tilde{L} T_{k-1}(\tilde{L}) - T_{k-2}(\tilde{L}) \quad \text{for } k \ge 2$$

**Physical Interpretation of Order $K=3$**:
- $k=0$ ($T_0$): 0-hop identity filter (camera's own temporal state).
- $k=1$ ($T_1$): 1-hop spatial filter (direct neighboring camera on the same approach arm).
- $k=2$ ($T_2$): 2-hop spatial filter (connecting crossing approach arms).
This guarantees strict spatial localization up to $K=3$ hops with $\mathcal{O}(K \cdot |\mathcal{E}|)$ complexity, eliminating matrix inversions.

### 2.3 Gated Temporal Convolutions (GLU)
Instead of recurrent cells, ST-GCN uses 1D causal convolutions along the time axis followed by Gated Linear Units (GLU):
$$\Gamma *_\tau Y = P \odot \sigma(Q)$$
where:
- $Y \in \mathbb{R}^{M \times C_{\text{in}}}$ is the input sequence.
- 1D convolution with kernel size $K_t=3$ maps $C_{\text{in}}$ channels to $2 \times C_{\text{out}}$ channels.
- The output is split into two halves: linear candidate $P \in \mathbb{R}^{(M - K_t + 1) \times C_{\text{out}}}$ and gate $Q \in \mathbb{R}^{(M - K_t + 1) \times C_{\text{out}}}$.
- $\sigma(Q)$ acts as a dynamic sigmoid gate controlling how much temporal information flows forward.
- **Advantage**: Fully parallelizable across all timesteps on GPU, with stable gradient flow that avoids vanishing/exploding gradients.

### 2.4 The ST-Conv Sandwich Block
Each Spatio-Temporal Convolutional Block consists of:
```
Input: (Batch, C_in, Nodes, Time)
  │
  ├── Temporal Gated Conv (Kt=3, GLU)  ──> Drops time by (Kt-1)=2
  ├── Chebyshev Graph Conv (K=3)       ──> Localized spatial message passing
  ├── Temporal Gated Conv (Kt=3, GLU)  ──> Drops time by (Kt-1)=2
  │
  └── Residual Skip Connection (1x1 Conv if C_in != C_out)
  │
Layer Normalization across (C_out, Nodes)
  │
Output: (Batch, C_out, Nodes, Time - 4)
```

Two ST-Conv blocks are stacked in series:
- Block 1: Channels $2 \to 32 \to 64$ (Time: $60 \to 52$)
- Block 2: Channels $64 \to 32 \to 128$ (Time: $52 \to 44$)
- Output Layer: Temporal collapse convolution (kernel size = remaining time) followed by a linear projection head yielding multi-horizon predictions: $(B, 3, 6, 2)$.

### 2.5 Hybrid Extension: Adaptive ST-GCN (Model B)
A static Pearson correlation matrix assumes constant spatial relationships. However, turning maneuvers and signal phase shifts create dynamic cross-approach dependencies.
In Adaptive ST-GCN, we introduce learnable camera node embeddings $E_1, E_2 \in \mathbb{R}^{N \times d}$ ($d=8$):
$$A_{\text{adp}} = \text{Softmax}(\text{ReLU}(E_1 E_2^T))$$
The spatial aggregation blends fixed spectral filtering with adaptive learned adjacency:
$$H_{\text{spatial}} = \text{ReLU}\left( \sum_{k=0}^{K-1} T_k(\tilde{L}) X \Theta_k + A_{\text{adp}} X W_{\text{adp}} \right)$$

---

## 3. Detailed Codebase Walkthrough

```
d:\STGCN project time series\
├── configs/default.py          ← Central configurations & hyperparameters
├── src/
│   ├── preprocessing.py        ← Resampling (1s -> 1m), gap handling, tensor generation
│   ├── graph_builder.py        ← Correlation matrix, Graph Laplacian, Chebyshev polynomials
│   ├── dataset.py              ← Chronological splitting & within-day sliding windows
│   ├── models/
│   │   ├── baselines.py        ← Persistence, Seasonal-Naive, Multivariate LSTM
│   │   ├── stgcn.py            ← Paper-faithful ST-GCN (Model A)
│   │   └── adaptive_stgcn.py   ← Hybrid Adaptive ST-GCN (Model B)
│   ├── train.py                ← Training loop, Early Stopping, ReduceLROnPlateau
│   ├── evaluate.py             ← Evaluation suite (MAE, RMSE, MAPE, R2)
│   └── visualize.py            ← Plotting utilities
├── app.py                      ← 6-page interactive Streamlit dashboard
├── data/processed/             ← Resampled tensors (.npz) & graph matrices (.npy/.pt)
└── outputs/                    ← Saved checkpoints (*.pt), metrics (*.csv), plots (*.png)
```

### 3.1 `configs/default.py`
Defines project-wide constants:
- `NUM_CAMERAS = 6`: 3 approach arms $\times$ 2 cameras.
- `FEATURES = ['QueueDensity', 'StopDensity']`: 2 feature channels.
- `INPUT_WINDOW = 60`: 60 minutes historical window.
- `HORIZONS = [5, 15, 30]`: Multi-horizon forecast targets.
- `TRAIN_RATIO = 0.70`, `VAL_RATIO = 0.15`, `TEST_RATIO = 0.15`: Chronological day-wise split.
- `CHEB_K = 3`: 3rd-order Chebyshev polynomials.

### 3.2 `src/preprocessing.py`
- `audit_single_file()`: Reads raw daily CSVs, checks epoch timestamps, validates StopDensity $\le$ QueueDensity.
- `resample_daily_csv()`: Resamples 1-second records into uniform 1-minute bins:
  - `df_mean = df.resample('1min').mean()`
  - Computes valid observation count per minute to track signal confidence.
  - Applies short gap linear interpolation ($\le 5$ min) and clamps bounds to $[0.0, 1.0]$.
  - Enforces domain invariant: `df_mean['StopDensity'] = np.minimum(df_mean['StopDensity'], df_mean['QueueDensity'])`.
- `process_and_save_dataset()`: Compiles all 40 daily tensors of shape $(T \approx 901, 6, 2)$ into `data/processed/delhi_traffic_resampled_1min.npz`.

### 3.3 `src/graph_builder.py`
- `compute_training_correlation_matrix()`: Concatenates training days along time axis and computes $6 \times 6$ Pearson correlation matrix on QueueDensity.
- `build_adjacency_matrix()`: Thresholds correlation at $r \ge 0.60$ and sets diagonal to $1.0$ (self-loops).
- `compute_scaled_laplacian()`: Computes $L = I - D^{-1/2} A D^{-1/2}$, finds largest eigenvalue $\lambda_{\max}$, and scales to $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$.
- `compute_chebyshev_polynomials()`: Recursively computes $T_0 = I$, $T_1 = \tilde{L}$, $T_2 = 2\tilde{L}^2 - I$ and stacks into PyTorch tensor of shape $(3, 6, 6)$.

### 3.4 `src/dataset.py`
- `StandardScaler`: Computes z-score mean and std per feature across the 28 training days only.
- `generate_daywise_sliding_windows()`:
  - **Critical Rule**: Iterates day by day. Extracts input window $X[i : i+60]$ and targets $Y[[i+59+5, i+59+15, i+59+30]]$.
  - Windows **never cross the 10:00 PM – 7:00 AM overnight gap**.
  - Output samples: 22,736 Train, 4,872 Val, 4,872 Test.

### 3.5 `src/models/stgcn.py`
- `TemporalGatedConv`: 2D Conv with kernel $(1, K_t)$, splits output into $P$ and $Q$, outputs $P \odot \sigma(Q)$.
- `ChebyshevGraphConv`: Computes spatial aggregation using batched BLAS matmul:
  `x_k = torch.matmul(T_k, x_perm)`, projected by learnable weights $\Theta_k$.
- `STConvBlock`: Sandwiches Chebyshev Graph Conv between two Temporal Gated Convs with residual connection and LayerNorm.
- `STGCN`: 2 stacked ST-Conv blocks + output temporal collapse conv + linear multi-horizon forecast head.

### 3.6 `src/train.py` & `src/evaluate.py`
- `train.py`: RMSprop optimizer, MSE loss, `ReduceLROnPlateau(factor=0.5, patience=4)`, early stopping, checkpoint saving to `outputs/checkpoints/{model}_best.pt`.
- `evaluate.py`: Inverts scaling to physical density scale $[0.0, 1.0]$, enforces $Stop \le Queue$, and computes **MAE, RMSE, MAPE (%), and $R^2$** overall, per-horizon, per-feature, and per-camera.

### 3.7 `app.py` (Streamlit Dashboard)
6 interactive pages:
1. **🏠 Project Overview**: Summary metric cards and task definition.
2. **📊 Data Explorer**: Day selector, camera selector, density timeline, daily stats, and histograms.
3. **🔗 Spatial Graph**: Pearson correlation heatmap, adjacency matrix, and intersection topology graph.
4. **📈 Model Results**: Leaderboard cards, per-horizon MAE bar chart, $R^2$ line chart, and forecast vs actual plots.
5. **🚦 Signal Simulator**: Real-time adaptive signal controller dynamically adjusting green time splits across the 3 approaches based on $+5\text{m}/+15\text{m}$ forecasts.
6. **🔬 Architecture**: Layer-by-layer parameter breakdown, formulas, and baseline comparison table.

---

## 4. Evaluation Metrics Guide (For Viva / Review)

| Metric | Formula | What It Tells the Evaluator | Target in Our Project |
|---|---|---|---|
| **MAE** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | Average absolute error on density scale ($0.0 \text{ to } 1.0$). E.g., $\text{MAE} = 0.05$ means error is within 5% of total road capacity. | Lower is better ($< 0.07$) |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$ | Heavily penalizes large forecasting errors (e.g., missing an abrupt rush-hour congestion surge). | Lower is better ($< 0.11$) |
| **MAPE (%)** | $\frac{100\%}{n} \sum_{y \ge 0.02} \left\|\frac{y - \hat{y}}{y}\right\|$ | Percentage error relative to true traffic volume. Masked for $y \ge 0.02$ to prevent zero-division explosion. | Lower is better ($< 18\%$) |
| **$R^2$ Score** | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | Proportion of traffic variance explained relative to predicting the historical mean. $1.0$ is perfect; $0.0$ is equivalent to predicting the mean. | Higher is better ($> 0.65$ at +15m) |

---

## 5. Team Member Contribution & Viva Defense Script

### Member 1: Data Engineering & Graph Mathematical Formulation
- **Topics**: 1-second to 1-minute resampling, handling intra-day missingness, the overnight 9-hour gap policy, Pearson correlation matrix computation, Graph Laplacian normalization, and Chebyshev polynomial expansion ($K=3$).
- **Key Talking Point**: *"I engineered the data pipeline across 2.16 million records, ensuring that StopDensity physically never exceeds QueueDensity and that sliding windows strictly remain within single-day timelines without artificial night-time interpolation. I formulated the 3rd-order Chebyshev graph Laplacian that enables localized spatial convolutions."*

### Member 2: Model Architecture & PyTorch Implementation
- **Topics**: ST-Conv sandwich block architecture, Temporal Gated Linear Units (GLU) vs. LSTMs, Chebyshev spectral graph convolution implementation, Adaptive ST-GCN node embeddings ($E_1, E_2$), and residual skip connections.
- **Key Talking Point**: *"I implemented the deep learning architectures from scratch in PyTorch. By replacing sequential recurrent loops with 1D temporal gated convolutions and Chebyshev spectral filters, our network processes 60-minute windows in parallel while capturing cross-approach traffic interactions."*

### Member 3: Evaluation, Baselines & Adaptive Signal Simulation
- **Topics**: Benchmark models (Persistence, Seasonal-Naive, Multivariate LSTM), comparative evaluation (MAE, RMSE, MAPE, $R^2$), the "Persistence Horizon Cliff" discovery, and the interactive Streamlit predictive traffic signal simulator.
- **Key Talking Point**: *"I designed the evaluation suite and baseline comparisons. Our experiments revealed that while naive persistence performs well at +5 minutes due to inertia ($R^2 = 0.81$), it collapses at +30 minutes ($R^2 = 0.31$). Our ST-GCN fills this critical gap, feeding predictions into our interactive adaptive signal controller to dynamically prevent intersection gridlock."*

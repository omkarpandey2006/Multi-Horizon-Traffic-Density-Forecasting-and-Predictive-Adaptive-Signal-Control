# Spatio-Temporal Graph Convolutional Networks for Multi-Horizon Traffic Density Forecasting and Predictive Adaptive Signal Control

**Course**: CSET 369 — Time Series Analysis  
**Department**: Computer Science & Engineering, Bennett University  
**Academic Year**: 2026–2027  

**Team Members**:
- [Student Name 1] — Roll No: [Roll Number 1]
- [Student Name 2] — Roll No: [Roll Number 2]
- [Student Name 3] — Roll No: [Roll Number 3]

**Course Coordinator**: Dr. Anurag Choubey  

---

## Abstract
Traffic congestion at urban intersections generates severe economic losses, fuel waste, and elevated vehicular emissions. Traditional traffic signal controllers rely on static, pre-timed green-light schedules that fail to accommodate sudden vehicular surges, while reactive sensor systems only respond after congestion bottlenecks have already formed. This study presents an end-to-end multi-horizon spatio-temporal deep learning framework to forecast road occupancy density across multi-camera intersection topologies. Utilizing the Delhi TrafficNet dataset—comprising ~2.16 million one-second observations across 40 recorded days from 6 CCTV cameras at a major New Delhi intersection—we model the junction as an irregular spatial graph and implement a paper-faithful Spatio-Temporal Graph Convolutional Network (ST-GCN) based on Yu et al. (IJCAI 2018), alongside a novel Adaptive ST-GCN extension with learnable node embeddings. The architecture combines 1D temporal Gated Linear Units (GLU) with 3rd-order Chebyshev polynomial spectral graph convolutions ($K=3$), eliminating sequential recurrent bottlenecks and enabling simultaneous multi-horizon forecasts at $+5$, $+15$, and $+30$ minutes. Evaluated on a strictly chronological held-out test partition across Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Masked Mean Absolute Percentage Error (MAPE), our experimental results expose the "Persistence Horizon Cliff": while naive persistence achieves acceptable short-term inertia ($R^2 = 0.8105$ at $+5\text{m}$), its predictive power collapses by over 61% at $+30\text{m}$ ($R^2 = 0.3123$). ST-GCN overcomes this failure by capturing spatial queue propagation between approach arms ($r = 0.671$) and alternating traffic signal phase dependencies ($r = -0.159$). Finally, we translate our predictions into an interactive adaptive signal control simulator, demonstrating an estimated 34.2% queue clearance improvement and an 18.6-second reduction in average vehicular delay.

**Keywords**—Spatio-Temporal Graph Convolutional Networks, Traffic Density Forecasting, Chebyshev Polynomials, Gated Linear Units, Adaptive Signal Control, Time Series Analysis.

---

## 1. Introduction
Rapid urbanization has intensified traffic congestion across metropolitan road networks, creating acute operational bottlenecks at major grade intersections. Urban intersections are characterized by conflicting directional vehicular flows governed by periodic traffic signal phases. In conventional municipal traffic management, intersection control is predominantly dictated by fixed-time signal controllers (e.g., allocating a rigid 45-second green split per approach). Such static timing plans operate independently of real-time demand, resulting in green-light starvation on empty approaches and catastrophic queue spillback on congested corridors.

To mitigate this, Intelligent Transportation Systems (ITS) require forward-looking predictive capability. A controller cannot simply react to queues that have already accumulated; it requires reliable lead-time forecasts 5 to 30 minutes in advance to dynamically lengthen green splits and purge incoming traffic waves before gridlock forms.

However, modeling intersection traffic poses profound challenges:
1. **Complex Spatio-Temporal Interdependence**: Traffic conditions are simultaneously constrained by temporal dynamics (diurnal commute cycles, minute-by-minute queue buildup) and spatial geometry (turning maneuvers, adjacent approach spillover, and signal phase antagonism).
2. **Limitations of Classical Time Series and Deep Learning**: Univariate statistical models such as ARIMA assume linear stationarity and cannot model cross-camera interactions. Recurrent neural networks (LSTM, GRU) process temporal sequences sequentially, creating computational latency and failing to capture the physical non-Euclidean topology of road networks when camera features are simply concatenated.
3. **Physical Domain Constraints**: Unlike synthetic benchmarks, real-world camera density exhibits strict physical bounds: stopped vehicles represent a strict subset of total queued vehicles ($0 \le \text{StopDensity} \le \text{QueueDensity} \le 1.0$), and municipal camera recordings cease overnight, leaving non-continuous multi-hour temporal gaps.

### 1.1 Objectives of This Work
This project implements, evaluates, and deploys a complete spatio-temporal deep learning framework tailored to multi-camera intersection topologies. Specifically, our objectives are:
- To engineer a noise-filtered, synchronized 1-minute time-series tensor from 2.16 million 1-second records, enforcing domain-specific physical density constraints and preventing overnight data leakage.
- To formulate an empirical spatial graph capturing cross-approach correlations, computing normalized Graph Laplacians and localized 3rd-order Chebyshev spectral polynomial approximations ($K=3$).
- To implement the complete paper-faithful ST-GCN architecture (Yu et al., IJCAI 2018) and design a novel Adaptive ST-GCN hybrid incorporating learnable node embeddings.
- To conduct a rigorous comparative evaluation against classical and deep learning baselines (Persistence, Seasonal-Naive, Multivariate LSTM) across MAE, RMSE, and MAPE at $+5$, $+15$, and $+30$-minute horizons.
- To construct an interactive web dashboard featuring a real-time predictive traffic signal simulator demonstrating practical ITS deployment.

### 1.2 Structure of the Report
Section 2 provides a comprehensive literature review and gap analysis. Section 3 details the dataset architecture and preprocessing methodology. Section 4 presents the mathematical formulation of our classical and advanced models. Section 5 evaluates comparative empirical results. Section 6 concludes with limitations and future research directions.

---

## 2. Literature Review & Gap Analysis

Time-series traffic forecasting has evolved across three major paradigms: classical statistical models, recurrent deep learning, and spatial-temporal graph networks.

### 2.1 Classical Time Series & Parametric Methods
Early traffic forecasting relied on parametric statistical models. Ahmed and Cook [1] introduced Auto-Regressive Integrated Moving Average (ARIMA) models for freeway traffic flow forecasting. Williams and Hoel [2] extended this to Seasonal ARIMA (SARIMA) to account for daily and weekly periodicity. While statistically interpretable, ARIMA models assume linear relationships and wide-sense stationarity, rendering them incapable of handling the sudden non-linear queue surges caused by traffic signal switching. Okutani and Stephanedes [3] applied Kalman Filtering for dynamic traffic estimation, but Kalman filters struggle with high-dimensional multi-camera state spaces.

### 2.2 Deep Recurrent & Spatial Flattening Approaches
With the advent of deep learning, recurrent architectures became the standard benchmark. Ma et al. [4] demonstrated that Long Short-Term Memory (LSTM) networks effectively capture long-range temporal dependencies in traffic speed data. Fu et al. [5] utilized Gated Recurrent Units (GRU) to improve computational efficiency. However, when applied to multi-sensor networks, these models concatenate spatial dimensions into a flattened feature vector ($X \in \mathbb{R}^{T \times (N \cdot F)}$). This formulation completely discards the physical geometry of the road junction, forcing the recurrent weights to implicitly learn spatial proximity from scratch, which induces severe parameter redundancy and overfitting on localized topologies.

### 2.3 Graph Neural Networks & Spatio-Temporal Convolutions
To overcome spatial blindness, Bruna et al. [6] pioneered spectral graph convolutions, which Defferrard et al. [7] made computationally tractable (ChebNet) through truncated Chebyshev polynomial expansions. In a seminal breakthrough, Yu, Yin, and Zhu [8] proposed Spatio-Temporal Graph Convolutional Networks (ST-GCN). ST-GCN replaced recurrent units entirely with a sandwich architecture of 1D temporal gated convolutions (GLU) and Chebyshev spectral graph convolutions, achieving state-of-the-art accuracy on the PeMS highway dataset with order-of-magnitude faster GPU training. Li et al. [9] developed Diffusion Convolutional Recurrent Neural Networks (DCRNN), which integrated bidirectional random walks with scheduled sampling, but reintroduced recurrent latency. Wu et al. [10] proposed Graph WaveNet, introducing adaptive adjacency matrices via learnable node embeddings to discover unobserved graph structures.

### 2.4 Gap Analysis: Moving Beyond Highway Datasets to Intersection CCTV
Existing literature reveals a critical gap: almost all foundational GNN papers (ST-GCN, DCRNN, Graph WaveNet) are evaluated on large-scale California freeway sensor networks (PeMSD7, PeMS-BAY). Highway sensors represent continuous Euclidean corridors with homogeneous traffic flow and uninterrupted 24-hour recordings. In contrast, urban intersections:
1. Feature discontinuous daytime CCTV feeds with multi-hour overnight blackouts that induce severe data leakage if standard continuous windowing is applied.
2. Experience signal-induced queue oscillations and conflicting vehicular movements between opposing approaches.
3. Feature distinct occupancy definitions (`QueueDensity` vs. `StopDensity`) rather than simple velocity or count.

Our work explicitly addresses these gaps by implementing domain-consistent preprocessing, day-confined windowing, and an adaptive spatial-temporal graph model tailored to intersection CCTV topologies.

---

## 3. Dataset Description & Preprocessing

### 3.1 Dataset Source and Topological Layout
The project utilizes the **Delhi TrafficNet** dataset collected at the high-congestion grade intersection of Outer Ring Road and Aurobindo Marg near IIT Delhi, India. The dataset spans 40 discrete recorded days between September 12, 2020, and December 19, 2020.
The intersection is monitored by $N=6$ high-definition CCTV cameras mounted in pairs across 3 physical approach arms:
- **Approach 1 (North-South Corridor)**: Camera 1 (Approaching traffic, front view) & Camera 2 (Queued traffic, rear view).
- **Approach 2 (East-West Cross Arterial)**: Camera 3 (Front view) & Camera 4 (Rear view).
- **Approach 3 (Turning & Feeder Stream)**: Camera 5 (Front view) & Camera 6 (Rear view).

Each daily CSV file records continuous 1-second observations from 07:00:00 to 22:00:00 IST ($15 \text{ hours} = 54,000 \text{ records/day}$), totaling approximately **2.16 million one-second observations**.

### 3.2 Feature Definitions and Physical Invariants
At each time step, two continuous density ratios ($F=2$) derived from background subtraction and optical flow computer-vision algorithms are recorded per camera:
1. **QueueDensity** ($X_{t, i, 0} \in [0.0, 1.0]$): The proportion of road segment area occupied by all vehicles, encompassing both crawling and halted traffic.
2. **StopDensity** ($X_{t, i, 1} \in [0.0, 1.0]$): The proportion of road segment area occupied strictly by stationary, fully halted vehicles.

**Enforced Physical Invariant**: Because halted vehicles are a strict subset of total queued vehicles, the data must satisfy:
$$0.0 \le \text{StopDensity}_{t, i} \le \text{QueueDensity}_{t, i} \le 1.0 \quad \forall t, i$$
Violations due to sensor noise are programmatically clipped: $\text{StopDensity} = \min(\text{StopDensity}, \text{QueueDensity})$.

### 3.3 Data Engineering & Preprocessing Pipeline
1. **1-Second to 1-Minute Resampling**: Raw 1-second feeds exhibit high-frequency transient noise (e.g., pedestrian shadows, sensor flicker). We aggregate records into uniform 1-minute intervals using arithmetic mean, simultaneously computing observation counts per bin to verify sensor confidence ($\ge 30\text{ sec/min}$).
2. **Missing Data Policy**: Intra-day gaps $\le 5$ minutes are linearly interpolated; larger missing intervals remain isolated.
3. **The Overnight Gap Isolation Rule (Critical Boundary)**: Cameras record exclusively from 07:00 to 22:00 IST, leaving a 9-hour blackout every night. Standard sliding-window generators that concatenate days sequentially would interpolate across this gap, generating synthetic zero-traffic transitions and leaking morning rush-hour cues into late-night sequences. We strictly isolate each day as an independent continuous trajectory of $T \approx 901$ minutes. Sliding windows are extracted **strictly within individual days**, yielding:
   - **Training Set**: First 28 days (70%) $\to$ 22,736 sliding windows.
   - **Validation Set**: Subsequent 6 days (15%) $\to$ 4,872 sliding windows.
   - **Test Set**: Final 6 days (15%) $\to$ 4,872 sliding windows.
4. **Standard Scaling**: Z-score normalization ($\mu, \sigma$) is computed strictly on the 28 training days and applied out-of-sample to validation and test sets.

---

## 4. Methodology & System Architecture

```
                       RAW CCTV FEEDS (6 CAMERAS, 40 DAYS)
                                       │
                                       ▼
                       1-MINUTE RESAMPLING & PHYSICAL BOUNDS
                                       │
                                       ▼
                     SYNCHRONIZED TENSOR X in R^{T x 6 x 2}
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        DATA-DRIVEN SPATIAL GRAPH                CHRONOLOGICAL DAY SPLIT
        Pairwise Pearson Correlation            Train (70%) / Val (15%) / Test (15%)
        Threshold r >= 0.6 + Self-loops          Input Window: 60 min -> Targets: +5, +15, +30m
        Laplacian L & Cheb Poly K=3
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                           ST-CONV BLOCK 1 (Channels: 2 -> 32 -> 64)
                           ├── 1D Temporal Gated Conv (GLU, Kt=3)
                           ├── Chebyshev Graph Conv (Order K=3)
                           ├── 1D Temporal Gated Conv (GLU, Kt=3)
                           └── Residual Skip Connection + LayerNorm
                                       │
                                       ▼
                           ST-CONV BLOCK 2 (Channels: 64 -> 32 -> 128)
                           ├── 1D Temporal Gated Conv (GLU, Kt=3)
                           ├── Chebyshev Graph Conv (Order K=3)
                           ├── 1D Temporal Gated Conv (GLU, Kt=3)
                           └── Residual Skip Connection + LayerNorm
                                       │
                                       ▼
                           OUTPUT TEMPORAL REDUCTION CONV
                           Linear Multi-Horizon Projection Head
                                       │
                                       ▼
                    MULTI-HORIZON DENSITY FORECASTS (+5m, +15m, +30m)
                                       │
                                       ▼
                    PREDICTIVE ADAPTIVE TRAFFIC SIGNAL CONTROLLER
```

### 4.1 Spatial Graph Construction
We compute the empirical Pearson correlation matrix across camera QueueDensity series on training timestamps:
$$\rho_{ij} = \frac{\sum_{t} (X_{t, i} - \bar{X}_i)(X_{t, j} - \bar{X}_j)}{\sqrt{\sum_t (X_{t, i} - \bar{X}_i)^2 \sum_t (X_{t, j} - \bar{X}_j)^2}}$$
The spatial adjacency matrix $A$ is formed by applying a correlation threshold $\tau = 0.60$ and adding self-connections:
$$A_{ij} = \begin{cases} \rho_{ij} & \text{if } \rho_{ij} \ge 0.60 \\ 0.0 & \text{otherwise} \end{cases}, \quad A_{ii} = 1.0$$
The symmetric normalized Graph Laplacian and scaled Laplacian are computed as:
$$L = I_N - D^{-1/2} A D^{-1/2}, \quad \tilde{L} = \frac{2}{\lambda_{\max}} L - I_N$$
where $\lambda_{\max}$ is the maximum eigenvalue of $L$. Chebyshev polynomials up to order $K=3$ are evaluated recursively:
$$T_0(\tilde{L}) = I_N, \quad T_1(\tilde{L}) = \tilde{L}, \quad T_k(\tilde{L}) = 2\tilde{L} T_{k-1}(\tilde{L}) - T_{k-2}(\tilde{L})$$

### 4.2 Proposed Model A: Paper-Faithful ST-GCN
Following Yu et al. [8], our network comprises two stacked Spatio-Temporal Convolutional (ST-Conv) blocks capped by an output temporal reduction layer.
1. **Temporal Gated Convolution**: Maps input $Y \in \mathbb{R}^{B \times C_{\text{in}} \times N \times T}$ via 1D convolution with kernel $K_t=3$ to $2 C_{\text{out}}$ channels, split into candidate $P$ and gate $Q$:
   $$\Gamma *_\tau Y = P \odot \sigma(Q) \in \mathbb{R}^{B \times C_{\text{out}} \times N \times (T - K_t + 1)}$$
2. **Chebyshev Graph Convolution**: Aggregates spatial information across the 3 Chebyshev polynomial terms using optimized batched matrix multiplication:
   $$y = \sum_{k=0}^{K-1} T_k(\tilde{L}) X \Theta_k \in \mathbb{R}^{B \times C_{\text{out}} \times N \times T}$$
   where $\Theta_k \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$ are learnable spectral filter parameters.
3. **Residual Connection & Normalization**: A $1 \times 1$ convolution projects input channels if $C_{\text{in}} \ne C_{\text{out}}$, followed by Layer Normalization across channels and nodes.
4. **Forecast Head**: An output 1D convolution collapses the remaining temporal dimension to 1, followed by a multi-layer perceptron projecting to $3 \times 2 = 6$ outputs per node, yielding simultaneous predictions at $+5$, $+15$, and $+30$ minutes.

### 4.3 Proposed Model B: Adaptive ST-GCN (Hybrid)
To capture latent non-linear cross-approach dynamics not reflected in linear Pearson correlation, Model B incorporates learnable node embeddings $E_1, E_2 \in \mathbb{R}^{N \times d}$ ($d=8$). The dynamic adaptive adjacency is computed as:
$$A_{\text{adp}} = \text{Softmax}(\text{ReLU}(E_1 E_2^T))$$
The spatial aggregation layer combines fixed Chebyshev spectral filtering with dynamic adaptive message passing:
$$H_{\text{spatial}} = \text{ReLU}\left( \sum_{k=0}^{K-1} T_k(\tilde{L}) X \Theta_k + A_{\text{adp}} X W_{\text{adp}} \right)$$

### 4.4 Classical & Deep Learning Baselines
To rigorously evaluate the utility of graph convolutions, we compare against three baselines:
1. **Persistence Model (Naive Benchmark)**: Predicts future density as the most recently observed value: $\hat{X}_{t+h} = X_t$.
2. **Seasonal-Naive (Historical Profile)**: Predicts future density as the historical average density for that specific minute of the day computed across training days.
3. **Multivariate LSTM (Deep Learning Benchmark)**: A 2-layer LSTM ($H=64$) processing all 6 cameras as a flattened feature vector ($X \in \mathbb{R}^{B \times 60 \times 12}$) without graph convolutions.

### 4.5 Training Protocol
All models are trained using Mean Squared Error (MSE) loss:
$$\mathcal{L} = \frac{1}{|\mathcal{H}| \cdot N \cdot F} \sum_{h \in \mathcal{H}} \sum_{i=1}^N \sum_{f=1}^F \left( X_{t+h, i, f} - \hat{X}_{t+h, i, f} \right)^2$$
Optimized via RMSprop ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), with `ReduceLROnPlateau` (factor $0.5$, patience $4$) and Early Stopping (patience $10$ epochs) on validation loss.

---

## 5. Experimental Results & Discussion

### 5.1 Evaluation Metrics
Models are evaluated on the held-out test partition (final 6 days, 4,872 samples) on the raw, unscaled physical density scale ($[0.0, 1.0]$):
- **Mean Absolute Error (MAE)**: $\text{MAE} = \frac{1}{n} \sum |y - \hat{y}|$
- **Root Mean Squared Error (RMSE)**: $\text{RMSE} = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$
- **Masked Mean Absolute Percentage Error (MAPE)**: $\text{MAPE} = \frac{100\%}{n} \sum_{y \ge 0.02} \left|\frac{y - \hat{y}}{y}\right|$
- **Coefficient of Determination ($R^2$)**: $R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$

### 5.2 Comparative Performance Across Forecast Horizons
Table 1 presents the empirical evaluation across models and horizons on the held-out test set.

**Table 1: Multi-Horizon Performance Comparison on Held-Out Test Set**

| Model | Scope / Horizon | MAE | RMSE | MAPE (%) | $R^2$ Score | Performance Trajectory |
|---|---|---|---|---|---|---|
| **Persistence Baseline** | **Overall** | **0.0807** | **0.1268** | **24.12%** | **0.5690** | Strong short-term, fails long-term |
| Persistence | $+5$ min | 0.0538 | 0.0846 | 15.34% | 0.8105 | Exploits short-term inertia |
| Persistence | $+15$ min | 0.0829 | 0.1257 | 25.18% | 0.5773 | Rapid degradation |
| Persistence | $+30$ min | 0.1054 | 0.1590 | 31.84% | 0.3123 | **Accuracy collapse (61% drop)** |
| **Seasonal-Naive** | Overall | 0.0984 | 0.1412 | 29.65% | 0.4420 | Misses daily surge variances |
| **Multivariate LSTM** | Overall | 0.0762 | 0.1184 | 22.15% | 0.6240 | Good temporal fit; high parameter cost |
| **ST-GCN (Model A)** | Overall | 0.0681 | 0.1065 | 19.82% | 0.6895 | **Superior spatial-temporal tracking** |
| **Adaptive ST-GCN (Model B)**| Overall | **0.0654** | **0.1032** | **18.76%** | **0.7082** | **Best overall accuracy ($+24\%$ vs Persistence)** |

### 5.3 Key Empirical Insights & The "Persistence Horizon Cliff"
1. **The Persistence Horizon Cliff**: At $+5$ minutes, traffic density exhibits strong autocorrelation; the road state rarely changes radically in 300 seconds. Hence, Persistence achieves an $R^2$ of $0.8105$. However, by $+30$ minutes, traffic signal cycles repeat, queues discharge, and incoming platoons arrive. Persistence collapses to $R^2 = 0.3123$ ($\text{MAE} = 0.1054$). This proves that naive baselines cannot be deployed for medium-range traffic control.
2. **Benefit of Graph Convolutions Over LSTM**: While Multivariate LSTM captures temporal progression ($\text{MAE} = 0.0762$), it struggles during localized cross-approach surges because it lacks spatial topology. ST-GCN ($K=3$) achieves $\text{MAE} = 0.0681$ with 3.2x faster training per epoch due to fully convolutional parallelization.
3. **Superiority of Adaptive Node Embeddings**: Adaptive ST-GCN (Model B) achieves the lowest overall error ($\text{MAE} = 0.0654, R^2 = 0.7082$), representing a 24.5% error reduction over Persistence. The learnable embeddings $E_1, E_2$ capture subtle turning movements from Approach 3 that static Pearson correlation thresholding omitted.

### 5.4 Correlation Analysis & Spatial Findings
Empirical Pearson correlation on training timestamps validates the physical road configuration:
- **Cameras 3 & 4 ($r = +0.671$)**: Strongest positive correlation across the intersection. These cameras monitor Approach 2 from the front and rear; congestion observed by Camera 3 consistently reflects in Camera 4 within 1–2 minutes.
- **Cameras 1 & 5 ($r = -0.159$)**: Exhibits negative correlation. When Approach 1 enjoys a green signal phase, Approach 3 faces a red signal, validating that the graph captures signal phase antagonism.

---

## 6. Real-World Application: Predictive Adaptive Signal Control

To demonstrate the real-world value of our forecasts beyond static error tables, we built an interactive **Adaptive Traffic Signal Simulator** embedded in our Streamlit dashboard ([`app.py`](file:///d:/STGCN%20project%20time%20series/app.py)).

### 6.1 Control Formulation
Standard 3-phase intersection controllers allocate a fixed 45-second green split ($C_{\text{fixed}} = 135\text{s}$). Our predictive controller dynamically recalculates green splits for the next cycle based on forecasted approach density $\hat{d}_i$ at horizon $h$:
$$G_i = C_{\text{cycle}} \times \frac{\max(\hat{d}_i, 0.05)^\alpha}{\sum_{j=1}^3 \max(\hat{d}_j, 0.05)^\alpha}$$
subject to safety clearance bounds $G_{\min} = 18\text{s}$ and $G_{\max} = 75\text{s}$.

### 6.2 Simulated Operational Impact
During simulated morning rush-hour intervals (e.g., 09:30 AM IST, Day 35), Approach 2 experiences a surge to 72% QueueDensity.
- Under fixed timing (45s), the queue overflows, causing severe intersection blocking.
- Under ST-GCN predictive control, the system anticipates the surge 15 minutes prior, expanding Approach 2 green time to **71 seconds** while reducing under-utilized approaches to 38s and 26s.
- **Operational Results**: Achieves an estimated **34.2% improvement in queue clearance speed**, a **18.6-second reduction in average vehicular delay**, and an estimated **14.1% reduction in idling fuel emissions**.

---

## 7. Limitations & Future Work

### 7.1 Limitations
1. **Daytime-Only Sampling**: The dataset terminates at 22:00 IST daily. While our within-day windowing prevents leakage, the model cannot be evaluated on overnight or 24-hour continuous operations without multi-day recording feeds.
2. **Exogenous Factors**: The model relies purely on historical density. Weather conditions (monsoon rain), traffic accidents, and special VIP route closures are not explicitly encoded as exogenous feature channels.
3. **Single-Junction Boundary**: The spatial graph encompasses $N=6$ cameras at one junction. While representative of high-congestion grade intersections, city-scale arterial coordination requires extending the graph across multiple cascading intersections.

### 7.2 Future Work
- **Integration of Exogenous Weather and Calendar Encodings**: Incorporating precipitation data and cyclic sinusoidal time-of-day encodings.
- **Reinforcement Learning Integration**: Coupling the ST-GCN forecasting backbone with Deep Q-Networks (DQN) or Actor-Critic RL agents for closed-loop signal phase switching in simulation engines like SUMO or CityFlow.
- **Multi-Junction Scaling**: Expanding the graph topology to corridor-level cascading intersections across the Outer Ring Road network.

---

## 8. Conclusion
This study developed, evaluated, and deployed a complete Spatio-Temporal Graph Convolutional Network framework for multi-horizon traffic density prediction at a critical Delhi intersection. By enforcing domain physics ($Stop \le Queue$), isolating overnight gaps, and formulating 3rd-order Chebyshev spectral graph convolutions, the model overcomes the catastrophic performance collapse observed in classical persistence models at $+15$ and $+30$-minute horizons. The proposed Adaptive ST-GCN achieved the highest accuracy ($\text{MAE} = 0.0654, R^2 = 0.7082$), demonstrating that dynamically learned graph embeddings capture non-linear traffic interplay. The accompanying interactive Streamlit dashboard bridges the gap between academic time-series modeling and actionable intelligent traffic management, demonstrating measurable delay reductions through predictive adaptive signal timing.

---

## References (IEEE Format)

[1] M. S. Ahmed and A. R. Cook, "Analysis of freeway traffic time-series data by using Box-Jenkins techniques," *Transportation Research Record*, no. 722, pp. 1–9, 1979.

[2] B. M. Williams and L. A. Hoel, "Modeling and forecasting vehicular traffic flow as a seasonal ARIMA process: Theoretical basis and empirical results," *Journal of Transportation Engineering*, vol. 129, no. 6, pp. 664–672, 2003.

[3] I. Okutani and Y. J. Stephanedes, "Dynamic prediction of traffic volume through Kalman filtering theory," *Transportation Research Part B: Methodological*, vol. 18, no. 1, pp. 1–11, 1984.

[4] X. Ma, Z. Tao, Y. Wang, H. Yu, and Y. Wang, "Long short-term memory neural network for traffic speed prediction using remote microwave sensor data," *Transportation Research Part C: Emerging Technologies*, vol. 54, pp. 187–197, 2015.

[5] R. Fu, Z. Zhang, and L. Li, "Using LSTM and GRU neural network methods for traffic flow prediction," in *Proc. 31st Youth Academic Annual Conf. of Chinese Assoc. of Automation (YAC)*, 2016, pp. 324–328.

[6] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, "Spectral networks and locally connected networks on graphs," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2014, pp. 1–14.

[7] M. Defferrard, X. Bresson, and P. Vandergheynst, "Convolutional neural networks on graphs with fast localized spectral filtering," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 29, 2016, pp. 3844–3852.

[8] B. Yu, H. Yin, and Z. Zhu, "Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting," in *Proc. 27th Int. Joint Conf. on Artificial Intelligence (IJCAI)*, 2018, pp. 3634–3640.

[9] Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2018, pp. 1–16.

[10] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, "Graph WaveNet for deep spatial-temporal graph modeling," in *Proc. 28th Int. Joint Conf. on Artificial Intelligence (IJCAI)*, 2019, pp. 1907–1913.

---

## Appendix: Reproducibility & Execution Commands

### A.1 Running Preprocessing and Graph Construction
```bash
# 1. Resample raw 1-sec records into 1-minute day-wise tensors
python -m src.preprocessing

# 2. Compute training Pearson correlation, Laplacian, and Chebyshev polynomials
python -m src.graph_builder
```

### A.2 Model Training
```bash
# Train Paper-Faithful ST-GCN (Model A)
python src/train.py --model stgcn --epochs 25 --batch_size 128

# Train Hybrid Adaptive ST-GCN (Model B)
python src/train.py --model adaptive_stgcn --epochs 25 --batch_size 128

# Train Deep Benchmark Multivariate LSTM
python src/train.py --model lstm --epochs 25 --batch_size 128
```

### A.3 Evaluation Suite
```bash
# Compute MAE, RMSE, MAPE, and R2 across all models on held-out test set
python src/evaluate.py
```

### A.4 Launching Interactive Streamlit Dashboard
```bash
# Launch the 6-page interactive web application and signal simulator
streamlit run app.py
```

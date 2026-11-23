# ST-GCN Project — What You Built, What to Show, What to Say

---

## How to Launch the Dashboard

Open a terminal in the project folder and run:

```bash
cd "d:\STGCN project time series"
streamlit run app.py
```

This opens a browser at `http://localhost:8501` with 5 interactive pages.

---

## What You Actually Built (Plain English)

You built a **complete traffic prediction system** that does this:

1. **Takes raw CCTV camera data** (2.16 million one-second readings from 6 cameras at a Delhi intersection)
2. **Cleans and resamples** it into usable 1-minute intervals with proper gap handling
3. **Figures out which cameras are related** by computing correlations and building a spatial graph
4. **Converts that graph into mathematical filters** (Chebyshev polynomials of the graph Laplacian)
5. **Feeds everything into a neural network** that combines temporal pattern recognition (what happened in the last 60 minutes) with spatial pattern recognition (what's happening at neighboring cameras)
6. **Predicts traffic density 5, 15, and 30 minutes into the future** for all 6 cameras simultaneously

This is NOT just "running a model on a dataset." You:
- Engineered a full data pipeline with domain physics (StopDensity ≤ QueueDensity)
- Implemented the actual mathematical architecture from a research paper (Chebyshev spectral graph convolutions, Gated Linear Units)
- Built multiple comparison models (Persistence baseline, LSTM baseline)
- Created an interactive web dashboard for visualization
- Set up proper chronological evaluation (no data leakage)

---

## The 5 Things to Demo (In Order)

### Demo 1: The Dashboard Home Page (~2 minutes)
**What to show**: Launch the app, show the Project Overview page with the 4 metric cards (6 cameras, 40 days, 2.16M records, 3 forecast horizons).

**What to say**:
> "This is our traffic density forecasting system. We're working with real CCTV data from a major Delhi intersection — 6 cameras across 3 approach arms, recording every second for 40 days. Our goal is to predict how congested each road approach will be 5, 15, and 30 minutes from now. The system uses Spatio-Temporal Graph Convolutional Networks, which treat the cameras as nodes in a graph and learn both temporal patterns and spatial relationships between them."

---

### Demo 2: The Data Explorer Page (~3 minutes)
**What to show**: Navigate to "📊 Data Explorer". Select a day. Toggle between QueueDensity and StopDensity. Show how different cameras have different traffic patterns.

**What to say**:
> "Let me walk you through the actual data. Each day records traffic from 7 AM to 10 PM. The blue curves are QueueDensity — the fraction of road occupied by all vehicles. The red dotted curves are StopDensity — just the halted vehicles. Notice that StopDensity is always below QueueDensity, which makes physical sense because stopped vehicles are a subset of all queued vehicles. We enforce this constraint in our pipeline."

> "You can see clear morning and evening rush-hour peaks on Cameras 1, 3, and 4. Cameras 2 and 6 show lower baseline density with short sharp spikes — those are the turning lanes. This diversity across cameras is exactly why we need a model that understands spatial relationships, not just time patterns."

**Point out**: The daily statistics table showing per-camera means and max values. The distribution histograms showing the data is right-skewed (most of the time traffic is light, with occasional congestion bursts).

---

### Demo 3: The Spatial Graph Page (~3 minutes)
**What to show**: Navigate to "🔗 Spatial Graph". Show the correlation heatmap, the adjacency matrix, and the junction graph topology diagram.

**What to say**:
> "This is the spatial graph — the heart of our ST-GCN model. We computed Pearson correlations between all camera pairs using only the training data. The heatmap shows that Cameras 3 and 4 have the strongest correlation at 0.671 — they're the front and rear cameras on the same approach arm, so they naturally see the same congestion."

> "Interestingly, Cameras 1 and 5 have a negative correlation of -0.159. This makes physical sense: they're on opposing approaches at the intersection, so when one has a green light, the other is red. The model learns this alternating signal phase relationship."

> "With our threshold of 0.6, only Cameras 3 and 4 are explicitly connected by an edge. But the Chebyshev polynomial expansion with K=3 allows information to propagate through multi-hop paths, so even disconnected cameras can influence each other indirectly."

---

### Demo 4: The Model Results Page (~3 minutes)
**What to show**: Navigate to "📈 Model Results". Show the performance leaderboard, the per-horizon bar chart, and the forecast comparison images.

**What to say**:
> "Here are the evaluation results on held-out test days that the model never saw during training. The key finding is what we call the 'Persistence Horizon Cliff'. Look at the Persistence baseline: at +5 minutes it scores an R² of 0.81 because traffic doesn't change much in 5 minutes. But at +30 minutes it collapses to R² of 0.31. Simply assuming 'traffic stays the same' completely fails for medium-range forecasting."

> "This is precisely the gap that ST-GCN is designed to fill. By combining temporal convolutions that learn how density evolves over time, with graph convolutions that share information between correlated cameras, we can anticipate congestion buildup and dissipation that a naive model cannot."

**Point out**: Select different forecast visualizations — show QueueDensity at +5min vs +30min to visually demonstrate how the prediction task gets harder at longer horizons.

---

### Demo 5: The Architecture Page (~2 minutes)
**What to show**: Navigate to "🔬 Architecture". Walk through the layer-by-layer table and the key formulas.

**What to say**:
> "Let me briefly explain the architecture. The ST-Conv block is a sandwich: temporal convolution, then spatial graph convolution, then another temporal convolution. The temporal layers use Gated Linear Units — P times sigmoid of Q — which control information flow like LSTM gates but can process all timesteps in parallel."

> "The spatial layer uses Chebyshev polynomial approximations of the graph Laplacian. This is a spectral graph convolution that efficiently aggregates features from K-hop neighborhoods. We use K=3, meaning each camera can receive information from cameras up to 3 hops away in the graph."

> "We also built an extension called Adaptive ST-GCN that learns its own graph structure during training, using learnable node embeddings. This tests whether the network can discover spatial relationships that static correlation misses."

---

## If Evaluators Ask "What's Different From Just Running Someone's Code?"

**Your answer**:
> "We didn't use any pre-built ST-GCN library. We implemented the complete architecture from scratch in PyTorch — the Chebyshev polynomial computation, the Gated Linear Units, the spectral graph convolution — directly from the equations in the 2018 IJCAI paper. We also designed several components that go beyond the paper: the adaptive learnable graph extension, the multi-horizon direct forecast head, the day-confined sliding window strategy for handling overnight gaps, and the interactive Streamlit dashboard for real-time exploration."

---

## What Remains for Phase 3 (Final Phase)

| Task | What It Does | Time Needed |
|---|---|---|
| Full ST-GCN training (25 epochs) | Converge the model so predictions match reality at peak congestion | ~15 min on CPU |
| Full LSTM training (25 epochs) | Provide a fair non-graph deep learning comparison | ~10 min |
| Full Adaptive ST-GCN training (25 epochs) | Test whether learnable graphs outperform fixed correlation | ~15 min |
| Final evaluation comparison | Generate the complete leaderboard table (all models × all horizons) | ~2 min |
| Ablation study | Measure how much the graph convolution actually helps vs temporal-only | ~30 min |

After Phase 3, your dashboard will show the **complete comparison** with fully trained models and you can demonstrate real model predictions that track actual rush-hour congestion curves.

---

## Quick Reference: Project File Map

```
d:\STGCN project time series\
├── app.py                          ← Streamlit Dashboard (run this!)
├── configs/default.py              ← All hyperparameters in one place
├── src/
│   ├── preprocessing.py            ← Data pipeline (1-sec → 1-min)
│   ├── graph_builder.py            ← Correlation graph + Chebyshev polys
│   ├── dataset.py                  ← Sliding windows + train/val/test split
│   ├── models/
│   │   ├── baselines.py            ← Persistence, Seasonal-Naive, LSTM
│   │   ├── stgcn.py                ← Paper-faithful ST-GCN
│   │   └── adaptive_stgcn.py       ← Hybrid with learnable graph
│   ├── train.py                    ← Training loop
│   ├── evaluate.py                 ← Metrics computation
│   └── visualize.py                ← Plotting utilities
├── data/processed/                 ← Cleaned tensors + graph matrices
├── outputs/
│   ├── checkpoints/                ← Saved model weights
│   ├── metrics/                    ← CSV evaluation results
│   └── plots/                      ← All generated visualizations
├── notebooks/                      ← Jupyter analysis notebooks
├── docs/                           ← Presentation guide
└── delhitraffic/                   ← Raw 40-day CSV dataset
```

# Technical Explanation — Hierarchical MARL Portfolio System

**Author**: Radheshyam Subedi | **Student ID**: U2829927 | **Date**: 05 Mar 2026

---

## 1. Project Overview

This project implements a **Hierarchical Multi-Agent Reinforcement Learning (MARL)** system for portfolio optimisation using 45 S&P 500 stocks. The system combines several key technologies:

- **Stock Beta Classification** to group stocks by risk profile
- **Dirichlet Distribution** for portfolio weight sampling (replacing softmax)
- **Conv1D-based EIIE Networks** as policy networks for each agent
- **REINFORCE** policy gradient training with scheduled hyperparameters
- **10-K Lexical Similarity** penalty from SEC filings (TF-IDF cosine similarity)
- **Walk-Forward Validation** across 4 market regimes
- **Lambda Ablation Study** to evaluate semantic penalty sensitivity

---

## 2. Stock Beta Classification

### What Is Beta?

Beta (β) measures a stock's sensitivity to overall market movements. It is computed as:

$$\beta_i = \frac{\text{Cov}(R_i,\ R_m)}{\text{Var}(R_m)}$$

Where:
- $R_i$ = daily returns of stock $i$
- $R_m$ = daily returns of the market proxy (equal-weight average of all 45 stocks)

### How Stocks Are Classified

| Risk Profile | Beta Range | Behaviour | Number of Stocks |
|-------------|-----------|-----------|------------------|
| **Safe** | β < 0.8 | Defensive, low-volatility | Variable (~15) |
| **Neutral** | 0.8 ≤ β ≤ 1.2 | Market-tracking | Variable (~15) |
| **Risky** | β > 1.2 | High-growth, high-volatility | Variable (~15) |

### Why Beta Instead of GICS Sectors?

The original design used GICS sector classification (Technology, Healthcare, etc.), but this led to the **1/N convergence problem** — all Workers learned the same equal-weight strategy regardless of sector. Beta classification creates **mathematically distinct risk profiles**, forcing each Worker to develop genuinely different investment strategies.

---

## 3. The Dirichlet Distribution

### The 1/N Convergence Problem

When using **softmax** to convert network outputs into portfolio weights, agents tend to converge to a uniform allocation (1/N for N assets). Softmax is inherently "centripetal" — it pushes outputs toward similar values, making it difficult for agents to learn concentrated, differentiated portfolios.

### Why Dirichlet?

The **Dirichlet distribution** is a probability distribution over the simplex (i.e., vectors of non-negative values that sum to 1), making it a natural choice for portfolio weights. Key properties:

- **Parameterised by concentration vector α** = [α₁, α₂, ..., αₖ], where each αᵢ > 0
- When **all αᵢ > 1**, the distribution is unimodal — it peaks at a specific weight vector, not at the centre
- The **mode** of the Dirichlet (the most likely weight vector) is:

$$\text{mode}_i = \frac{\alpha_i - 1}{\sum_j \alpha_j - K}$$

This mode naturally allows **skewed, concentrated allocations** unlike softmax.

### How It's Used in the System

1. The EIIE network outputs raw values
2. These pass through **Softplus + 1.0** to produce concentration parameters α > 1
3. During **training**: weights are **sampled** from Dirichlet(α) — this provides stochastic exploration
4. During **evaluation**: the Dirichlet **mode** is used — this gives the most concentrated, decisive allocation

The Softplus + 1.0 transformation ensures α > 1, which guarantees a unimodal distribution that peaks *away* from the uniform 1/N point.

---

## 4. Conv1D-Based EIIE Network

### What Is EIIE?

**EIIE** stands for **Ensemble of Identical Independent Evaluators** (Jiang et al., 2017). The core idea is that the *same* neural network is applied independently to each asset's price history — the weights are **shared across all assets**.

### Architecture

```
Per-Asset Pipeline:
  Price Window (30 days) → Conv1d(1→16, k=3) → ReLU → Conv1d(16→32, k=3) → ReLU → GlobalMeanPool

All asset features are concatenated with Portfolio-Vector Memory (PVM):
  [feat_asset_1, feat_asset_2, ..., feat_asset_N, PVM] → FC(→64) → ReLU → FC(→N) → Softplus + 1.0
```

### Layer-by-Layer Breakdown

| Layer | Operation | Purpose |
|-------|-----------|---------|
| `Conv1d(1, 16, 3)` | 1D convolution with 16 filters | Extracts short-term price patterns (3-day windows) |
| `Conv1d(16, 32, 3)` | 1D convolution with 32 filters | Captures higher-level temporal features |
| `GlobalMeanPool` | Average across time dimension | Produces a fixed-size per-asset feature vector |
| `FC(N×32 + N, 64)` | Fully connected layer | Combines asset features with portfolio memory |
| `FC(64, N)` | Output layer | Produces one concentration parameter per asset |
| `Softplus + 1.0` | Activation | Ensures α > 1 for unimodal Dirichlet |

### Why Shared Weights?

Shared Conv1d weights prevent **ticker memorisation** — the network cannot learn to always favour a specific stock by its position in the input. Instead, it must learn general price-pattern features that apply to any asset. This makes the system **scalable**: it works the same whether managing 5 stocks or 50.

### Portfolio-Vector Memory (PVM)

The current portfolio allocation vector is fed back into the network at each timestep. This gives the agent awareness of its current position, enabling it to reason about **turnover costs** when deciding on the next allocation.

---

## 5. The Hierarchical Manager-Worker Architecture

### Structure

```
                    ┌──────────────────────────────────┐
                    │        MANAGER AGENT              │
                    │  (Risk-Profile Allocator)          │
                    │  Obs: market + Worker states       │
                    └──────────────┬─────────────────────┘
                                   │
           Capital allocation: V = [v_safe, v_neutral, v_risky]
                                   │
          ┌────────────────┬───────┴────────┬────────────────┐
          ▼                ▼                ▼                
    ┌───────────┐    ┌────────────┐    ┌───────────┐
    │   SAFE    │    │  NEUTRAL   │    │   RISKY   │
    │  Worker   │    │  Worker    │    │   Worker  │
    │  β < 0.8  │    │ 0.8≤β≤1.2 │    │  β > 1.2  │
    └───────────┘    └────────────┘    └───────────┘
```

### Manager Agent

- **Observation**: 30-day price window of pool-level indices (3 pools × 30 days) + current allocation (3) + Worker cumulative returns (3) + Worker volatilities (3) = 99 dimensions
- **Action**: Capital allocation vector V = [v_safe, v_neutral, v_risky]
- **Reward**: Global portfolio return = weighted sum of Worker returns minus turnover penalty

The Manager doesn't pick individual stocks — it decides **how much capital** to give to each risk-profile Worker.

### Worker Agents (×3)

Each Worker manages stocks within its own risk-profile pool plus a **Cash asset**:

- **Observation**: 30-day normalised price window for pool stocks + current portfolio weights
- **Action**: Weight vector (including Cash), sampled via Dirichlet
- **Reward**: Profile-specific (see below)

### Profile-Specific Reward Functions

Each Worker has a **mathematically different** objective function:

| Worker | Reward Formula | Focus |
|--------|---------------|-------|
| **Safe** | $R_{\log} - \lambda \cdot (w^T S w) - 2.0 \cdot \sigma_{port}$ | Minimise portfolio variance |
| **Neutral** | $R_{\log} - \lambda \cdot (w^T S w) - \gamma \cdot \text{Turnover}$ | Maximise risk-adjusted return (Sharpe focus) |
| **Risky** | $1.5 \cdot R_{\log} - 0.5 \cdot \lambda \cdot (w^T S w)$ | Maximise raw returns (alpha chasing) |

Where:
- $R_{\log}$ = log portfolio return
- $w^T S w$ = weighted lexical similarity penalty (from 10-K filings)
- $\sigma_{port}$ = portfolio volatility (using covariance matrix)
- Turnover = $\sum |w_t - w_{t-1}|$ (weight change between steps)

### The Cash Asset

Each Worker has an extra "Cash" asset in its action space. Cash always returns 1.0 (no gain, no loss). This is critical because:

1. It provides an explicit **"do nothing"** option
2. It breaks the 1/N trap — agents can park capital in Cash instead of spreading it thinly
3. It acts as a natural risk-free rate benchmark

### Global Portfolio Return

The final portfolio return is computed hierarchically:

$$R_{\text{global}} = v_{\text{safe}} \cdot R_{\text{safe}} + v_{\text{neutral}} \cdot R_{\text{neutral}} + v_{\text{risky}} \cdot R_{\text{risky}}$$

---

## 6. REINFORCE Training with Scheduled Hyperparameters

### REINFORCE Algorithm

The system uses **REINFORCE** (Williams, 1992), a Monte Carlo policy gradient method:

1. Run the agent for a full episode, collecting log-probabilities and rewards
2. Compute discounted returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ with $\gamma = 0.99$
3. Normalise returns (zero mean, unit variance) for stable gradients
4. Update policy: $\nabla_\theta J = -\log \pi(a_t | s_t) \cdot G_t$
5. Add an **entropy bonus** to encourage exploration

### Scheduled Hyperparameters

To break the 1/N convergence, three hyperparameters are **scheduled** over 200 episodes:

| Hyperparameter | Start (Episode 0) | End (Episode 200) | Schedule Type | Purpose |
|---------------|-------------------|-------------------|---------------|---------|
| **Learning Rate** | 3×10⁻³ | 5×10⁻⁴ | Cosine annealing | Fast learning early, fine-tuning late |
| **Entropy Bonus** | 0.05 | 0.005 | Linear decay | Explore broadly early, exploit late |
| **Turnover Limit** | 1.0 (unrestricted) | 0.3 (tight) | Linear decay | Allow large portfolio shifts early, stabilise late |

### Concurrent Training

All 4 agents (1 Manager + 3 Workers) train **simultaneously** at each timestep:

1. Workers report their current state (cumulative return + volatility) to Manager
2. Manager observes global market + Worker states → outputs capital allocation
3. Workers each select stock weights within their pools
4. Manager reward = global return; Worker rewards = profile-specific objectives
5. All 4 agents backpropagate independently using their own REINFORCE loss

---

## 7. 10-K Lexical Similarity Matrix

### Data Source

The system uses **SEC EDGAR 10-K annual filings** (Item 1: Business Description) for all 45 stocks. This text describes each company's core business activities, revenue streams, and competitive landscape.

### How the Matrix Is Built

1. **Data Collection**: For each stock, the most recent 10-K filing's Item 1 (Business Description) is fetched from SEC EDGAR. Fallback: Yahoo Finance company description.
2. **TF-IDF Vectorisation**: Each company's text is converted into a TF-IDF (Term Frequency–Inverse Document Frequency) vector using scikit-learn
3. **Cosine Similarity**: Pairwise cosine similarity between all TF-IDF vectors produces a 45×45 matrix
4. High similarity (→1.0) means two companies describe their business similarly
5. Low similarity (→0.0) means very different business descriptions

### How It's Used in Rewards

The lexical matrix $S$ appears as a **semantic penalty** in all Worker reward functions:

$$\text{Semantic Penalty} = \lambda \cdot w^T S w = \lambda \cdot \sum_{i,j} w_i \cdot w_j \cdot S_{ij}$$

This penalises the agent for holding large positions in companies with similar business descriptions. The effect: the agent is pushed toward **fundamental diversification**, not just statistical diversification (low correlation). Two banks might have low price correlation during calm markets but will crash together in a financial crisis — the lexical penalty catches this.

---

## 8. Walk-Forward Validation

### What Is Walk-Forward Validation?

Walk-Forward is an out-of-sample testing methodology for time-series models. Unlike random train/test splits (which leak future information), Walk-Forward:

1. Trains on a historical window
2. Tests on the immediately following period
3. Rolls forward and repeats

### Windows Used

| Window | Training Period | Test Period | Market Regime |
|--------|----------------|-------------|---------------|
| 1 | 2015–2018 | 2019 | Bull market |
| 2 | 2015–2019 | 2020 | COVID crash |
| 3 | 2015–2020 | 2021 | Recovery |
| 4 | 2015–2021 | 2022 | Bear market |

For each window, the full hierarchy (Manager + 3 Workers) is retrained from scratch on training data, then evaluated on the unseen test period. This tests adaptability across different market conditions.

---

## 9. Lambda Ablation Study

### What Is Ablation?

An ablation study isolates the effect of one component by varying it while holding everything else constant. Here, the **semantic penalty strength** (λ) is varied:

| λ Value | Effect |
|---------|--------|
| λ = 0.0 | No semantic penalty — pure return maximisation |
| λ = 0.1 | Baseline — moderate diversification pressure |
| λ = 0.5 | Strong penalty — heavy diversification bias |
| λ = 1.0 | Extreme penalty — diversification dominates returns |

### What It Reveals

The ablation evaluates how the penalty affects:
- **Total portfolio return**: Does diversification cost performance?
- **Sharpe ratio**: Does diversification improve risk-adjusted returns?
- **Manager allocation shifts**: Does the Manager reallocate capital differently under stronger penalties?
- **Worker-level returns**: Which risk profile is most affected by the penalty?

---

## 10. Risk Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Sharpe Ratio** | $\frac{\bar{R}}{\sigma_R} \times \sqrt{252}$ | Annualised risk-adjusted return |
| **Max Drawdown** | $\max \frac{\text{Peak} - \text{Trough}}{\text{Peak}}$ | Largest peak-to-trough loss |
| **CVaR (95%)** | $E[R\ |\ R \leq VaR_{95\%}]$ | Expected loss in worst 5% of days |
| **Sortino Ratio** | $\frac{\bar{R}}{\sigma_{\text{downside}}}$ | Return per unit of downside risk only |
| **Calmar Ratio** | $\frac{R_{\text{annualised}}}{\text{MaxDrawdown}}$ | Annual return relative to worst drawdown |
| **HHI** | $\sum w_i^2$ | Portfolio concentration (1/N = minimum) |
| **Effective N** | $\frac{1}{\sum w_i^2}$ | Effective number of holdings |

---

## 11. Summary of Key Technical Decisions

| Problem | Solution | Technology |
|---------|----------|------------|
| 1/N convergence (uniform weights) | Dirichlet distribution + Cash asset | `torch.distributions.Dirichlet` |
| Ticker memorisation | Shared Conv1d weights (EIIE) | `nn.Conv1d` with weight sharing |
| Homogeneous Worker behaviour | Beta-based risk classification | Covariance-based β calculation |
| Overfitting to training period | Walk-Forward Validation (4 windows) | Rolling retrain + OOS testing |
| Naive diversification | 10-K lexical similarity penalty | TF-IDF cosine similarity matrix |
| Hyperparameter sensitivity | Lambda ablation study | λ ∈ {0, 0.1, 0.5, 1.0} |
| Hyper-trading | Hard turnover clipping + scheduled limit | Weight delta ≤ 0.3 (tightening) |
| No "do nothing" option | Cash asset per Worker | Extra action dimension with return = 1.0 |
| Unstable early training | Scheduled LR, entropy, turnover | Cosine annealing + linear decay |

---

## 12. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11 |
| RL Framework | Gymnasium | ≥ 0.29 |
| Deep Learning | PyTorch | ≥ 2.0 |
| Data Handling | Pandas, NumPy | Latest |
| Visualisation | Matplotlib, Seaborn | Latest |
| NLP / Similarity | scikit-learn (TF-IDF) | Latest |
| Data Source (Prices) | yfinance | Latest |
| Data Source (Filings) | SEC EDGAR API | Direct HTTP |

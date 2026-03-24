# Implementation Plan — 24 Feb 2026
## Hierarchical MARL Gymnasium Notebook with Semantic Diversification

### Goal
Create a **lightweight, educational** Jupyter notebook that implements the Hierarchical MARL portfolio optimization system from the research proposal, using **10 stocks** in **two comparative instances**.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Stock count | **10** (5 per instance) | Lightweight, fast to train on CPU |
| Instances | **Two**: semantically similar vs. semantically different | Proves the semantic penalty matters |
| Framework | **Gymnasium** (imported) | Standard RL env API |
| Policy network | **EIIE (PyTorch)** | Custom shared-weight architecture from Jiang et al. 2017 |
| Training | **Lightweight** REINFORCE / simple policy gradient | Runs in minutes, not hours |
| Tone | **Educational** — markdown explanations between every code cell | Helps reader understand each component |

---

## Two-Instance Experiment Design

### Instance A — "Semantically Similar" Portfolio
Pick ~5 stocks from the **same sector/business** (e.g., 5 big-tech companies that all do cloud/software).
The lexical similarity between these stocks will be **high** (~0.5+).
**Expected**: The semantic penalty heavily penalizes this portfolio → forces diversification or accepts lower reward.

### Instance B — "Semantically Diverse" Portfolio
Pick ~5 stocks from **different sectors** with genuinely different business models.
The lexical similarity between these stocks will be **low** (~0.1–0.2).
**Expected**: The semantic penalty barely fires → agent can maximize returns freely.

### Comparison
Train the **same agent architecture** on both instances. Compare:
- Raw returns
- Semantic penalty magnitude
- Portfolio concentration (HHI)
- Effective number of assets

---

## Notebook Sections

### Section 1: Setup & Data Loading
- Import: `gymnasium`, `numpy`, `pandas`, `torch`, `matplotlib`, `sklearn`
- Load existing `data/raw/top_20_prices.csv` and `data/processed/lexical_matrix_20.csv`
- Select two groups of 5 stocks (similar vs. diverse) from the 20 available
- Display summary stats + lexical similarity comparison

### Section 2: The Reward Function (Math Explained)
Full explanation with LaTeX formulas:

```
R_total = R_portfolio − λ · Σᵢⱼ(wᵢ · wⱼ · Sᵢⱼ) − γ · Σ|wₜ − wₜ₋₁|
```

- **R_portfolio** = `log(Σ wᵢ · yᵢ)` — log portfolio return
- **Weight-Aware Lexical Penalty** = `λ · Σ(wᵢ wⱼ Sᵢⱼ)` — penalizes holding semantically similar companies, weighted by capital
- **Turnover Penalty** = `γ · Σ|wₜ − wₜ₋₁|` — transaction cost proxy
- Explain λ and γ hyperparameters and their trade-offs

### Section 3: Gymnasium Environment
Custom `LexicalPortfolioEnv(gymnasium.Env)` with:
- **Observation**: Rolling price window (normalized) + portfolio-vector memory (PVM)
- **Action**: Continuous weights → softmax normalized
- **Reward**: The formula from Section 2
- **Done**: When we reach the end of the price series
- Run `gymnasium.utils.env_checker.check_env()` to validate

### Section 4: EIIE Policy Network (PyTorch)
Implement the **Ensemble of Identical Independent Evaluators** architecture:
- **Conv1D** layers with **shared weights** across all assets (the "identical" part)
- Each asset gets the same neural network applied to its price window
- **Portfolio-Vector Memory (PVM)** concatenated before the final layer
- **Softmax** output for weight allocation
- Explain WHY shared weights prevent the agent from memorizing ticker symbols

### Section 5: Training Loop
- Simple **REINFORCE (policy gradient)** training
- Train on Instance A (similar stocks)
- Train on Instance B (diverse stocks)
- Track and plot: episode reward, portfolio value, weight distributions
- Keep number of episodes small (~50–100) for fast CPU execution

### Section 6: Metrics & Comparison
Calculate for both instances:
- **CVaR (95%)**: Expected loss in worst 5% of scenarios
- **Sortino Ratio**: Return / Downside Deviation
- **HHI**: Σ wᵢ² (portfolio concentration)
- **Effective N**: 1 / Σ wᵢ² (effective diversification)
- Side-by-side bar charts comparing Instance A vs Instance B

### Section 7: Visualizations
- Lexical similarity heatmap for each instance
- Portfolio weight evolution over time
- Cumulative return curves (A vs B)
- Semantic penalty magnitude over time

---

## Verification Plan

1. **`check_env`**: Both environments pass Gymnasium's API validator
2. **Reward sanity**: Assert reward is finite, not NaN, for random actions
3. **Shape checks**: Observation/action dimensions match expected values
4. **Training convergence**: Reward should trend upward over episodes
5. **Comparison validity**: Instance B (diverse) should have lower semantic penalty than Instance A

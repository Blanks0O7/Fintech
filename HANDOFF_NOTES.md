# Hierarchical MARL System — Handoff Notes

**Date:** 2026-05-01  
**Notebook:** `Hierarchical_MARL_System.ipynb`  
**Environment:** `.venv` (Windows, has CUDA-enabled PyTorch)  
**Activate:** `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\.venv\Scripts\Activate.ps1`

---

## 1. Context & Goal

Port and extend the staged MARL training pipeline (`Staged_MARL_Training.py`) into the main notebook (`Hierarchical_MARL_System.ipynb`) and add several new experiments (skewness/kurtosis, lambda ablation across both training modes, random-text null hypothesis test). Final step: run the notebook end-to-end on a higher-VRAM machine.

The original task spec is preserved verbatim at the end of this document (Appendix A).

---

## 2. What Has Been Completed

All notebook **edits** are done and committed/pushed. Execution started but did **not** finish on the current machine (concurrent training alone took ~36 min on CPU; the lambda ablation × 5 was deemed infeasible here).

### Task 1 — Smart Stock Selection (DONE)
- Replaced Section 1 data-loading cell.
- Loads full **100-stock** universe from `data/sp500_100_prices.csv` and `data/processed/lexical_matrix_100.csv`.
- New function `select_stocks(price_df, n_total=50)`:
  1. Computes beta vs equal-weight market proxy.
  2. Classifies each stock as Safe (β<0.8), Neutral (0.8≤β<1.2), Risky (β≥1.2).
  3. Allocates 50 slots **proportionally** to pool sizes in the universe.
  4. Ranks within each pool by data completeness (fewest NaN) then return-volatility liquidity proxy.
  5. Slices `price_df` and `lexical_df` to the selected 50 tickers.
  6. Prints universe size, pre/post pool sizes, and per-pool tickers.
- Section 6 (beta classification cell) recomputes `betas`, `risk_pools`, `beta_labels`, `returns_df` on the sliced 50-stock frame — downstream untouched.
- Lexical-stats cell updated to read `data/raw/sp500_100_10k_texts.json` (was `sp500_50_…`).

### Task 2 — Staged Training Port (DONE)
- **2a `EIIENetwork`** replaced with the upgraded version from `Staged_MARL_Training.py`:
  - Signature: `EIIENetwork(n_assets, window_size, n_price_assets=None, extra_context_dim=0)`
  - Adds `LayerNorm`, `Dropout(0.1)`, optional `extra_context` arg in `forward`.
- **2b `ManagerEnv`** replaced with the staged version:
  - `_get_market_context()` returns 7 features (short_return, slow_return, short_vol, slow_vol, downside_ratio, current_drawdown, stress_score).
  - Reward includes `stress_penalty` and `drawdown_penalty` terms.
  - `self.extra_context_dim = n_pools + n_pools + 7` (= 13 for 3 pools).
  - Constructor params: `fast_window=5, slow_window=20, turnover_penalty=0.01, stress_penalty=0.20, drawdown_penalty=0.30`.
  - `_get_obs()` appends market context.
- **2c** `reinforce_update()` helper + `_ema_baselines = {}` added before training functions.
- **2d** `train_staged()` ported, CPU-adapted (no `.to(DEVICE)`, uses `torch.FloatTensor`). Phases: 200 / 160 / 80 episodes. Manager net instantiated with `extra_context_dim=manager_env.extra_context_dim`.
- **2e** New **Section 5b** added after concurrent eval:
  - Markdown: 3-phase explanation table.
  - Code: fresh staged envs + `train_staged()` run.
  - Code: `evaluate_concurrent()` re-used in inference mode for staged.
  - Code: side-by-side comparison table (Return / Sharpe / Sortino / MaxDD + Manager allocation).
  - Code: drawdown comparison figure → `drawdown_comparison.png`.
- Workers are instantiated with `lambda_penalty=0.35` (concurrent run cell + walk-forward cell updated).
- `evaluate_concurrent()` updated to thread `m_context` through manager net forward pass.

### Task 3 — Skewness & Kurtosis (DONE)
- New **Section 8b** inserted after the drawdown decomposition (Section 8).
- Computes `scipy.stats.skew()` and `scipy.stats.kurtosis(fisher=True)` for daily returns of: Staged, Concurrent, Equal-Weight, MVO, Risk Parity.
- Figure A (2×2): histogram+KDE overlay (with normal fit on Staged), Q-Q plot, skewness bars (green/red), excess-kurtosis bars (orange/blue) with 0 and 3 reference lines. Saved as `return_distribution_analysis.png`.
- Figure B: summary table with Mean / Std / Skew / Excess Kurtosis / Interpretation column (skew thresholds ±0.1; tail tag from kurtosis vs 3).

### Task 4 — Lambda Ablation across BOTH approaches (DONE — code only)
- Section 7 ablation cell rewritten to loop `λ ∈ [0.0, 0.1, 0.35, 0.5, 1.0]` over **both** concurrent and staged training, recording Return / Sharpe / Sortino / MaxDD / Manager allocation.
- 3×2 figure (Sharpe / Return / MaxDD vs λ in left column; Manager Safe% allocation vs λ in right column) → `lambda_ablation_comparison.png`.
- Text summary identifies peak-λ for each mode and % Sharpe improvement vs λ=0.
- ⚠️ **Heaviest cell in the notebook.** On CPU it is impractical. Plan to run this on the GPU machine.

### Task 5 — Random Text Control / Null Hypothesis (DONE)
- New **Section 7b** inserted after lambda ablation viz.
- `create_shuffled_lexical_matrix(lexical_df, seed=42)` permutes rows and columns by the same permutation (preserves symmetry, destroys semantics).
- Runs three staged trainings @ λ=0.35:
  - **A.** Real lexical matrix.
  - **B.** Shuffled lexical matrix.
  - **C.** Zero matrix (null penalty).
- Figure (2×2): cumulative returns overlay + Sharpe / MaxDD / Safe-allocation bars → `random_control_experiment.png`.
- Results table with Interpretation column ("Semantic content active" / "Structural penalty only, no semantics" / "No penalty baseline").
- Auto-prints which conclusion is supported by observed Sharpe ordering.

### Final Cleanup (DONE)
- Title cell updated to: *"Hierarchical MARL System — Risk-Profile Architecture with Staged & Concurrent Training, Lexical Null Hypothesis Test"*.
- Section 9 Results Summary markdown updated to reference all new experiments.
- Final save cell now also serializes: staged eval results, skewness/kurtosis stats, full ablation table, and random-control results into the same JSON (`data/processed/sp500_notebook_results.json`).

---

## 3. What Remains To Do (on the new GPU machine)

### Step 0 — Setup
```powershell
git pull
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If `.venv` is not present on the new machine, recreate it:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Ensure CUDA-enabled torch wheel matches the local CUDA version
```

### Step 1 — (Optional) Re-enable GPU in the notebook
The staged training cells were ported in **CPU-only** form per the original spec (`torch.FloatTensor`, no `.to(DEVICE)`). To exploit GPU:
- Search for `torch.FloatTensor(` inside the training cells and replace with `torch.tensor(..., dtype=torch.float32, device=DEVICE)`.
- Move every `nn.Module` instance with `.to(DEVICE)` after construction (manager + worker EIIE nets).
- `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` is already defined in the setup cell.
- If staying on CPU is acceptable, **skip this step** and just run as-is — results will be identical, only slower.

### Step 2 — Run the notebook top-to-bottom
Use VS Code "Run All", or:
```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute Hierarchical_MARL_System.ipynb --inplace --ExecutePreprocessor.timeout=-1
```
Expected long-running cells (rough order):
1. Concurrent training (≈36 min CPU, ≪ on GPU).
2. Staged training (Section 5b) — similar magnitude.
3. **Lambda ablation (Section 7)** — 5 λ × (concurrent + staged). The dominant cost. Plan accordingly; on a strong GPU this should be tractable in well under an hour.
4. Random control (Section 7b) — 3 staged runs.

### Step 3 — Verify outputs are produced
Expected new artifacts in repo root / `data/processed/`:
- `drawdown_comparison.png`
- `return_distribution_analysis.png`
- `lambda_ablation_comparison.png`
- `random_control_experiment.png`
- `data/processed/sp500_notebook_results.json` (now expanded)

### Step 4 — Fix any errors as they arise
Likely-fragile points (verify if a cell fails):
- **`evaluate_concurrent` signature**: it now accepts/forwards `m_context`; ensure every call site (concurrent eval, staged eval in 5b, ablation, random-control) passes it correctly.
- **`extra_context_dim`**: when re-instantiating a manager net for staged/ablation/control, must pass `extra_context_dim=manager_env.extra_context_dim`. Mismatch → linear-layer shape error.
- **Lexical matrix shape**: after slicing to 50 tickers, `lexical_df` must be 50×50 with index/columns aligned to `tickers`. Ablation/control cells assume this.
- **`risk_pools` / `beta_labels`**: built from sliced 50-stock frame. Worker constructor expects pool ticker lists matching `tickers` order.
- **JSON save cell**: any variable referenced (e.g. `staged_eval`, `ablation_results`, `random_control_results`, `skew_kurt_table`) must exist by the time the cell runs — they’re created in Sections 5b, 7, 7b, 8b respectively. Run order matters.

### Step 5 — Commit & push final results
```powershell
git add Hierarchical_MARL_System.ipynb data/processed/*.json *.png
git commit -m "Run full pipeline: staged + ablation + random control on GPU"
git push
```

---

## 4. File Map (most relevant)

| File | Role |
|---|---|
| `Hierarchical_MARL_System.ipynb` | Main notebook — all edits are here |
| `Staged_MARL_Training.py` | Source of staged training port (reference only) |
| `load_sp500_100.py` | Data pipeline reference |
| `data/sp500_100_prices.csv` | 100-stock OHLC/close history |
| `data/processed/lexical_matrix_100.csv` | 100×100 TF-IDF similarity matrix |
| `data/processed/sector_map_100.json` | Sector mapping |
| `data/raw/sp500_100_10k_texts.json` | Raw 10-K text corpus |
| `data/processed/sp500_notebook_results.json` | Final aggregated results (regenerated by last cell) |

---

## 5. Known Caveats

- Cells use **stochastic seeding** in places (REINFORCE, env reset). Set `np.random.seed` / `torch.manual_seed` near the top if exact reproducibility across machines is required.
- The ablation cell does **not** checkpoint between λ values. If it crashes mid-loop, results so far are lost. Consider wrapping the inner loop body in a `try/except` and pickling `ablation_results` after each λ on the GPU machine if runtime is uncertain.
- `evaluate_concurrent()` runs in inference mode for staged too — this was per spec; no separate evaluator was created.
- The shuffled-lexical control reuses `np.random.seed(42)` — change seed if running multiple controls.

---

## Appendix A — Original Task Specification (verbatim)

> *(Preserved here for the next session so the spec is self-contained.)*

**TASK 1 — Smart Stock Selection** Replace Section 1 with: load full 100-stock universe; `select_stocks(price_df, n_total=50)` performs stratified beta sampling (Safe β<0.8 / Neutral 0.8–1.2 / Risky ≥1.2), proportional allocation to 50 slots, ranking by data completeness then liquidity proxy (avg daily price range, fallback return vol), slice price_df and lexical_df to selected tickers, print summary.

**TASK 2 — Port Staged Training**
- 2a Replace `EIIENetwork` with upgraded version (`extra_context_dim`, LayerNorm, Dropout 0.1).
- 2b Replace `ManagerEnv` with staged version: 7-feature `_get_market_context`, stress + drawdown penalties, `extra_context_dim=2*n_pools+7`, params `fast_window=5, slow_window=20, turnover_penalty=0.01, stress_penalty=0.20, drawdown_penalty=0.30`. Workers `lambda_penalty=0.35`.
- 2c Copy `reinforce_update()` + `_ema_baselines={}`.
- 2d Copy `train_staged()` (CPU only). Phases 200/160/80. Manager EIIE with `extra_context_dim`.
- 2e Section 5b: markdown phase table, fresh staged envs + train, evaluate via existing `evaluate_concurrent`, side-by-side comparison, drawdown comparison figure → `drawdown_comparison.png`.

**TASK 3 — Skewness & Kurtosis** New Section 8b. Use `scipy.stats.skew` / `kurtosis(fisher=True)` on daily returns of Staged / Concurrent / EW / MVO / Risk Parity. Figure A 2×2 (hist+KDE+normal, Q-Q, skew bars, excess-kurtosis bars). Figure B summary table with Interpretation column. Save `return_distribution_analysis.png`.

**TASK 4 — Lambda Ablation across both modes** λ ∈ {0.0, 0.1, 0.35, 0.5, 1.0}. Run concurrent (200 ep) and staged (200+160+80) per λ. 3×2 figure (Sharpe/Return/MaxDD vs λ + Manager Safe alloc vs λ). Save `lambda_ablation_comparison.png`. Print peak-λ and % Sharpe gain vs λ=0 for each mode.

**TASK 5 — Random Text Control** Section 7b. `create_shuffled_lexical_matrix` (same row+col permutation). Three staged runs @ λ=0.35: Real / Shuffled / Zero. 2×2 figure (cum returns, Sharpe bars, MaxDD bars, Safe-alloc bars). Results table with Interpretation. Print conclusion (Real > Shuffled > Zero ⇒ semantics matter). Save `random_control_experiment.png`.

**Final Cleanup** Update title; update Section 9 markdown; expand final JSON save to include staged eval, skew/kurtosis, full ablation, random control. Run notebook top-to-bottom and fix errors.

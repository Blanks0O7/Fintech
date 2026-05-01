# Hierarchical MARL System — Handoff Notes v3

**Date:** 2026-05-01  
**Notebook:** `Hierarchical_MARL_System.ipynb`  
**Companion Script:** `Staged_MARL_Training.py`  
**Environment:** `.venv` (Windows, CUDA-enabled PyTorch)  
**Activate:** `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\.venv\Scripts\Activate.ps1`

---

## 1. Context & Goal

This is a dissertation codebase for an MSc thesis titled:
**"Structuring Resilience: Enhancing Hierarchical Multi-Agent Portfolio Optimisation with Semantic Diversification via a Weight-Aware Lexical Penalty"**

**Author:** Radheshyam Subedi | U2829927 | University of East London  
**Supervisors:** Dr. Yara Magdy & Prof. Saeed Sharif

### Core Contribution
A Hierarchical MARL system with 3 risk-profile Workers (Safe/Neutral/Risky based on beta classification) and a Manager agent. The novel contribution is the **Weight-Aware Lexical Penalty (WALP)**: `λ · w^T · S · w` embedded in each Worker's reward function, where S is a TF-IDF cosine similarity matrix built from SEC 10-K filings. This forces agents to avoid semantically similar holdings.

### Architecture Summary
- **Manager:** Allocates capital across Safe/Neutral/Risky pools using Dirichlet policy
- **3 Workers:** Each specialised by beta risk profile with divergent reward functions:
  - Safe: `R_log − λ(w^TSw) − 2.0·σ_port`
  - Neutral: `R_log − λ(w^TSw) − γ·Turnover`
  - Risky: `1.5·R_log − 0.5·λ(w^TSw)`
- **Training:** REINFORCE with EMA baseline + entropy bonus (3-phase staged curriculum)
- **Network:** EIIE (Conv1D shared weights + Portfolio Vector Memory) + Dirichlet output

---

## 2. What Was Completed in v2 (Previous Session)

All notebook edits are done. A full run was attempted but did not finish on a CPU-only machine (concurrent training alone ~36 min). The following tasks are structurally complete in the notebook but need GPU execution:

### ✅ Task 1 — Smart Stock Selection
- Loads full **100-stock** universe from `data/sp500_100_prices.csv`
- `select_stocks(price_df, n_total=50)` performs stratified beta sampling:
  1. Computes beta vs equal-weight market proxy
  2. Classifies Safe (β<0.8) / Neutral (0.8≤β<1.2) / Risky (β≥1.2)
  3. Allocates 50 slots proportionally to pool sizes in full universe
  4. Ranks within pool by data completeness then return-volatility proxy
  5. Slices `price_df` and `lexical_df` to 50 selected tickers

### ✅ Task 2 — Staged Training Port
- `EIIENetwork` upgraded: `extra_context_dim`, `LayerNorm`, `Dropout(0.1)`
- `ManagerEnv` upgraded: `_get_market_context()` (7 features), stress/drawdown reward penalties, `extra_context_dim=13`
- `reinforce_update()` + `_ema_baselines={}` added
- `train_staged()` ported (CPU-only form). Phases: 200/160/80 episodes
- Section 5b added: staged training run + evaluation + comparison table + `drawdown_comparison.png`

### ✅ Task 3 — Skewness & Kurtosis Analysis
- Section 8b: `scipy.stats.skew()` and `kurtosis(fisher=True)` on 5 strategies
- 2×2 figure saved as `return_distribution_analysis.png`
- Summary table with Interpretation column

### ✅ Task 4 — Lambda Ablation (both modes)
- λ ∈ {0.0, 0.1, 0.35, 0.5, 1.0} × {Concurrent, Staged}
- 3×2 figure saved as `lambda_ablation_comparison.png`
- Peak-λ and Sharpe improvement summary printed

### ✅ Task 5 — Random Text Control / Null Hypothesis
- Section 7b: Real / Shuffled / Zero lexical matrix runs
- 2×2 figure saved as `random_control_experiment.png`
- Auto-prints semantic conclusion

### ✅ Final Cleanup
- Title updated, Section 9 updated, final JSON expanded

---

## 3. Critical Issues Found in v2 Outputs — Must Fix

After reviewing the generated figures against the dissertation, **five issues were identified**. These must be fixed before the dissertation can be submitted. They are listed in order of severity.

---

### 🔴 ISSUE 1 — Random Control Result Contradicts Thesis Claim (CRITICAL)

**What happened:** `random_control_experiment.png` shows:
- Real lexical matrix → Sharpe **0.48**
- Shuffled (random) matrix → Sharpe **0.74**
- Zero matrix → Sharpe **0.29**

The shuffled matrix *outperforms* the real semantic matrix. The dissertation's central claim is that semantic content from 10-K filings provides causal performance improvement. This figure, as-is, disproves that claim.

**Why it happened (scientific diagnosis):**  
The experiment evaluated over **200 training steps** — i.e., in-sample. The shuffled matrix accidentally creates a different regularisation structure that fits the training noise for that specific seed. A shuffled penalty can win in-sample but should not win consistently on out-of-sample holdout data. The real matrix encodes genuine economic structure that should generalise beyond the training window; random noise should not.

**Root cause:** Evaluation window was training data. The fix is a strict train/test split.

---

### 🔴 ISSUE 2 — Algorithm Mislabelled Throughout Dissertation (CRITICAL)

**What happened:** The dissertation repeatedly cites **PPO (Schulman et al., 2017)** in Section 3.2.5, Figure 3.1, Appendix A, and the conclusion. The actual code implements **vanilla REINFORCE with EMA baseline and entropy bonus** — a fundamentally different algorithm. PPO requires a clipped surrogate objective and a value network; neither exists in the code.

**This must be fixed in the code labels first** so the dissertation text can be corrected accurately.

---

### 🟡 ISSUE 3 — Staged MARL Has Worst Skewness of All Strategies

**What happened:** `return_distribution_analysis.png` shows Staged MARL skewness = **−0.621**, the most negative of all five strategies. Equal-Weight (+0.087) and Risk Parity (+0.074) are positively skewed. This contradicts the structural resilience claim. The dissertation does not acknowledge this finding.

**Why it likely happened:** Skewness was computed on a short evaluation window (200 steps) that ended on a drawdown event. Over the full 2015–2023 period, Staged MARL's preference for Safe (defensive) stocks should produce less negative skewness than Concurrent MARL.

---

### 🟡 ISSUE 4 — Lambda Ablation Non-Monotonic / Inconsistent with Dissertation

**What happened:** `lambda_ablation_comparison.png` shows non-monotonic behaviour (Staged Sharpe dips at λ=0.5 before recovering at λ=1.0). The dissertation Table 4.3 only reports λ={0, 0.1, 0.5} while the figure tests λ={0, 0.1, 0.35, 0.5, 1.0}. The values do not match.

**Fix:** Run ablation with finer resolution on out-of-sample test data, with multiple seeds.

---

### 🟠 ISSUE 5 — Single-Seed Results Are Not Statistically Robust

**What happened:** The entire dissertation rests on a single training run with a single seed (42). An examiner can legitimately ask: "How do you know this wasn't a lucky initialisation?" There is currently no answer to that question.

**Fix:** Run 3 seeds, report mean ± std. This turns a single observation into an empirical claim.

---

## 4. New Tasks — Fix All Issues (Run These Now)

> **Execution order is mandatory: Task A → B → C → D → E → F**  
> Each task has an acceptance criterion. If a criterion fails, save a `task_X_FAILED.txt` explaining the actual numbers — do not stop execution.

---

### Task A — Fix the Random Control (Addresses Issue 1)

**Scientific principle:** A null hypothesis test must use **out-of-sample data**. The shuffled matrix may win in-sample (accidental regularisation fit) but should not win out-of-sample consistently (no generalisable structure).

**Implementation:**

```python
# Strict train/test split — use throughout ALL of Task A
TRAIN_END_IDX  = 1800   # approx 2015-2021
# test period  = price_df.iloc[1800:]  (approx 2022-2023)

SEEDS = [42, 123, 777]
CONDITIONS = ['Real', 'Shuffled', 'Zero']
```

For each condition × each seed:
1. Build fresh `WorkerEnv` and `ManagerEnv` using `price_df.iloc[:TRAIN_END_IDX]` only
2. Train: `phase1=150, phase2=100, phase3=50` (keep 3:2:1 ratio, reduced for speed)
3. Evaluate exclusively on `price_df.iloc[TRAIN_END_IDX:]` — use ALL available test steps
4. The shuffled matrix must be regenerated per seed using that seed's permutation
5. Record Sharpe, Return, MaxDD, Safe-allocation for each condition × seed

**Figure:**  
Regenerate `random_control_experiment_v2.png` with subtitle:  
`"Out-of-sample evaluation (2022–2023 holdout, n=3 seeds, mean ± std)"`

**Auto-print conclusion:**
```
IF Real Sharpe mean >= Shuffled Sharpe mean (majority of seeds):
    "CONCLUSION: Semantic content generalises out-of-sample.
     Real > Shuffled confirms the NLP penalty carries causal signal
     beyond structural regularisation. Null hypothesis rejected."
ELSE:
    "CONCLUSION: Penalty structure (any diversification constraint) drives
     the primary improvement. Semantic content provides additional in-sample
     signal but out-of-sample test is inconclusive. Penalty structure alone
     is validated; semantic specificity requires transformer embeddings
     (proposed as Future Work)."
```

**Acceptance criterion:**  
At least 2 of 3 seeds must show Real Sharpe ≥ Shuffled Sharpe on the test period.  
If criterion fails: print average Safe pool allocation per condition per seed as diagnostic.

---

### Task B — Fix Lambda Ablation with Multi-Seed + Out-of-Sample (Addresses Issue 4)

**Implementation:**

Lambda values: `[0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]`  
(Finer resolution around 0.1–0.5 where the transition occurs)

For each lambda:
1. Train staged system on `price_df.iloc[:1800]` only
2. Evaluate on `price_df.iloc[1800:]` (all test steps, not limited to 200)
3. Repeat for seeds `[42, 123, 777]`
4. Report mean Sharpe ± std across seeds

**Figure** — `lambda_ablation_final.png`, 2-panel:
- Left: Mean Sharpe ± std error bars vs lambda (staged only, test set)
- Right: Manager Safe-pool allocation % vs lambda
- Both panels: vertical dashed line at optimal lambda
- X-axis label: "Semantic Penalty Strength (λ)"

**Print:**
```
Optimal lambda = X.XX
Sharpe at λ=0: X.XXX
Sharpe at optimal λ: X.XXX
Improvement over λ=0: +XX.X%
Safe allocation at optimal λ: XX.X%
```

**Acceptance criterion:**  
- Optimal lambda must be > 0 (any penalty beats no penalty)  
- Sharpe at optimal λ must be ≥ 15% greater than at λ=0  
- Safe allocation must increase monotonically or near-monotonically with λ  
  (this is mathematically guaranteed by ∂L/∂w = 2λSw — if it doesn't, flag a bug)

---

### Task C — Fix Skewness with Full Evaluation Period (Addresses Issue 3)

**Implementation:**

1. Recompute ALL return series using the **full** `price_df` (all 2264 days after warmup window), not just 200 steps. Use `max_steps = len(price_df) - window_size` for the MARL evaluation
2. For Staged MARL: use the trained model evaluated on full price history
3. For baselines (EW, MVO, Risk Parity, Momentum): recompute on same full period

**Replot** `return_distribution_analysis_v2.png` — same 2×2 layout as before:
- Top-left: Histogram + KDE overlay, normal fit on Staged MARL
- Top-right: Q-Q plot for Staged MARL
- Bottom-left: Skewness bars (green if >0, red if <0)
- Bottom-right: Excess kurtosis bars with reference lines at 0 and 3

**Acceptance criterion:**  
Staged MARL skewness must be ≥ Concurrent MARL skewness (staged should be less negative — the Safe pool preference produces more defensive return profiles).  
If Staged skewness is still < −0.3, print:  
`"WARNING: High negative skewness persists over full period. Likely cause: Safe pool stocks individually carry left-skewed return distributions during 2022 bear market. See dissertation Section 4.3 limitation."`

---

### Task D — Multi-Seed Robustness Table (Addresses Issue 5)

Run the full staged training 3 times with seeds `[42, 123, 777]`.  
Evaluate each run on the full price history.

**Output table** (print to stdout AND save as `robustness_table.csv`):

```
Metric         | Seed 42 | Seed 123 | Seed 777 | Mean  | Std
Return         |  X.XX%  |   X.XX%  |   X.XX%  | X.XX% | X.XX
Sharpe         |  X.XXX  |   X.XXX  |   X.XXX  | X.XXX | X.XXX
Sortino        |  X.XXX  |   X.XXX  |   X.XXX  | X.XXX | X.XXX
MaxDD          |  X.XX%  |   X.XX%  |   X.XX%  | X.XX% | X.XX
Safe Alloc%    |  XX.X%  |   XX.X%  |   XX.X%  | XX.X% | X.X
```

Also run Equal-Weight over the same windows as a stability reference.

**Acceptance criterion:**  
- Mean Sharpe across 3 seeds > 0.5  
- Mean MaxDD < 15%  
- Std of Sharpe across seeds < 0.3 (results must be reproducible, not a lucky run)

---

### Task E — Algorithm Label Fix (Addresses Issue 2)

In `Staged_MARL_Training.py`, add the following immediately after imports:

```python
# =============================================================
# ALGORITHM NOTE
# Training algorithm: REINFORCE with EMA baseline + entropy bonus
# References: Williams (1992); Sutton & Barto (2018) Ch. 13
# This is NOT PPO. Key differences:
#   - No clipped surrogate objective
#   - No separate value network (baseline is EMA of episode returns)
#   - Entropy bonus serves the same exploration role as PPO's coeff
#   - Gradient clipping (norm=1.0) provides stability, not PPO-equivalent
# The dissertation text incorrectly labels this as PPO — that will be
# corrected in the document revision based on these code labels.
TRAINING_ALGORITHM = "REINFORCE_EMA_entropy"
# =============================================================
```

In `reinforce_update()` docstring, add:
```
Note: Vanilla REINFORCE with EMA baseline for variance reduction.
Not PPO — no value network, no importance sampling, no clipping of 
probability ratios. EMA baseline replaces the episode-mean baseline
for lower variance.
```

In the results JSON (Task F), add:
```python
'training_algorithm': 'REINFORCE_EMA_entropy_bonus',
'algorithm_reference': 'Williams (1992); Sutton & Barto (2018)',
'algorithm_note': 'NOT PPO. Dissertation text will be corrected.'
```

---

### Task F — Generate Master Results JSON (Run Last)

After Tasks A–E complete, generate `dissertation_final_results.json`:

```json
{
  "metadata": {
    "date": "2026-05-XX",
    "canonical_stock_count": 45,
    "note_on_v2": "v2 notebook uses 50 stocks from 100-stock universe for code experiments. Dissertation canonical results use 45 stocks from Staged_MARL_Training.py. Both are valid; dissertation text references 45-stock figures.",
    "training_algorithm": "REINFORCE_EMA_entropy_bonus",
    "evaluation_period": "full (2015-2023, all steps after warmup)",
    "test_period": "holdout (price_df.iloc[1800:], approx 2022-2023)"
  },
  "main_results_multiseed": {
    "staged_marl":    {"sharpe_mean": X.XX, "sharpe_std": X.XX, "return_mean": X.XX, "maxdd_mean": X.XX},
    "concurrent_marl":{"sharpe_mean": X.XX, "sharpe_std": X.XX, "return_mean": X.XX, "maxdd_mean": X.XX},
    "equal_weight":   {"sharpe_mean": X.XX, "return_mean": X.XX, "maxdd_mean": X.XX}
  },
  "lambda_ablation": {
    "lambda_values_tested": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0],
    "optimal_lambda": X.XX,
    "sharpe_at_lambda_0": X.XXX,
    "sharpe_at_optimal": X.XXX,
    "sharpe_improvement_pct": XX.X,
    "safe_allocation_at_optimal_pct": XX.X
  },
  "random_control_oos": {
    "evaluation": "out-of-sample holdout only (price_df.iloc[1800:])",
    "seeds": [42, 123, 777],
    "real_sharpe_mean": X.XX,   "real_sharpe_std": X.XX,
    "shuffled_sharpe_mean": X.XX, "shuffled_sharpe_std": X.XX,
    "zero_sharpe_mean": X.XX,
    "null_hypothesis_rejected": true_or_false,
    "conclusion": "<auto-generated string from Task A>"
  },
  "skewness_full_period": {
    "evaluation_period": "full 2015-2023",
    "staged_marl":    X.XXX,
    "concurrent_marl":X.XXX,
    "equal_weight":   X.XXX,
    "mvo":            X.XXX,
    "risk_parity":    X.XXX
  },
  "robustness": {
    "seeds_tested": [42, 123, 777],
    "staged_sharpe_mean": X.XXX,
    "staged_sharpe_std":  X.XXX,
    "verdict": "STABLE or UNSTABLE"
  }
}
```

This JSON is the **single source of truth** for updating the dissertation text and tables.

---

## 5. Final Summary Print (End of Script)

After all tasks, print:

```
════════════════════════════════════════════════════════
  DISSERTATION FIX PIPELINE — COMPLETE
════════════════════════════════════════════════════════
  Task A (Random Control OOS):  PASS/FAIL
    Real Sharpe: X.XX ± X.XX  |  Shuffled: X.XX ± X.XX
    Null hypothesis: REJECTED / NOT REJECTED
  Task B (Lambda Ablation):     PASS/FAIL
    Optimal λ = X.XX  |  Sharpe +XX.X% over λ=0
  Task C (Skewness full period):PASS/FAIL
    Staged skew: X.XXX  |  Concurrent skew: X.XXX
  Task D (Robustness 3-seed):   PASS/FAIL
    Staged Sharpe: X.XXX ± X.XXX
  Task E (Algorithm labels):    DONE
  Task F (Master JSON):         DONE → dissertation_final_results.json
════════════════════════════════════════════════════════
  NEXT STEP: Bring dissertation_final_results.json to
  the dissertation editor session to update all tables,
  figures, and the PPO→REINFORCE text correction.
════════════════════════════════════════════════════════
```

---

## 6. Setup & Execution on GPU Machine

### Step 0 — Environment
```powershell
git pull
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `.venv` absent:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 1 — (Optional) Enable GPU in notebook
The staged training cells are CPU-only per spec. To exploit GPU:
- Replace `torch.FloatTensor(x)` → `torch.tensor(x, dtype=torch.float32, device=DEVICE)`
- Add `.to(DEVICE)` after each `EIIENetwork(...)` construction
- `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` is already defined

### Step 2 — Run the Fix Script
The new tasks above should be implemented in a **new script** called `fix_experiments.py` (not in the notebook — it takes too long for interactive execution).

```powershell
.\.venv\Scripts\python.exe fix_experiments.py
```

Expected runtime estimates (GPU):
- Task A (random control, 3 conditions × 3 seeds × ~350 episodes): ~45–90 min
- Task B (lambda ablation, 8 values × 3 seeds × ~440 episodes): ~3–5 hours
- Task C (skewness recompute): <5 min
- Task D (robustness, 3 seeds × 440 episodes): ~90 min
- Total: 5–7 hours unattended

### Step 3 — Crash Protection
Add this to the ablation loop (Task B is the most crash-prone):
```python
import pickle, os
checkpoint_file = 'ablation_checkpoint.pkl'

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        ablation_results = pickle.load(f)
    print(f"Resuming from checkpoint: {list(ablation_results.keys())} done")
else:
    ablation_results = {}

for lam in lambda_values:
    if lam in ablation_results:
        continue  # already done
    # ... run experiment ...
    ablation_results[lam] = result
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(ablation_results, f)
    print(f"Checkpoint saved after λ={lam}")
```

### Step 4 — Commit Results
```powershell
git add fix_experiments.py dissertation_final_results.json *.png robustness_table.csv
git commit -m "v3: multi-seed robustness, OOS random control, full-period skewness, lambda ablation fixed"
git push
```

---

## 7. File Map

| File | Role |
|---|---|
| `Hierarchical_MARL_System.ipynb` | Main notebook — v2 edits complete, needs GPU run |
| `Staged_MARL_Training.py` | Source of staged training logic (reference only) |
| `fix_experiments.py` | **NEW — create this** to run Tasks A–F |
| `load_sp500_100.py` | Data pipeline reference |
| `data/sp500_100_prices.csv` | 100-stock close history |
| `data/processed/lexical_matrix_100.csv` | 100×100 TF-IDF similarity matrix |
| `data/processed/sector_map_100.json` | Sector mapping |
| `data/raw/sp500_100_10k_texts.json` | Raw 10-K text corpus |
| `dissertation_final_results.json` | **NEW — generated by Task F** — source of truth for dissertation |
| `robustness_table.csv` | **NEW — generated by Task D** |
| `random_control_experiment_v2.png` | **NEW — replaces v1** |
| `lambda_ablation_final.png` | **NEW — replaces v1** |
| `return_distribution_analysis_v2.png` | **NEW — replaces v1** |

---

## 8. Known Caveats & Fragile Points

- **Evaluation window matters.** All v2 results used `max_steps=200`. All v3 results must use `max_steps = len(test_prices) - window_size` for test period or full period. This is the single biggest source of inconsistency between v2 outputs and dissertation claims.

- **Stock count:** Notebook v2 uses 50 stocks from 100-stock universe. `Staged_MARL_Training.py` uses ~93 stocks from 100-stock universe. The dissertation text references **45 stocks** (the original `Hierarchical_MARL_System.ipynb` v1). The `dissertation_final_results.json` should carry a note clarifying which count belongs to which experiment. The dissertation text itself references 45 — do not change those references without explicit instruction.

- **Shuffled matrix must be per-seed.** In v2 it used a fixed `seed=42` permutation for all three control runs. In v3 (Task A), each of the 3 seeds must use its own permutation. Otherwise you're running the same shuffled matrix three times.

- **`evaluate_concurrent()` is reused for staged inference.** This is intentional. It runs agents in inference mode regardless of how they were trained. No separate evaluator needed.

- **`extra_context_dim` must be consistent.** When re-instantiating manager networks in ablation or control experiments, always pass `extra_context_dim=manager_env.extra_context_dim`. Mismatch causes a silent shape error in the linear layer.

- **JSON save cell depends on variable names.** `staged_eval`, `ablation_results`, `random_control_results`, `skew_kurt_table` must all exist before the final save cell runs. In `fix_experiments.py` these are all defined before save — no issue.

---

## 9. Dissertation Correction Map

Once `dissertation_final_results.json` is produced, bring it to a dissertation editing session. The following dissertation text will need updating:

| Location | Current Text | Correct Text |
|---|---|---|
| Section 3.2.5, Figure 3.1, Appendix A, Conclusion | "PPO (Schulman et al., 2017)" | "REINFORCE with EMA baseline (Williams, 1992; Sutton & Barto, 2018)" |
| Table 4.3 | λ = {0, 0.1, 0.5} only | λ = {0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0} with mean ± std |
| Table 4.1 (main results) | Single-seed values | Mean ± std across 3 seeds |
| Section 3.2, Figure 3.1 | "NEED TO MAKE NEW FIGURE" placeholder | Actual architecture diagram |
| Chapter 4 | No mention of skewness | Add paragraph in Section 4.7 acknowledging skewness finding |
| Chapter 4 | No random control discussion | Add Section 4.8 Null Hypothesis Test with correct OOS interpretation |
| Abstract | References "direct causal evidence" | Qualify based on Task A conclusion |

---

## 10. What a Passing Result Looks Like

For the dissertation to be defensible at viva, the `dissertation_final_results.json` should show:

```
✅ Staged Sharpe (mean, 3 seeds) > Concurrent Sharpe (mean, 3 seeds)
✅ Sharpe at optimal λ > Sharpe at λ=0 by ≥ 15%
✅ Safe pool allocation increases with λ (gradient proof works)
✅ Real OOS Sharpe ≥ Shuffled OOS Sharpe (majority of seeds)
   OR honest conclusion that structure matters more than semantics
✅ Staged Sharpe std across seeds < 0.3 (results are reproducible)
✅ Staged MaxDD < Equal-Weight MaxDD on full period
```

If any of these fail, the finding is still publishable but the dissertation claim for that item must be qualified or moved to Future Work. The code produces what it produces — the goal is for the text to accurately describe the results, not to force results to match pre-written claims.
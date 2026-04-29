"""
Walk-Forward Production Retrain
================================
Trains a fresh Staged-MARL model every year on an expanding window,
evaluates the next year out-of-sample (with transaction costs and the
defensive stress overlay already baked into the env), then advances.

This is the proper production setup: no single frozen model — markets
drift, so the model is retrained as new data arrives.

Outputs:
  - checkpoints/wf_<YEAR>.pt          fresh model per fold
  - checkpoints/staged_marl_full.pt   = the latest fold (used by
                                        realmarket_backtest.py)
  - data/processed/walk_forward_oos.csv   stitched OOS daily returns
  - data/processed/walk_forward_summary.json
"""

import os, json, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
import torch

# Reuse everything from the main training file (same envs, nets, helpers)
import importlib.util, sys
_SPEC = importlib.util.spec_from_file_location("smt", os.path.join(os.path.dirname(__file__), "Staged_MARL_Training.py"))
# Avoid re-running its __main__ block by injecting a stub flag — simplest
# guard: we import only the needed symbols by execing the upper portion.
# But Staged_MARL_Training.py runs everything top-level. Easier: factor out.
# Instead, re-implement the small bits we need without re-running training.

import numpy as np, pandas as pd, json, os, time
import torch, torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Re-import the env classes / training fn from the main script in a
# safe way: we copy the module's source then strip the bottom main block.
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Staged_MARL_Training.py")
with open(_PATH, "r", encoding="utf-8") as f:
    _SRC = f.read()
# Cut everything after the MAIN EXECUTION marker so importing has no
# side-effects beyond defining classes/functions.
_MARKER = "# MAIN EXECUTION"
_idx = _SRC.find(_MARKER)
if _idx == -1:
    raise RuntimeError("Could not find MAIN EXECUTION marker in Staged_MARL_Training.py")
_SAFE_SRC = _SRC[:_idx]
_ns = {"__name__": "smt_lib", "__file__": _PATH}
exec(compile(_SAFE_SRC, _PATH, "exec"), _ns)

WorkerEnv          = _ns["WorkerEnv"]
ManagerEnv         = _ns["ManagerEnv"]
EIIENetwork        = _ns["EIIENetwork"]
make_worker_envs   = _ns["make_worker_envs"]
make_manager_env   = _ns["make_manager_env"]
train_staged       = _ns["train_staged"]
evaluate_system    = _ns["evaluate_system"]
equal_weight_baseline = _ns["equal_weight_baseline"]
compute_cumulative_return = _ns["compute_cumulative_return"]
compute_sharpe     = _ns["compute_sharpe"]
compute_sortino    = _ns["compute_sortino"]
compute_max_drawdown = _ns["compute_max_drawdown"]
CHECKPOINT_DIR     = _ns["CHECKPOINT_DIR"]
TRANSACTION_COST_BPS = _ns["TRANSACTION_COST_BPS"]

# ---------------- Data ----------------
price_df = pd.read_csv("data/sp500_100_prices.csv", index_col=0, parse_dates=True)
lexical_df = pd.read_csv("data/processed/lexical_matrix_100.csv", index_col=0)
tickers = list(price_df.columns)
returns_df = price_df.pct_change().dropna()
market_returns = returns_df.mean(axis=1)
market_var = market_returns.var()
betas = {t: returns_df[t].cov(market_returns) / (market_var + 1e-10) for t in tickers}
risk_pools = {'Safe': [], 'Neutral': [], 'Risky': []}
for t, b in betas.items():
    if b < 0.8:    risk_pools['Safe'].append(t)
    elif b <= 1.2: risk_pools['Neutral'].append(t)
    else:          risk_pools['Risky'].append(t)

print("=" * 70)
print(" WALK-FORWARD PRODUCTION RETRAIN")
print("=" * 70)
print(f"Device: {DEVICE} | Stocks: {len(tickers)} | Range: {price_df.index[0].date()} -> {price_df.index[-1].date()}")
print(f"Transaction cost: {TRANSACTION_COST_BPS} bps per turnover unit")

# ---------------- Walk-forward folds ----------------
# Annual retrain: train on all data up to Jan-1-Y, evaluate on year Y.
years = sorted({d.year for d in price_df.index})
folds = []
first_train_end_year = years[0] + 3   # at least 3 yrs of training to start
for y in years:
    if y < first_train_end_year: continue
    train_end = pd.Timestamp(f"{y}-01-01")
    test_end  = pd.Timestamp(f"{y+1}-01-01")
    train = price_df.loc[:train_end]
    test  = price_df.loc[train_end:test_end]
    if len(train) < 200 or len(test) < 30:
        continue
    folds.append((y, train, test))

print(f"\nFolds: {len(folds)} (years {folds[0][0]} → {folds[-1][0]})")

oos_daily_returns = []   # stitched
oos_dates = []
fold_summaries = []
last_models = None        # (mgr_net, w_nets, mgr_env_for_layout, worker_envs_for_layout)
t0 = time.time()

for y, train_prices, test_prices in folds:
    fold_t = time.time()
    print(f"\n--- Fold {y}: train≤{train_prices.index[-1].date()}  test {test_prices.index[0].date()}→{test_prices.index[-1].date()} ---")

    tr_w_envs = make_worker_envs(train_prices, lexical_df, risk_pools)
    tr_m_env  = make_manager_env(train_prices, lexical_df, risk_pools, worker_envs_ref=tr_w_envs)

    # Modest episode count per fold; the cumulative compute is what matters
    mgr_net, w_nets, _ = train_staged(
        tr_m_env, tr_w_envs,
        phase1_episodes=140, phase2_episodes=100, phase3_episodes=50,
        verbose=False
    )

    te_w_envs = make_worker_envs(test_prices, lexical_df, risk_pools)
    te_m_env  = make_manager_env(test_prices, lexical_df, risk_pools, worker_envs_ref=te_w_envs)
    ev = evaluate_system(te_m_env, te_w_envs, mgr_net, w_nets, max_steps=400)

    rets = ev['global_returns']
    # Stitch with proper dates
    test_idx = test_prices.index[te_w_envs[next(iter(te_w_envs))].window_size : te_w_envs[next(iter(te_w_envs))].window_size + len(rets)]
    oos_daily_returns.extend(list(rets))
    oos_dates.extend(list(test_idx))

    # Equal-weight bench on same window
    valid = [t for t in tickers if t in test_prices.columns]
    ew_rets, _ = equal_weight_baseline(test_prices, valid, window_size=30)
    n = min(len(rets), len(ew_rets))

    summary = {
        'year': y,
        'days': int(n),
        'staged_return': float(compute_cumulative_return(rets[:n])),
        'staged_sharpe': float(compute_sharpe(rets[:n])),
        'staged_sortino': float(compute_sortino(rets[:n])),
        'staged_maxdd': float(compute_max_drawdown(rets[:n])),
        'ew_return': float(compute_cumulative_return(ew_rets[:n])),
        'ew_sharpe': float(compute_sharpe(ew_rets[:n])),
        'ew_maxdd': float(compute_max_drawdown(ew_rets[:n])),
        'beats_ew': bool(compute_cumulative_return(rets[:n]) > compute_cumulative_return(ew_rets[:n])),
        'fold_seconds': float(time.time() - fold_t),
    }
    fold_summaries.append(summary)
    print(f"  staged: ret={summary['staged_return']:+.4f} sharpe={summary['staged_sharpe']:.3f} maxdd={summary['staged_maxdd']:.4f}"
          f"  | EW: ret={summary['ew_return']:+.4f} maxdd={summary['ew_maxdd']:.4f}"
          f"  | beats_EW={summary['beats_ew']}  ({summary['fold_seconds']:.0f}s)")

    # Save fold checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"wf_{y}.pt")
    torch.save({
        'manager_state_dict': mgr_net.state_dict(),
        'worker_state_dicts': {n: net.state_dict() for n, net in w_nets.items()},
        'manager_n_assets': tr_m_env.action_space.shape[0],
        'manager_window_size': tr_m_env.window_size,
        'manager_extra_context_dim': tr_m_env.extra_context_dim,
        'worker_specs': {n: {'n_assets': tr_w_envs[n].n_assets,
                             'n_total': tr_w_envs[n].n_total,
                             'window_size': tr_w_envs[n].window_size,
                             'tickers': tr_w_envs[n].tickers}
                         for n in w_nets},
        'risk_pools': {k: list(v) for k, v in risk_pools.items()},
        'tickers': tickers,
        'pool_names': ['Safe', 'Neutral', 'Risky'],
        'training_end_date': str(train_prices.index[-1].date()),
        'transaction_cost_bps': TRANSACTION_COST_BPS,
    }, ckpt_path)
    last_models = (mgr_net, w_nets, tr_m_env, tr_w_envs, train_prices)

# Latest fold = the model to use for paper trading going forward
if last_models is not None:
    mgr_net, w_nets, tr_m_env, tr_w_envs, train_prices = last_models
    final_path = os.path.join(CHECKPOINT_DIR, "staged_marl_full.pt")
    torch.save({
        'manager_state_dict': mgr_net.state_dict(),
        'worker_state_dicts': {n: net.state_dict() for n, net in w_nets.items()},
        'manager_n_assets': tr_m_env.action_space.shape[0],
        'manager_window_size': tr_m_env.window_size,
        'manager_extra_context_dim': tr_m_env.extra_context_dim,
        'worker_specs': {n: {'n_assets': tr_w_envs[n].n_assets,
                             'n_total': tr_w_envs[n].n_total,
                             'window_size': tr_w_envs[n].window_size,
                             'tickers': tr_w_envs[n].tickers}
                         for n in w_nets},
        'risk_pools': {k: list(v) for k, v in risk_pools.items()},
        'tickers': tickers,
        'pool_names': ['Safe', 'Neutral', 'Risky'],
        'training_end_date': str(train_prices.index[-1].date()),
        'transaction_cost_bps': TRANSACTION_COST_BPS,
    }, final_path)
    print(f"\nLatest fold checkpoint copied to {final_path}")

# Stitched OOS analysis
oos = pd.Series(oos_daily_returns, index=pd.DatetimeIndex(oos_dates), name='oos_ret')
oos = oos[~oos.index.duplicated(keep='last')]
oos.to_csv("data/processed/walk_forward_oos.csv", header=True)
oos_arr = oos.values

print("\n" + "=" * 70)
print(" STITCHED WALK-FORWARD OOS PERFORMANCE")
print("=" * 70)
print(f"  Days OOS:        {len(oos_arr)}")
print(f"  Cum return:      {compute_cumulative_return(oos_arr):+.4f}")
print(f"  Sharpe:          {compute_sharpe(oos_arr):.4f}")
print(f"  Sortino:         {compute_sortino(oos_arr):.4f}")
print(f"  Max drawdown:    {compute_max_drawdown(oos_arr):.4f}")
fy = pd.Timestamp(f"{folds[0][0]}-01-01")
ly = oos.index[-1]
years_span = max((ly - fy).days / 365.25, 1e-6)
cagr = (1 + compute_cumulative_return(oos_arr)) ** (1/years_span) - 1
print(f"  CAGR:            {cagr:+.4f}")

summary_out = {
    'folds': fold_summaries,
    'stitched_oos': {
        'days': int(len(oos_arr)),
        'cumulative_return': float(compute_cumulative_return(oos_arr)),
        'cagr': float(cagr),
        'sharpe': float(compute_sharpe(oos_arr)),
        'sortino': float(compute_sortino(oos_arr)),
        'max_drawdown': float(compute_max_drawdown(oos_arr)),
        'start': str(oos.index[0].date()),
        'end': str(oos.index[-1].date()),
    },
    'transaction_cost_bps': TRANSACTION_COST_BPS,
    'total_seconds': float(time.time() - t0),
}
with open("data/processed/walk_forward_summary.json", "w") as f:
    json.dump(summary_out, f, indent=2)
print(f"\nWrote data/processed/walk_forward_summary.json (total {time.time()-t0:.0f}s)")

"""
Real-Market Paper-Trading Backtest
===================================
Loads the latest checkpoint produced by Staged_MARL_Training.py or
walk_forward_train.py and simulates daily trading from `--start` (default:
day after training-end) through today using LIVE yfinance data.

Mechanics:
  * Same WorkerEnv / ManagerEnv as training (so the model sees the same
    observation tensors it was trained on)
  * Same transaction-cost overlay (TRANSACTION_COST_BPS) baked into envs
  * Same defensive stress overlay (cash floor + Safe-pool boost)
  * Compares the model's $1 equity curve against SPY buy-and-hold

Usage:
  python realmarket_backtest.py
  python realmarket_backtest.py --start 2024-01-02
  python realmarket_backtest.py --checkpoint checkpoints/wf_2024.pt
"""

import os, sys, json, argparse, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
import torch

# ---- Import env/net machinery from main script (no side effects) ----
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Staged_MARL_Training.py")
with open(_PATH, "r", encoding="utf-8") as f:
    _SRC = f.read()
_idx = _SRC.find("# MAIN EXECUTION")
if _idx == -1:
    raise RuntimeError("Could not find MAIN EXECUTION marker in Staged_MARL_Training.py")
_ns = {"__name__": "smt_lib", "__file__": _PATH}
exec(compile(_SRC[:_idx], _PATH, "exec"), _ns)

WorkerEnv          = _ns["WorkerEnv"]
ManagerEnv         = _ns["ManagerEnv"]
EIIENetwork        = _ns["EIIENetwork"]
make_worker_envs   = _ns["make_worker_envs"]
make_manager_env   = _ns["make_manager_env"]
evaluate_system    = _ns["evaluate_system"]
compute_cumulative_return = _ns["compute_cumulative_return"]
compute_sharpe     = _ns["compute_sharpe"]
compute_sortino    = _ns["compute_sortino"]
compute_max_drawdown = _ns["compute_max_drawdown"]
TRANSACTION_COST_BPS = _ns["TRANSACTION_COST_BPS"]
DEVICE = _ns["DEVICE"]

# ---- CLI ----
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/staged_marl_full.pt")
ap.add_argument("--start", default=None,
                help="ISO date (YYYY-MM-DD). Default: day after training_end_date in checkpoint")
ap.add_argument("--end", default=None,
                help="ISO date. Default: today")
ap.add_argument("--benchmark", default="SPY")
ap.add_argument("--output", default="data/processed/realmarket_backtest.csv")
args = ap.parse_args()

# ---- Load checkpoint ----
print("=" * 70)
print(" REAL-MARKET PAPER-TRADING BACKTEST")
print("=" * 70)
print(f"Checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
training_end = pd.Timestamp(ckpt['training_end_date'])
tickers      = ckpt['tickers']
risk_pools   = {k: list(v) for k, v in ckpt['risk_pools'].items()}
worker_specs = ckpt['worker_specs']
print(f"Training data ended: {training_end.date()}")
print(f"Stocks in universe:  {len(tickers)}")
print(f"Pools: " + ", ".join(f"{k}={len(v)}" for k, v in risk_pools.items()))

start = pd.Timestamp(args.start) if args.start else (training_end + pd.Timedelta(days=1))
end   = pd.Timestamp(args.end)   if args.end   else pd.Timestamp.today().normalize()
# Need lookback before `start` for the rolling windows in the env
WINDOW_PAD = 60
fetch_start = start - pd.Timedelta(days=WINDOW_PAD * 2)
print(f"Trading window:      {start.date()} -> {end.date()}")

# ---- Fetch live data ----
import yfinance as yf
print(f"\nDownloading {len(tickers)} tickers + {args.benchmark} from yfinance...")
all_syms = list(tickers) + [args.benchmark]
data = yf.download(all_syms, start=fetch_start.strftime("%Y-%m-%d"),
                   end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                   auto_adjust=True, progress=False, threads=True)
if isinstance(data.columns, pd.MultiIndex):
    px = data['Close'].copy()
else:
    px = data[['Close']].copy(); px.columns = all_syms[:1]
px = px.ffill().dropna(how='all')
px = px.dropna(axis=1, how='any')  # drop tickers with any missing
available = [t for t in tickers if t in px.columns]
missing = [t for t in tickers if t not in px.columns]
if missing:
    print(f"  Skipping {len(missing)} tickers with missing data: {missing[:8]}{'...' if len(missing)>8 else ''}")
print(f"  Got {len(available)} tickers + benchmark, {len(px)} trading days")

if args.benchmark not in px.columns:
    raise RuntimeError(f"Benchmark {args.benchmark} not retrieved")

bench_px = px[args.benchmark]
stock_px = px[available]

# Filter risk pools to available tickers
risk_pools_avail = {k: [t for t in v if t in available] for k, v in risk_pools.items()}
print(f"Pool sizes after filtering: " + ", ".join(f"{k}={len(v)}" for k, v in risk_pools_avail.items()))

# ---- Lexical matrix (must already exist on disk) ----
lexical_df = pd.read_csv("data/processed/lexical_matrix_100.csv", index_col=0)

# ---- Build envs from the LIVE price slice ----
worker_envs = make_worker_envs(stock_px, lexical_df, risk_pools_avail)
manager_env = make_manager_env(stock_px, lexical_df, risk_pools_avail, worker_envs_ref=worker_envs)

# ---- Rebuild networks with the same shapes as in checkpoint ----
m_n  = ckpt['manager_n_assets']
m_ws = ckpt['manager_window_size']
m_ec = ckpt['manager_extra_context_dim']
manager_net = EIIENetwork(m_n, m_ws, extra_context_dim=m_ec).to(DEVICE)
manager_net.load_state_dict(ckpt['manager_state_dict'])
manager_net.eval()

worker_nets = {}
for name, spec in worker_specs.items():
    if name not in worker_envs:
        print(f"  Warning: pool {name} has no available stocks now; skipping")
        continue
    env = worker_envs[name]
    # The env's n_assets may differ from training if some tickers dropped
    if env.n_total != spec['n_total']:
        print(f"  Pool {name}: live n_total={env.n_total} differs from trained {spec['n_total']} — recreating env to trained tickers")
        # Force env to use exactly the trained tickers (drop missing)
        trained_tickers = [t for t in spec['tickers'] if t in stock_px.columns]
        if len(trained_tickers) < 2:
            print(f"    Pool {name}: insufficient trained tickers in live data; skipping")
            continue
        env = WorkerEnv(stock_px, lexical_df, trained_tickers, profile=name.lower())
        worker_envs[name] = env
        if env.n_total != spec['n_total']:
            print(f"    Pool {name}: still mismatch ({env.n_total} vs {spec['n_total']}); skipping")
            continue
    net = EIIENetwork(env.n_total, env.window_size, n_price_assets=env.n_assets).to(DEVICE)
    net.load_state_dict(ckpt['worker_state_dicts'][name])
    net.eval()
    worker_nets[name] = net

# Manager env may also need rebuilding if pool sizes shifted
manager_env = make_manager_env(stock_px, lexical_df, risk_pools_avail, worker_envs_ref=worker_envs)
if manager_env.action_space.shape[0] != m_n or manager_env.window_size != m_ws or manager_env.extra_context_dim != m_ec:
    print(f"  Manager env shape changed: action={manager_env.action_space.shape[0]} (trained {m_n}),"
          f" window={manager_env.window_size}/{m_ws}, ctx={manager_env.extra_context_dim}/{m_ec}")
    print("  Manager network will not load cleanly — aborting.")
    sys.exit(1)

# ---- Run the model forward over the live slice ----
print("\nRunning model forward over live data...")
max_steps = len(stock_px) - manager_env.window_size - 2
ev = evaluate_system(manager_env, worker_envs, manager_net, worker_nets, max_steps=max_steps)
model_rets = ev['global_returns']
allocs     = ev['allocations']

# ---- Align dates & restrict to start->end ----
# The env starts trading at index = window_size and produces len(model_rets) daily returns
trade_dates = stock_px.index[manager_env.window_size : manager_env.window_size + len(model_rets)]
df = pd.DataFrame({'model_ret': model_rets}, index=trade_dates)
df = df[df.index >= start]
df = df[df.index <= end]
print(f"Backtest produced {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")

# Benchmark daily returns over the same dates
bench_aligned = bench_px.reindex(df.index).pct_change().fillna(0.0)
df['bench_ret'] = bench_aligned

df['model_equity'] = (1.0 + df['model_ret']).cumprod()
df['bench_equity'] = (1.0 + df['bench_ret']).cumprod()

# Save
os.makedirs(os.path.dirname(args.output), exist_ok=True)
df.to_csv(args.output)

# ---- Summary ----
m = df['model_ret'].values
b = df['bench_ret'].values
years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-6)

def cagr(rets):
    if len(rets) == 0: return 0.0
    return float((1 + compute_cumulative_return(rets)) ** (1/years) - 1)

print("\n" + "=" * 70)
print(f" REAL-MARKET RESULTS: {df.index[0].date()} -> {df.index[-1].date()}  ({years:.2f} yrs)")
print("=" * 70)
fmt = "  {:<22} | model: {:>+8.4f}   bench ({}): {:>+8.4f}"
print(fmt.format("Cumulative return", compute_cumulative_return(m), args.benchmark, compute_cumulative_return(b)))
print(fmt.format("CAGR",              cagr(m),                       args.benchmark, cagr(b)))
print(fmt.format("Sharpe (ann.)",     compute_sharpe(m),             args.benchmark, compute_sharpe(b)))
print(fmt.format("Sortino (ann.)",    compute_sortino(m),            args.benchmark, compute_sortino(b)))
print(fmt.format("Max drawdown",      compute_max_drawdown(m),       args.benchmark, compute_max_drawdown(b)))

# Extra context
out_perf = compute_cumulative_return(m) - compute_cumulative_return(b)
print(f"\n  Excess return vs {args.benchmark}: {out_perf:+.4f}")
verdict = "EARNED MONEY" if compute_cumulative_return(m) > 0 else "LOST MONEY"
print(f"  Model {verdict} over the period (after {TRANSACTION_COST_BPS} bps costs).")
beat_bench = "BEAT" if compute_cumulative_return(m) > compute_cumulative_return(b) else "LOST TO"
print(f"  Model {beat_bench} the {args.benchmark} buy-and-hold benchmark.")

# Per-year breakdown
print("\n  Per calendar year:")
yearly = df[['model_ret', 'bench_ret']].groupby(df.index.year).apply(
    lambda g: pd.Series({
        'model_ret': float((1 + g['model_ret']).prod() - 1),
        'bench_ret': float((1 + g['bench_ret']).prod() - 1),
        'days': len(g),
    })
)
print(yearly.to_string(float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)))

print(f"\nDaily equity curve saved to {args.output}")

# Save JSON summary too
json_path = args.output.replace(".csv", "_summary.json")
with open(json_path, "w") as f:
    json.dump({
        'start': str(df.index[0].date()), 'end': str(df.index[-1].date()),
        'years': years,
        'model': {
            'cumulative_return': float(compute_cumulative_return(m)),
            'cagr': cagr(m),
            'sharpe': float(compute_sharpe(m)),
            'sortino': float(compute_sortino(m)),
            'max_drawdown': float(compute_max_drawdown(m)),
        },
        'benchmark': {
            'symbol': args.benchmark,
            'cumulative_return': float(compute_cumulative_return(b)),
            'cagr': cagr(b),
            'sharpe': float(compute_sharpe(b)),
            'sortino': float(compute_sortino(b)),
            'max_drawdown': float(compute_max_drawdown(b)),
        },
        'transaction_cost_bps': TRANSACTION_COST_BPS,
        'checkpoint': args.checkpoint,
        'training_end_date': str(training_end.date()),
    }, f, indent=2)
print(f"Summary JSON saved to {json_path}")

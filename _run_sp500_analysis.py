"""
Phase 3: S&P 500 Analysis — Hierarchical MARL with 10-K Lexical Diversification
================================================================================
- Manager-Worker with real GICS sectors
- Walk-Forward Validation (4 windows)
- Lambda Ablation (0, 0.1, 0.5, 1.0) — key test with 45 stocks
- Drawdown Decomposition + Risk Metrics (CVaR, Sortino, Calmar)
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# HELPER: JSON-safe conversion
# ═══════════════════════════════════════════════════════════════════════
def to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════
# EIIE NETWORK
# ═══════════════════════════════════════════════════════════════════════
class EIIENetwork(nn.Module):
    def __init__(self, n_assets, window_size):
        super().__init__()
        self.n_assets = n_assets
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(n_assets * 32 + n_assets, 64)
        self.fc2 = nn.Linear(64, n_assets)

    def forward(self, pw, pvm):
        feats = []
        for i in range(self.n_assets):
            x = pw[:, i, :].unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.mean(dim=2)
            feats.append(x)
        c = torch.cat(feats + [pvm], dim=1)
        x = F.relu(self.fc1(c))
        return F.softmax(self.fc2(x), dim=1)


# ═══════════════════════════════════════════════════════════════════════
# GYMNASIUM ENVIRONMENTS
# ═══════════════════════════════════════════════════════════════════════
class WorkerEnv(gym.Env):
    """Worker environment for stock selection within a sector."""
    metadata = {"render_modes": []}

    def __init__(self, price_df, lexical_matrix, tickers,
                 window_size=30, lambda_penalty=0.1, gamma_penalty=0.01):
        super().__init__()
        self.tickers = [t for t in tickers
                        if t in price_df.columns and t in lexical_matrix.index]
        self.n_assets = len(self.tickers)
        if self.n_assets == 0:
            raise ValueError("No valid tickers for WorkerEnv")

        self.prices = price_df[self.tickers].values
        self.lexical_matrix = np.nan_to_num(
            lexical_matrix.loc[self.tickers, self.tickers].values, nan=0.0
        )
        self.window_size = window_size
        self.lambda_penalty = lambda_penalty
        self.gamma_penalty = gamma_penalty

        obs_dim = window_size * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.current_step = 0
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        self.max_steps = len(self.prices) - window_size - 1
        self.portfolio_value = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        return self._get_obs(), {}

    def step(self, action):
        exp_a = np.exp(action - np.max(action))
        w = exp_a / np.sum(exp_a)
        curr = self.prices[self.current_step]
        nxt = self.prices[self.current_step + 1]
        pr = nxt / (curr + 1e-8)
        port_ret = max(np.dot(w, pr), 1e-8)
        log_ret = np.log(port_ret)
        sem_pen = np.dot(w.T, np.dot(self.lexical_matrix, w))
        turnover = np.sum(np.abs(w - self.portfolio_weights))
        reward = float(
            log_ret - self.lambda_penalty * sem_pen - self.gamma_penalty * turnover
        )
        self.portfolio_weights = w
        self.portfolio_value *= port_ret
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_obs(), reward, done, False, {
            "portfolio_return": port_ret,
            "semantic_penalty": sem_pen,
            "turnover": turnover,
            "weights": w.copy(),
        }

    def _get_obs(self):
        s = self.current_step - self.window_size
        win = self.prices[s : self.current_step]
        norm = win / (win[0, :] + 1e-8)
        return np.concatenate([norm.flatten(), self.portfolio_weights]).astype(
            np.float32
        )


class ManagerEnv(gym.Env):
    """Manager environment for sector capital allocation."""
    metadata = {"render_modes": []}

    def __init__(self, price_df, lexical_matrix, sectors, window_size=30):
        super().__init__()
        self.sectors = sectors
        self.sector_names = list(sectors.keys())
        self.n_sectors = len(self.sector_names)

        self.sector_prices = {}
        for name, tickers in sectors.items():
            valid_t = [t for t in tickers if t in price_df.columns]
            if valid_t:
                self.sector_prices[name] = price_df[valid_t].mean(axis=1).values

        self.window_size = window_size
        self.all_prices = np.column_stack(
            [self.sector_prices[s] for s in self.sector_names]
        )

        # Sector-level similarity = avg intra-sector lexical similarity
        self.sector_sim = np.zeros(self.n_sectors)
        for i, name in enumerate(self.sector_names):
            tickers = [t for t in sectors[name] if t in lexical_matrix.index]
            if len(tickers) > 1:
                sim = lexical_matrix.loc[tickers, tickers].values
                mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
                self.sector_sim[i] = np.nanmean(sim[mask])

        obs_dim = window_size * self.n_sectors + self.n_sectors + self.n_sectors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_sectors,), dtype=np.float32
        )

        self.current_step = 0
        self.sector_weights = np.ones(self.n_sectors) / self.n_sectors
        self.max_steps = len(self.all_prices) - window_size - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.sector_weights = np.ones(self.n_sectors) / self.n_sectors
        return self._get_obs(), {}

    def step(self, action):
        exp_a = np.exp(action - np.max(action))
        w = exp_a / np.sum(exp_a)
        curr = self.all_prices[self.current_step]
        nxt = self.all_prices[self.current_step + 1]
        pr = nxt / (curr + 1e-8)
        port_ret = max(np.dot(w, pr), 1e-8)
        log_ret = np.log(port_ret)
        sim_penalty = np.dot(w, self.sector_sim)
        turnover = np.sum(np.abs(w - self.sector_weights))
        reward = float(log_ret - 0.1 * sim_penalty - 0.01 * turnover)
        self.sector_weights = w
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_obs(), reward, done, False, {
            "sector_weights": w.copy(),
            "sim_penalty": sim_penalty,
        }

    def _get_obs(self):
        s = self.current_step - self.window_size
        win = self.all_prices[s : self.current_step]
        norm = win / (win[0, :] + 1e-8)
        return np.concatenate(
            [norm.flatten(), self.sector_weights, self.sector_sim]
        ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# REINFORCE TRAINING
# ═══════════════════════════════════════════════════════════════════════
def train_reinforce(env, n_episodes=50, lr=1e-3, max_steps=200, verbose=False):
    n = env.action_space.shape[0]
    ws = env.window_size
    net = EIIENetwork(n, ws)
    opt = optim.Adam(net.parameters(), lr=lr)
    hist = {"rewards": [], "penalties": [], "values": []}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        lps, rews, sps = [], [], []
        done, steps = False, 0

        while not done and steps < max_steps:
            obs_dim = ws * n
            pw = torch.FloatTensor(obs[:obs_dim].reshape(n, ws)).unsqueeze(0)
            pvm = torch.FloatTensor(obs[obs_dim : obs_dim + n]).unsqueeze(0)
            weights = net(pw, pvm)
            dist = torch.distributions.Normal(weights.squeeze(), 0.1)
            action = dist.sample()
            lps.append(dist.log_prob(action).sum())
            obs, reward, terminated, truncated, info = env.step(
                action.detach().numpy()
            )
            done = terminated or truncated
            rews.append(reward)
            sps.append(info.get("semantic_penalty", info.get("sim_penalty", 0)))
            steps += 1

        G, rets = 0, []
        for r in reversed(rews):
            G = r + 0.99 * G
            rets.insert(0, G)
        rets = torch.FloatTensor(rets)
        if len(rets) > 1:
            rets = (rets - rets.mean()) / (rets.std() + 1e-8)

        loss = sum(-lp * G for lp, G in zip(lps, rets))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        hist["rewards"].append(sum(rews))
        hist["penalties"].append(float(np.mean(sps)))
        hist["values"].append(getattr(env, "portfolio_value", 1.0))

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"    Ep {ep+1:3d} | R: {sum(rews):.4f} "
                f"| Pen: {np.mean(sps):.4f} | Val: {hist['values'][-1]:.4f}"
            )

    return net, hist


# ═══════════════════════════════════════════════════════════════════════
# RISK METRICS
# ═══════════════════════════════════════════════════════════════════════
def compute_max_drawdown(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def compute_cvar(returns, alpha=0.05):
    sorted_r = np.sort(returns)
    cutoff = max(int(alpha * len(sorted_r)), 1)
    return float(np.mean(sorted_r[:cutoff]))


def compute_sortino(returns, target=0):
    excess = returns - target
    downside = returns[returns < target]
    if len(downside) == 0:
        return float("inf")
    return float(np.mean(excess) / (np.std(downside) + 1e-8))


def compute_calmar(returns, max_dd=None):
    if max_dd is None:
        max_dd = compute_max_drawdown(returns)
    ann_ret = np.mean(returns) * 252
    return float(ann_ret / max_dd) if max_dd > 0 else 0.0


def compute_hhi(weights):
    return float(np.sum(weights ** 2))


def compute_effective_n(weights):
    hhi = compute_hhi(weights)
    return float(1.0 / hhi) if hhi > 0 else 0.0


def drawdown_decomposition(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    drawdowns = []
    in_dd = False
    dd_start = max_depth = trough_idx = 0
    for i in range(len(dd)):
        if dd[i] > 0.001:
            if not in_dd:
                dd_start = i
                in_dd = True
                max_depth = dd[i]
                trough_idx = i
            elif dd[i] > max_depth:
                max_depth = dd[i]
                trough_idx = i
        else:
            if in_dd:
                drawdowns.append({
                    "start": dd_start, "trough": trough_idx, "end": i,
                    "depth": float(max_depth),
                    "duration": i - dd_start,
                    "recovery": i - trough_idx,
                })
                in_dd = False
    if in_dd:
        drawdowns.append({
            "start": dd_start, "trough": trough_idx, "end": len(dd) - 1,
            "depth": float(max_depth),
            "duration": len(dd) - 1 - dd_start,
            "recovery": len(dd) - 1 - trough_idx,
        })
    return drawdowns


def evaluate_portfolio(env, net, max_steps=200):
    """Run a trained agent through the environment and collect returns."""
    obs, _ = env.reset()
    returns, weights_hist, penalties = [], [], []
    done, steps = False, 0
    n = env.action_space.shape[0]
    ws = env.window_size

    while not done and steps < max_steps:
        obs_dim = ws * n
        pw = torch.FloatTensor(obs[:obs_dim].reshape(n, ws)).unsqueeze(0)
        pvm = torch.FloatTensor(obs[obs_dim : obs_dim + n]).unsqueeze(0)
        with torch.no_grad():
            w = net(pw, pvm).squeeze().numpy()
        obs, reward, terminated, truncated, info = env.step(w)
        done = terminated or truncated
        returns.append(info.get("portfolio_return", 1.0) - 1.0)
        weights_hist.append(info.get("weights", w))
        penalties.append(info.get("semantic_penalty", 0))
        steps += 1

    returns = np.array(returns)
    avg_weights = np.mean(weights_hist, axis=0) if weights_hist else np.zeros(n)
    return {
        "returns": returns,
        "total_return": float(np.prod(1 + returns) - 1),
        "sharpe": float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)),
        "max_drawdown": compute_max_drawdown(returns),
        "cvar_95": compute_cvar(returns),
        "sortino": compute_sortino(returns),
        "calmar": compute_calmar(returns),
        "hhi": compute_hhi(avg_weights),
        "effective_n": compute_effective_n(avg_weights),
        "avg_penalty": float(np.mean(penalties)),
        "avg_weights": avg_weights.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 3: S&P 500 Hierarchical MARL with 10-K Semantic Analysis")
    print("=" * 65)

    # ─── Load Data ────────────────────────────────────────────────────
    print("\n1. Loading data...")
    price_df = pd.read_csv("data/raw/sp500_50_prices.csv", index_col=0, parse_dates=True)
    lex_df = pd.read_csv("data/processed/lexical_matrix_50.csv", index_col=0)
    with open("data/processed/sector_map_50.json") as f:
        sector_map = json.load(f)

    tickers = list(price_df.columns)
    n_stocks = len(tickers)
    print(f"  {n_stocks} stocks, {len(price_df)} days, {len(set(sector_map.values()))} sectors")

    # Build sector groups
    sectors = {}
    for t, s in sector_map.items():
        if t in tickers:
            sectors.setdefault(s, []).append(t)

    # Filter to sectors with 2+ stocks (Workers need at least 2)
    sectors = {s: ts for s, ts in sectors.items() if len(ts) >= 2}
    valid_tickers = [t for ts in sectors.values() for t in ts]
    price_df = price_df[valid_tickers]
    print(f"  After filtering: {len(valid_tickers)} stocks in {len(sectors)} sectors")
    for s, ts in sorted(sectors.items()):
        print(f"    {s}: {ts}")

    results = {}

    # ─── 2. Hierarchical Training ────────────────────────────────────
    print("\n2. Training Hierarchical Manager-Worker System...")

    # Train Manager
    print("  Training Manager (sector allocator)...")
    mgr_env = ManagerEnv(price_df, lex_df, sectors, window_size=30)
    mgr_net, mgr_hist = train_reinforce(mgr_env, n_episodes=50, verbose=True)

    # Train Workers (one per sector)
    worker_nets = {}
    worker_hists = {}
    for sec_name, sec_tickers in sectors.items():
        if len(sec_tickers) < 2:
            continue
        print(f"  Training Worker: {sec_name} ({len(sec_tickers)} stocks)...")
        wenv = WorkerEnv(price_df, lex_df, sec_tickers, window_size=30, lambda_penalty=0.1)
        wnet, whist = train_reinforce(wenv, n_episodes=50, verbose=False)
        worker_nets[sec_name] = wnet
        worker_hists[sec_name] = whist

    # Evaluate hierarchy
    print("  Evaluating hierarchical system...")
    mgr_eval = evaluate_portfolio(mgr_env, mgr_net, max_steps=200)
    
    worker_evals = {}
    for sec_name, sec_tickers in sectors.items():
        if sec_name in worker_nets:
            wenv = WorkerEnv(price_df, lex_df, sec_tickers, window_size=30, lambda_penalty=0.1)
            worker_evals[sec_name] = evaluate_portfolio(wenv, worker_nets[sec_name], max_steps=200)

    results["hierarchical"] = {
        "manager": {k: to_native(v) for k, v in mgr_eval.items() if k != "returns"},
        "workers": {
            s: {k: to_native(v) for k, v in ev.items() if k != "returns"}
            for s, ev in worker_evals.items()
        },
        "manager_training": {
            "final_reward": float(mgr_hist["rewards"][-1]),
            "avg_reward": float(np.mean(mgr_hist["rewards"][-10:])),
        },
    }

    print(f"  Manager: Return={mgr_eval['total_return']:.4f}, Sharpe={mgr_eval['sharpe']:.4f}")
    for s, ev in worker_evals.items():
        print(f"  Worker {s}: Return={ev['total_return']:.4f}, Sharpe={ev['sharpe']:.4f}, EffN={ev['effective_n']:.1f}")

    # ─── 3. Walk-Forward Validation ──────────────────────────────────
    print("\n3. Walk-Forward Validation...")
    wf_windows = [
        ("2015-2018 → 2019", "2015-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
        ("2015-2019 → 2020", "2015-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
        ("2015-2020 → 2021", "2015-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
        ("2015-2021 → 2022", "2015-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
    ]

    wf_results = {}
    for wf_name, train_s, train_e, test_s, test_e in wf_windows:
        print(f"  Window: {wf_name}")
        train_prices = price_df.loc[train_s:train_e]
        test_prices = price_df.loc[test_s:test_e]

        if len(train_prices) < 60 or len(test_prices) < 20:
            print(f"    Skipping — insufficient data")
            continue

        # Train on training window
        wf_env = WorkerEnv(train_prices, lex_df, valid_tickers,
                           window_size=30, lambda_penalty=0.1)
        wf_net, _ = train_reinforce(wf_env, n_episodes=40, verbose=False)

        # Test on test window
        test_env = WorkerEnv(test_prices, lex_df, valid_tickers,
                             window_size=30, lambda_penalty=0.1)
        wf_eval = evaluate_portfolio(test_env, wf_net, max_steps=200)

        wf_results[wf_name] = {k: to_native(v) for k, v in wf_eval.items() if k != "returns"}
        print(f"    Return: {wf_eval['total_return']:.4f}, Sharpe: {wf_eval['sharpe']:.4f}, "
              f"MaxDD: {wf_eval['max_drawdown']:.4f}")

    results["walk_forward"] = wf_results

    # ─── 4. Lambda Ablation Study ────────────────────────────────────
    print("\n4. Lambda Ablation Study (KEY TEST with 45 stocks)...")
    lambdas = [0.0, 0.1, 0.5, 1.0]
    ablation_results = {}

    for lam in lambdas:
        print(f"  λ = {lam}...")
        abl_env = WorkerEnv(price_df, lex_df, valid_tickers,
                            window_size=30, lambda_penalty=lam)
        abl_net, abl_hist = train_reinforce(abl_env, n_episodes=50, verbose=False)
        abl_eval = evaluate_portfolio(abl_env, abl_net, max_steps=200)

        ablation_results[str(lam)] = {
            k: to_native(v) for k, v in abl_eval.items() if k != "returns"
        }
        ablation_results[str(lam)]["training"] = {
            "final_reward": float(abl_hist["rewards"][-1]),
            "avg_penalty": float(np.mean(abl_hist["penalties"][-10:])),
        }
        print(f"    Return={abl_eval['total_return']:.4f}, Sharpe={abl_eval['sharpe']:.4f}, "
              f"HHI={abl_eval['hhi']:.4f}, EffN={abl_eval['effective_n']:.1f}, "
              f"Penalty={abl_eval['avg_penalty']:.4f}")

    results["ablation"] = ablation_results

    # ─── 5. Drawdown Decomposition ───────────────────────────────────
    print("\n5. Drawdown Decomposition...")
    # Use the baseline (λ=0.1) returns
    base_env = WorkerEnv(price_df, lex_df, valid_tickers,
                         window_size=30, lambda_penalty=0.1)
    base_net, _ = train_reinforce(base_env, n_episodes=50, verbose=False)
    base_eval = evaluate_portfolio(base_env, base_net, max_steps=200)
    dds = drawdown_decomposition(base_eval["returns"])

    results["drawdown"] = {
        "num_events": len(dds),
        "events": to_native(dds[:10]),  # Top 10
        "max_depth": float(max(d["depth"] for d in dds)) if dds else 0.0,
        "avg_depth": float(np.mean([d["depth"] for d in dds])) if dds else 0.0,
        "avg_duration": float(np.mean([d["duration"] for d in dds])) if dds else 0.0,
        "avg_recovery": float(np.mean([d["recovery"] for d in dds])) if dds else 0.0,
    }
    print(f"  {len(dds)} drawdown events")
    if dds:
        print(f"  Deepest: {results['drawdown']['max_depth']:.4f}")
        print(f"  Avg depth: {results['drawdown']['avg_depth']:.4f}, "
              f"Avg duration: {results['drawdown']['avg_duration']:.1f} days")

    # ─── 6. Comprehensive Risk Metrics ───────────────────────────────
    print("\n6. Risk Metrics Summary...")
    results["risk_metrics"] = {
        "total_return": base_eval["total_return"],
        "sharpe": base_eval["sharpe"],
        "max_drawdown": base_eval["max_drawdown"],
        "cvar_95": base_eval["cvar_95"],
        "sortino": base_eval["sortino"],
        "calmar": base_eval["calmar"],
        "hhi": base_eval["hhi"],
        "effective_n": base_eval["effective_n"],
    }
    for k, v in results["risk_metrics"].items():
        print(f"  {k}: {v:.4f}")

    # ─── 7. Generate Plots ───────────────────────────────────────────
    print("\n7. Generating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 3: S&P 500 MARL Analysis (45 Stocks, 10-K Lexical)", fontsize=14, fontweight="bold")

    # 7a: Manager sector allocation
    ax = axes[0, 0]
    if "avg_weights" in mgr_eval:
        sector_names = list(sectors.keys())
        weights = mgr_eval["avg_weights"][:len(sector_names)]
        bars = ax.barh(sector_names, weights, color=plt.cm.Set3(np.linspace(0, 1, len(sector_names))))
        ax.set_xlabel("Avg Weight")
        ax.set_title("Manager: Sector Allocation")
        ax.axvline(1.0 / len(sector_names), color="red", linestyle="--", alpha=0.5, label="Equal Weight")
        ax.legend(fontsize=8)

    # 7b: Walk-Forward returns
    ax = axes[0, 1]
    if wf_results:
        wf_names = list(wf_results.keys())
        wf_returns = [wf_results[w]["total_return"] for w in wf_names]
        colors = ["green" if r > 0 else "red" for r in wf_returns]
        ax.bar(range(len(wf_names)), wf_returns, color=colors)
        ax.set_xticks(range(len(wf_names)))
        ax.set_xticklabels([w.split("→")[1].strip() for w in wf_names], fontsize=8)
        ax.set_ylabel("Return")
        ax.set_title("Walk-Forward: OOS Returns")
        ax.axhline(0, color="black", linewidth=0.5)

    # 7c: Lambda ablation — HHI and Effective N
    ax = axes[0, 2]
    lam_vals = [float(l) for l in ablation_results.keys()]
    hhi_vals = [ablation_results[str(l)]["hhi"] for l in lam_vals]
    eff_n_vals = [ablation_results[str(l)]["effective_n"] for l in lam_vals]
    ax2 = ax.twinx()
    l1 = ax.bar(np.array(range(len(lam_vals))) - 0.15, hhi_vals, 0.3, color="steelblue", label="HHI")
    l2 = ax2.bar(np.array(range(len(lam_vals))) + 0.15, eff_n_vals, 0.3, color="coral", label="Eff. N")
    ax.set_xticks(range(len(lam_vals)))
    ax.set_xticklabels([f"λ={l}" for l in lam_vals])
    ax.set_ylabel("HHI (↓ better)", color="steelblue")
    ax2.set_ylabel("Effective N (↑ better)", color="coral")
    ax.set_title("Lambda Ablation: Concentration")
    ax.legend([l1, l2], ["HHI", "Eff. N"], loc="upper left", fontsize=8)

    # 7d: Lambda ablation — Returns and Sharpe
    ax = axes[1, 0]
    ret_vals = [ablation_results[str(l)]["total_return"] for l in lam_vals]
    sharpe_vals = [ablation_results[str(l)]["sharpe"] for l in lam_vals]
    ax.plot(lam_vals, ret_vals, "o-", color="green", label="Return")
    ax2 = ax.twinx()
    ax2.plot(lam_vals, sharpe_vals, "s-", color="purple", label="Sharpe")
    ax.set_xlabel("λ")
    ax.set_ylabel("Total Return", color="green")
    ax2.set_ylabel("Sharpe Ratio", color="purple")
    ax.set_title("Lambda Ablation: Performance")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # 7e: Lambda ablation — Semantic Penalty
    ax = axes[1, 1]
    pen_vals = [ablation_results[str(l)]["avg_penalty"] for l in lam_vals]
    ax.bar(range(len(lam_vals)), pen_vals, color="orange")
    ax.set_xticks(range(len(lam_vals)))
    ax.set_xticklabels([f"λ={l}" for l in lam_vals])
    ax.set_ylabel("Avg Semantic Penalty")
    ax.set_title("Lambda Ablation: Semantic Penalty")

    # 7f: Drawdown timeline
    ax = axes[1, 2]
    cum_returns = np.cumprod(1 + base_eval["returns"])
    peak = np.maximum.accumulate(cum_returns)
    dd_series = (peak - cum_returns) / peak
    ax.fill_between(range(len(dd_series)), 0, -dd_series, color="red", alpha=0.3)
    ax.plot(range(len(dd_series)), -dd_series, color="red", linewidth=0.5)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Drawdown")
    ax.set_title(f"Drawdown Timeline (MaxDD={base_eval['max_drawdown']:.2%})")

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "sp500_50_analysis.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close()

    # ─── 8. Save Results ─────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, "sp500_50_results.json")
    with open(results_path, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"  Saved: {results_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

"""
Staged MARL Training — Curriculum Learning for Hierarchical Portfolio Management
================================================================================
Author: Radheshyam Subedi | Student ID: U2829927

KEY IMPROVEMENT: Implements staged/curriculum training to solve the
instability of training interdependent agents simultaneously.

Training Phases:
  Phase 1 — Train Workers (Manager frozen at 1/3 equal allocation)
  Phase 2 — Train Manager (Workers frozen with learned policies)
  Phase 3 — Fine-tune all agents together (joint optimization)

Evaluation: 5-Metric Thesis Table
  1. Cumulative Return    — raw total profit
  2. Sharpe Ratio         — risk-adjusted return (total volatility)
  3. Sortino Ratio        — risk-adjusted return (downside volatility only)
  4. Maximum Drawdown     — worst peak-to-trough loss
  5. Lexical Ratio (HHI)  — semantic diversification measure

Comparisons:
  - Staged MARL vs Concurrent MARL (existing)
  - Both vs Equal-Weight (1/N) baseline
  - Both vs Mean-Variance Optimization (MVO) baseline
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings, json, time, copy
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("STAGED MARL TRAINING — Curriculum Learning")
print("=" * 70)

price_df = pd.read_csv("data/sp500_50_prices.csv", index_col=0, parse_dates=True)
lexical_df = pd.read_csv("data/processed/lexical_matrix_50.csv", index_col=0)
with open("data/processed/sector_map_50.json") as f:
    sector_map = json.load(f)

tickers = list(price_df.columns)
n_stocks = len(tickers)
returns_df = price_df.pct_change().dropna()
market_returns = returns_df.mean(axis=1)

# Beta classification
market_var = market_returns.var()
betas = {}
for ticker in tickers:
    cov = returns_df[ticker].cov(market_returns)
    betas[ticker] = cov / (market_var + 1e-10)

risk_pools = {'Safe': [], 'Neutral': [], 'Risky': []}
beta_labels = {}
for ticker, beta in betas.items():
    if beta < 0.8:
        risk_pools['Safe'].append(ticker)
        beta_labels[ticker] = 'Safe'
    elif beta <= 1.2:
        risk_pools['Neutral'].append(ticker)
        beta_labels[ticker] = 'Neutral'
    else:
        risk_pools['Risky'].append(ticker)
        beta_labels[ticker] = 'Risky'

print(f"\nStocks: {n_stocks}")
print(f"Date range: {price_df.index[0].date()} → {price_df.index[-1].date()}")
for profile in ['Safe', 'Neutral', 'Risky']:
    avg_b = np.mean([betas[t] for t in risk_pools[profile]])
    print(f"  {profile}: {len(risk_pools[profile])} stocks (avg β={avg_b:.3f})")


# ══════════════════════════════════════════════════════════════
# ENVIRONMENTS
# ══════════════════════════════════════════════════════════════
class WorkerEnv(gym.Env):
    """Worker agent: picks stocks within a risk-profile pool + Cash asset."""
    metadata = {"render_modes": []}
    TURNOVER_LIMIT = 0.5

    def __init__(self, price_df, lexical_matrix, tickers, profile="neutral",
                 window_size=30, lambda_penalty=0.1, gamma_penalty=0.01):
        super().__init__()
        self.profile = profile
        self.tickers = [t for t in tickers if t in price_df.columns and t in lexical_matrix.index]
        self.n_assets = len(self.tickers)
        self.n_total = self.n_assets + 1  # +1 for Cash
        self.prices = price_df[self.tickers].values
        self.returns_matrix = np.diff(self.prices, axis=0) / (self.prices[:-1] + 1e-8)
        self.lexical_matrix = np.nan_to_num(
            lexical_matrix.loc[self.tickers, self.tickers].values, nan=0.0)
        self.window_size = window_size
        self.lambda_penalty = lambda_penalty
        self.gamma_penalty = gamma_penalty

        if len(self.returns_matrix) > 1:
            self.cov_matrix = np.cov(self.returns_matrix.T)
            if self.cov_matrix.ndim == 0:
                self.cov_matrix = np.array([[float(self.cov_matrix)]])
        else:
            self.cov_matrix = np.eye(self.n_assets) * 0.01

        obs_dim = window_size * self.n_assets + self.n_total
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.01, high=10.0, shape=(self.n_total,), dtype=np.float32)

        self.current_step = 0
        self.portfolio_weights = np.ones(self.n_total) / self.n_total
        self.max_steps = len(self.prices) - window_size - 1
        self.portfolio_value = 1.0
        self.cum_return = 0.0
        self.recent_returns = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_weights = np.ones(self.n_total) / self.n_total
        self.portfolio_value = 1.0
        self.cum_return = 0.0
        self.recent_returns = []
        return self._get_obs(), {}

    def step(self, action):
        w_full = np.abs(action) + 1e-8
        w_full = w_full / np.sum(w_full)

        delta = w_full - self.portfolio_weights
        turnover = np.sum(np.abs(delta))
        if turnover > self.TURNOVER_LIMIT:
            delta = delta * (self.TURNOVER_LIMIT / turnover)
            w_full = self.portfolio_weights + delta
            w_full = np.maximum(w_full, 0.0)
            w_full = w_full / np.sum(w_full)
            turnover = self.TURNOVER_LIMIT

        w_stocks = w_full[:self.n_assets]
        w_cash = w_full[self.n_assets]

        curr = self.prices[self.current_step]
        nxt = self.prices[self.current_step + 1]
        pr = nxt / (curr + 1e-8)
        stock_ret = np.dot(w_stocks, pr)
        port_ret = max(stock_ret + w_cash * 1.0, 1e-8)
        log_ret = np.log(port_ret)

        if np.sum(w_stocks) > 1e-8:
            w_norm = w_stocks / (np.sum(w_stocks) + 1e-8)
            sem_pen = np.dot(w_norm.T, np.dot(self.lexical_matrix, w_norm))
        else:
            sem_pen = 0.0

        if self.profile == "safe":
            if np.sum(w_stocks) > 1e-8:
                w_norm = w_stocks / (np.sum(w_stocks) + 1e-8)
                port_vol = np.sqrt(max(np.dot(w_norm.T, np.dot(self.cov_matrix, w_norm)), 1e-10))
            else:
                port_vol = 0.0
            reward = float(log_ret - self.lambda_penalty * sem_pen - 2.0 * port_vol)
        elif self.profile == "risky":
            # Asymmetric reward: amplify gains, dampen losses to avoid crash amplification
            reward = float(1.5 * max(log_ret, 0) + 0.5 * min(log_ret, 0) - 0.5 * self.lambda_penalty * sem_pen)
        else:
            reward = float(log_ret - self.lambda_penalty * sem_pen - self.gamma_penalty * turnover)

        self.portfolio_weights = w_full
        self.portfolio_value *= port_ret
        self.cum_return += (port_ret - 1.0)
        self.recent_returns.append(port_ret - 1.0)
        if len(self.recent_returns) > 30:
            self.recent_returns.pop(0)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {
            'portfolio_return': port_ret, 'semantic_penalty': sem_pen,
            'turnover': turnover, 'weights': w_full.copy(),
            'stock_weights': w_stocks.copy(), 'cash_weight': w_cash,
            'cum_return': self.cum_return,
            'volatility': float(np.std(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0}

    def _get_obs(self):
        s = self.current_step - self.window_size
        win = self.prices[s:self.current_step]
        norm = win / (win[0, :] + 1e-8)
        return np.concatenate([norm.flatten(), self.portfolio_weights]).astype(np.float32)

    def get_worker_state(self):
        vol = float(np.std(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0
        return np.array([self.cum_return, vol], dtype=np.float32)


class ManagerEnv(gym.Env):
    """Manager agent: allocates capital across 3 risk-profile Workers."""
    metadata = {"render_modes": []}

    def __init__(self, price_df, lexical_matrix, risk_pools, window_size=30):
        super().__init__()
        self.risk_pools = risk_pools
        self.pool_names = ['Safe', 'Neutral', 'Risky']
        self.n_pools = 3

        self.pool_prices = {}
        for name in self.pool_names:
            pool_tickers = [t for t in risk_pools.get(name, []) if t in price_df.columns]
            if pool_tickers:
                self.pool_prices[name] = price_df[pool_tickers].mean(axis=1).values
            else:
                self.pool_prices[name] = np.ones(len(price_df))

        self.window_size = window_size
        self.all_prices = np.column_stack([self.pool_prices[p] for p in self.pool_names])

        self.pool_sim = np.zeros(self.n_pools)
        for i, name in enumerate(self.pool_names):
            pool_tickers = [t for t in risk_pools.get(name, []) if t in lexical_matrix.index]
            if len(pool_tickers) > 1:
                sim = lexical_matrix.loc[pool_tickers, pool_tickers].values
                mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
                self.pool_sim[i] = np.nanmean(sim[mask])

        obs_dim = window_size * self.n_pools + self.n_pools + self.n_pools + self.n_pools
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_pools,), dtype=np.float32)

        self.current_step = 0
        self.pool_weights = np.ones(self.n_pools) / self.n_pools
        self.max_steps = len(self.all_prices) - window_size - 1
        self.worker_cum_returns = np.zeros(self.n_pools, dtype=np.float32)
        self.worker_volatilities = np.zeros(self.n_pools, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.pool_weights = np.ones(self.n_pools) / self.n_pools
        self.worker_cum_returns = np.zeros(self.n_pools, dtype=np.float32)
        self.worker_volatilities = np.zeros(self.n_pools, dtype=np.float32)
        return self._get_obs(), {}

    def update_worker_states(self, worker_states):
        for i, name in enumerate(self.pool_names):
            if name in worker_states:
                self.worker_cum_returns[i] = worker_states[name][0]
                self.worker_volatilities[i] = worker_states[name][1]

    def step(self, action, worker_returns=None):
        v = np.abs(action) + 1e-8
        v = v / np.sum(v)

        if worker_returns is not None:
            global_ret = sum(v[i] * worker_returns.get(name, 1.0)
                             for i, name in enumerate(self.pool_names))
            global_ret = max(global_ret, 1e-8)
        else:
            curr = self.all_prices[self.current_step]
            nxt = self.all_prices[self.current_step + 1]
            pr = nxt / (curr + 1e-8)
            global_ret = max(np.dot(v, pr), 1e-8)

        log_ret = np.log(global_ret)
        turnover = np.sum(np.abs(v - self.pool_weights))
        reward = float(log_ret - 0.01 * turnover)

        self.pool_weights = v
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {
            'pool_weights': v.copy(), 'global_return': global_ret}

    def _get_obs(self):
        s = self.current_step - self.window_size
        win = self.all_prices[s:self.current_step]
        norm = win / (win[0, :] + 1e-8)
        return np.concatenate([
            norm.flatten(), self.pool_weights,
            self.worker_cum_returns, self.worker_volatilities
        ]).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# EIIE NETWORK
# ══════════════════════════════════════════════════════════════
class EIIENetwork(nn.Module):
    """EIIE with shared Conv1D weights, LayerNorm, Dropout + Dirichlet output."""
    def __init__(self, n_assets, window_size, n_price_assets=None):
        super().__init__()
        self.n_assets = n_assets
        self.n_price_assets = n_price_assets or n_assets
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        fc_in = self.n_price_assets * 32 + n_assets
        self.fc1 = nn.Linear(fc_in, 64)
        self.ln1 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, n_assets)

    def forward(self, pw, pvm):
        feats = []
        for i in range(self.n_price_assets):
            x = pw[:, i, :].unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.mean(dim=2)
            feats.append(x)
        c = torch.cat(feats + [pvm], dim=1)
        x = self.dropout(F.relu(self.ln1(self.fc1(c))))
        return F.softplus(self.fc2(x)) + 1.0


# ══════════════════════════════════════════════════════════════
# RISK METRICS
# ══════════════════════════════════════════════════════════════
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
        return float('inf')
    down_std = np.std(downside)
    if down_std < 1e-10:
        return float('inf')
    return float(np.mean(excess) / down_std * np.sqrt(252))

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

def compute_cumulative_return(returns):
    return float(np.prod(1 + returns) - 1)

def compute_sharpe(returns):
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

def dirichlet_mode(alpha):
    K = len(alpha)
    if np.all(alpha > 1.0) and np.sum(alpha) > K:
        return (alpha - 1.0) / (np.sum(alpha) - K)
    else:
        return alpha / np.sum(alpha)

def drawdown_decomposition(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    drawdowns = []
    in_dd, dd_start, max_depth, trough_idx = False, 0, 0, 0
    for i in range(len(dd)):
        if dd[i] > 0.001:
            if not in_dd:
                dd_start, in_dd, max_depth, trough_idx = i, True, dd[i], i
            elif dd[i] > max_depth:
                max_depth, trough_idx = dd[i], i
        else:
            if in_dd:
                drawdowns.append({'start': dd_start, 'trough': trough_idx, 'end': i,
                                  'depth': float(max_depth), 'duration': i - dd_start,
                                  'recovery': i - trough_idx})
                in_dd = False
    if in_dd:
        drawdowns.append({'start': dd_start, 'trough': trough_idx, 'end': len(dd)-1,
                          'depth': float(max_depth), 'duration': len(dd)-1-dd_start,
                          'recovery': len(dd)-1-trough_idx})
    return drawdowns


def five_metric_table(returns, weights_history=None, lexical_matrix=None, label=""):
    """Compute the 5 thesis-standard metrics."""
    cum_ret = compute_cumulative_return(returns)
    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)
    max_dd = compute_max_drawdown(returns)

    # Lexical ratio (HHI of average weights)
    if weights_history is not None and len(weights_history) > 0:
        avg_w = np.mean(weights_history, axis=0)
        hhi = compute_hhi(avg_w)
        eff_n = compute_effective_n(avg_w)
    else:
        hhi = 0.0
        eff_n = 0.0

    return {
        'label': label,
        'cumulative_return': cum_ret,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'hhi': hhi,
        'effective_n': eff_n,
    }


# ══════════════════════════════════════════════════════════════
# REINFORCE HELPER (used by all training phases)
# ══════════════════════════════════════════════════════════════
# EMA baselines: per-agent running average of episode returns for variance reduction
_ema_baselines = {}

def reinforce_update(log_probs, rewards, entropies, optimizer, net, entropy_bonus, gamma=0.99):
    """REINFORCE with EMA baseline and entropy bonus (lower variance than episode-mean)."""
    if not log_probs:
        return
    G, rets = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        rets.insert(0, G)
    rets = torch.FloatTensor(rets)

    # EMA baseline: use per-network running average instead of episode mean
    net_id = id(net)
    ep_mean = float(rets.mean())
    if net_id not in _ema_baselines:
        _ema_baselines[net_id] = ep_mean
    else:
        _ema_baselines[net_id] = 0.9 * _ema_baselines[net_id] + 0.1 * ep_mean
    baseline = _ema_baselines[net_id]

    rets = rets - baseline
    std = rets.std()
    if std > 1e-8:
        rets = rets / (std + 1e-8)

    policy_loss = sum(-lp * g for lp, g in zip(log_probs, rets))
    ent_loss = -entropy_bonus * sum(entropies)
    total_loss = policy_loss + ent_loss
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()


# ══════════════════════════════════════════════════════════════
# STAGED TRAINING — THE KEY IMPROVEMENT
# ══════════════════════════════════════════════════════════════
def train_staged(manager_env, worker_envs, 
                 phase1_episodes=300, phase2_episodes=200, phase3_episodes=100,
                 lr=3e-3, max_steps=200, verbose=True):
    """
    Curriculum/Staged Training:
      Phase 1: Workers train with Manager FROZEN at equal allocation (1/3 each)
      Phase 2: Manager trains with Workers FROZEN (using learned policies)
      Phase 3: Fine-tune all agents jointly (unfrozen)
    
    This prevents the instability of training interdependent agents simultaneously.
    Workers learn stable trading strategies first, then Manager learns to allocate
    across already-competent Workers.
    """
    pool_names = ['Safe', 'Neutral', 'Risky']
    
    # Initialize all networks
    m_n = manager_env.action_space.shape[0]
    m_ws = manager_env.window_size
    manager_net = EIIENetwork(m_n, m_ws)
    
    worker_nets, worker_opts, worker_schedulers = {}, {}, {}
    for name in pool_names:
        if name in worker_envs:
            env = worker_envs[name]
            worker_nets[name] = EIIENetwork(env.n_total, env.window_size, n_price_assets=env.n_assets)

    hist = {'phase': [], 'manager_rewards': [], 'worker_rewards': {n: [] for n in pool_names},
            'global_returns': [], 'allocations': [], 'phase_boundaries': []}

    # ─────────────────────────────────────────────────────────
    # PHASE 1: Train Workers, Manager frozen at 1/3 equal
    # ─────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: Training Workers (Manager FROZEN at 1/3 equal)")
        print("=" * 60)

    for name in pool_names:
        if name in worker_nets:
            worker_opts[name] = optim.Adam(worker_nets[name].parameters(), lr=lr)
            worker_schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(
                worker_opts[name], T_max=phase1_episodes, eta_min=5e-4)

    fixed_allocation = np.array([1.0/3, 1.0/3, 1.0/3])

    for ep in range(phase1_episodes):
        progress = ep / max(phase1_episodes - 1, 1)
        entropy_bonus = 0.05 * (1.0 - progress) + 0.005 * progress
        turnover_limit = 1.0 * (1.0 - progress) + 0.5 * progress
        for name in pool_names:
            if name in worker_envs:
                worker_envs[name].TURNOVER_LIMIT = turnover_limit

        # Reset envs
        m_obs, _ = manager_env.reset()
        w_obs = {}
        for name in pool_names:
            if name in worker_envs:
                w_obs[name], _ = worker_envs[name].reset()

        w_lps = {n: [] for n in pool_names}
        w_rews = {n: [] for n in pool_names}
        w_entropies = {n: [] for n in pool_names}
        m_rews = []
        done, steps = False, 0

        while not done and steps < max_steps:
            # Manager is FROZEN: always outputs equal allocation
            v = fixed_allocation.copy()

            worker_returns = {}
            for name in pool_names:
                if name not in worker_envs:
                    continue
                env = worker_envs[name]
                net = worker_nets[name]
                obs = w_obs[name]
                w_obs_dim = env.window_size * env.n_assets
                w_pw = torch.FloatTensor(obs[:w_obs_dim].reshape(env.n_assets, env.window_size)).unsqueeze(0)
                w_pvm = torch.FloatTensor(obs[w_obs_dim:w_obs_dim+env.n_total]).unsqueeze(0)
                w_alpha = net(w_pw, w_pvm).squeeze()
                w_alpha = torch.clamp(w_alpha, min=1.01, max=100.0)
                w_dist = torch.distributions.Dirichlet(w_alpha)
                w_action = w_dist.sample()
                w_lps[name].append(w_dist.log_prob(w_action))
                w_entropies[name].append(w_dist.entropy())

                obs_new, w_reward, w_term, _, w_info = env.step(w_action.detach().numpy())
                w_obs[name] = obs_new
                w_rews[name].append(w_reward)
                worker_returns[name] = w_info.get('portfolio_return', 1.0)
                if w_term:
                    done = True

            # Step manager env (for bookkeeping only, no gradient)
            m_obs_new, m_reward, m_term, _, m_info = manager_env.step(v, worker_returns=worker_returns)
            m_obs = m_obs_new
            m_rews.append(m_reward)
            if m_term:
                done = True
            steps += 1

        # Update Workers only (Manager frozen)
        for name in pool_names:
            if name not in worker_nets or not w_lps[name]:
                continue
            reinforce_update(w_lps[name], w_rews[name], w_entropies[name],
                             worker_opts[name], worker_nets[name], entropy_bonus)
            worker_schedulers[name].step()

        hist['phase'].append(1)
        hist['manager_rewards'].append(sum(m_rews))
        for name in pool_names:
            hist['worker_rewards'][name].append(sum(w_rews[name]) if w_rews[name] else 0)
        hist['allocations'].append(fixed_allocation.tolist())

        if verbose and (ep + 1) % 50 == 0:
            w_str = " | ".join(f"{n}={sum(w_rews.get(n, [])):+.3f}" for n in pool_names if n in worker_envs)
            print(f"  P1 Ep {ep+1:3d}/{phase1_episodes} | {w_str} | ent={entropy_bonus:.4f}")

    hist['phase_boundaries'].append(len(hist['manager_rewards']))
    if verbose:
        print(f"  Phase 1 complete: {phase1_episodes} episodes")

    # ─────────────────────────────────────────────────────────
    # PHASE 2: Train Manager, Workers frozen
    # ─────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Training Manager (Workers FROZEN)")
        print("=" * 60)

    manager_opt = optim.Adam(manager_net.parameters(), lr=lr)
    m_scheduler = optim.lr_scheduler.CosineAnnealingLR(manager_opt, T_max=phase2_episodes, eta_min=5e-4)

    # Freeze Worker networks
    for name in pool_names:
        if name in worker_nets:
            for param in worker_nets[name].parameters():
                param.requires_grad = False

    for ep in range(phase2_episodes):
        progress = ep / max(phase2_episodes - 1, 1)
        entropy_bonus = 0.04 * (1.0 - progress) + 0.005 * progress

        m_obs, _ = manager_env.reset()
        w_obs = {}
        for name in pool_names:
            if name in worker_envs:
                w_obs[name], _ = worker_envs[name].reset()
                worker_envs[name].TURNOVER_LIMIT = 0.5  # Workers use tight turnover

        m_lps, m_rews, m_entropies = [], [], []
        w_rews_track = {n: [] for n in pool_names}
        done, steps = False, 0

        while not done and steps < max_steps:
            # Worker states for Manager observation
            worker_states = {}
            for name in pool_names:
                if name in worker_envs:
                    worker_states[name] = worker_envs[name].get_worker_state()
            manager_env.update_worker_states(worker_states)
            m_obs = manager_env._get_obs()

            # Manager samples action
            obs_dim = m_ws * m_n
            m_pw = torch.FloatTensor(m_obs[:obs_dim].reshape(m_n, m_ws)).unsqueeze(0)
            m_pvm = torch.FloatTensor(m_obs[obs_dim:obs_dim+m_n]).unsqueeze(0)
            m_alpha = manager_net(m_pw, m_pvm).squeeze()
            m_alpha = torch.clamp(m_alpha, min=1.01, max=100.0)
            m_dist = torch.distributions.Dirichlet(m_alpha)
            m_action = m_dist.sample()
            m_lps.append(m_dist.log_prob(m_action))
            m_entropies.append(m_dist.entropy())

            # Workers act with FROZEN policies (no gradient)
            worker_returns = {}
            for name in pool_names:
                if name not in worker_envs:
                    continue
                env = worker_envs[name]
                net = worker_nets[name]
                obs = w_obs[name]
                w_obs_dim = env.window_size * env.n_assets
                w_pw = torch.FloatTensor(obs[:w_obs_dim].reshape(env.n_assets, env.window_size)).unsqueeze(0)
                w_pvm = torch.FloatTensor(obs[w_obs_dim:w_obs_dim+env.n_total]).unsqueeze(0)
                with torch.no_grad():
                    w_alpha = net(w_pw, w_pvm).squeeze().numpy()
                    w_w = dirichlet_mode(w_alpha)

                obs_new, _, w_term, _, w_info = env.step(w_w)
                w_obs[name] = obs_new
                worker_returns[name] = w_info.get('portfolio_return', 1.0)
                w_rews_track[name].append(w_info.get('portfolio_return', 1.0) - 1.0)
                if w_term:
                    done = True

            m_obs_new, m_reward, m_term, _, m_info = manager_env.step(
                m_action.detach().numpy(), worker_returns=worker_returns)
            m_obs = m_obs_new
            m_rews.append(m_reward)
            if m_term:
                done = True
            steps += 1

        # Update Manager only
        reinforce_update(m_lps, m_rews, m_entropies, manager_opt, manager_net, entropy_bonus)
        m_scheduler.step()

        hist['phase'].append(2)
        hist['manager_rewards'].append(sum(m_rews))
        for name in pool_names:
            hist['worker_rewards'][name].append(sum(w_rews_track[name]) if w_rews_track[name] else 0)
        if m_info and 'pool_weights' in m_info:
            hist['allocations'].append(m_info['pool_weights'].tolist())

        if verbose and (ep + 1) % 50 == 0:
            alloc = m_info['pool_weights'] if m_info and 'pool_weights' in m_info else fixed_allocation
            print(f"  P2 Ep {ep+1:3d}/{phase2_episodes} | MgrR={sum(m_rews):+.3f}"
                  f" | V=[{alloc[0]:.2f},{alloc[1]:.2f},{alloc[2]:.2f}]"
                  f" | lr={manager_opt.param_groups[0]['lr']:.5f}")

    hist['phase_boundaries'].append(len(hist['manager_rewards']))

    # Unfreeze Workers for Phase 3
    for name in pool_names:
        if name in worker_nets:
            for param in worker_nets[name].parameters():
                param.requires_grad = True

    if verbose:
        print(f"  Phase 2 complete: {phase2_episodes} episodes")

    # ─────────────────────────────────────────────────────────
    # PHASE 3: Joint fine-tuning (all unfrozen)
    # ─────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 3: Joint Fine-Tuning (all agents unfrozen)")
        print("=" * 60)

    # Reduce LR on existing optimizers (preserve momentum state from P1/P2)
    fine_lr = 5e-4
    for pg in manager_opt.param_groups:
        pg['lr'] = fine_lr
    for name in pool_names:
        if name in worker_opts:
            for pg in worker_opts[name].param_groups:
                pg['lr'] = fine_lr

    for ep in range(phase3_episodes):
        progress = ep / max(phase3_episodes - 1, 1)
        entropy_bonus = 0.01 * (1.0 - progress) + 0.002 * progress
        for name in pool_names:
            if name in worker_envs:
                worker_envs[name].TURNOVER_LIMIT = 0.5

        m_obs, _ = manager_env.reset()
        w_obs = {}
        for name in pool_names:
            if name in worker_envs:
                w_obs[name], _ = worker_envs[name].reset()

        m_lps, m_rews, m_entropies = [], [], []
        w_lps = {n: [] for n in pool_names}
        w_rews = {n: [] for n in pool_names}
        w_entropies = {n: [] for n in pool_names}
        done, steps = False, 0

        while not done and steps < max_steps:
            worker_states = {}
            for name in pool_names:
                if name in worker_envs:
                    worker_states[name] = worker_envs[name].get_worker_state()
            manager_env.update_worker_states(worker_states)
            m_obs = manager_env._get_obs()

            obs_dim = m_ws * m_n
            m_pw = torch.FloatTensor(m_obs[:obs_dim].reshape(m_n, m_ws)).unsqueeze(0)
            m_pvm = torch.FloatTensor(m_obs[obs_dim:obs_dim+m_n]).unsqueeze(0)
            m_alpha = manager_net(m_pw, m_pvm).squeeze()
            m_alpha = torch.clamp(m_alpha, min=1.01, max=100.0)
            m_dist = torch.distributions.Dirichlet(m_alpha)
            m_action = m_dist.sample()
            m_lps.append(m_dist.log_prob(m_action))
            m_entropies.append(m_dist.entropy())

            worker_returns = {}
            for name in pool_names:
                if name not in worker_envs:
                    continue
                env = worker_envs[name]
                net = worker_nets[name]
                obs = w_obs[name]
                w_obs_dim = env.window_size * env.n_assets
                w_pw = torch.FloatTensor(obs[:w_obs_dim].reshape(env.n_assets, env.window_size)).unsqueeze(0)
                w_pvm = torch.FloatTensor(obs[w_obs_dim:w_obs_dim+env.n_total]).unsqueeze(0)
                w_alpha = net(w_pw, w_pvm).squeeze()
                w_alpha = torch.clamp(w_alpha, min=1.01, max=100.0)
                w_dist = torch.distributions.Dirichlet(w_alpha)
                w_action = w_dist.sample()
                w_lps[name].append(w_dist.log_prob(w_action))
                w_entropies[name].append(w_dist.entropy())

                obs_new, w_reward, w_term, _, w_info = env.step(w_action.detach().numpy())
                w_obs[name] = obs_new
                w_rews[name].append(w_reward)
                worker_returns[name] = w_info.get('portfolio_return', 1.0)
                if w_term:
                    done = True

            m_obs_new, m_reward, m_term, _, m_info = manager_env.step(
                m_action.detach().numpy(), worker_returns=worker_returns)
            m_obs = m_obs_new
            m_rews.append(m_reward)
            if m_term:
                done = True
            steps += 1

        # Update ALL agents
        reinforce_update(m_lps, m_rews, m_entropies, manager_opt, manager_net, entropy_bonus)
        for name in pool_names:
            if name not in worker_nets or not w_lps[name]:
                continue
            reinforce_update(w_lps[name], w_rews[name], w_entropies[name],
                             worker_opts[name], worker_nets[name], entropy_bonus)

        hist['phase'].append(3)
        hist['manager_rewards'].append(sum(m_rews))
        for name in pool_names:
            hist['worker_rewards'][name].append(sum(w_rews[name]) if w_rews[name] else 0)
        if m_info and 'pool_weights' in m_info:
            hist['allocations'].append(m_info['pool_weights'].tolist())

        if verbose and (ep + 1) % 25 == 0:
            alloc = m_info['pool_weights'] if m_info and 'pool_weights' in m_info else np.ones(3)/3
            w_str = " | ".join(f"{n}={sum(w_rews.get(n, [])):+.3f}" for n in pool_names if n in worker_envs)
            print(f"  P3 Ep {ep+1:3d}/{phase3_episodes} | MgrR={sum(m_rews):+.3f}"
                  f" | V=[{alloc[0]:.2f},{alloc[1]:.2f},{alloc[2]:.2f}] | {w_str}")

    hist['phase_boundaries'].append(len(hist['manager_rewards']))
    total_eps = phase1_episodes + phase2_episodes + phase3_episodes
    if verbose:
        print(f"  Phase 3 complete: {phase3_episodes} episodes")
        print(f"\n  STAGED TRAINING COMPLETE: {total_eps} total episodes")

    return manager_net, worker_nets, hist


# ══════════════════════════════════════════════════════════════
# CONCURRENT TRAINING (existing baseline for comparison)
# ══════════════════════════════════════════════════════════════
def train_concurrent(manager_env, worker_envs, n_episodes=200, lr=3e-3, max_steps=200, verbose=False):
    """Original concurrent training (all agents train simultaneously)."""
    pool_names = ['Safe', 'Neutral', 'Risky']
    m_n = manager_env.action_space.shape[0]
    m_ws = manager_env.window_size
    manager_net = EIIENetwork(m_n, m_ws)
    manager_opt = optim.Adam(manager_net.parameters(), lr=lr)
    m_scheduler = optim.lr_scheduler.CosineAnnealingLR(manager_opt, T_max=n_episodes, eta_min=5e-4)

    worker_nets, worker_opts, worker_schedulers = {}, {}, {}
    for name in pool_names:
        if name in worker_envs:
            env = worker_envs[name]
            worker_nets[name] = EIIENetwork(env.n_total, env.window_size, n_price_assets=env.n_assets)
            worker_opts[name] = optim.Adam(worker_nets[name].parameters(), lr=lr)
            worker_schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(
                worker_opts[name], T_max=n_episodes, eta_min=5e-4)

    hist = {'manager_rewards': [], 'worker_rewards': {n: [] for n in pool_names},
            'global_returns': [], 'allocations': []}

    for ep in range(n_episodes):
        progress = ep / max(n_episodes - 1, 1)
        entropy_bonus = 0.05 * (1.0 - progress) + 0.005 * progress
        turnover_limit = 1.0 * (1.0 - progress) + 0.5 * progress
        for name in pool_names:
            if name in worker_envs:
                worker_envs[name].TURNOVER_LIMIT = turnover_limit

        m_obs, _ = manager_env.reset()
        w_obs = {}
        for name in pool_names:
            if name in worker_envs:
                w_obs[name], _ = worker_envs[name].reset()

        m_lps, m_rews, m_entropies = [], [], []
        w_lps = {n: [] for n in pool_names}
        w_rews = {n: [] for n in pool_names}
        w_entropies = {n: [] for n in pool_names}
        done, steps = False, 0

        while not done and steps < max_steps:
            worker_states = {}
            for name in pool_names:
                if name in worker_envs:
                    worker_states[name] = worker_envs[name].get_worker_state()
            manager_env.update_worker_states(worker_states)
            m_obs = manager_env._get_obs()

            obs_dim = m_ws * m_n
            m_pw = torch.FloatTensor(m_obs[:obs_dim].reshape(m_n, m_ws)).unsqueeze(0)
            m_pvm = torch.FloatTensor(m_obs[obs_dim:obs_dim+m_n]).unsqueeze(0)
            m_alpha = manager_net(m_pw, m_pvm).squeeze()
            m_alpha = torch.clamp(m_alpha, min=1.01, max=100.0)
            m_dist = torch.distributions.Dirichlet(m_alpha)
            m_action = m_dist.sample()
            m_lps.append(m_dist.log_prob(m_action))
            m_entropies.append(m_dist.entropy())

            worker_returns = {}
            for name in pool_names:
                if name not in worker_envs:
                    continue
                env = worker_envs[name]
                net = worker_nets[name]
                obs = w_obs[name]
                w_obs_dim = env.window_size * env.n_assets
                w_pw = torch.FloatTensor(obs[:w_obs_dim].reshape(env.n_assets, env.window_size)).unsqueeze(0)
                w_pvm = torch.FloatTensor(obs[w_obs_dim:w_obs_dim+env.n_total]).unsqueeze(0)
                w_alpha = net(w_pw, w_pvm).squeeze()
                w_alpha = torch.clamp(w_alpha, min=1.01, max=100.0)
                w_dist = torch.distributions.Dirichlet(w_alpha)
                w_action = w_dist.sample()
                w_lps[name].append(w_dist.log_prob(w_action))
                w_entropies[name].append(w_dist.entropy())

                obs_new, w_reward, w_term, _, w_info = env.step(w_action.detach().numpy())
                w_obs[name] = obs_new
                w_rews[name].append(w_reward)
                worker_returns[name] = w_info.get('portfolio_return', 1.0)
                if w_term:
                    done = True

            m_obs_new, m_reward, m_term, _, m_info = manager_env.step(
                m_action.detach().numpy(), worker_returns=worker_returns)
            m_obs = m_obs_new
            m_rews.append(m_reward)
            if m_term:
                done = True
            steps += 1

        # Update all agents
        reinforce_update(m_lps, m_rews, m_entropies, manager_opt, manager_net, entropy_bonus)
        m_scheduler.step()
        for name in pool_names:
            if name not in worker_nets or not w_lps[name]:
                continue
            reinforce_update(w_lps[name], w_rews[name], w_entropies[name],
                             worker_opts[name], worker_nets[name], entropy_bonus)
            worker_schedulers[name].step()

        hist['manager_rewards'].append(sum(m_rews))
        for name in pool_names:
            hist['worker_rewards'][name].append(sum(w_rews[name]) if w_rews[name] else 0)
        if m_info and 'pool_weights' in m_info:
            hist['allocations'].append(m_info['pool_weights'].tolist())

        if verbose and (ep + 1) % 50 == 0:
            alloc = m_info['pool_weights'] if m_info and 'pool_weights' in m_info else np.ones(3)/3
            print(f"  Ep {ep+1:3d} | MgrR={sum(m_rews):+.3f} | V=[{alloc[0]:.2f},{alloc[1]:.2f},{alloc[2]:.2f}]")

    return manager_net, worker_nets, hist


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════
def evaluate_system(manager_env, worker_envs, manager_net, worker_nets, max_steps=200):
    """Evaluate full hierarchy. Returns daily returns + weight history."""
    pool_names = ['Safe', 'Neutral', 'Risky']
    m_obs, _ = manager_env.reset()
    w_obs = {}
    for name in pool_names:
        if name in worker_envs:
            w_obs[name], _ = worker_envs[name].reset()

    global_returns, worker_returns_hist = [], {n: [] for n in pool_names}
    allocations, worker_weights = [], {n: [] for n in pool_names}
    done, steps = False, 0
    m_n = manager_env.action_space.shape[0]
    m_ws = manager_env.window_size

    while not done and steps < max_steps:
        worker_states = {}
        for name in pool_names:
            if name in worker_envs:
                worker_states[name] = worker_envs[name].get_worker_state()
        manager_env.update_worker_states(worker_states)
        m_obs = manager_env._get_obs()

        obs_dim = m_ws * m_n
        m_pw = torch.FloatTensor(m_obs[:obs_dim].reshape(m_n, m_ws)).unsqueeze(0)
        m_pvm = torch.FloatTensor(m_obs[obs_dim:obs_dim+m_n]).unsqueeze(0)
        with torch.no_grad():
            m_alpha = manager_net(m_pw, m_pvm).squeeze().numpy()
            v = dirichlet_mode(m_alpha)

        w_rets = {}
        for name in pool_names:
            if name not in worker_envs:
                continue
            env = worker_envs[name]
            net = worker_nets[name]
            obs = w_obs[name]
            w_od = env.window_size * env.n_assets
            w_pw = torch.FloatTensor(obs[:w_od].reshape(env.n_assets, env.window_size)).unsqueeze(0)
            w_pvm = torch.FloatTensor(obs[w_od:w_od+env.n_total]).unsqueeze(0)
            with torch.no_grad():
                w_alpha = net(w_pw, w_pvm).squeeze().numpy()
                w_w = dirichlet_mode(w_alpha)
            obs_new, _, w_term, _, w_info = env.step(w_w)
            w_obs[name] = obs_new
            w_rets[name] = w_info.get('portfolio_return', 1.0)
            worker_returns_hist[name].append(w_rets[name] - 1.0)
            worker_weights[name].append(w_info.get('stock_weights', w_w[:env.n_assets]).copy())
            if w_term:
                done = True

        m_obs_new, _, m_term, _, m_info = manager_env.step(v, worker_returns=w_rets)
        m_obs = m_obs_new

        global_ret = sum(v[i] * w_rets.get(name, 1.0) for i, name in enumerate(pool_names))
        global_returns.append(global_ret - 1.0)
        allocations.append(v.copy())

        if m_term:
            done = True
        steps += 1

    global_returns = np.array(global_returns)
    avg_alloc = np.mean(allocations, axis=0) if allocations else np.ones(3)/3

    result = {
        'global_returns': global_returns,
        'total_return': compute_cumulative_return(global_returns),
        'sharpe': compute_sharpe(global_returns),
        'sortino': compute_sortino(global_returns),
        'max_drawdown': compute_max_drawdown(global_returns),
        'cvar_95': compute_cvar(global_returns),
        'calmar': compute_calmar(global_returns),
        'avg_allocation': avg_alloc,
        'allocations': allocations,
        'worker_weights': worker_weights,
        'worker_results': {}
    }
    for name in pool_names:
        wr = np.array(worker_returns_hist[name]) if worker_returns_hist[name] else np.array([0.0])
        avg_ww = np.mean(worker_weights[name], axis=0) if worker_weights[name] else np.zeros(1)
        result['worker_results'][name] = {
            'total_return': compute_cumulative_return(wr),
            'sharpe': compute_sharpe(wr),
            'sortino': compute_sortino(wr),
            'max_drawdown': compute_max_drawdown(wr),
            'hhi': compute_hhi(avg_ww),
            'effective_n': compute_effective_n(avg_ww),
            'returns': wr,
        }
    return result


# ══════════════════════════════════════════════════════════════
# BASELINES: Equal-Weight and Mean-Variance Optimization
# ══════════════════════════════════════════════════════════════
def equal_weight_baseline(price_df, tickers, window_size=30):
    """Simple 1/N equal-weight buy-and-hold baseline."""
    prices = price_df[tickers].values
    n = len(tickers)
    w = np.ones(n) / n
    rets = []
    for t in range(window_size, len(prices) - 1):
        curr = prices[t]
        nxt = prices[t + 1]
        pr = nxt / (curr + 1e-8)
        port_ret = np.dot(w, pr) - 1.0
        rets.append(port_ret)
    return np.array(rets), w


def mvo_baseline(price_df, tickers, window_size=30, lookback=120):
    """Mean-Variance Optimization baseline (Markowitz).
    Rebalances daily using rolling lookback window.
    Uses minimum-variance portfolio with target return constraint."""
    prices = price_df[tickers].values
    n = len(tickers)
    rets_all = []
    weights_all = []
    
    for t in range(window_size, len(prices) - 1):
        # Rolling lookback for covariance estimation
        start = max(0, t - lookback)
        hist_prices = prices[start:t+1]
        if len(hist_prices) < 20:
            w = np.ones(n) / n
        else:
            hist_rets = np.diff(hist_prices, axis=0) / (hist_prices[:-1] + 1e-8)
            mu = np.mean(hist_rets, axis=0)
            cov = np.cov(hist_rets.T)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            # Add regularization
            cov += np.eye(n) * 1e-6
            
            try:
                cov_inv = np.linalg.inv(cov)
                ones = np.ones(n)
                # Minimum variance portfolio
                w = cov_inv @ ones / (ones @ cov_inv @ ones)
                # Ensure non-negative (long-only)
                w = np.maximum(w, 0)
                if np.sum(w) > 1e-8:
                    w = w / np.sum(w)
                else:
                    w = np.ones(n) / n
            except np.linalg.LinAlgError:
                w = np.ones(n) / n
        
        curr = prices[t]
        nxt = prices[t + 1]
        pr = nxt / (curr + 1e-8)
        port_ret = np.dot(w, pr) - 1.0
        rets_all.append(port_ret)
        weights_all.append(w.copy())
    
    return np.array(rets_all), np.array(weights_all)


# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════
t0 = time.time()

# Create environments
worker_envs = {}
for profile in ['Safe', 'Neutral', 'Risky']:
    pool_tickers = risk_pools[profile]
    if len(pool_tickers) >= 2:
        worker_envs[profile] = WorkerEnv(
            price_df, lexical_df, pool_tickers,
            profile=profile.lower(), window_size=30,
            lambda_penalty=0.1, gamma_penalty=0.01)
        print(f"  {profile} Worker: {len(pool_tickers)} stocks")

manager_env = ManagerEnv(price_df, lexical_df, risk_pools, window_size=30)
print(f"  Manager: allocating across {len(worker_envs)} risk pools\n")

# ─────────────────────────────────────────────────────────
# A) STAGED TRAINING (new approach)
# ─────────────────────────────────────────────────────────
print("\n" + "█" * 70)
print("  EXPERIMENT A: STAGED/CURRICULUM TRAINING")
print("█" * 70)

staged_mgr_net, staged_w_nets, staged_hist = train_staged(
    manager_env, worker_envs,
    phase1_episodes=200,  # Workers learn stable strategies
    phase2_episodes=160,  # Manager learns to allocate across competent Workers
    phase3_episodes=80,   # Joint fine-tuning
    lr=3e-3, max_steps=200, verbose=True
)

# Evaluate staged system
staged_eval = evaluate_system(manager_env, worker_envs, staged_mgr_net, staged_w_nets, max_steps=200)
print(f"\nStaged Training Time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────
# B) CONCURRENT TRAINING (existing approach — for comparison)
# ─────────────────────────────────────────────────────────
print("\n" + "█" * 70)
print("  EXPERIMENT B: CONCURRENT TRAINING (baseline)")
print("█" * 70)

# Recreate fresh environments for fair comparison
worker_envs_c = {}
for profile in ['Safe', 'Neutral', 'Risky']:
    pool_tickers = risk_pools[profile]
    if len(pool_tickers) >= 2:
        worker_envs_c[profile] = WorkerEnv(
            price_df, lexical_df, pool_tickers,
            profile=profile.lower(), window_size=30,
            lambda_penalty=0.1, gamma_penalty=0.01)

manager_env_c = ManagerEnv(price_df, lexical_df, risk_pools, window_size=30)

t1 = time.time()
conc_mgr_net, conc_w_nets, conc_hist = train_concurrent(
    manager_env_c, worker_envs_c, n_episodes=440, lr=3e-3, max_steps=200, verbose=True
)

# Recreate envs for eval
worker_envs_ce = {}
for profile in ['Safe', 'Neutral', 'Risky']:
    pool_tickers = risk_pools[profile]
    if len(pool_tickers) >= 2:
        worker_envs_ce[profile] = WorkerEnv(
            price_df, lexical_df, pool_tickers,
            profile=profile.lower(), window_size=30,
            lambda_penalty=0.1, gamma_penalty=0.01)
manager_env_ce = ManagerEnv(price_df, lexical_df, risk_pools, window_size=30)

conc_eval = evaluate_system(manager_env_ce, worker_envs_ce, conc_mgr_net, conc_w_nets, max_steps=200)
print(f"\nConcurrent Training Time: {time.time()-t1:.1f}s")

# ─────────────────────────────────────────────────────────
# HOLDOUT TEST SET EVALUATION (last 20% of data, unseen)
# ─────────────────────────────────────────────────────────
print("\n" + "█" * 70)
print("  HOLDOUT TEST-SET EVALUATION (last 20% of data)")
print("█" * 70)

split_idx = int(len(price_df) * 0.8)
test_prices = price_df.iloc[split_idx:]
print(f"  Test period: {test_prices.index[0].date()} -> {test_prices.index[-1].date()} ({len(test_prices)} days)")

# Staged on holdout
holdout_worker_envs = {}
for profile in ['Safe', 'Neutral', 'Risky']:
    pool_tickers = risk_pools[profile]
    valid_t = [t for t in pool_tickers if t in test_prices.columns]
    if len(valid_t) >= 2:
        holdout_worker_envs[profile] = WorkerEnv(
            test_prices, lexical_df, valid_t,
            profile=profile.lower(), window_size=30,
            lambda_penalty=0.1, gamma_penalty=0.01)
holdout_mgr_env = ManagerEnv(test_prices, lexical_df, risk_pools, window_size=30)
holdout_staged_eval = evaluate_system(holdout_mgr_env, holdout_worker_envs, staged_mgr_net, staged_w_nets, max_steps=200)

# Concurrent on holdout
holdout_worker_envs_c = {}
for profile in ['Safe', 'Neutral', 'Risky']:
    pool_tickers = risk_pools[profile]
    valid_t = [t for t in pool_tickers if t in test_prices.columns]
    if len(valid_t) >= 2:
        holdout_worker_envs_c[profile] = WorkerEnv(
            test_prices, lexical_df, valid_t,
            profile=profile.lower(), window_size=30,
            lambda_penalty=0.1, gamma_penalty=0.01)
holdout_mgr_env_c = ManagerEnv(test_prices, lexical_df, risk_pools, window_size=30)
holdout_conc_eval = evaluate_system(holdout_mgr_env_c, holdout_worker_envs_c, conc_mgr_net, conc_w_nets, max_steps=200)

# EW/MVO on holdout
holdout_valid_tickers = [t for t in tickers if t in test_prices.columns]
holdout_ew_rets, _ = equal_weight_baseline(test_prices, holdout_valid_tickers, window_size=30)
holdout_mvo_rets, _ = mvo_baseline(test_prices, holdout_valid_tickers, window_size=30)

holdout_min_len = min(len(holdout_staged_eval['global_returns']), len(holdout_ew_rets), len(holdout_mvo_rets))
print(f"\n  HOLDOUT RESULTS (unseen test data):")
print(f"  {'Method':<20} | {'Return':>10} | {'Sharpe':>10} | {'Sortino':>10} | {'MaxDD':>10}")
print(f"  {'-'*70}")
for label, rets in [('Staged MARL', holdout_staged_eval['global_returns'][:holdout_min_len]),
                     ('Concurrent MARL', holdout_conc_eval['global_returns'][:holdout_min_len]),
                     ('Equal-Weight', holdout_ew_rets[:holdout_min_len]),
                     ('MVO', holdout_mvo_rets[:holdout_min_len])]:
    r = np.array(rets)
    print(f"  {label:<20} | {compute_cumulative_return(r):>+10.4f} | {compute_sharpe(r):>10.4f} | "
          f"{compute_sortino(r):>10.4f} | {compute_max_drawdown(r):>10.4f}")

# ─────────────────────────────────────────────────────────
# C) BASELINES
# ─────────────────────────────────────────────────────────
print("\n" + "█" * 70)
print("  BASELINES: Equal-Weight (1/N) & Mean-Variance Optimization")
print("█" * 70)

# All tickers for baselines
valid_tickers = [t for t in tickers if t in price_df.columns]
ew_returns, ew_weights = equal_weight_baseline(price_df, valid_tickers, window_size=30)
mvo_returns, mvo_weights = mvo_baseline(price_df, valid_tickers, window_size=30)

# Truncate to same length as MARL eval for fair comparison
min_len = min(len(staged_eval['global_returns']), len(ew_returns), len(mvo_returns))
ew_returns_trim = ew_returns[:min_len]
mvo_returns_trim = mvo_returns[:min_len]
staged_returns_trim = staged_eval['global_returns'][:min_len]
conc_returns_trim = conc_eval['global_returns'][:min_len]


# ══════════════════════════════════════════════════════════════
# 5-METRIC THESIS TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FIVE-METRIC THESIS TABLE — Master's Degree Standard")
print("=" * 90)

# Compute HHI for MARL systems (using Worker stock weights)
def compute_global_hhi(eval_result, worker_envs, lexical_df):
    """Compute portfolio-level HHI across all stocks."""
    pool_names = ['Safe', 'Neutral', 'Risky']
    all_stock_weights = []
    for name in pool_names:
        if name in eval_result['worker_results'] and name in worker_envs:
            wr = eval_result['worker_results'][name]
            alloc = eval_result['avg_allocation'][pool_names.index(name)]
            if 'returns' in wr and name in eval_result.get('worker_weights', {}):
                ww_list = eval_result['worker_weights'][name]
                if ww_list:
                    avg_ww = np.mean(ww_list, axis=0)
                    all_stock_weights.extend(alloc * avg_ww)
    if all_stock_weights:
        all_stock_weights = np.array(all_stock_weights)
        if np.sum(all_stock_weights) > 1e-8:
            all_stock_weights = all_stock_weights / np.sum(all_stock_weights)
        return compute_hhi(all_stock_weights), compute_effective_n(all_stock_weights)
    return 0.0, 0.0


staged_hhi, staged_effn = compute_global_hhi(staged_eval, worker_envs, lexical_df)
conc_hhi, conc_effn = compute_global_hhi(conc_eval, worker_envs_ce, lexical_df)
ew_hhi = compute_hhi(np.ones(len(valid_tickers)) / len(valid_tickers))
ew_effn = len(valid_tickers)
mvo_avg_w = np.mean(mvo_weights, axis=0) if len(mvo_weights) > 0 else np.ones(len(valid_tickers)) / len(valid_tickers)
mvo_hhi = compute_hhi(mvo_avg_w)
mvo_effn = compute_effective_n(mvo_avg_w)

# Build table rows
metrics_table = {
    'Staged MARL': {
        'Cumulative Return': compute_cumulative_return(staged_returns_trim),
        'Sharpe Ratio': compute_sharpe(staged_returns_trim),
        'Sortino Ratio': compute_sortino(staged_returns_trim),
        'Max Drawdown': compute_max_drawdown(staged_returns_trim),
        'HHI (Lexical)': staged_hhi,
        'Effective N': staged_effn,
    },
    'Concurrent MARL': {
        'Cumulative Return': compute_cumulative_return(conc_returns_trim),
        'Sharpe Ratio': compute_sharpe(conc_returns_trim),
        'Sortino Ratio': compute_sortino(conc_returns_trim),
        'Max Drawdown': compute_max_drawdown(conc_returns_trim),
        'HHI (Lexical)': conc_hhi,
        'Effective N': conc_effn,
    },
    'Equal-Weight (1/N)': {
        'Cumulative Return': compute_cumulative_return(ew_returns_trim),
        'Sharpe Ratio': compute_sharpe(ew_returns_trim),
        'Sortino Ratio': compute_sortino(ew_returns_trim),
        'Max Drawdown': compute_max_drawdown(ew_returns_trim),
        'HHI (Lexical)': ew_hhi,
        'Effective N': ew_effn,
    },
    'MVO (Markowitz)': {
        'Cumulative Return': compute_cumulative_return(mvo_returns_trim),
        'Sharpe Ratio': compute_sharpe(mvo_returns_trim),
        'Sortino Ratio': compute_sortino(mvo_returns_trim),
        'Max Drawdown': compute_max_drawdown(mvo_returns_trim),
        'HHI (Lexical)': mvo_hhi,
        'Effective N': mvo_effn,
    },
}

# Print formatted table
header = f"{'Metric':<22} | {'Staged MARL':>14} | {'Concurrent MARL':>16} | {'Equal-Weight':>14} | {'MVO':>14}"
print(header)
print("-" * len(header))

metric_keys = ['Cumulative Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'HHI (Lexical)', 'Effective N']
for mk in metric_keys:
    vals = []
    for method in ['Staged MARL', 'Concurrent MARL', 'Equal-Weight (1/N)', 'MVO (Markowitz)']:
        v = metrics_table[method][mk]
        if mk == 'Cumulative Return':
            vals.append(f"{v:+.2%}")
        elif mk == 'Max Drawdown':
            vals.append(f"{v:.2%}")
        elif mk == 'Effective N':
            vals.append(f"{v:.1f}")
        else:
            vals.append(f"{v:.4f}")
    print(f"{mk:<22} | {vals[0]:>14} | {vals[1]:>16} | {vals[2]:>14} | {vals[3]:>14}")

# ─────── Worker-level breakdown ───────
print("\n" + "=" * 90)
print("  WORKER-LEVEL PERFORMANCE (Staged MARL)")
print("=" * 90)
print(f"{'Worker':<10} | {'Return':>10} | {'Sharpe':>10} | {'Sortino':>10} | {'MaxDD':>10} | {'HHI':>10} | {'Eff N':>8}")
print("-" * 80)
for name in ['Safe', 'Neutral', 'Risky']:
    wr = staged_eval['worker_results'].get(name, {})
    print(f"{name:<10} | {wr.get('total_return',0):>+10.4f} | {wr.get('sharpe',0):>10.4f} | "
          f"{wr.get('sortino',0):>10.4f} | {wr.get('max_drawdown',0):>10.4f} | "
          f"{wr.get('hhi',0):>10.4f} | {wr.get('effective_n',0):>8.1f}")

print(f"\nManager Allocation (Staged): Safe={staged_eval['avg_allocation'][0]:.2%}, "
      f"Neutral={staged_eval['avg_allocation'][1]:.2%}, Risky={staged_eval['avg_allocation'][2]:.2%}")

# ══════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION (with staged training)
# ══════════════════════════════════════════════════════════════
print("\n" + "█" * 70)
print("  WALK-FORWARD VALIDATION (Staged Training)")
print("█" * 70)

wf_windows = [
    ("2015-2018 → 2019 (Bull)", "2015-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
    ("2015-2019 → 2020 (COVID)", "2015-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
    ("2015-2020 → 2021 (Recovery)", "2015-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
    ("2015-2021 → 2022 (Bear)", "2015-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
]

wf_results = {}
for wf_name, train_s, train_e, test_s, test_e in wf_windows:
    print(f"\n--- {wf_name} ---")
    train_prices = price_df.loc[train_s:train_e]
    test_prices = price_df.loc[test_s:test_e]

    if len(train_prices) < 60 or len(test_prices) < 20:
        print(f"  Skipping — insufficient data")
        continue

    wf_worker_envs = {}
    for profile in ['Safe', 'Neutral', 'Risky']:
        pool_tickers = risk_pools[profile]
        valid_t = [t for t in pool_tickers if t in train_prices.columns]
        if len(valid_t) >= 2:
            wf_worker_envs[profile] = WorkerEnv(
                train_prices, lexical_df, valid_t,
                profile=profile.lower(), window_size=30,
                lambda_penalty=0.1, gamma_penalty=0.01)

    wf_mgr_env = ManagerEnv(train_prices, lexical_df, risk_pools, window_size=30)

    # Staged training for walk-forward (reduced episodes for speed)
    wf_mgr_net, wf_w_nets, _ = train_staged(
        wf_mgr_env, wf_worker_envs,
        phase1_episodes=120, phase2_episodes=80, phase3_episodes=40,
        verbose=False
    )

    # Test
    test_worker_envs = {}
    for profile in ['Safe', 'Neutral', 'Risky']:
        pool_tickers = risk_pools[profile]
        valid_t = [t for t in pool_tickers if t in test_prices.columns]
        if len(valid_t) >= 2:
            test_worker_envs[profile] = WorkerEnv(
                test_prices, lexical_df, valid_t,
                profile=profile.lower(), window_size=30,
                lambda_penalty=0.1, gamma_penalty=0.01)

    test_mgr_env = ManagerEnv(test_prices, lexical_df, risk_pools, window_size=30)
    wf_eval = evaluate_system(test_mgr_env, test_worker_envs, wf_mgr_net, wf_w_nets, max_steps=200)

    wf_results[wf_name] = wf_eval
    print(f"  Return: {wf_eval['total_return']:+.4f}, Sharpe: {wf_eval['sharpe']:.4f}, "
          f"Sortino: {wf_eval['sortino']:.4f}, MaxDD: {wf_eval['max_drawdown']:.4f}")

# Walk-Forward summary table
if wf_results:
    print("\n" + "=" * 90)
    print("  WALK-FORWARD: 5-METRIC TABLE PER REGIME")
    print("=" * 90)
    print(f"{'Window':<30} | {'CumRet':>10} | {'Sharpe':>10} | {'Sortino':>10} | {'MaxDD':>10} | {'CVaR95':>10}")
    print("-" * 95)
    for wf_name, ev in wf_results.items():
        short = wf_name.split('→')[1].strip()[:20] if '→' in wf_name else wf_name[:20]
        print(f"{short:<30} | {ev['total_return']:>+10.4f} | {ev['sharpe']:>10.4f} | "
              f"{ev['sortino']:>10.4f} | {ev['max_drawdown']:>10.4f} | {ev['cvar_95']:>10.4f}")

# ══════════════════════════════════════════════════════════════
# LAMBDA ABLATION (with staged training)
# ══════════════════════════════════════════════════════════════
print("\n" + "█" * 70)
print("  LAMBDA ABLATION (Staged Training)")
print("█" * 70)

lambdas = [0.0, 0.1, 0.5]
ablation_results = {}

for lam in lambdas:
    print(f"\n  λ = {lam}...")
    abl_worker_envs = {}
    for profile in ['Safe', 'Neutral', 'Risky']:
        pool_tickers = risk_pools[profile]
        if len(pool_tickers) >= 2:
            abl_worker_envs[profile] = WorkerEnv(
                price_df, lexical_df, pool_tickers,
                profile=profile.lower(), window_size=30,
                lambda_penalty=lam, gamma_penalty=0.01)

    abl_mgr_env = ManagerEnv(price_df, lexical_df, risk_pools, window_size=30)
    abl_mgr_net, abl_w_nets, _ = train_staged(
        abl_mgr_env, abl_worker_envs,
        phase1_episodes=120, phase2_episodes=80, phase3_episodes=40,
        verbose=False
    )

    # Recreate for eval
    abl_worker_envs_e = {}
    for profile in ['Safe', 'Neutral', 'Risky']:
        pool_tickers = risk_pools[profile]
        if len(pool_tickers) >= 2:
            abl_worker_envs_e[profile] = WorkerEnv(
                price_df, lexical_df, pool_tickers,
                profile=profile.lower(), window_size=30,
                lambda_penalty=lam, gamma_penalty=0.01)
    abl_mgr_env_e = ManagerEnv(price_df, lexical_df, risk_pools, window_size=30)

    abl_eval = evaluate_system(abl_mgr_env_e, abl_worker_envs_e, abl_mgr_net, abl_w_nets, max_steps=200)
    ablation_results[lam] = abl_eval
    print(f"    Return={abl_eval['total_return']:+.4f}, Sharpe={abl_eval['sharpe']:.4f}, "
          f"Sortino={abl_eval['sortino']:.4f}, Alloc=[{abl_eval['avg_allocation'][0]:.2f},"
          f"{abl_eval['avg_allocation'][1]:.2f},{abl_eval['avg_allocation'][2]:.2f}]")

# Ablation table
print("\n" + "=" * 90)
print("  LAMBDA ABLATION: 5-METRIC TABLE")
print("=" * 90)
print(f"{'Lambda':>8} | {'CumRet':>10} | {'Sharpe':>10} | {'Sortino':>10} | {'MaxDD':>10} | {'Alloc S/N/R':>18}")
print("-" * 75)
for lam in lambdas:
    ev = ablation_results[lam]
    alloc = ev['avg_allocation']
    print(f"{lam:>8.2f} | {ev['total_return']:>+10.4f} | {ev['sharpe']:>10.4f} | "
          f"{ev['sortino']:>10.4f} | {ev['max_drawdown']:>10.4f} | "
          f"{alloc[0]:>5.1%}/{alloc[1]:>5.1%}/{alloc[2]:>5.1%}")


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
print("\n\nGenerating plots...")

# Figure 1: Training curves by phase
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Staged MARL Training — Curriculum Learning", fontsize=14, fontweight='bold')

# Manager reward by phase
ax = axes[0, 0]
phase_colors = {1: 'blue', 2: 'orange', 3: 'green'}
for phase_num, color in phase_colors.items():
    idxs = [i for i, p in enumerate(staged_hist['phase']) if p == phase_num]
    if idxs:
        ax.plot(idxs, [staged_hist['manager_rewards'][i] for i in idxs],
                color=color, alpha=0.6, label=f'Phase {phase_num}')
for b in staged_hist['phase_boundaries'][:-1]:
    ax.axvline(b, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Manager Reward')
ax.set_title('Manager Reward (by Training Phase)')
ax.legend()

# Worker rewards
ax = axes[0, 1]
for name, color in [('Safe', 'green'), ('Neutral', 'steelblue'), ('Risky', 'red')]:
    vals = staged_hist['worker_rewards'][name]
    if vals:
        # Smoothed
        smooth = pd.Series(vals).rolling(20, min_periods=1).mean()
        ax.plot(smooth, color=color, label=name, alpha=0.8)
for b in staged_hist['phase_boundaries'][:-1]:
    ax.axvline(b, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Worker Reward (smoothed)')
ax.set_title('Worker Rewards (by Training Phase)')
ax.legend()

# Equity curves comparison
ax = axes[1, 0]
ax.plot(np.cumprod(1 + staged_returns_trim), label='Staged MARL', color='blue', linewidth=2)
ax.plot(np.cumprod(1 + conc_returns_trim), label='Concurrent MARL', color='orange', linewidth=1.5, alpha=0.8)
ax.plot(np.cumprod(1 + ew_returns_trim), label='Equal-Weight', color='grey', linewidth=1, alpha=0.7)
ax.plot(np.cumprod(1 + mvo_returns_trim), label='MVO', color='purple', linewidth=1, alpha=0.7)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Cumulative Value ($1)')
ax.set_title('Equity Curves: Staged vs Concurrent vs Baselines')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 5-metric bar chart
ax = axes[1, 1]
methods = ['Staged\nMARL', 'Concurrent\nMARL', 'Equal\nWeight', 'MVO']
sharpes = [metrics_table[m]['Sharpe Ratio'] for m in ['Staged MARL', 'Concurrent MARL', 'Equal-Weight (1/N)', 'MVO (Markowitz)']]
sortinos = [metrics_table[m]['Sortino Ratio'] for m in ['Staged MARL', 'Concurrent MARL', 'Equal-Weight (1/N)', 'MVO (Markowitz)']]
x = np.arange(len(methods))
w_bar = 0.35
ax.bar(x - w_bar/2, sharpes, w_bar, label='Sharpe', color='steelblue', alpha=0.8)
ax.bar(x + w_bar/2, sortinos, w_bar, label='Sortino', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Ratio')
ax.set_title('Sharpe & Sortino Comparison')
ax.legend()
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('staged_training_results.png', dpi=150, bbox_inches='tight')
print("  Saved: staged_training_results.png")

# Figure 2: Walk-Forward + Ablation
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

if wf_results:
    ax = axes[0]
    wf_names_short = [w.split('→')[1].strip()[:15] if '→' in w else w[:15] for w in wf_results.keys()]
    wf_rets = [wf_results[w]['total_return'] for w in wf_results]
    wf_sharpes = [wf_results[w]['sharpe'] for w in wf_results]
    colors = ['green' if r > 0 else 'red' for r in wf_rets]
    x = np.arange(len(wf_names_short))
    bars = ax.bar(x - 0.15, wf_rets, 0.3, color=colors, alpha=0.7, label='Return')
    ax2_wf = ax.twinx()
    ax2_wf.plot(x, wf_sharpes, 's-', color='purple', label='Sharpe', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(wf_names_short, fontsize=8)
    ax.set_ylabel('OOS Return')
    ax2_wf.set_ylabel('Sharpe', color='purple')
    ax.set_title('Walk-Forward (Staged Training)')
    ax.legend(loc='upper left', fontsize=8)
    ax2_wf.legend(loc='upper right', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)

if ablation_results:
    ax = axes[1]
    lam_vals = list(ablation_results.keys())
    abl_sharpes = [ablation_results[l]['sharpe'] for l in lam_vals]
    abl_sortinos = [ablation_results[l]['sortino'] for l in lam_vals]
    ax.plot(lam_vals, abl_sharpes, 'o-', color='steelblue', label='Sharpe', linewidth=2)
    ax.plot(lam_vals, abl_sortinos, 's-', color='coral', label='Sortino', linewidth=2)
    ax.set_xlabel('λ (Semantic Penalty)')
    ax.set_ylabel('Ratio')
    ax.set_title('Lambda Ablation (Staged Training)')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('walkforward_ablation_results.png', dpi=150, bbox_inches='tight')
print("  Saved: walkforward_ablation_results.png")

# Figure 3: Drawdown comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

cum_staged = np.cumprod(1 + staged_returns_trim)
peak_staged = np.maximum.accumulate(cum_staged)
dd_staged = (peak_staged - cum_staged) / peak_staged

cum_conc = np.cumprod(1 + conc_returns_trim)
peak_conc = np.maximum.accumulate(cum_conc)
dd_conc = (peak_conc - cum_conc) / peak_conc

axes[0].plot(cum_staged, label='Staged MARL', color='blue', linewidth=1.5)
axes[0].plot(cum_conc, label='Concurrent MARL', color='orange', linewidth=1.5, alpha=0.8)
axes[0].axhline(1.0, color='grey', linestyle='--', alpha=0.5)
axes[0].set_ylabel('Cumulative Value')
axes[0].set_title('Staged vs Concurrent: Equity Curves')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].fill_between(range(len(dd_staged)), 0, -dd_staged, color='blue', alpha=0.3, label='Staged')
axes[1].fill_between(range(len(dd_conc)), 0, -dd_conc, color='orange', alpha=0.3, label='Concurrent')
axes[1].set_xlabel('Trading Day')
axes[1].set_ylabel('Drawdown')
axes[1].set_title('Drawdown Comparison')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('drawdown_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: drawdown_comparison.png")


# ══════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ══════════════════════════════════════════════════════════════
def to_serializable(obj):
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

all_results = {
    'experiment': 'Staged vs Concurrent MARL Training',
    'total_runtime_seconds': time.time() - t0,
    'staged_training': {
        'phases': '200W + 160M + 80J = 440 total',
        'global_metrics': {k: to_serializable(v) for k, v in staged_eval.items()
                          if k not in ('global_returns', 'allocations', 'worker_weights', 'worker_results')},
        'allocation': {name: float(staged_eval['avg_allocation'][i])
                      for i, name in enumerate(['Safe', 'Neutral', 'Risky'])},
        'worker_results': {name: {k: to_serializable(v) for k, v in wr.items() if k != 'returns'}
                          for name, wr in staged_eval['worker_results'].items()},
    },
    'concurrent_training': {
        'episodes': 440,
        'global_metrics': {k: to_serializable(v) for k, v in conc_eval.items()
                          if k not in ('global_returns', 'allocations', 'worker_weights', 'worker_results')},
        'allocation': {name: float(conc_eval['avg_allocation'][i])
                      for i, name in enumerate(['Safe', 'Neutral', 'Risky'])},
    },
    'five_metric_table': metrics_table,
    'walk_forward': {w: {'total_return': float(ev['total_return']),
                         'sharpe': float(ev['sharpe']),
                         'sortino': float(ev['sortino']),
                         'max_drawdown': float(ev['max_drawdown']),
                         'cvar_95': float(ev['cvar_95'])}
                    for w, ev in wf_results.items()},
    'lambda_ablation': {str(l): {'total_return': float(ev['total_return']),
                                  'sharpe': float(ev['sharpe']),
                                  'sortino': float(ev['sortino']),
                                  'allocation': [float(a) for a in ev['avg_allocation']]}
                       for l, ev in ablation_results.items()},
    'baselines': {
        'equal_weight': {
            'cumulative_return': float(compute_cumulative_return(ew_returns_trim)),
            'sharpe': float(compute_sharpe(ew_returns_trim)),
            'sortino': float(compute_sortino(ew_returns_trim)),
            'max_drawdown': float(compute_max_drawdown(ew_returns_trim)),
        },
        'mvo': {
            'cumulative_return': float(compute_cumulative_return(mvo_returns_trim)),
            'sharpe': float(compute_sharpe(mvo_returns_trim)),
            'sortino': float(compute_sortino(mvo_returns_trim)),
            'max_drawdown': float(compute_max_drawdown(mvo_returns_trim)),
        },
    },
    'holdout_test': {
        'test_period': f"{test_prices.index[0].date()} to {test_prices.index[-1].date()}",
        'staged': {
            'cumulative_return': float(compute_cumulative_return(holdout_staged_eval['global_returns'][:holdout_min_len])),
            'sharpe': float(compute_sharpe(holdout_staged_eval['global_returns'][:holdout_min_len])),
            'sortino': float(compute_sortino(holdout_staged_eval['global_returns'][:holdout_min_len])),
            'max_drawdown': float(compute_max_drawdown(holdout_staged_eval['global_returns'][:holdout_min_len])),
        },
        'concurrent': {
            'cumulative_return': float(compute_cumulative_return(holdout_conc_eval['global_returns'][:holdout_min_len])),
            'sharpe': float(compute_sharpe(holdout_conc_eval['global_returns'][:holdout_min_len])),
        },
        'equal_weight': {
            'cumulative_return': float(compute_cumulative_return(holdout_ew_rets[:holdout_min_len])),
            'sharpe': float(compute_sharpe(holdout_ew_rets[:holdout_min_len])),
        },
    },
}

with open("data/processed/staged_training_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nResults saved to data/processed/staged_training_results.json")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("\n" + "=" * 70)
print("  DONE — Staged MARL Training Complete")
print("=" * 70)

# Full Project Context Summary

## Project Title Context
Structuring Resilience: Hierarchical Multi-Agent Portfolio Optimization with Semantic Diversification using a Weight-Aware Lexical Penalty.

## 1. Project Objective
The project builds and evaluates a hierarchical multi-agent reinforcement learning (MARL) system for portfolio optimization.

Core objective:
- Improve out-of-sample risk-adjusted performance versus simple and classical baselines.
- Reduce structural concentration risk that price-only methods miss.
- Improve robustness under stressed market regimes.

Research intent:
- Move from statistical diversification (correlation-based) toward structural diversification (business-similarity aware).

## 2. Problem Being Solved
Traditional methods and many RL portfolio models face key weaknesses:
- 1/N trap: model converges to near-uniform allocations.
- Softmax output bias: difficult to generate high-conviction, sparse allocations.
- Price-only blindness: no awareness of fundamental business overlap between firms.
- Instability in flat or jointly trained multi-agent setups.

## 3. Technology Stack
Primary stack:
- Python
- PyTorch
- Gymnasium
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- yfinance
- requests

Data artifacts:
- Price data CSVs for 50-stock and 100-stock universes.
- Processed lexical similarity matrices (TF-IDF cosine similarity).
- Sector maps and diagnostics JSON outputs.

## 4. Data and Feature Pipeline
### Financial data
- Daily price data (primarily from Yahoo Finance).
- Returns and beta are computed for each stock.
- Stocks are grouped by risk profile using beta thresholds:
  - Safe
  - Neutral
  - Risky

### Semantic data
- Business-description text is collected from SEC EDGAR 10-K (Item 1), with fallback text sources in some runs.
- Text is cleaned and vectorized using TF-IDF.
- Pairwise cosine similarity creates lexical matrix S.

### Why this matters
- Matrix S is used in reward shaping to penalize semantically similar holdings.
- This encourages structural diversification beyond price correlations.

## 5. MARL System Architecture
### Hierarchical structure
- Manager agent:
  - Allocates portfolio capital across Safe/Neutral/Risky workers.
- Worker agents:
  - Allocate inside their own asset pool.
  - Include a cash dimension in action space for defensive behavior.

### Policy model
- EIIE-style Conv1D shared feature extraction.
- Portfolio vector memory input.
- Dirichlet output policy (instead of softmax) for non-uniform allocations.

### Reward design
- Portfolio return term.
- Weight-aware lexical penalty: w^T S w.
- Turnover penalty.
- Profile-specific shaping:
  - Safe: stronger risk control.
  - Neutral: balanced objective.
  - Risky: return-seeking objective.

## 6. Training Methods
### Concurrent baseline
- All agents trained simultaneously.
- Used as a reference for comparison.

### Staged curriculum training (key improvement)
- Phase 1: train workers, manager fixed.
- Phase 2: train manager, workers frozen.
- Phase 3: joint fine-tuning.

Purpose:
- Reduce inter-agent instability.
- Improve specialization and coordination.

## 7. Evaluation Framework
Main evaluation methods:
- Walk-forward validation across multiple market regimes.
- Holdout testing on unseen period.
- Lambda ablation study for semantic penalty strength.
- Baseline comparisons:
  - Equal-weight (1/N)
  - Markowitz MVO
  - In some scripts: risk parity and momentum

Common metrics used:
- Cumulative return
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- CVaR / Calmar where applicable
- HHI and Effective N (concentration/diversification indicators)

## 8. Thought Process Behind the Design
Design logic is coherent and layered:
1. Financial markets are regime-shifting and noisy.
2. Flat RL often converges to low-conviction behavior.
3. Structural overlap risk is not visible in pure price tensors.
4. Therefore:
   - use hierarchy to separate macro and micro allocation tasks,
   - use Dirichlet policy to reduce softmax-induced uniformity,
   - use NLP penalty to discourage redundant fundamental exposure,
   - use staged training for stability.

## 9. Performance Summary (Important)
There are multiple result snapshots in the repository from different phases.

### Thesis narrative and interpretation docs indicate
- Staged MARL improved clearly over concurrent.
- Holdout performance around +8.86% return and Sharpe around 0.577 in the strongest reported run.
- Outperformed equal-weight and MVO in that holdout setting.

### Some diagnostics snapshots indicate
- More modest global performance in certain runs (for example around +2.15% return, Sharpe around 0.26).
- Manager allocation heavily concentrated in Neutral pool in some outputs.
- Residual issues (for example near-1/N behavior in specific workers, or low manager movement in earlier diagnostics).

Interpretation:
- The project is technically strong and directionally successful.
- Results in the workspace represent multiple experiment versions and maturity levels, so not all files reflect the final best run.

## 10. Current State of the Thesis Workspace
- Core design and implementation are substantial and research-grade.
- Training code evolved from concurrent to staged approach.
- Supporting chapter markdown files exist, but some chapter files are placeholders while the consolidated thesis report is much more developed.
- Result consistency across all files is not fully synchronized yet.

## 11. Practical Next Step
To finalize and use this work effectively, create one canonical final-results package:
- One frozen experiment configuration.
- One definitive metrics table set.
- One canonical JSON artifact referenced by thesis chapters.
- One reproducibility checklist (data files, seeds, command to run, expected outputs).

This will align code outputs, interpretation notes, and thesis chapters into a single defendable final narrative.

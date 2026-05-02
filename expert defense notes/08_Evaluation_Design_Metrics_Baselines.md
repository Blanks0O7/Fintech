# Evaluation Design, Metrics, and Baselines

## Why evaluation design is critical in finance
A finance model is easy to overstate if evaluation is weak. Strong evaluation is as important as model design.

## Main evaluation principle
- The project must be tested out of sample.
- Time order must be preserved.
- Multiple risk measures must be reported.

## Why walk-forward validation was used
### Simple answer
- Financial data is time ordered.
- Random cross-validation would leak future information.

### Strong answer
- "Walk-forward validation preserves temporal structure, avoids leakage, and tests the strategy across different market regimes."

## Why not random k-fold cross-validation
- It breaks chronology.
- It lets training data indirectly include future behavior.
- In finance, that creates unrealistic optimism.

## Why a holdout period matters
- A final holdout period gives an extra layer of honesty.
- It is data the model did not see during development.
- It helps reduce hidden overfitting.

## Baselines used in the dissertation
### Equal-weight (1/N)
Why included:
- It is the classic benchmark.
- It is surprisingly hard to beat out of sample.
- It directly tests whether the model escapes the 1/N trap.

### Mean-variance optimization (MVO)
Why included:
- It is the classical benchmark from modern portfolio theory.
- It shows how the new system compares with traditional optimization.

### Risk parity
Why included:
- It is a risk-based allocation benchmark.
- It is useful because the dissertation emphasizes structural resilience.

### Momentum
Why included:
- It is a strong rules-based active benchmark.
- It provides a non-RL alternative.

### Concurrent MARL
Why included:
- It is the internal architectural control.
- It tests whether staged training helps.

## Why these baselines are a strong set
- They cover naive, classical, risk-based, rules-based, and internal-ML comparisons.
- That makes the comparison more balanced than using only one or two baselines.

## Performance metrics
The dissertation uses multiple metrics because no single metric is enough.

### Cumulative return
What it tells you:
- Total growth over the period.

What it misses:
- It ignores how much risk was taken.

### Sharpe ratio
What it tells you:
- Return per unit of total volatility.

Why experts care:
- It is the standard risk-adjusted metric.

### Sortino ratio
What it tells you:
- Return per unit of downside volatility only.

Why it matters here:
- The project cares about downside protection.

### Maximum drawdown
What it tells you:
- Worst peak-to-trough loss.

Why it is central in this dissertation:
- The dissertation emphasizes resilience and capital protection.

### HHI and Effective N
What they tell you:
- How concentrated the portfolio weights are.

Why they are useful:
- They help show that diversification is not only about return but also about concentration structure.

### CVaR
What it tells you:
- Average loss in the worst tail region.

Why it matters:
- It is a stronger tail-risk measure than variance alone.

### Calmar ratio
What it tells you:
- Return relative to maximum drawdown.

Why useful:
- Good for evaluating return versus worst-case downside.

## Why so many metrics are necessary
Best answer:
- "Portfolio quality is multi-dimensional. A strategy can look good on return but poor on drawdown, or good on Sharpe but bad on tail risk. Multiple metrics prevent misleading conclusions."

## Why Sharpe is still emphasized
- It is widely recognized.
- It provides a common comparison language.
- But it is not treated as sufficient by itself.

## Why drawdown is especially important here
- The thesis is about structural resilience.
- Drawdown is one of the clearest practical measures of resilience.

## Why not evaluate only on average return
- High return could come from concentrated risk.
- That would contradict the diversification goal.

## Why not use only one crisis window
- One crisis is not enough to generalize.
- That is why walk-forward windows and regime analysis are useful.

## Why regime analysis matters
- It shows whether performance is stable across different market environments.
- A model that works only in one regime is less convincing.

## Common expert challenges
### "Why did you not do statistical significance tests everywhere?"
Answer:
- This is a valid extension. The project prioritizes structural comparisons, ablations, and out-of-sample evaluation. Multi-seed and formal statistical inference would strengthen future work.

### "Why should we trust backtest metrics?"
Answer:
- You should be cautious. That is exactly why the dissertation includes walk-forward design, holdout testing, multiple baselines, and control experiments rather than a single in-sample backtest.

### "Why compare to equal-weight if the goal is semantic diversification?"
Answer:
- Because equal-weight is the most important practical sanity check in the portfolio literature. If the model cannot justify itself against 1/N, the sophistication has limited value.

## Best summary sentence
- "The evaluation design was built to answer not only whether the model earns returns, but whether it earns them in a robust, risk-aware, and scientifically defensible way."

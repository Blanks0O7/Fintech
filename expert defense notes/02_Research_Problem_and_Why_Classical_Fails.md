# Research Problem and Why Classical Diversification Fails

## The core problem
Traditional portfolio diversification usually relies on price covariance or correlation. That works reasonably well in stable periods, but it often fails in crises. Assets that looked independent before may suddenly crash together.

## The simple explanation
- Price history tells you how assets moved.
- It does not always tell you why they moved.
- In a crisis, the hidden economic links become visible.
- Then the portfolio turns out to be less diversified than it looked.

## The dissertation's problem statement
The dissertation tackles two linked problems:
1. The 1/N trap in portfolio optimization and RL.
2. Price-blind diversification that ignores business similarity.

## Problem 1: the 1/N trap
### What it means
- The 1/N trap means complex optimization methods often fail to outperform a simple equal-weight portfolio out of sample.
- In RL, this can happen because the model learns that spreading weights evenly is a safe way to reduce noisy reward variance.

### Why experts care
- If a sophisticated model collapses to near-equal weight, then the model is not adding much value.
- A valid portfolio model should justify why it deviates from equal weighting.

### Why this project cares
- The dissertation wants concentrated but justified allocations.
- That is why it replaces Softmax with Dirichlet and adds a staged training schedule.

## Problem 2: price blindness
### What it means
- Price-blindness means the model only sees market prices and ignores what companies actually do.
- If many firms depend on the same supply chain, same customers, or same interest-rate sensitivity, a price-only model may miss that common exposure.

### Example you can use in the defense
- "If I buy many technology firms that all depend on semiconductors, that may look diversified by ticker count, but it may still be one concentrated economic bet."

## Why Markowitz-style diversification can fail
### What Markowitz gets right
- Mean-variance optimization is mathematically elegant.
- It formalized the trade-off between expected return and variance.
- It introduced the efficient frontier.

### What Markowitz gets wrong in practice
- It needs estimates of expected return and covariance.
- Those estimates are noisy.
- Small input errors can create large weight changes.
- Correlations are unstable, especially in crises.

### Strong defense answer
- "Markowitz is not wrong mathematically. Its weakness is that real financial inputs are noisy and unstable. The method is very sensitive to estimation error and regime change."

## Tail dependence and crisis co-movement
### Key idea
- In bad markets, correlations tend to increase.
- Assets fall together more than historical averages suggested.
- This is why apparent diversification disappears when it is needed most.

### Keywords to use
- Tail dependence
- Correlation breakdown
- Regime shift
- Stress co-movement
- Systemic shock

## Why business text is a reasonable solution
### Simple version
- Business descriptions are more stable than daily prices.
- They reflect products, operations, customers, and strategy.
- Similar descriptions may signal similar fundamental exposures.

### Stronger academic version
- Business-description similarity is a proxy for economic overlap.
- That overlap may explain co-movement that industry labels or short-term covariance do not fully capture.

## Why not just use sector labels?
### Good answer
- Sector labels are too coarse.
- Two firms in the same sector may be very different.
- Two firms in different sectors may still share business exposure.
- Text gives a more continuous similarity measure than a hard label.

### Short version for fast answering
- "Sector labels are categorical. Text similarity is continuous and richer."

## Why not just use fundamental ratios?
### Good answer
- Ratios like P/E, debt, or book-to-market describe financial condition.
- They do not directly describe business overlap.
- This project specifically needs a signal for operational similarity, not only valuation.

## Why not only use news or sentiment?
### Good answer
- News and sentiment are high frequency but noisy.
- They can be event-driven and unstable.
- 10-K filings are formal, audited, and less noisy.
- They better represent long-term business structure.

## The real motivation behind the project
- Not just to earn more return.
- Not just to build a more complex model.
- The real goal is to reduce hidden concentration risk and improve resilience.

## Best "why this dissertation matters" answer
- "The dissertation matters because it questions whether diversification should be defined only by price history. It proposes that a portfolio is safer when the businesses inside it are genuinely different, not only historically uncorrelated."

## What experts may challenge here
### Challenge 1
- "Aren't prices already forward-looking?"

Answer:
- Prices are forward-looking, but they are compressed summaries of many effects. They do not explain the economic structure behind those effects. Two firms can look independent in price data until a common structural shock reveals the overlap.

### Challenge 2
- "Why should text from annual filings help with daily trading?"

Answer:
- The purpose is not daily alpha forecasting from text. The purpose is structural diversification. Annual filings are appropriate because business models change slowly compared with daily prices.

### Challenge 3
- "Is this a return-maximization project or a risk-control project?"

Answer:
- Primarily a risk-control and resilience project. Return matters, but drawdown protection and structural diversification are central goals.

## One line to remember
- "The project starts where classical diversification becomes weakest: when price independence is only apparent, not fundamental."

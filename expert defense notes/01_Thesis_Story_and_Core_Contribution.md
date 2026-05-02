# Thesis Story and Core Contribution

## The simplest possible summary
- Classical diversification often fails during market stress because it only looks at historical price co-movement.
- This project tries to diversify at the business level, not only at the price level.
- It does that by reading company 10-K business descriptions and penalizing portfolios that put too much weight on companies with similar business models.
- That semantic signal is inserted into a Hierarchical Multi-Agent Reinforcement Learning system.
- The final idea is structural diversification, not only statistical diversification.

## The full story in simple language
The project starts from a practical finance problem. In normal times, assets may look uncorrelated based on past prices. But in crisis periods, many of those same assets fall together. That means historical covariance can be misleading. It measures what happened in prices, but not why those price movements happened.

The dissertation argues that true independence should be measured partly through business fundamentals. If two companies have similar products, supply chains, customers, and regulatory exposures, then they may carry hidden common risk even if their past prices do not show strong correlation.

To capture that hidden common risk, the project uses company business descriptions from SEC 10-K filings. It converts those texts into TF-IDF vectors, then builds a cosine similarity matrix. That matrix becomes a diversification penalty inside the reinforcement learning reward.

The reinforcement learning system is hierarchical. A manager agent allocates capital across three worker agents. Those workers operate in beta-based risk pools:
- Safe
- Neutral
- Risky

This architecture separates high-level risk budgeting from low-level stock selection. It also lets each worker specialize.

## What is genuinely novel here
If an examiner asks, "What is the real contribution?", answer with this structure.

### Contribution 1: semantic diversification inside RL
- Many portfolio systems use only price data.
- This project injects NLP-derived business similarity directly into the reward function.
- The lexical penalty is not just a report statistic after training.
- It is part of the learning signal during training.

Best answer:
- "The main novelty is that semantic business similarity is used as a trainable reward penalty inside reinforcement learning, rather than as a separate descriptive analysis."

### Contribution 2: hierarchical manager-worker structure
- Flat systems make one agent solve everything at once.
- This project separates global capital allocation from local stock selection.
- The manager chooses the risk-pool mix.
- The workers choose assets within their pools.

Best answer:
- "The hierarchy improves credit assignment and specialization. The manager focuses on top-level risk allocation, while workers focus on stock selection under profile-specific objectives."

### Contribution 3: staged curriculum training
- Concurrent training can be unstable because every agent changes at the same time.
- This project trains workers first, then the manager, then fine-tunes jointly.
- That reduces non-stationarity.

Best answer:
- "Staged training was chosen to reduce the moving-target problem in MARL. The manager should learn against stable workers, not workers that are still changing every step."

### Contribution 4: Dirichlet policy instead of Softmax
- Softmax often encourages smoother, more uniform allocations.
- The Dirichlet policy is more natural on the simplex and can represent high-conviction portfolios.

Best answer:
- "Dirichlet is a better distribution for portfolio weights because it lives on the simplex directly and supports concentrated as well as diversified allocations."

## The central research questions
### Research Question 1
- Can a Hierarchical MARL system with a lexical penalty improve risk-adjusted performance and drawdown protection compared with price-only approaches?

### Research Question 2
- Does the strength of the semantic penalty, controlled by lambda, causally affect portfolio quality?

## The one-sentence thesis claim
- "Semantic business similarity provides useful diversification information that historical prices alone do not fully capture, and embedding that information inside a staged hierarchical RL system improves structural resilience."

## Keywords experts may expect you to use
- Structural diversification
- Statistical diversification
- Hidden common exposure
- Semantic similarity
- Reward shaping
- Hierarchical control
- Credit assignment
- Non-stationarity
- Simplex-constrained policy
- Risk-adjusted performance
- Drawdown control

## Why this problem matters academically
- It challenges the assumption that price covariance is enough.
- It connects finance with NLP and RL.
- It tests whether text-derived business information has practical value in portfolio optimization.
- It addresses a real gap: many RL portfolio papers remain price-only.

## Why this problem matters practically
- Investors care about capital protection during stress periods.
- Institutions care about hidden concentration risk.
- Business-description overlap may reveal concentration that sector labels or price history miss.

## If asked "Why should anyone trust this?"
Answer:
- The project does not just propose an idea.
- It tests the idea through ablations, walk-forward validation, baseline comparison, and a null-hypothesis control.
- The null-hypothesis control is especially important because it asks whether the semantic content matters, not just the existence of a penalty term.

## If asked "What is the project not claiming?"
Answer carefully:
- It is not claiming perfect prediction.
- It is not claiming it beats every baseline in every regime.
- It is not claiming that text alone drives returns.
- It is claiming that business-text similarity can improve diversification quality, especially in terms of structural risk control.

## Short defense close
- "This dissertation is not only about building an RL trader. It is about testing whether we can build more meaningful diversification by combining market data with business semantics."

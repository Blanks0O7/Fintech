# Methodology Chapter Walkthrough

## Why this file exists
If an examiner says, "Walk me through your methodology chapter step by step," use this file.

## Step 1: define the problem clearly
- Classical diversification uses price covariance.
- That can fail in stress periods.
- I therefore wanted a diversification signal based on business similarity.

## Step 2: choose the data
- Daily price data for tradable portfolio behavior
- 10-K business descriptions for semantic structure

## Step 3: build the semantic matrix
- Clean text
- TF-IDF vectorize
- Compute cosine similarity
- Use that matrix as S in the lexical penalty

## Step 4: build risk pools
- Estimate beta
- Split assets into Safe, Neutral, Risky
- Use those pools as worker universes

## Step 5: define the hierarchy
- Manager allocates across pools
- Workers allocate within pools
- This separates risk budgeting from stock picking

## Step 6: define the policy representation
- Use an EIIE-style feature extractor
- Feed previous weights through portfolio vector memory
- Output Dirichlet concentration parameters
- Sample simplex-valid portfolio weights

## Step 7: define the rewards
- Every worker has its own objective
- All workers include the lexical penalty
- The safe worker penalizes volatility more heavily
- The neutral worker penalizes turnover
- The risky worker emphasizes return more strongly

## Step 8: train the system
- Phase 1: workers only
- Phase 2: manager only
- Phase 3: joint fine-tuning

## Step 9: evaluate the system
- Compare against multiple baselines
- Use walk-forward and holdout logic
- Measure return, Sharpe, Sortino, MaxDD, CVaR, HHI, Effective N, and Calmar where relevant

## Step 10: test causal claims
- Lambda ablation tests whether semantic strength matters
- Null-hypothesis control tests whether semantic structure matters

## Why this methodology is coherent
Each step solves one earlier problem:
- Data choice solves information-source mismatch
- Hierarchy solves role separation
- Dirichlet solves action-space mismatch
- WALP solves hidden semantic concentration
- Staged training solves non-stationarity
- Walk-forward solves time-series evaluation realism

## If asked "What would break if one part were removed?"
Use this table:
- Remove text matrix -> back to price-only diversification
- Remove hierarchy -> lose specialization and interpretability
- Remove Dirichlet -> weaker support for concentrated simplex allocations
- Remove staged training -> more unstable manager-worker co-learning
- Remove walk-forward -> weaker evaluation credibility

## Best short answer
- "The methodology chapter is a chain of design decisions, where each choice answers a weakness identified in the literature review."

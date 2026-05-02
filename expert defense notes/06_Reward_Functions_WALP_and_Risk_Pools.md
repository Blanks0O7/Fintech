# Reward Functions, WALP, and Risk Pools

## Why reward design matters so much
In reinforcement learning, the reward is the objective the agent actually learns. If the reward is poorly designed, the behavior will also be poor.

Best answer:
- "In portfolio RL, reward design is almost equivalent to investment philosophy. It defines what the agent considers a good portfolio."

## Why the workers have different rewards
### Simple answer
- The workers should not all learn the same behavior.
- If they do, the hierarchy becomes pointless.

### Strong answer
- "Profile-specific rewards are used to create purposeful behavioral diversity. The hierarchy only adds value if the workers specialize rather than converge to identical policies."

## Safe worker reward
### Goal
- Capital preservation
- Lower volatility
- Defensive behavior

### How it is shaped
- Log return
- Minus lexical penalty
- Minus portfolio volatility term

### Why this makes sense
- A defensive worker should prefer stable, lower-risk positions.
- It still seeks return, but under stronger risk control.

## Neutral worker reward
### Goal
- Balanced risk-return trade-off
- Turnover-aware behavior

### How it is shaped
- Log return
- Minus lexical penalty
- Minus turnover penalty

### Why turnover matters here
- Neutral strategies often benefit from stability and lower friction.
- Turnover penalty discourages unnecessary rebalancing.

## Risky worker reward
### Goal
- Pursue alpha more aggressively
- Accept more risk

### How it is shaped
- Return term is amplified
- Lexical penalty is still present but lighter

### Why this makes sense
- The risky worker should not behave like the safe worker.
- It must be allowed to seek higher-upside opportunities.

## Why use log return
### Good answer
- Log return is numerically stable and common in finance.
- It behaves well under multiplicative compounding.
- It is a reasonable step reward for sequential portfolio growth.

## What WALP is
WALP stands for Weight-Aware Lexical Penalty.

Formula:
- lambda times w transpose S w

Where:
- w = current portfolio weight vector
- S = semantic similarity matrix
- lambda = penalty strength

## The intuition behind WALP
### Very simple version
- If two companies are very similar and you put a lot of money in both, the penalty becomes large.
- If they are dissimilar, or if their weights are small, the penalty is smaller.

### Why this is smart
- It is not enough to know similarity alone.
- Similarity becomes risky when capital concentration is also high.
- The quadratic form captures both at the same time.

## Why the penalty is weight-aware
- Because a tiny weight in a similar company is not the same as a huge weight.
- A good diversification penalty must care about exposure size, not just pair similarity.

## Why the quadratic form is elegant
### Good answer
- "The form w transpose S w is compact, differentiable, and naturally penalizes concentration in semantically overlapping assets."

### Why differentiability matters
- The penalty can be integrated directly into gradient-based RL training.
- It becomes part of the learning signal, not only a post-hoc evaluation metric.

## Why not use HHI alone
- HHI measures concentration by weights only.
- It does not know whether the concentrated assets are economically similar or different.
- WALP is richer because it includes both concentration and business overlap.

## Why not use sector diversification only
- Sector diversification is a coarse rule.
- WALP is a continuous similarity penalty rather than a hard category constraint.

## Why not use covariance as the penalty
- Covariance is price-derived and backward-looking.
- The whole point of WALP is to add a non-price diversification signal.

## Why lambda exists
### What lambda does
- It controls how strongly the agent cares about semantic overlap.

### Why this parameter is important
- If lambda is zero, there is no semantic penalty.
- If lambda is too high, the penalty may dominate return seeking.
- The ablation study tests where the useful balance lies.

### Best answer
- "Lambda is the tuning knob that decides how much structural diversification pressure the model feels."

## Why the penalty is applied at the worker level
- Workers choose actual stock weights.
- That is where semantic overlap is directly created.
- The manager allocates across pools, but the workers create intra-pool stock concentration.

## Risk-pool logic
### Why Safe, Neutral, Risky pools exist
- They make the hierarchy meaningful.
- They let workers specialize.
- They make manager decisions interpretable.

### Why not use random pools
- Random pools would remove finance meaning.
- Experts want the hierarchy to map to a real risk concept.

### Why beta-based pools are defensible
- Beta is a standard finance measure of market sensitivity.
- It creates an interpretable ladder of systematic risk.

## Common expert challenges
### "Is WALP just another regularizer?"
Answer:
- It is a regularizer in mathematical form, but its information source is meaningful. The null-hypothesis control is designed to test whether the semantic content matters beyond generic regularization.

### "Does WALP reduce return too much?"
Answer:
- That is exactly why lambda is tuned through ablation. The project does not assume stronger penalty is always better.

### "Can semantic similarity really proxy economic co-exposure?"
Answer:
- Not perfectly, but it is a defensible approximation. It is more informative than ticker count and often richer than sector labels. The literature supports that business-text similarity reflects real competitive and product overlap.

## One line to remember
- "WALP penalizes hidden sameness, not just visible concentration."

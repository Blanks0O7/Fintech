# Architecture: Manager, Workers, EIIE, and Dirichlet Policy

## High-level architecture
The system is hierarchical.
- One manager agent
- Three worker agents
- Safe worker
- Neutral worker
- Risky worker

## Why hierarchical instead of one big agent
### Simple answer
- A single agent must solve too many tasks at once.
- The hierarchy separates global allocation from local stock selection.

### Strong answer
- "Hierarchy improves specialization and reduces decision complexity. The manager solves inter-pool capital allocation, while workers solve intra-pool selection under profile-specific objectives."

## What each agent does
### Manager
- Decides how much capital goes to Safe, Neutral, and Risky pools.
- It is a top-level allocator.
- It does not pick individual stocks directly.

### Workers
- Each worker only sees its own risk-pool assets.
- Each worker builds a sub-portfolio inside that pool.
- Each worker has a different reward emphasis.

## Why this helps
- Safe worker can learn defensive behavior.
- Risky worker can learn aggressive behavior.
- Neutral worker can focus on balance and turnover.
- The manager can learn when to prefer one style over another.

## Why not use separate agents for every stock
- That creates too much agent interference.
- Credit assignment becomes hard.
- Communication becomes noisy.
- The system may collapse into fragmented or unstable learning.

## Why not use only one worker and no manager
- Then the model loses modularity and interpretability.
- You cannot clearly separate risk budgeting from stock selection.
- It becomes harder to explain why the portfolio shifts toward safer or riskier exposures.

## EIIE backbone
### What EIIE means
- Ensemble of Identical Independent Evaluators

### Core idea
- The same convolutional filters are applied across assets.
- The network learns common local patterns rather than memorizing asset identities.

### Why this was chosen
Best answer:
- "EIIE is useful because it promotes cross-asset pattern learning. It reduces the chance that the model simply memorizes one specific ticker's history."

## Portfolio Vector Memory
### What it is
- The previous portfolio weights are fed back into the network.

### Why it matters
- Portfolio decisions are path dependent.
- The cost of changing weights matters.
- Previous weights help the model understand turnover and continuity.

### Why not ignore past weights
- Then the model behaves as if reallocation is free and memoryless.
- That is unrealistic in portfolio management.

## Manager observation design
### What the manager observes
- Pool-level market features
- Worker cumulative return summaries
- Worker volatility summaries
- In the updated code, market stress context is also included

### Why this is important
- The manager should not allocate blindly.
- It needs to know how workers are performing and how market conditions are changing.

### Good defense line
- "The manager must see both opportunity and stress. Without worker summaries, it cannot do meaningful top-level allocation."

## Worker observation design
### What the workers observe
- Rolling price windows for assets in their own pool
- Their own portfolio memory weights

### Why pool-specific observation is useful
- Keeps the state smaller.
- Supports specialization.
- Makes the worker's role clearer.

## Dirichlet policy
### What it does
- Produces portfolio weights on the simplex.
- All weights are non-negative.
- All weights sum to one.

### Why Dirichlet is a good fit
Best answer:
- "Portfolio weights naturally live on the simplex. Dirichlet is a probability distribution defined exactly on that space."

### Why not Softmax
- Softmax converts logits into a normalized vector, but it is not a full distribution over portfolios in the same way.
- It can encourage smoother, less concentrated outputs.
- The dissertation argues this contributes to the 1/N tendency.

### Why concentration matters
- Good portfolio decisions sometimes require conviction.
- A model that cannot move away from nearly equal weights is limited.

## Why a cash asset is included in workers
### Simple answer
- Cash gives the worker a "do nothing" or "step back" option.

### Why this is important
- Without cash, the worker is forced to allocate across risky assets at every step.
- Cash supports downside control and more realistic behavior.

### Best answer
- "Cash prevents forced investment. That matters because sometimes the best portfolio action is partial de-risking, not choosing the least bad risky asset."

## Why this architecture is interpretable
The architecture is more interpretable than many black-box portfolio systems because you can separately inspect:
- Manager allocation across risk profiles
- Worker behavior inside each pool
- Average Safe, Neutral, and Risky capital shares
- Effect of the lexical penalty on worker decisions

## Common expert challenges
### "Is the hierarchy really necessary?"
Answer:
- The hierarchy is justified because the problem is naturally two-level: risk budgeting and asset selection. A flat design would mix those roles and make learning less stable.

### "Why exactly three workers?"
Answer:
- Three workers are a pragmatic balance. They create meaningful specialization without making the system too fragmented. Safe, Neutral, and Risky also map cleanly onto a finance interpretation.

### "Could more pools be better?"
Answer:
- Possibly, but more pools increase complexity, reduce per-pool sample size, and make training harder. Three is a defensible first design.

## Final one-line summary
- "The architecture is designed so that allocation decisions become modular, interpretable, and better aligned with the finance structure of the problem."

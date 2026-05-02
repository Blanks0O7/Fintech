# Literature Review: Reinforcement Learning and Multi-Agent RL

## Core message
Reinforcement learning is attractive for portfolio management because investing is sequential, but RL and MARL create their own problems: instability, noisy rewards, and poor coordination.

## Why RL is a natural fit
- Portfolio management is sequential.
- Decisions today affect future state and reward.
- That matches the reinforcement-learning framework.

## Why RL is not automatically superior
- Financial rewards are noisy.
- Markets are non-stationary.
- Policy learning can become unstable.
- The agent may learn overly smooth or trivial allocations.

## EIIE literature
### Why it matters
- EIIE is one of the most influential deep RL portfolio architectures.
- It uses shared convolutional filters across assets.
- It reduces ticker-specific memorization.

### Why your dissertation uses it
- Because it provides a sensible feature-extraction backbone.
- You keep this backbone but change the higher-level policy and training structure.

## Portfolio Vector Memory
### Why relevant
- Sequential allocation should know previous weights.
- Rebalancing and turnover matter.

### How to explain its role
- "Portfolio Vector Memory makes the model path-aware. Without it, the agent would ignore the cost and meaning of changing positions."

## Flat MARL literature
### Why it matters
- It shows multi-agent portfolio design is possible.
- But it also shows coordination can be difficult.

### Your dissertation's position
- Flat MARL is useful, but hierarchy is a better fit for separating risk budgeting and stock picking.

## Why MARL is appealing here
- Different agents can specialize.
- Safe, Neutral, and Risky are naturally different roles.
- The hierarchy mirrors real investment decision layers.

## Why MARL is dangerous here
- Agent interference
- Moving-target learning
- Difficult credit assignment
- Potential convergence to trivial behavior

## Why curriculum learning literature matters
- It supports the idea that training order can stabilize learning.
- Your staged training is a practical curriculum for the manager-worker hierarchy.

## Why not use a single-agent RL model
Best answer:
- "A single agent can work, but it mixes top-level risk budgeting and bottom-level stock selection into one decision problem. I used hierarchy to make the structure clearer and more specialized."

## Why not use actor-critic or a larger value-based MARL design
Best answer:
- "That is possible, but I wanted a policy-gradient setup directly compatible with portfolio weights on the simplex. The focus of the dissertation was on semantic diversification and hierarchical control, not exhausting every RL algorithm family."

## What the RL literature contributes to your design
- EIIE -> shared feature extraction
- MARL -> agent specialization
- Curriculum/stabilization -> staged training
- Dirichlet policy work -> better action representation

## Final one-line summary
- "RL gives adaptivity, but hierarchy and staged training are what make that adaptivity usable in this project."

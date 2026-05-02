# Training Strategy: Staged vs Concurrent

## Why training strategy is a major issue here
In multi-agent reinforcement learning, agents affect each other's environment. That means the environment is not fully stable while training. This is called non-stationarity.

## What non-stationarity means in simple words
- If one agent changes its policy, the other agents suddenly face a different world.
- So each agent is trying to learn while the target keeps moving.

## Why this matters in this dissertation
- The manager depends on worker behavior.
- The workers are also learning.
- If all four agents train together from the start, the manager may never get a stable picture of which worker is actually best.

## Concurrent training
### What it is
- All agents learn at the same time.

### Why it is attractive
- It is simpler to set up.
- It sounds efficient.
- It allows co-adaptation from the beginning.

### Why it is risky
- The manager sees moving-target workers.
- Workers also react while the manager changes.
- This can create noisy gradients and unstable allocations.

## Staged training
### What it is
The system trains in three phases:
1. Train workers while the manager stays fixed.
2. Train the manager while workers stay fixed.
3. Fine-tune all together with smaller learning rates.

### Why this is powerful
- Phase 1 gives each worker a stable objective.
- Phase 2 gives the manager stable worker policies to evaluate.
- Phase 3 allows coordination without destroying the structure already learned.

## Phase 1: worker training
### What happens
- The manager is frozen at equal allocation.
- Workers learn inside their own pools.

### Why that helps
- It prevents the manager from changing capital flows while workers are still discovering basic behavior.
- Workers can specialize first.

### Best answer
- "Phase 1 is about role formation. Each worker learns what it means to be Safe, Neutral, or Risky before the manager starts ranking them."

## Phase 2: manager training
### What happens
- Workers are frozen.
- The manager learns how to allocate across already-formed workers.

### Why that helps
- The manager now sees a stable performance mapping.
- It can learn a meaningful allocator policy instead of chasing noise.

### Best answer
- "Phase 2 isolates the manager's learning problem. It turns a moving-target MARL problem into a much cleaner allocation problem."

## Phase 3: joint fine-tuning
### What happens
- All agents are unfrozen.
- Learning rates are reduced.
- The whole hierarchy is fine-tuned together.

### Why this is necessary
- The first two phases may create a good structure, but not perfect coordination.
- Fine-tuning lets the system adjust jointly without starting from chaos.

## Why not train staged forever and skip joint tuning
- Then the agents may remain slightly misaligned.
- A final joint phase helps coordination.

## Why not only concurrent training
- Because the dissertation argues concurrent training is exactly where hierarchical instability shows up.

## Why not only train workers and use a rule-based manager
- Then the top-level allocator would not learn from data.
- You would lose one of the main advantages of RL.

## Why reduced learning rate in Phase 3
- Once the structure is learned, you want refinement, not disruption.
- Large learning rates could destroy earlier specialization.

## What training algorithm is actually used in code
This is a very important defense point.

### In the code
- The submitted notebook and script use REINFORCE with an EMA baseline and an entropy bonus.
- The code has a function called `reinforce_update` and uses policy gradients directly.

### Why this matters
- Some dissertation text mentions PPO.
- But the submitted code is REINFORCE-style, not PPO in the full clipped-surrogate sense.

### How to answer honestly if asked
- "The implementation I am submitting uses REINFORCE with an EMA baseline and entropy regularization. Some dissertation wording references PPO, but the code I prepared and defended is a simpler policy-gradient implementation. The core architecture and experiments remain the same, but I would present the code as REINFORCE-based."

### Why this answer is strong
- It is honest.
- It prevents the examiner from catching a mismatch before you explain it.
- It shows you understand your own implementation.

## Why REINFORCE is still reasonable here
- Simpler to implement and inspect.
- Directly fits policy-gradient learning with stochastic Dirichlet policies.
- Enough to test the architectural hypothesis.

## Why entropy bonus is used
- It encourages exploration.
- Without it, the agent may become too deterministic too early.

## Why an EMA baseline is used
- It reduces variance in policy-gradient updates.
- Policy-gradient estimators are noisy; a baseline stabilizes learning.

## Why gradient clipping is used
- It prevents unstable update spikes.
- This is especially useful in multi-agent settings with noisy rewards.

## Common expert challenges
### "How do you know staged training really helped?"
Answer:
- The staged vs concurrent comparison is included precisely to test that. The argument is not only theoretical. It is evaluated empirically.

### "Is staged training just manual engineering?"
Answer:
- Yes, it is a deliberate training design choice. But it is theoretically motivated by MARL non-stationarity and empirically testable.

### "Could staged training over-constrain learning?"
Answer:
- It could if the final joint phase were omitted. That is why the design includes a fine-tuning phase after stabilization.

## Best summary sentence
- "Staged training was used because the system is hierarchical in structure, and the learning schedule should respect that hierarchy rather than ignore it."

# Literature Review: Classical Optimization, RL, and Semantic Diversification

## How to present the literature review
Do not present the literature review as a list of papers. Present it as a logical path.

Use this structure:
1. Classical optimization gave the first formal diversification framework.
2. Empirical work showed classical methods often fail out of sample.
3. Reinforcement learning offered adaptivity, but brought new issues.
4. Semantic and textual research suggested business text contains economically useful information.
5. This dissertation combines those strands.

## Part 1: Classical portfolio optimization literature
### Markowitz (1952)
Why it matters:
- It is the foundation of modern portfolio theory.
- It introduced diversification as a mean-variance problem.
- It gave a mathematical definition of efficient portfolios.

What to say if asked:
- "Markowitz is the right starting point because every later portfolio model either extends it, reacts to it, or tries to fix its weaknesses."

### Michaud (1989)
Why it matters:
- Shows that optimization can amplify estimation error.
- Very important because the dissertation argues that instability is structural, not accidental.

Strong answer:
- "Michaud matters because he explains why mathematically optimal weights can be empirically poor. The optimizer is very sensitive to noisy inputs."

### Ledoit and Wolf (2004)
Why it matters:
- Shows attempts to stabilize covariance estimation.
- Important because it proves researchers tried to repair MVO rather than simply reject it.

### DeMiguel, Garlappi, and Uppal (2009)
Why it matters:
- Famous empirical result: many sophisticated methods fail to beat 1/N out of sample.
- This is one of the strongest motivations for the dissertation.

Best short answer:
- "This paper is important because it reminds us that sophistication does not automatically translate into out-of-sample superiority."

### Chua, Flint, Ang and Chen
Why they matter:
- They support the idea that diversification weakens in crisis regimes.
- They justify the dissertation's focus on stress periods and drawdown.

## Part 2: Reinforcement learning and MARL literature
### Why RL entered portfolio management
- RL can adapt sequentially.
- Portfolio management is a sequential decision problem.
- The agent observes a state, acts, receives reward, and updates policy.

### Jiang, Xu, and Liang (EIIE)
Why it matters:
- EIIE is a major RL portfolio architecture.
- It uses shared convolutional filters across assets.
- It helps avoid ticker memorization.
- It also introduces Portfolio Vector Memory.

What to say:
- "I did not discard EIIE. I kept its strongest idea, which is cross-asset feature sharing, and then changed the policy and hierarchy around it."

### Lee et al. (MAPS) and flat MARL
Why it matters:
- Shows multi-agent methods in portfolio management.
- But flat multi-agent competition can create instability and poor coordination.

Defense answer:
- "Flat MARL gives adaptivity but often weak coordination. My hierarchy was designed to reduce direct interference between agents."

### Trend-regularized MARL papers
Why they matter:
- They try to diversify or regularize agent behavior.
- But many still rely on price-only information.
- This creates the opening for a semantic penalty.

### MARL survey papers
Why they matter:
- They justify the non-stationarity concern.
- They support the need for training stabilization strategies.

## Part 3: Semantic and textual finance literature
### Loughran and McDonald
Why it matters:
- Domain-specific financial text matters.
- Generic sentiment dictionaries are often misleading in finance.
- Supports using 10-K filings as serious financial text, not casual text.

### Tetlock
Why it matters:
- One of the core papers showing text has market relevance.
- Important as general evidence that language contains financial signal.

### Hoberg and Phillips
Why it matters most for this dissertation:
- They show that text-based business similarity has economic meaning.
- This directly supports using business-description similarity for diversification.

Best answer:
- "Hoberg and Phillips is central because it connects text similarity to real economic overlap between firms. That is exactly the mechanism this dissertation needs."

### Cohen, Malloy, and Nguyen
Why it matters:
- Supports the idea that filing text contains persistent information, not just short-lived noise.

### Mohseni et al. and the lexical ratio idea
Why it matters:
- Gives direct inspiration for semantic diversification.
- The dissertation extends this from static optimization into dynamic RL reward shaping.

## Part 4: Dirichlet policy literature
### Why this literature matters
- Most portfolio models use Softmax.
- The dissertation argues this contributes to overly smooth allocations.
- Dirichlet provides a natural distribution on the simplex.

### Tian et al., André, and related work
What they contribute:
- Theoretical and practical support for Dirichlet policies.
- Justification for concentration-parameter control.
- Support for simplex-based allocation modeling.

Best answer:
- "Dirichlet was not chosen just because it is different. It was chosen because it is mathematically aligned with weight vectors that must be non-negative and sum to one."

## How the literature review supports the dissertation design
### Classical literature gives the problem
- Estimation error
- Crisis correlation instability
- 1/N trap

### RL literature gives the opportunity and the challenge
- Sequential adaptivity
- But instability and non-stationarity

### Semantic literature gives the missing information source
- Business descriptions
- Economic overlap
- Persistent structural information

### Policy literature gives the action mechanism
- Dirichlet instead of Softmax

## The main literature gap
If asked "What exact gap are you filling?" use this:
- "There is a gap at the intersection of three areas: hierarchical portfolio RL, semantic business similarity, and trainable reward shaping. Prior work usually has one or two of these, but not all three together in a single system."

## Why this literature review is coherent
- It does not randomly combine topics.
- Every literature strand maps to one design choice:
- Classical failure -> motivation
- RL/MARL -> architecture
- NLP/10-K -> semantic signal
- Dirichlet literature -> action generation
- Curriculum/MARL stabilization -> training method

## Common expert challenge and answer
### Challenge
- "Did you just combine popular methods without a strong theory?"

### Answer
- "No. Each component answers a specific weakness identified in the literature. The hierarchy addresses agent specialization and credit assignment. Dirichlet addresses simplex allocation behavior. WALP addresses price blindness. Staged training addresses MARL non-stationarity."

## Best final sentence for this section
- "The literature review is not only background. It is the design logic of the system."

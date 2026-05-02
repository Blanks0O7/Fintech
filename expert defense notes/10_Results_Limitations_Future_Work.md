# Results, Limitations, and Future Work

## How to talk about results well
Do not oversell. Experts respect balanced interpretation.

Use this structure:
1. What improved
2. What did not improve
3. What the results imply
4. What limits the conclusions
5. What should be done next

## What the results are trying to show
The results are not only about highest return. They are mainly about:
- Structural resilience
- Risk-adjusted quality
- Drawdown behavior
- Evidence that semantic diversification changes allocation behavior

## How to describe positive findings
### Good answer format
- "The model showed evidence of improved structural resilience, especially in drawdown-oriented metrics, and the semantic penalty affected allocation behavior in a measurable way."

## How to talk about staged vs concurrent results
### Main message
- Staged training is intended to stabilize the hierarchy.
- Even if concurrent training sometimes produces similar or slightly better raw metrics in one slice, staged training is still valuable if it gives better downside control or more coherent allocation behavior.

### Best answer
- "The staged system is not claimed to dominate every metric in every period. Its main advantage is more stable hierarchical learning and better alignment with the intended risk-control objective."

## How to talk about lambda ablation results
### Main message
- A moderate semantic penalty can help.
- Too little penalty means semantics are not used.
- Too much penalty may overwhelm the return signal.

### Expert-friendly explanation
- "The semantic penalty behaves like a calibrated structural prior. If it is too weak, it has little effect. If it is too strong, it can over-constrain the allocator."

## How to talk about the null-hypothesis results
### Best framing
- If the real matrix beats the shuffled matrix, that supports the idea that semantic content matters.
- If the result is mixed, say that honestly and frame it as partial evidence with a clear next step.

### Strong honest answer
- "The control experiment is designed to separate semantic meaning from generic regularization. It provides stronger evidence than a simple on-off penalty comparison, but I would still strengthen it with more seeds and out-of-sample repeats."

## How to talk about holdout underperformance if asked
This is very important.

### Good answer
- "The system is not claimed to win in every holdout period on raw return. In a macro shock like 2022, the semantic matrix may still help with diversification structure while not fully capturing rapid rate-driven repricing. That is why the thesis emphasizes resilience and drawdown, not unconditional outperformance."

### Why this answer works
- It is honest.
- It connects to the project's stated objective.
- It does not pretend the model is universally superior.

## Limitations you should be ready to state clearly
### Limitation 1: annual text update frequency
- 10-K text is slow-moving.
- It captures structural business similarity well.
- It may miss fast-changing macro shocks.

### Limitation 2: TF-IDF is simple but shallow
- It is interpretable.
- But it may miss deeper semantic equivalence.
- Firms can describe similar businesses using different words.

### Limitation 3: evaluation robustness can be extended
- More seeds would help.
- Formal significance testing would help.
- More out-of-sample controls would help.

### Limitation 4: code/dissertation wording mismatch
- Some dissertation text references PPO.
- Submitted code uses REINFORCE with EMA baseline and entropy bonus.
- Be prepared to explain the implementation honestly.

### Limitation 5: backtesting limits
- Any backtest is an approximation.
- Transaction costs, slippage, and execution realism can always be improved.

## Why admitting limitations helps you
- It makes you sound credible.
- It shows scientific maturity.
- It prevents the examiner from framing the limitation before you do.

## Future work directions
### 10-Q updates
Why important:
- Higher-frequency business updates
- Better response to changing sector conditions

### Transformer embeddings
Why important:
- Richer semantic capture
- Better handling of synonymous business language

### Adaptive lambda
Why important:
- Penalty strength may need to vary by regime
- Stress periods may need stronger structural diversification

### Multi-seed and significance testing
Why important:
- Stronger empirical credibility
- Better reproducibility

### More realistic market frictions
Why important:
- Better practical deployment realism

## Best answer to "What would you improve first?"
- "I would first strengthen the semantic representation and evaluation protocol: transformer-based text embeddings, multi-seed out-of-sample tests, and dynamic lambda by market regime."

## Best answer to "What is the biggest weakness?"
- "The biggest weakness is the low frequency and simplicity of the semantic input. Annual TF-IDF 10-K representations are transparent and useful, but they are not the richest possible business-similarity signal."

## Best closing line
- "The project should be judged as a strong proof of concept for semantic diversification inside hierarchical RL, not as the final word on portfolio management."

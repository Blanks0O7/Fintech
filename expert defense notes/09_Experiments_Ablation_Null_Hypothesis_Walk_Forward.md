# Experiments: Ablation, Null Hypothesis, and Walk-Forward

## Why this file matters
Experts often care less about the model itself and more about whether the experiments really test the claim. This file prepares you for that level of questioning.

## The three most important experiment types
1. Staged vs concurrent comparison
2. Lambda ablation study
3. Null-hypothesis control experiment

Walk-forward validation is the evaluation frame around them.

## Experiment 1: staged vs concurrent
### What question it answers
- Does the staged training schedule improve the hierarchical system relative to concurrent learning?

### Why it matters
- If staged training is central to the dissertation, it must be tested directly rather than assumed.

### Best answer
- "This comparison isolates the value of the curriculum schedule. It asks whether hierarchical stabilization changes performance and allocation behavior."

## Experiment 2: lambda ablation study
### What question it answers
- Does the semantic penalty matter?
- How much penalty is helpful?

### Why lambda is ablated
- Because turning lambda on and off lets you test causality more directly.
- If lambda changes outcomes while other settings stay fixed, then semantic regularization is influencing portfolio construction.

### Why not choose one lambda and stop
- That would be weak scientifically.
- The examiner could say the chosen value was arbitrary.
- Ablation shows the effect across a range.

### Best answer
- "Ablation turns a design choice into a testable variable. It shows whether the model improves because of semantics or merely because of the rest of the architecture."

## Experiment 3: null-hypothesis control with shuffled matrix
### What question it answers
- Does the semantic content of the matrix matter, or is any quadratic penalty enough?

### Why this experiment is so important
- Without this control, an examiner can say: maybe the improvement comes from regularization alone.
- The shuffled matrix preserves the same values but destroys the true company-to-company meaning.

### Why not compare only real matrix versus zero matrix
- Real vs zero tells you whether having a penalty matters.
- It does not tell you whether the semantic structure matters.
- Shuffled vs real is what tests semantic causality more directly.

### Best short answer
- "The shuffled control is stronger than a zero control because it preserves the mathematics while destroying the meaning."

## Why call it a null-hypothesis test
- The null hypothesis is that the semantic identity of the matrix does not matter.
- In other words, any matrix-shaped regularization would do the same job.
- The experiment tests whether the real semantic structure outperforms the shuffled structure.

## Why not use permutation importance or SHAP instead
Good answer:
- Those tools explain feature influence after fitting. They do not directly test whether semantic structure itself is the causal ingredient in the reward penalty. The shuffled control is more aligned with the scientific question here.

## Why walk-forward is paired with these experiments
- Because the real question is not only in-sample behavior.
- The project wants to know whether the architecture and semantic penalty remain useful across new time periods and different regimes.

## Walk-forward windows
### What they do
- Train on an earlier period.
- Test on a later unseen period.
- Repeat across different market phases.

### Why that is convincing
- It mimics realistic deployment.
- It tests robustness across bull, crisis, recovery, and bear periods.

## Holdout test set
### Why it exists
- It is a final honest check.
- Hyperparameters and design choices should not be tuned on it.

### Why this matters in a defense
- Experts want to see whether you understand the difference between model development and final evaluation.

## Worker-level analysis
### Why included
- It helps explain what each worker is actually contributing.
- It makes the hierarchy interpretable.

### Why not only report final portfolio metrics
- Because experts will ask whether the manager is learning meaningful allocations or just random mixtures.

## Drawdown decomposition
### Why included
- The dissertation claims resilience.
- Drawdown decomposition gives detailed evidence about loss depth, duration, and recovery.

## Return-distribution analysis
### Why included
- Mean and Sharpe can hide tail shape.
- Skewness and kurtosis help reveal asymmetry and fat tails.

## Common expert challenges
### "Is the null-hypothesis control really enough to prove semantics?"
Answer:
- It is strong evidence, not absolute proof. It is a well-designed control because it preserves penalty structure while removing semantic meaning. A stronger next step would be multi-seed out-of-sample shuffled tests and richer text embeddings.

### "Why not do more seeds?"
Answer:
- More seeds would improve robustness and are a valid extension. The current design still improves on simple single-run backtesting by using multiple evaluation modes and explicit controls.

### "Why not test other text representations?"
Answer:
- That is a good future-work direction. TF-IDF was chosen as the interpretable baseline representation. The first goal was to test whether semantic diversification is useful at all.

## Best sentence to remember
- "Every major experiment exists to answer a specific causal question, not just to create more plots."

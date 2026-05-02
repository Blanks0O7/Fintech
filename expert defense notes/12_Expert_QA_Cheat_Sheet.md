# Expert Q&A Cheat Sheet

## How to use this file
Read the question. Then say the short answer first. If needed, expand with the longer answer.

## 1. What is the main contribution of your dissertation?
Short answer:
- I integrate semantic business similarity into a hierarchical RL portfolio system through a trainable lexical penalty.

Longer answer:
- The main contribution is not just using RL or just using text. It is combining a manager-worker architecture, a Dirichlet policy, staged curriculum learning, and a weight-aware semantic penalty derived from 10-K filings.

## 2. Why is price-based diversification not enough?
Short answer:
- Because prices show co-movement, but not necessarily the underlying business overlap.

Longer answer:
- Historical covariance can underestimate hidden common exposure. Firms may look different in price data yet still share customers, supply chains, or macro sensitivities.

## 3. Why use SEC 10-K business descriptions?
Short answer:
- Because they are formal, stable, and directly describe the firm's business model.

Longer answer:
- I wanted structural business information, not noisy short-term sentiment. 10-K business descriptions are suitable for that role.

## 4. Why TF-IDF and not BERT?
Short answer:
- TF-IDF is simple, transparent, and easy to defend.

Longer answer:
- Transformer embeddings are a strong next step, but TF-IDF gives interpretability and reproducibility. For a dissertation proof of concept, that was a good starting point.

## 5. Why cosine similarity?
Short answer:
- It is a standard way to compare document vectors and is robust to document length differences.

## 6. What is WALP in one line?
Short answer:
- It is a penalty that discourages heavy investment in companies with similar business descriptions.

## 7. Why make the penalty weight-aware?
Short answer:
- Because similarity only becomes dangerous when the portfolio is also concentrated in those similar firms.

## 8. Why use a hierarchy?
Short answer:
- Because capital allocation and stock selection are different problems and should be separated.

## 9. Why three workers?
Short answer:
- Three workers give meaningful specialization without making the system too fragmented.

## 10. Why beta-based pools?
Short answer:
- Beta gives a standard, interpretable measure of systematic risk.

## 11. Why Dirichlet instead of Softmax?
Short answer:
- Dirichlet is naturally defined on the simplex and supports more expressive portfolio allocations.

Longer answer:
- Portfolio weights must be non-negative and sum to one. Dirichlet is a natural distribution over exactly that space.

## 12. Why include cash?
Short answer:
- Cash gives the model the option not to force risky allocation when conditions are poor.

## 13. Why staged training?
Short answer:
- Because concurrent MARL creates a moving-target learning problem.

Longer answer:
- Workers should specialize first, then the manager should learn against stable workers, and only after that should the system fine-tune jointly.

## 14. Why not just train everything together?
Short answer:
- Because the manager then tries to learn from workers that are still changing, which makes the learning signal unstable.

## 15. What algorithm do you actually use?
Short answer:
- The submitted code uses REINFORCE with an EMA baseline and entropy bonus.

Longer answer:
- Some dissertation wording references PPO, but the code I am defending is REINFORCE-style. I would present the implementation honestly that way.

## 16. Why not use PPO properly?
Short answer:
- A simpler policy-gradient implementation was enough to test the architecture, though PPO would be a valid extension.

## 17. Why use walk-forward validation?
Short answer:
- To preserve time order and avoid data leakage.

## 18. Why not k-fold cross-validation?
Short answer:
- Because random shuffling is unrealistic for financial time series.

## 19. Why compare against equal-weight?
Short answer:
- Because the 1/N portfolio is the most important empirical sanity check in portfolio research.

## 20. Why include MVO, risk parity, and momentum baselines?
Short answer:
- To compare against classical, risk-based, and rules-based alternatives, not only against another RL model.

## 21. Why do a lambda ablation?
Short answer:
- To test whether the semantic penalty actually changes performance and portfolio behavior.

## 22. Why do a null-hypothesis control?
Short answer:
- To test whether the semantic meaning matters, not just the mathematical presence of a penalty.

## 23. Why shuffled matrix instead of only zero matrix?
Short answer:
- Because shuffled preserves the same matrix values but destroys the firm-to-firm semantic meaning.

## 24. Why is that stronger scientifically?
Short answer:
- Because it isolates semantic structure from generic regularization.

## 25. What if the shuffled matrix performs similarly?
Short answer:
- Then the penalty may be acting more like a structural regularizer than a semantic signal, and I would say that honestly.

## 26. What is the main practical benefit you are claiming?
Short answer:
- Better structural diversification and stronger downside resilience.

## 27. Are you claiming this beats all baselines everywhere?
Short answer:
- No.

Longer answer:
- I am claiming it is a strong proof of concept for semantic diversification inside hierarchical RL, especially for resilience, not universal domination.

## 28. What is the biggest limitation?
Short answer:
- The semantic signal is low frequency and based on a simple TF-IDF representation.

## 29. Why might the model struggle in a shock like 2022?
Short answer:
- Because an annual business-description matrix may not fully capture rapid macro repricing driven by interest-rate shocks.

## 30. What would you improve first?
Short answer:
- Higher-frequency semantic updates, richer embeddings, and more robust multi-seed evaluation.

## 31. Why is this an academic contribution and not just engineering?
Short answer:
- Because it tests a scientific claim: whether semantic business similarity provides useful diversification information beyond price-only inputs.

## 32. What is your strongest defense sentence?
- "My dissertation does not only ask whether AI can trade. It asks whether diversification becomes more meaningful when the model understands what the companies actually do."

## 33. What is your safest honest sentence when challenged?
- "That is a fair criticism, and I would frame it as a limitation rather than deny it. The project is a strong proof of concept, but several parts can be strengthened in future work."

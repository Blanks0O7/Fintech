# Read First

This folder is a defense-preparation pack for the dissertation and the submitted code.

Goal:
- Help you explain the project clearly to expert examiners.
- Give you simple but technically correct language.
- Prepare you for questions like why this method, why not another method, what assumptions were made, and what the limitations are.

How to use this pack:
1. Read `01_Thesis_Story_and_Core_Contribution.md` first.
2. Read the literature and methodology files next.
3. Read the experiment and results files after that.
4. Keep `12_Expert_QA_Cheat_Sheet.md` open before your presentation or viva.
5. Use `13_Glossary_and_Key_Terms.md` when you want short definitions.

Important rule for the defense:
- Do not try to sound overly complicated.
- Use simple words first.
- Then add the technical keyword.
- Then explain why that keyword matters.

Good answer structure:
1. State the idea in one simple sentence.
2. Give the technical term.
3. Explain why it was chosen.
4. Explain what trade-off it creates.
5. Say what you would improve next.

Example:
- Simple: "I used walk-forward validation because financial data is time ordered."
- Technical: "This avoids temporal leakage and preserves regime structure."
- Why: "In finance, random shuffling makes the model learn from the future."
- Trade-off: "It gives fewer folds than normal cross-validation, but it is much more realistic."
- Future improvement: "I would add more rolling windows and multiple seeds."

What this project is about in one paragraph:
- The dissertation argues that normal portfolio diversification often fails because it looks only at historical price relationships. The proposed system tries to build stronger diversification by also looking at what companies actually do. It uses SEC 10-K business descriptions, converts them into a text-similarity matrix, and uses that matrix inside a Hierarchical Multi-Agent Reinforcement Learning system. The manager allocates money across Safe, Neutral, and Risky worker agents. The workers select stocks inside their own risk pools. A lexical penalty discourages the model from putting too much weight into companies with similar business descriptions.

Very important defense point:
- If examiners ask whether this is only an AI project, say no.
- It is a finance, machine learning, and NLP integration project.
- The finance part is portfolio construction and risk control.
- The machine learning part is hierarchical reinforcement learning.
- The NLP part is the semantic similarity matrix from 10-K filings.

Another very important defense point:
- If asked whether this is just "another backtest," say no.
- The core scientific question is whether semantic business similarity adds useful diversification information beyond price-only signals.
- The null-hypothesis control is there to test exactly that.

File guide:
- `01_Thesis_Story_and_Core_Contribution.md`: your presentation story.
- `02_Research_Problem_and_Why_Classical_Fails.md`: why the problem matters.
- `03_Literature_Review_Classical_RL_Semantic.md`: literature review logic.
- `04_Data_and_Preprocessing.md`: datasets and preprocessing pipeline.
- `05_Architecture_Manager_Workers_EIIE_Dirichlet.md`: model architecture.
- `06_Reward_Functions_WALP_and_Risk_Pools.md`: reward design and lexical penalty.
- `07_Training_Strategy_Staged_vs_Concurrent.md`: staged curriculum learning.
- `08_Evaluation_Design_Metrics_Baselines.md`: evaluation logic.
- `09_Experiments_Ablation_Null_Hypothesis_Walk_Forward.md`: why each experiment exists.
- `10_Results_Limitations_Future_Work.md`: interpretation, weaknesses, next steps.
- `11_Code_Map_and_Implementation_Defense.md`: how the code is organized and what to say about it.
- `12_Expert_QA_Cheat_Sheet.md`: likely viva questions and strong answers.
- `13_Glossary_and_Key_Terms.md`: short definitions.

# Code Map and Implementation Defense

## Why this file matters
Experts may ask you where each idea appears in the code. This file helps you connect dissertation claims to actual implementation.

## Main files in the submitted code
### `Hierarchical_MARL_System.ipynb`
What it is:
- The dissertation-facing notebook.
- Includes stock selection, architecture definition, training, staged training, ablation, null-hypothesis control, walk-forward evaluation, and figure generation.

Why it matters:
- It is the most complete experiment narrative.

### `Staged_MARL_Training.py`
What it is:
- Standalone script version of the main implementation.
- Cleaner for code inspection.

Why it matters:
- Easier to defend as the main implementation surface.

### `load_sp500_100.py`
What it is:
- Data pipeline for price download, text extraction, lexical matrix construction, and sector mapping.

Why it matters:
- Shows the semantic input is built by a reproducible pipeline.

## Where key ideas live in the code
### Worker environment
Look for:
- `class WorkerEnv`

What it contains:
- Asset selection inside a risk pool
- Cash asset
- Reward definitions
- Lexical penalty usage
- Turnover handling

### Manager environment
Look for:
- `class ManagerEnv`

What it contains:
- Allocation across Safe, Neutral, Risky pools
- Worker performance summaries
- Market-context features
- Top-level reward and penalties

### Network
Look for:
- `class EIIENetwork`

What it contains:
- Shared Conv1D processing
- Portfolio vector memory input
- Extra context support
- Dirichlet concentration output

### Training
Look for:
- `train_concurrent`
- `train_staged`
- `reinforce_update`

What they contain:
- Concurrent multi-agent training
- Three-phase staged training
- Policy-gradient update logic

### Evaluation
Look for:
- `evaluate_concurrent`

What it contains:
- Deterministic evaluation using Dirichlet mode
- Global and worker-level metrics

## How to explain the notebook versus script
Best answer:
- "The notebook is the experimental and presentation workflow. The script is the cleaner standalone implementation. I included both because one is easier for reproducing the dissertation flow, while the other is easier for auditing the implementation."

## Important implementation truth: REINFORCE vs PPO
### What the code actually does
- The code uses REINFORCE with EMA baseline and entropy bonus.
- It does not implement a full PPO clipped objective with a critic network.

### How to say this confidently
- "My submitted implementation is policy-gradient based and uses REINFORCE-style updates. If you compare the code to canonical PPO, you will see it is not using the clipped surrogate objective."

### Why you should say this first
- Because an expert may spot it.
- If you say it first, it becomes a strength: you know your own code.

## Data files in the submission
### `data/sp500_100_prices.csv`
- Main price dataset

### `data/processed/lexical_matrix_100.csv`
- Semantic similarity matrix

### `data/processed/sector_map_100.json`
- Metadata and sector mapping

### `data/raw/sp500_100_10k_texts.json`
- Business-description source text

## Figures in the submission
The `images` folder contains dissertation figures currently available in the workspace, including:
- Drawdown comparison
- Lambda ablation comparison
- Random control experiment
- Return distribution analysis
- Staged training results
- Walk-forward ablation results

## If asked "How reproducible is the code?"
Good answer:
- "The code is reasonably reproducible because the main architecture, data inputs, and experiment pipeline are explicit. However, full reproducibility in RL would be stronger with multi-seed reporting, fixed environment exports, and a documented runtime manifest."

## If asked "Is the code production-grade?"
Best answer:
- "No, it is research-grade code aimed at proving the thesis idea. It is organized enough for reproduction and defense, but production deployment would need stronger testing, configuration handling, monitoring, and execution logic."

## If asked "How do you know the code matches the dissertation?"
Good answer:
- "The main conceptual components match: hierarchy, Dirichlet policy, lexical penalty, staged training, and evaluation suite. The main implementation point to clarify is the training algorithm naming, because the code is REINFORCE-based rather than canonical PPO."

## Best final sentence
- "When I defend the code, I focus on implementation honesty: which conceptual claims are fully implemented, which are approximated, and which parts remain future extensions."

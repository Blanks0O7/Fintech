# Code To Submit

This folder contains the minimal project files needed to represent the dissertation implementation in [Dissertation-final-v1.5-converted.md](Z:/Fintech/Dissertation-final-v1.5-converted.md).

Included files:

- `Hierarchical_MARL_System.ipynb`: notebook version of the project, including concurrent training, staged/curriculum training, ablations, random-control experiment, walk-forward analysis, and figure generation.
- `Staged_MARL_Training.py`: main Hierarchical MARL implementation, training, evaluation, and figure generation script.
- `load_sp500_100.py`: data preparation pipeline used to build the S&P 500 price dataset, 10-K text corpus, lexical similarity matrix, and sector map.
- `requirements.txt`: Python dependencies.
- `data/sp500_100_prices.csv`: price dataset used by the main training script.
- `data/processed/lexical_matrix_100.csv`: TF-IDF cosine similarity matrix used by WALP.
- `data/processed/sector_map_100.json`: ticker-to-sector metadata used by the project.
- `data/raw/sp500_100_10k_texts.json`: extracted business-description corpus backing the semantic component.
- `images/*.png`: generated dissertation figures currently available in the workspace.

Excluded files:

- Helper scripts beginning with `_`: local utility scripts for notebook/export/debug workflows, not part of the dissertation implementation.
- Older notebooks and planning/notes documents: supporting development artifacts rather than the core submission code.

Notes:

- The notebook and the Python script overlap. The notebook is useful as the dissertation-facing experimental workflow, while the script is the cleaner standalone implementation surface.
- The converted dissertation markdown references placeholder image paths like `media/image1.png`. The actual generated figures available in this workspace are stored in the `images` folder with descriptive filenames.
- `Staged_MARL_Training.py` expects to be run from this folder root so the relative `data/...` paths resolve correctly.

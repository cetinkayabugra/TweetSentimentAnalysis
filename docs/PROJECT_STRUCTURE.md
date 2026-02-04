# Project Structure

This repository uses a simple, standard layout for NLP/ML projects.

```
.
├── data/
│   ├── external/   # Third-party datasets (read-only snapshots)
│   ├── processed/  # Cleaned or feature-engineered datasets
│   └── raw/        # Original, immutable data dumps
├── docs/           # Project documentation
├── models/         # Trained model artifacts and checkpoints
├── notebooks/      # Exploratory analysis and experiments
├── reports/        # Reports, writeups, and figures
├── scripts/        # One-off scripts and CLI utilities
└── src/            # Reusable source code (datasets, training, evaluation)
```

Add files to the appropriate folder as the project grows.

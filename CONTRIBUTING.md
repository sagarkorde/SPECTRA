# Contributing

Contributions are welcome, whether that's a bug fix, a new baseline, a
reproducibility report, or a documentation improvement.

## Getting started

1. Fork the repository and create a branch from `main`.
2. Set up the environment (see `README.md` → Installation).
3. Make your change. Keep it focused: one logical change per pull request.
4. If you touch `spectra/`, run the smoke tests (`pytest tests/`) and confirm
   `python run_spectra.py --phase data` still completes without error.
5. Open a pull request describing what changed and why.

## Reporting issues

Please include:
- The command you ran and the full traceback.
- Python version, OS, and whether a GPU was used (`torch.cuda.is_available()`).
- Whether you modified `configs/config.yaml`, and if so, what changed.

## Code style

- Follow the existing module layout in `spectra/` (one pipeline stage per file).
- Prefer explicit, typed function signatures over `**kwargs` passthroughs.
- Docstrings follow the NumPy style already used throughout the codebase.

## What not to open a PR for

- Changes to `configs/config.yaml` defaults that would silently alter
  published results without a corresponding note in the PR description.
- Committing large binaries (datasets, checkpoints) directly to the repo;
  see `data/DATASET.md` and `outputs/checkpoints/README.md` for how these
  are distributed instead.

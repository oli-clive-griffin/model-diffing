# Model Diffing

A library for training and analyzing Cross-Coders (and by extension, Sparse Autoencoders) on transformer model activations.

# Quickstart

```bash
python \
    model_diffing/scripts/train_jan_update_crosscoder/run.py \
    model_diffing/scripts/train_jan_update_crosscoder/example_config.yaml
```

# Development

Suggested extensions and settings for VSCode are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

Development commands:

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```
# Signalium Trainer — Developer Guide

## Getting Started

### Prerequisites

- Python 3.12+
- `pip` or `pipx`
- Docker (optional, for container builds)
- Git + pre-commit

### Install in Development Mode

```bash
git clone https://github.com/signalium/signalium-trainer
cd signalium-trainer
pip install -e ".[dev]"
pre-commit install
```

### Run the Test Suite

```bash
pytest tests/ -v --cov=trainer
```

### Run Linters

```bash
black trainer/ tests/
isort trainer/ tests/
ruff check trainer/ tests/
mypy trainer/
```

## Project Structure

See `docs/architecture.md` for the full package map.

## Adding a New Dataset Format

1. Create `trainer/datasets/my_format.py` with a class inheriting `BaseDataset`.
2. Implement `feature_dim`, `num_classes`, `__len__`, and `__getitem__`.
3. Export from `trainer/datasets/__init__.py`.
4. Add unit tests in `tests/unit/test_my_format.py`.

## Configuration

All configuration lives in `conf/` (Hydra YAML) and is mapped to typed dataclasses in
`trainer/config/schema.py`. To add a new config option:

1. Add the field to the appropriate dataclass in `schema.py`.
2. Add the corresponding key to `conf/config.yaml`.
3. Override per-environment in `conf/environments/*.yaml`.

## Secrets

Never hard-code credentials. Always use environment variables:
- `SIGNALIUM_API_KEY` — API key
- `SIGNALIUM_TOKEN` — Bearer token

The `trainer doctor` command checks whether credentials are present.

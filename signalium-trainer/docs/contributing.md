# Signalium Trainer — Contribution Guide

## Code of Conduct

Treat all contributors with respect. Constructive feedback only.

## Getting Set Up

```bash
git clone https://github.com/signalium/signalium-trainer
cd signalium-trainer
pip install -e ".[dev]"
pre-commit install
```

## Branching Strategy

- `main` — stable releases only. Protected — no direct pushes.
- `develop` — integration branch for feature PRs.
- Feature branches: `feature/<short-description>`
- Bug fix branches: `fix/<short-description>`

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(api): add retry on 429 rate limit
fix(checkpoint): prevent race condition on best.pt copy
docs(architecture): update data flow diagram
test(early_stopping): add min_delta edge case
```

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New features include unit tests
- [ ] Linters pass (`black`, `isort`, `ruff`, `mypy`)
- [ ] Docstrings added for all public functions and classes
- [ ] No secrets, credentials, or dataset files committed
- [ ] `CHANGES.md` entry added (if user-facing)

## Code Quality Standards

- **SOLID principles** — single responsibility, open/closed, etc.
- **Type hints** — all function signatures must be fully typed.
- **Docstrings** — Google style, on every public class and function.
- **No duplication** — extract shared logic into utilities.
- **No placeholder implementations** — raise `NotImplementedError` or implement properly.
- **Strict linting** — `ruff` and `mypy` must pass on CI.

## Testing Policy

- **Unit tests** (`tests/unit/`) — test individual classes/functions in isolation.
- **Integration tests** (`tests/integration/`) — test interactions between components.
- **Mocks** (`tests/mocks/`) — mock API responses and datasets, never real data.
- Coverage target: **≥ 80%** for `trainer/` package.

## What NOT to Contribute

- Concrete neural network architectures (LSTM, Transformer, GRU, etc.)
- Direct PostgreSQL connections of any kind
- Prediction or inference logic — this repo is the training framework only

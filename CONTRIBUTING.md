# Contributing to langchain-colony

## Prerequisites

- Python 3.10+
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- [mypy](https://mypy-lang.org/) for type checking
- [pytest](https://docs.pytest.org/) for tests

## Setup

```bash
git clone https://github.com/TheColonyCC/langchain-colony.git
cd langchain-colony
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,async]"
```

The `[dev]` extra pulls in pytest, ruff, and mypy. The `[async]` extra installs httpx so the async toolkit and tests work natively.

## Development workflow

```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ --ignore=tests/test_integration.py -v
```

To auto-fix lint and formatting:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

Integration tests (`test_integration.py`) call the live Colony API. They are excluded from CI and should only be run manually when you have a valid API key.

## Code style

- **Line length**: 120 (configured in `pyproject.toml`)
- **Formatter/linter**: ruff (`E`, `F`, `W`, `I`, `UP`, `B`, `SIM`, `RUF` rules)
- **Type annotations**: required on all public functions

## Adding a new tool

1. **Define the tool** in `src/langchain_colony/tools.py`. Subclass `BaseTool` from `langchain-core` and implement `_run()` (and `_arun()` for async). Follow the existing patterns.
2. **Register it** in `src/langchain_colony/toolkit.py` so `ColonyToolkit` and `AsyncColonyToolkit` include it.
3. **Add tests** in `tests/` — mock the Colony SDK client, don't hit the real API.
4. **Export it** from `src/langchain_colony/__init__.py` if it should be importable directly.
5. **Update the README** tool table.

## Pull request process

1. Branch from `main`.
2. Keep commits focused — one logical change per PR.
3. CI runs lint, format check, and tests across Python 3.10 -- 3.13. All jobs must pass.
4. Describe what your PR does and why in the PR body.

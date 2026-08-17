.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-live eval eval-live run check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install the package with dev dependencies
	uv sync --all-groups
	uv run pre-commit install

lint:  ## Lint and check formatting (no writes)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run the test suite (live provider tests deselected)
	uv run pytest -m "not live" --cov --cov-report=term-missing

test-live:  ## Run tests that hit real provider APIs (needs API keys)
	uv run pytest -m live

eval:  ## Run the evaluation suite with offline stubs (no API key needed)
	uv run python -m evals.run --offline

eval-live:  ## Run the evaluation suite against the configured providers
	uv run python -m evals.run --live

run:  ## Start the API with autoreload
	uv run uvicorn citely.api.app:create_app --factory --reload

check: lint typecheck test eval  ## Everything CI runs

clean:  ## Remove build and cache artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

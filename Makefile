.PHONY: help
help:
	@echo "Available targets:"
	@echo "  sync               - Create venv and install dependencies (with all extras)"
	@echo "  install            - Install pre-commit hooks (depends on sync)"
	@echo "  format             - Format code with ruff"
	@echo "  lint               - Run all linters (ruff check + format check)"
	@echo "  typecheck          - Run all type checkers (basedpyright + mypy)"
	@echo "  test               - Run unit tests"
	@echo "  test-integration   - Run integration tests (gated by provider API keys; not in CI default)"
	@echo "  coverage           - Run unit tests with coverage report"
	@echo "  check              - Run lint + typecheck + test (mirrors CI)"
	@echo "  pre-commit         - Run all pre-commit hooks on all files"
	@echo "  clean              - Remove caches and build artifacts"

.PHONY: sync
sync:
	uv sync --group dev --all-extras

.PHONY: install
install: sync
	uv run pre-commit install

.PHONY: format
format:
	uv run ruff format

.PHONY: lint-ruff
lint-ruff:
	uv run ruff check

.PHONY: lint-format
lint-format:
	uv run ruff format --check

.PHONY: lint
lint: lint-ruff lint-format

.PHONY: typecheck-basedpyright
typecheck-basedpyright:
	uv run basedpyright

.PHONY: typecheck-mypy
typecheck-mypy:
	uv run mypy

.PHONY: typecheck
typecheck: typecheck-basedpyright typecheck-mypy

.PHONY: test
test:
	uv run pytest tests/unit

.PHONY: test-integration
test-integration:
	uv run pytest tests/integration

.PHONY: coverage
coverage:
	uv run pytest tests/unit --cov --cov-report=term

.PHONY: check
check: lint typecheck test

.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

.PHONY: clean
clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov coverage.xml .coverage build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

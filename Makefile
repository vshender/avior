.PHONY: help
help:
	@echo "Available targets:"
	@echo "  sync         - Create venv and install dependencies"
	@echo "  install      - Install pre-commit hooks (depends on sync)"
	@echo "  format       - Format code with ruff"
	@echo "  lint         - Run all linters (ruff check + format check)"
	@echo "  typecheck    - Run all type checkers (basedpyright + mypy)"
	@echo "  check        - Run lint + typecheck (mirrors CI)"
	@echo "  pre-commit   - Run all pre-commit hooks on all files"
	@echo "  clean        - Remove caches and build artifacts"

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

.PHONY: check
check: lint typecheck

.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

.PHONY: clean
clean:
	rm -rf .ruff_cache .mypy_cache build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

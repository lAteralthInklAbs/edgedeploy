.PHONY: test lint typecheck eval docker ci clean install dev

# Python settings
PYTHON := python
PYTEST := pytest
MYPY := mypy
RUFF := ruff

# Default target
all: ci

# Install dependencies
install:
	$(PYTHON) -m pip install -e .

# Install dev dependencies
dev:
	$(PYTHON) -m pip install -e ".[dev]"

# Run tests
test:
	$(PYTEST) tests/ -v --tb=short

# Run tests with coverage
test-cov:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run linter
lint:
	$(RUFF) check src/ tests/ scripts/
	$(RUFF) format --check src/ tests/ scripts/

# Fix linting issues
lint-fix:
	$(RUFF) check --fix src/ tests/ scripts/
	$(RUFF) format src/ tests/ scripts/

# Run type checker
typecheck:
	$(MYPY) src/

# Run evaluation scripts
eval: eval-qat eval-drift

eval-qat:
	$(PYTHON) scripts/run_qat_eval.py --epochs 2 --output results/qat_results.json

eval-drift:
	$(PYTHON) scripts/run_drift_eval.py --output results/drift_results.json

# Build Docker image
docker:
	docker build -t edgedeploy:latest .

# Run Docker container
docker-run:
	docker run --rm -it edgedeploy:latest

# Full CI pipeline
ci: lint typecheck test
	@echo "CI pipeline passed!"

# Clean build artifacts
clean:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf results/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Create results directory
results:
	mkdir -p results

# Help target
help:
	@echo "Available targets:"
	@echo "  install    - Install package"
	@echo "  dev        - Install with dev dependencies"
	@echo "  test       - Run tests"
	@echo "  test-cov   - Run tests with coverage"
	@echo "  lint       - Run linter"
	@echo "  lint-fix   - Fix linting issues"
	@echo "  typecheck  - Run type checker"
	@echo "  eval       - Run evaluation scripts"
	@echo "  docker     - Build Docker image"
	@echo "  ci         - Run full CI pipeline"
	@echo "  clean      - Clean build artifacts"



.PHONY: help install dev-install lint fmt typecheck test cov db-up db-down api ingest features labels clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime deps only
	pip install -e .

dev-install:  ## Install full stack + dev tooling
	pip install -e ".[all]"

lint:  ## Ruff lint
	ruff check src tests scripts

fmt:  ## Ruff format
	ruff format src tests scripts

typecheck:  ## mypy
	mypy src

test:  ## Run tests
	pytest

cov:  ## Run tests with coverage
	pytest --cov=aurax --cov-report=term-missing

db-up:  ## Start TimescaleDB
	docker compose up -d db

db-down:  ## Stop TimescaleDB
	docker compose down

api:  ## Run FastAPI dev server
	uvicorn aurax.api.main:app --reload --port 8000

ingest:  ## L0 — pull bars/ticks from MT5 into TimescaleDB
	python -m scripts.ingest

features:  ## L1 — build the feature matrix
	python -m scripts.build_features

labels:  ## L4 — generate triple-barrier labels + uniqueness weights
	python -m scripts.make_labels

validate:  ## §5 — run the CPCV/walk-forward/holdout gate (use ARGS="--demo")
	python -m scripts.validate $(ARGS)

clean:  ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml

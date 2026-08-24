# groundtruth — one-command workflows.
# `make help` lists targets. Everything assumes `uv` is installed and a `.env`
# exists (copy .env.example). DB targets expect the compose Postgres to be up.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Comma-separated CIKs for the demo ingest: Apple, Microsoft, NVIDIA.
DEMO_CIKS ?= 320193,789019,1045810
YEARS ?= 3

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Setup -----------------------------------------------------------------
.PHONY: install
install: ## Create the venv and install dev dependencies
	uv sync --extra dev

.PHONY: env
env: ## Create .env from the example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env — set SEC_USER_AGENT and ANTHROPIC_API_KEY")

# --- Quality (mirrors the CI fast lane) ------------------------------------
.PHONY: lint
lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

.PHONY: fmt
fmt: ## Auto-format
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: type
type: ## Mypy (strict)
	uv run mypy src/groundtruth

.PHONY: test
test: ## Unit tests (no DB, no API key)
	uv run pytest -m "not integration and not llm"

.PHONY: test-all
test-all: ## All tests including integration (needs DB)
	uv run pytest

.PHONY: check
check: lint type test ## Everything the CI fast lane runs

# --- Infrastructure --------------------------------------------------------
.PHONY: up
up: ## Start Postgres, API, Prometheus, Grafana
	docker compose up -d --build

.PHONY: down
down: ## Stop the stack (keeps volumes)
	docker compose down

.PHONY: db-shell
db-shell: ## psql into the compose Postgres
	docker compose exec postgres psql -U groundtruth -d groundtruth

# --- Data pipeline ---------------------------------------------------------
.PHONY: ingest
ingest: ## Ingest demo filings and activate the snapshot (DEMO_CIKS, YEARS)
	uv run python -m groundtruth.ingestion.ingest --ciks $(DEMO_CIKS) --years $(YEARS) --activate

.PHONY: labels
labels: ## Bootstrap golden queries + graded labels with the cheap model
	uv run python -m groundtruth.training.build_labels --per-snapshot 200

.PHONY: trainset
trainset: ## Build the reranker training JSONL from labels + live retrieval
	uv run python -m groundtruth.training.build_training_set

.PHONY: train
train: ## Train the LambdaMART reranker (writes models/reranker.ubj + sidecar)
	uv run python -m groundtruth.training.train_reranker

# --- Evaluation ------------------------------------------------------------
.PHONY: fixture
fixture: ## Load the hermetic eval fixture as an active snapshot
	uv run python -m groundtruth.eval.fixture --load

.PHONY: eval
eval: ## Run the eval harness against the active snapshot
	uv run python -m groundtruth.eval.run_eval --tag "local" --json-out eval/last_run.json

.PHONY: eval-baseline
eval-baseline: ## Run eval and pin the result as the CI regression baseline
	uv run python -m groundtruth.eval.run_eval --tag "baseline" --json-out eval/baseline.json
	@echo "Baseline updated. Commit eval/baseline.json to move the gate."

.PHONY: gate
gate: ## Compare eval/last_run.json against the committed baseline
	uv run python -m groundtruth.eval.gate --baseline eval/baseline.json --candidate eval/last_run.json

.PHONY: agreement
agreement: ## Cohen's kappa between machine and human labels
	uv run python -m groundtruth.eval.agreement

.PHONY: ablation
ablation: ## Reproduce the README ablation: RRF baseline vs learned reranker
	uv run python -m groundtruth.eval.run_eval --tag "rrf-only" --no-rerank --json-out eval/rrf_only.json
	uv run python -m groundtruth.eval.run_eval --tag "reranked" --json-out eval/reranked.json
	@echo "Compare eval/rrf_only.json and eval/reranked.json for the nDCG lift."

# --- Serving & Demo --------------------------------------------------------
.PHONY: serve
serve: ## Run the API locally (reload)
	uv run uvicorn groundtruth.api.main:app --reload --port 8000

.PHONY: demo
demo: ## Run the interactive Streamlit showcase app
	uv run --extra demo streamlit run demo/app.py

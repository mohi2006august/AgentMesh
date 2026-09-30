# AgentMesh Cloud — developer entry points.
# No `make` on your machine (e.g. Windows)? Every target is a thin wrapper over
# `python scripts/tasks.py <target>`, which works everywhere.

PY ?= python

.PHONY: install dev dev-local test test-isolation lint typecheck bench seed console clean

install:        ## Create .venv and install Python + console dependencies
	$(PY) scripts/tasks.py install

dev:            ## Full stack: kind + docker compose (Postgres, Redis, Qdrant, Kafka, ClickHouse, Temporal)
	$(PY) scripts/tasks.py dev

dev-local:      ## Zero-dependency stack: SQLite, in-process stores, mock model
	$(PY) scripts/tasks.py dev-local

test:           ## Unit + integration tests
	$(PY) scripts/tasks.py test

test-isolation: ## Cross-tenant isolation suite (retrieval, cache, adapters, secrets, audit)
	$(PY) scripts/tasks.py test-isolation

lint:           ## ruff + tenancy lint rules
	$(PY) scripts/tasks.py lint

typecheck:
	$(PY) scripts/tasks.py typecheck

bench:          ## Performance budgets (runtime step + gateway overhead)
	$(PY) scripts/tasks.py bench

seed:           ## Load demo plans, tenants and usage into the local stack
	$(PY) scripts/tasks.py seed

console:
	pnpm --filter console dev

clean:
	$(PY) scripts/tasks.py clean

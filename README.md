# AgentMesh Cloud

> A multi-tenant platform for deploying AI agents: an MCP tool gateway, per-tenant LoRA and RAG, shared caches and built-in usage billing with unit economics.

**Concepts:** `MCP`, `LLMs`, `RAG`, `LoRA`, `Caches`, `Context Window`, `Burn Rate`, `CAC`

AgentMesh is a platform (think 'Vercel for agents'). Developers define agents declaratively (model, tools via MCP servers, knowledge sources, adapters). The platform runs them with tenant isolation: an MCP gateway with per-tenant auth and policies, per-tenant RAG indexes, per-tenant LoRA adapters on shared base models, multi-level caching across tenants where safe, and context budgeting. Metering feeds usage-based billing and a unit-economics console (AI COGS, gross margin, burn impact and CAC payback per plan).

## Quick start

The local profile needs only Python 3.11+ and Node 20+. No Docker, no API keys: it runs on SQLite, in-process stores and a deterministic mock model.

```bash
python scripts/tasks.py install      # or: make install
python scripts/tasks.py test         # or: make test
```

| Command | What it does |
|---|---|
| `make dev` | Full stack: kind + docker compose (Postgres, Redis, Qdrant, Kafka, ClickHouse, Temporal) |
| `make dev-local` | Zero-dependency stack on SQLite and in-process stores |
| `mesh deploy examples/support-agent.yaml` | Deploy the sample agent |
| `make test-isolation` | Run the tenant isolation suite |
| `pnpm --filter console dev` | Console |

No `make`? Every target is `python scripts/tasks.py <target>`.

## Repository layout

```text
core/              # tenancy, tenant-scoped SQL + RLS, audit trail, secrets, config
sdk/               # mesh CLI, agent.yaml schema
control-plane/     # api, deploy, policies
runtime/           # durable agent loop, context manager
mcp-gateway/       # registry, authz, audit
retrieval/         # per-tenant indexes, ACL-aware search
inference-router/  # providers, vLLM multi-LoRA, fallbacks
cache/             # tenant-scoped exact + semantic caches
metering/          # meter events, billing sync
economics/         # margin, burn, CAC models
console/           # Next.js
tools/             # project lint rules
tests/             # unit, integration, isolation and benchmark suites
```

Each component directory holds one importable `mesh_*` Python package.

## Docs

| File | Contents |
|---|---|
| [BRAIN.md](BRAIN.md) | Concept primers, research, experiments, decision log |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI coding assistants |
| [DESIGN.md](DESIGN.md) | UX flows, screens, wireframes, visual language |
| [PRD.md](PRD.md) | Problem, users, requirements, metrics, roadmap |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment, failure modes |
| [TECHNICAL.md](TECHNICAL.md) | Stack, data models, APIs, algorithms, setup |

## Contributors

- mohi2006august: author and maintainer
- Claude (Anthropic): pair-programming co-author via [Claude Code](https://claude.com/claude-code)

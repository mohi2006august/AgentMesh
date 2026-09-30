# CLAUDE.md: AgentMesh Cloud

Guidance for Claude Code (and other AI assistants) working in this project.

## Project Summary

A multi-tenant platform for deploying AI agents: an MCP tool gateway, per-tenant LoRA and RAG, shared caches and built-in usage billing with unit economics. AgentMesh is a platform (think 'Vercel for agents'). Developers define agents declaratively (model, tools via MCP servers, knowledge sources, adapters). The platform runs them with tenant isolation: an MCP gateway with per-tenant auth and policies, per-tenant RAG indexes, per-tenant LoRA adapters on shared base models, multi-level caching across tenants where safe, and context budgeting. Metering feeds usage-based billing and a unit-economics console (AI COGS, gross margin, burn impact and CAC payback per plan).

## Read First

- `PRD.md`: what we are building and why; requirement IDs (FR-xx) are referenced in commits.
- `SYSTEM_ARCHITECTURE.md`: components, data flow and failure modes.
- `TECHNICAL.md`: stack, data models, APIs and algorithms.
- `DESIGN.md`: UX flows, screens and visual tokens.
- `BRAIN.md`: concept primers, experiments and the decision log.

## Commands

| Command | What it does |
|---|---|
| `make dev` | Start local stack (kind + docker compose) |
| `mesh deploy examples/support-agent.yaml` | Deploy sample agent |
| `make test-isolation` | Run tenant isolation suite |
| `pnpm --filter console dev` | Console |

## Code Conventions

- Every data access function takes `tenant_id` as its first argument; lint rule enforces it.
- Metering events are emitted from a single helper; never hand-roll events.
- Keep functions small and typed; no silent `except`. Log with structured fields.
- Every new feature references its FR-ID in the PR description and ships with tests.
- Secrets only come from environment variables. Never hard-code keys or commit `.env`.

## Gotchas

- Cache scope must include adapter id, or tenants can see each other's styled responses.
- Stripe usage records are idempotent only with a stable idempotency key per meter event.

## When Unsure

- Prefer the simplest design that satisfies the PRD; record bigger trade-offs in `BRAIN.md` → Decision Log.
- Run the eval / benchmark suite before claiming a quality or performance improvement.
- Ask before adding a new paid dependency, external service or data store.

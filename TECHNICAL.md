# AgentMesh Cloud: Technical Specification

> A multi-tenant platform for deploying AI agents: an MCP tool gateway, per-tenant LoRA and RAG, shared caches and built-in usage billing with unit economics.
>
> **Concepts:** `MCP`, `LLMs`, `RAG`, `LoRA`, `Caches`, `Context Window`, `Burn Rate`, `CAC`

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python + Temporal | Durable agent loops. |
| Gateway | Go | Performance + concurrency. |
| MCP | Official SDKs | Protocol compliance. |
| Inference | Provider APIs + vLLM multi-LoRA | Flexibility. |
| Data | Postgres, Qdrant, Redis, Kafka, ClickHouse | Proven. |
| Billing | Stripe Billing (usage-based) | Standard. |
| Frontend | Next.js | Console + widget. |

## 2. Repository Layout

```text
agentmesh/
├── control-plane/     # api, deploy, policies
├── runtime/           # temporal workflows, context manager
├── mcp-gateway/       # go: registry, authz, audit
├── retrieval/
├── inference-router/
├── cache/
├── metering/          # kafka consumers, clickhouse schemas
├── economics/         # margin, burn, cac models
├── sdk/               # mesh CLI, agent.yaml schema
└── console/           # Next.js
```

## 3. Data Models

```yaml
# agent.yaml
name: support-agent
model: {default: claude-sonnet, fallback: small-model, adapter: "{tenant}-tone"}
instructions: prompts/support.md
knowledge:
  - source: tenant_drive
    acl: inherit
tools:
  - mcp: zendesk        # tenant supplies OAuth at install
    policy: {create_ticket: allow, refund: approve}
  - mcp: stripe-readonly
context: {budget_tokens: 24000, memory: session+profile}
cache: {semantic: {threshold: 0.93, ttl: 1d}, scope: tenant}
limits: {max_steps: 12, max_usd_per_run: 0.40}
billing: {meter: [tokens, tool_calls, retrievals]}
```

## 4. API Surface

| Method | Endpoint / Tool | Purpose |
|---|---|---|
| cli | mesh deploy agent.yaml | Deploy/update agent. |
| POST | /v1/agents/{id}/runs | Start run (stream events). |
| POST | /v1/tenants/{id}/connections | Tenant connects MCP server / data source. |
| GET | /v1/economics/margins?by=plan | Margins by plan/tenant. |
| GET | /v1/audit?tenant= | Audit events. |

## 5. Core Algorithms

### Tenant-scoped cache key

`key = H(tenant_id, agent_version, model, adapter, tool_set_hash, normalized_messages)`. The global cache is allowed only if every retrieved source is marked public.

### Fair scheduling

Weighted fair queueing across tenants by plan weight. A token bucket per tenant for model tokens prevents noisy neighbours.

### Per-run COGS

`cogs = Σ model_cost(tokens, cached_tokens, model) + Σ tool_cost + retrieval_ops·unit_cost + gpu_seconds·rate`. Margin = revenue allocated by plan − cogs.

### Burn & CAC impact

Monthly AI COGS projected from usage growth feeds the burn forecast. Trial-tenant COGS are added to acquisition cost for CAC payback.

## 6. Configuration

| Variable | Purpose |
|---|---|
| MESH_REGION | Data residency region |
| TEMPORAL_ADDR | Workflow engine |
| STRIPE_KEY | Billing |
| VAULT_ADDR | Secrets |
| VLLM_ENDPOINTS | Inference fleet |

## 7. Testing Strategy

- Isolation suite: automated cross-tenant probes for retrieval, cache, adapters, secrets, audit.
- Policy matrix tests for MCP gateway.
- Chaos tests: provider outage, MCP server failure.
- Billing reconciliation tests vs meter totals.

## 8. Performance Budgets

- Runtime step overhead < 100 ms p95.
- Gateway overhead < 15 ms p95.
- Metering lag < 60 s.

## 9. Local Setup

```bash
make dev
mesh deploy examples/support-agent.yaml
make test-isolation
pnpm --filter console dev
```

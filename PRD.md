# AgentMesh Cloud: Product Requirements Document

> A multi-tenant platform for deploying AI agents: an MCP tool gateway, per-tenant LoRA and RAG, shared caches and built-in usage billing with unit economics.
>
> **Concepts:** `MCP`, `LLMs`, `RAG`, `LoRA`, `Caches`, `Context Window`, `Burn Rate`, `CAC`

## 1. Overview

AgentMesh is a platform (think 'Vercel for agents'). Developers define agents declaratively (model, tools via MCP servers, knowledge sources, adapters). The platform runs them with tenant isolation: an MCP gateway with per-tenant auth and policies, per-tenant RAG indexes, per-tenant LoRA adapters on shared base models, multi-level caching across tenants where safe, and context budgeting. Metering feeds usage-based billing and a unit-economics console (AI COGS, gross margin, burn impact and CAC payback per plan).

## 2. Problem Statement

Every SaaS company now wants to ship AI agents to its customers. Each team rebuilds the same infrastructure: tool integrations, tenant isolation, per-customer knowledge and customisation, cost control, usage-based billing and a way to know whether the AI feature is profitable. Getting this wrong leaks data across tenants or sinks margins.

## 3. Target Users & Personas

| Persona | Needs |
|---|---|
| SaaS product team | Ship agents to customers fast without rebuilding infra. |
| Platform/security | Guaranteed tenant isolation and audit. |
| Founder/finance | Know agent margins; price plans correctly. |

## 4. Goals

- Deploy an agent from YAML in < 10 minutes.
- Zero cross-tenant data access (formally tested).
- ≥ 30% infra cost reduction via shared base models + caches.
- Per-tenant margin visible in real time.

## 5. Non-Goals

- Building an LLM.
- End-user chat UI builder beyond an embeddable widget.

## 6. User Stories

- As a developer, I write `agent.yaml` with 3 MCP tools and a knowledge source, run `mesh deploy`, and get an API + widget.
- As a tenant admin, I connect my Google Drive; only my users' agents can retrieve from it.
- As a founder, I see Pro-plan customers have 71% gross margin but Enterprise heavy users are at 22%.
- As a tenant, my adapter `acme-tone` is applied to my agent's responses.

## 7. Functional Requirements

| ID | Priority | Feature | Description |
|---|---|---|---|
| FR-01 | P0 | Agent runtime | Declarative agents; tool-use loop; streaming; retries; human-approval steps. |
| FR-02 | P0 | MCP gateway | Registry of MCP servers; per-tenant OAuth credentials; policy (allow/approve/deny); audit. |
| FR-03 | P0 | Tenant isolation | Per-tenant namespaces for vectors, adapters, caches, secrets; row-level security. |
| FR-04 | P0 | Metering + billing | Tokens, tool calls, retrieval ops metered → Stripe usage billing. |
| FR-05 | P1 | Per-tenant RAG | Connectors → per-tenant indexes; ACL-aware retrieval. |
| FR-06 | P1 | Multi-LoRA | Tenant adapters on shared base models (vLLM). |
| FR-07 | P1 | Caches | Tenant-scoped semantic cache, global safe caches (public docs), provider prompt caching. |
| FR-08 | P1 | Context manager | Budgeted packing of memory, RAG results and tool outputs. |
| FR-09 | P1 | Unit-economics console | AI COGS, margin by plan/tenant, burn impact, CAC payback incl. trial agent costs. |
| FR-10 | P2 | Agent evals & canaries | Per-agent eval suites gate deploys. |

## 8. Non-Functional Requirements

- 99.9% availability; regional data residency (EU/US).
- p95 agent step overhead < 100 ms excluding model time.
- SOC 2-ready audit logging.
- Noisy-neighbour protection (per-tenant quotas).

## 9. Success Metrics

| Metric | Target |
|---|---|
| Time to first deployed agent | < 10 min |
| Cross-tenant leakage incidents | 0 |
| Infra cost per 1k agent runs vs naive | −30% |
| Customers using margin console monthly | ≥ 50% |

## 10. Release Plan

| Phase | Scope |
|---|---|
| M1 | Agent runtime + MCP gateway + metering (single tenant). |
| M2 | Multi-tenancy + isolation test suite + Stripe billing. |
| M3 | Per-tenant RAG + caches + context manager. |
| M4 | Multi-LoRA + unit-economics console. |
| M5 | Evals/canaries + data residency. |

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Cross-tenant leakage via shared cache | Scope keys always include tenant id; global cache only for content marked public; isolation fuzzing. |
| Prompt injection through tools/RAG causing data exfiltration | Tool policies, egress allowlists, approval for sensitive tools, output filtering. |
| Margin erosion from heavy users | Quotas, plan limits, routing to cheaper models, margin alerts. |

## 12. Open Questions

- Bring-your-own-model-key vs platform-managed keys pricing?
- Allow tenants to upload custom MCP servers (sandboxing)?

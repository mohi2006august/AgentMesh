# AgentMesh Cloud: Design Document

> A multi-tenant platform for deploying AI agents: an MCP tool gateway, per-tenant LoRA and RAG, shared caches and built-in usage billing with unit economics.
>
> **Concepts:** `MCP`, `LLMs`, `RAG`, `LoRA`, `Caches`, `Context Window`, `Burn Rate`, `CAC`

## 1. Design Principles

- Isolation is a feature customers can see (audit + scopes UI).
- Every agent run has a price tag.

## 2. Key User Flows

### Developer deploys agent

1. Write agent.yaml.
2. `mesh deploy`.
3. Test in console playground.
4. Embed widget/API.

### Founder checks margins

1. Open Economics.
2. Filter Enterprise plan.
3. See heavy tenants' margins.
4. Apply quota or routing policy.

## 3. Screens & Surfaces

| Screen | What it shows / does |
|---|---|
| Agents | List, versions, eval status, traffic. |
| Run trace | Step timeline: model calls, tools, retrievals, cost per step. |
| Tenants | Connections, adapters, quotas, residency. |
| Economics | COGS, margin by plan/tenant, burn impact, CAC payback. |
| Audit | Tool calls and data access by tenant/user. |

## 4. Wireframe (Primary Screen)

```text
Economics · Sep 2026                      AI COGS $38.4k  Revenue $214k  GM 82%
Plan         Tenants  Rev/mo   COGS/mo  Margin   Trend
Starter        412    $20.6k   $3.1k     85%     ▁▂▂▂
Pro            118    $94.4k   $12.7k    87%     ▂▂▃▃
Enterprise      14    $99.0k   $22.6k    77%  ⚠  ▃▄▅▆  (3 tenants < 40%)
Burn impact: AI COGS growth adds $6.2k/mo net burn by Mar · Trial agent cost adds $41 to CAC
```

## 5. Visual Language

| Token | Value |
|---|---|
| Primary accent | `#0891B2` (cyan-600) |
| Neutral surface | `#0F1117` dark / `#FAFAF9` light |
| Success / Warning / Danger | `#22C55E` / `#F59E0B` / `#EF4444` |
| Type | Inter for UI, JetBrains Mono for code, tokens and numbers |
| Spacing scale | 4 / 8 / 12 / 16 / 24 / 32 px |
| Radius | 8px cards, 6px inputs, full for pills |

## 6. States

| State | Treatment |
|---|---|
| Approval pending | Run paused with approver notification. |
| Quota near | Tenant banner + admin alert. |

## 7. Accessibility

- WCAG 2.2 AA contrast in both themes; never encode meaning with colour alone (pair with icon or label).
- Full keyboard navigation with visible focus rings; `/` focuses search, `?` opens shortcuts.
- Charts ship with a data-table fallback and `aria-describedby` summaries.
- Respect `prefers-reduced-motion`; streaming text is announced politely via `aria-live`.

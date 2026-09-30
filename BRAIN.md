# AgentMesh Cloud: Brain

> Second brain for this project: concepts, mental models, research, ideas and decisions. Keep it messy and alive.

## 1. Why This Project

It's the capstone of production AI engineering, spanning protocols (MCP), retrieval, customisation (LoRA), performance (caches, context), multi-tenant security and the business model (billing, margins, burn, CAC).

## 2. Concept Primers

### MCP

Model Context Protocol: an open JSON-RPC protocol that lets AI clients (Claude, IDEs, agents) discover and call tools, read resources and use prompts exposed by servers in a standard way.

- **Server**: Exposes tools, resources and prompts over stdio or Streamable HTTP.
- **Tool**: A typed function the model can call (a JSON Schema input and a structured result).
- **Resource**: Read-only context addressed by URI (files, rows, docs) that the client can attach.
- **Transport / auth**: stdio for local servers, Streamable HTTP + OAuth 2.1 for remote servers.

> **Mental model:** Think of MCP as USB-C for AI. Write the integration once and every compliant client can use it. Tool descriptions are prompts, so write them for the model.

### LLMs

Large Language Models: transformer networks trained on large text corpora to predict the next token, then aligned (SFT and RLHF/RLAIF) to follow instructions and use tools.

- **Prompting**: System prompts, few-shot examples, structured outputs, tool use.
- **Sampling**: Temperature, top-p and max tokens shape determinism and length.
- **Evals**: Task-specific test sets plus LLM-as-judge with calibrated rubrics.
- **Routing**: Sending each request to the cheapest model that meets its quality bar.

> **Mental model:** Treat the LLM as a probabilistic component. Wrap it with evals, guardrails, retries and observability, the same way you would a flaky network dependency.

### RAG

Retrieval-Augmented Generation: fetch relevant passages from an external corpus at query time and put them in the prompt so the LLM answers from evidence instead of memory.

- **Chunking**: Splitting documents into retrievable units (fixed tokens, sentence windows, semantic or structure-aware splits).
- **Hybrid retrieval**: Combining sparse (BM25) and dense (embedding) search, usually merged with Reciprocal Rank Fusion.
- **Re-ranking**: A cross-encoder rescoring the top-k candidates for precision before they reach the prompt.
- **Grounding / faithfulness**: Whether every claim in the answer is supported by a retrieved passage.

> **Mental model:** Retrieval sets the ceiling and generation sets the floor. If the right chunk never gets retrieved, no prompt can save the answer, so measure recall@k before tuning prompts.

### LoRA

Low-Rank Adaptation: freeze the base weights W and learn a low-rank update ΔW = B·A (rank r ≪ d), training well under 1% of parameters with small, swappable adapters.

- **Rank r / alpha**: Capacity of the update and its scaling (alpha/r).
- **Target modules**: Which projections get adapters (q,k,v,o, MLP).
- **QLoRA**: LoRA on top of a 4-bit quantised base model; fine-tune big models on one GPU.
- **Multi-LoRA serving**: One base model in memory serving many adapters per request (S-LoRA, Punica).

> **Mental model:** The base model is the OS and adapters are apps. They are cheap to train, store and switch, so give each tenant, task or style its own adapter.

### Caches

Storing results of expensive operations for reuse. In LLM systems this means exact response caches, semantic caches, embedding caches, and provider-side prompt/KV caching.

- **Hit ratio**: Fraction of requests served from cache; the core ROI metric.
- **Semantic cache**: Returns a cached answer if a new query is similar enough (embedding distance < threshold).
- **Prompt caching**: Provider reuses computed KV for a repeated prompt prefix, cutting cost and latency.
- **Invalidation / TTL**: When cached answers become stale; famously one of the two hard problems.

> **Mental model:** A cache is a bet that the future looks like the past. A wrong semantic cache hit costs more than a miss, so tune thresholds for precision.

### Context Window

The maximum number of tokens (prompt plus output) a model can attend to in one call. Everything the model 'knows' about the task at inference time must fit inside it.

- **Lost in the middle**: Models attend less reliably to information buried mid-context than at the start or end.
- **Compaction**: Summarising or pruning older turns so long sessions fit the budget.
- **KV cache**: Stored attention keys/values; grows linearly with context length and dominates memory.
- **Needle-in-a-haystack**: Eval that plants a fact at varying depths to test long-context recall.

> **Mental model:** Context is a budget, not a bucket. More tokens cost more money and latency and can dilute attention, so curate what goes in.

### Burn Rate

How fast a company spends cash. Gross burn is total monthly spend. Net burn is spend minus revenue. Runway = cash ÷ net burn.

- **Gross vs net burn**: Total outflow vs outflow minus inflow.
- **Runway**: Months until cash hits zero at the current (or projected) net burn.
- **Burn multiple**: Net burn ÷ net new ARR; efficiency of growth spending (< 1.5 is good).
- **Default alive**: Whether current trajectory reaches profitability before cash runs out.

> **Mental model:** Burn is a lagging indicator of decisions made months ago. Model the decisions (hires, contracts, pricing), not only the totals.

### CAC

Customer Acquisition Cost: total sales and marketing spend ÷ new customers acquired over a period. It is judged against LTV and payback period.

- **Blended vs paid CAC**: All customers vs only those attributed to paid channels.
- **LTV:CAC**: Lifetime value relative to acquisition cost; ~3:1 is a common healthy benchmark.
- **CAC payback**: Months of gross-margin-adjusted revenue needed to recover CAC.
- **Attribution**: Assigning credit to touchpoints (first/last touch, linear, data-driven, MMM).

> **Mental model:** Blended CAC hides channel truth. Always slice by channel, cohort and segment, and include fully loaded costs (salaries, tools).

## 3. What I Want to Learn

- [ ] Multi-tenant architecture and isolation testing.
- [ ] Durable agent execution.
- [ ] Usage-based billing.
- [ ] AI unit economics at platform scale.

## 4. Reading & Research List

- MCP specification (authorization, security best practices)
- Temporal docs: durable execution
- AWS SaaS Lens: tenant isolation
- Stripe usage-based billing docs
- OWASP LLM Top 10 (excessive agency, data leakage)

## 5. Experiments to Run

| # | Hypothesis | How to test |
|---|---|---|
| 1 | Shared base + LoRA cheaper than per-tenant fine-tuned endpoints by ≥ 5x | Cost model + load test. |
| 2 | Tenant-scoped semantic cache hit rate ≥ 20% for support agents | Replay traffic. |

## 6. Ideas Backlog

- Agent marketplace.
- Bring-your-own-cloud deployment.
- Automatic cheaper-model routing when margins drop.

## 7. Decision Log

| Date | Decision | Reason |
|---|---|---|
| YYYY-MM-DD | Temporal for runtime | Agent runs need durability, retries and human-approval pauses. |
| YYYY-MM-DD | Go for gateway | High-concurrency proxy with low overhead. |

## 8. Scratchpad

- _Dump raw thoughts, links and half-formed ideas here._

"""Retrieval service (FR-05): ingest into per-tenant namespaces, ACL-aware hybrid search.

Namespaces:
- each tenant's own sources live under its tenant id;
- public knowledge shared by every tenant (product docs) lives under the reserved
  `PUBLIC_NAMESPACE` tenant and is searched with a separate transaction scoped to it.

A search never opens a transaction for any tenant other than the caller and the public
namespace, so cross-tenant reads are structurally impossible here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mesh_core.audit import AuditStore
from mesh_core.ids import content_hash, stable_id
from mesh_core.tenancy import Principal
from mesh_retrieval.chunking import chunk_text
from mesh_retrieval.connectors import Connector
from mesh_retrieval.embedding import Embedder
from mesh_retrieval.index import LocalIndex, ScoredChunk, rrf
from mesh_retrieval.store import EVERYONE, Document, KnowledgeStore
from mesh_sdk.spec import KnowledgeSpec

PUBLIC_NAMESPACE = "mesh-public"


@dataclass(frozen=True)
class IngestReport:
    source: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0


@dataclass(frozen=True)
class Hit:
    source: str
    doc_id: str
    title: str
    text: str
    uri: str
    score: float
    public: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "uri": self.uri,
            "score": round(self.score, 6),
            "public": self.public,
        }


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    searched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    hidden_by_acl: int = 0

    @property
    def all_public(self) -> bool:
        """True when every hit came from public knowledge: the answer is safe to share across tenants."""
        return all(hit.public for hit in self.hits)


def can_read(principal: Principal, acl: tuple[str, ...]) -> bool:
    return EVERYONE in acl or bool(principal.acl_principals & set(acl))


def acl_filter(spec: KnowledgeSpec, principal: Principal) -> Callable[[tuple[str, ...]], bool]:
    """`acl: inherit` enforces each document's ACL; `acl: tenant` opens the source to the tenant."""
    if spec.acl == "tenant":
        return lambda acl: True
    return lambda acl: can_read(principal, acl)


class RetrievalService:
    def __init__(
        self,
        store: KnowledgeStore,
        embedder: Embedder,
        audit: AuditStore,
        *,
        max_chunk_chars: int = 900,
    ) -> None:
        self.store = store
        self._embedder = embedder
        self._audit = audit
        self._index = LocalIndex(store, embedder.dim)
        self._max_chunk_chars = max_chunk_chars

    # --- ingestion ----------------------------------------------------------------------

    def ingest(
        self,
        tenant_id: str,
        source: str,
        documents: Sequence[Document],
        *,
        kind: str = "inline",
        config: dict[str, Any] | None = None,
        prune: bool = False,
    ) -> IngestReport:
        """Upsert documents into a source; with `prune`, delete documents no longer present."""
        self.store.upsert_source(tenant_id, source, kind, config or {})
        existing = self.store.document_hashes(tenant_id, source)
        added = updated = unchanged = chunk_count = 0
        for document in documents:
            digest = content_hash(
                document.title, document.text, document.uri, ",".join(sorted(document.acl))
            )
            if existing.get(document.doc_id) == digest:
                unchanged += 1
                continue
            pieces = chunk_text(
                document.text, title=document.title, max_chars=self._max_chunk_chars
            )
            vectors = self._embedder.embed([piece.text for piece in pieces])
            chunks = [
                (
                    stable_id("chk", tenant_id, source, document.doc_id, piece.ordinal),
                    piece.ordinal,
                    piece.text,
                    vectors[i],
                )
                for i, piece in enumerate(pieces)
            ]
            self.store.replace_document(tenant_id, source, document, digest, chunks)
            chunk_count += len(chunks)
            if document.doc_id in existing:
                updated += 1
            else:
                added += 1
        removed = 0
        if prune:
            present = {document.doc_id for document in documents}
            for doc_id in set(existing) - present:
                self.store.delete_document(tenant_id, source, doc_id)
                removed += 1
        self.store.mark_synced(tenant_id, source)
        self._index.invalidate(tenant_id, source)
        report = IngestReport(source, added, updated, unchanged, removed, chunk_count)
        self._audit.record(
            tenant_id, actor="system", action="knowledge.ingest", resource=f"knowledge:{source}",
            decision="allow", detail=report.__dict__,
        )  # fmt: skip
        return report

    def sync(
        self,
        tenant_id: str,
        source: str,
        connector: Connector,
        config: dict[str, Any] | None = None,
    ) -> IngestReport:
        return self.ingest(
            tenant_id, source, connector.fetch(), kind=connector.kind, config=config, prune=True
        )

    # --- search ---------------------------------------------------------------------------

    def search(
        self,
        tenant_id: str,
        principal: Principal,
        query: str,
        specs: Sequence[KnowledgeSpec],
        *,
        top_k: int = 5,
        run_id: str | None = None,
    ) -> SearchResult:
        query_vector = self._embedder.embed([query])[0]
        per_source: list[list[tuple[ScoredChunk, bool]]] = []
        searched: list[str] = []
        missing: list[str] = []
        hidden = 0
        for spec in specs:
            if self.store.has_source(tenant_id, spec.source):
                results, blocked = self._index.search(
                    tenant_id, spec.source, query, query_vector,
                    allowed=acl_filter(spec, principal), limit=spec.top_k,
                )  # fmt: skip
                is_public = False
            elif self.store.has_source(PUBLIC_NAMESPACE, spec.source):
                results, blocked = self.search_public(spec.source, query, query_vector, spec.top_k)
                is_public = True
            else:
                missing.append(spec.source)
                continue
            searched.append(spec.source)
            hidden += blocked
            per_source.append([(result, is_public) for result in results])

        # Fuse across sources by rank so no single source's score scale dominates.
        flat = [item for ranking in per_source for item in ranking]
        offsets, start = [], 0
        for ranking in per_source:
            offsets.append(list(range(start, start + len(ranking))))
            start += len(ranking)
        fused = rrf(offsets)
        order = sorted(fused, key=lambda index: -fused[index])[:top_k]
        hits = [
            Hit(
                source=flat[i][0].chunk.source,
                doc_id=flat[i][0].chunk.doc_id,
                title=flat[i][0].chunk.title,
                text=flat[i][0].chunk.text,
                uri=flat[i][0].chunk.uri,
                score=fused[i],
                public=flat[i][1],
            )
            for i in order
        ]
        self._audit.record(
            tenant_id,
            actor=f"user:{principal.user_id}",
            action="data.retrieve",
            resource="knowledge:" + ",".join(searched or missing),
            decision="allow",
            run_id=run_id,
            detail={
                "query": query[:200],
                "documents": sorted({f"{hit.source}/{hit.doc_id}" for hit in hits}),
                "hidden_by_acl": hidden,
            },
        )
        return SearchResult(hits, searched, missing, hidden)

    def search_public(
        self, source: str, query: str, query_vector: Any, limit: int
    ) -> tuple[list[ScoredChunk], int]:
        """Public knowledge: readable by every tenant, so no ACL applies."""
        return self._index.search(
            PUBLIC_NAMESPACE, source, query, query_vector, allowed=lambda acl: True, limit=limit
        )

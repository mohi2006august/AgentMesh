"""Per-tenant knowledge storage: sources, documents with ACLs, and embedded chunks (FR-05)."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mesh_core.clock import Clock, SystemClock
from mesh_core.db import Database, Row, dumps, loads
from mesh_retrieval.embedding import Vector

# Everyone in the tenant may read a document whose ACL contains this entry.
EVERYONE = "*"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    uri: str = ""
    # Principals allowed to read: "user:<id>", "group:<name>", or "*" for the whole tenant.
    acl: tuple[str, ...] = (EVERYONE,)


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    source: str
    doc_id: str
    ordinal: int
    text: str
    title: str
    uri: str
    acl: tuple[str, ...]
    embedding: Vector = field(repr=False, compare=False)


def encode_vector(vector: Vector) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def decode_vector(data: str) -> Vector:
    return np.frombuffer(base64.b64decode(data), dtype=np.float32)


class KnowledgeStore:
    def __init__(self, db: Database, clock: Clock | None = None) -> None:
        self._db = db
        self._clock = clock or SystemClock()

    def upsert_source(self, tenant_id: str, source: str, kind: str, config: dict[str, Any]) -> None:
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                """
                INSERT INTO knowledge_sources (tenant_id, source, kind, config, created_at)
                VALUES (:tenant_id, :source, :kind, :config, :now)
                ON CONFLICT (tenant_id, source) DO UPDATE SET kind = excluded.kind, config = excluded.config
                """,
                {"source": source, "kind": kind, "config": dumps(config), "now": self._clock.now()},
            )

    def sources(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                """
                SELECT s.source, s.kind, s.config, s.created_at, s.synced_at,
                       (SELECT COUNT(*) FROM documents d WHERE d.tenant_id = :tenant_id AND d.source = s.source) AS documents
                FROM knowledge_sources s WHERE s.tenant_id = :tenant_id ORDER BY s.source
                """
            )
        return [{**row, "config": loads(row["config"], {})} for row in rows]

    def has_source(self, tenant_id: str, source: str) -> bool:
        with self._db.tenant_tx(tenant_id) as tx:
            row = tx.query_one(
                "SELECT 1 AS present FROM knowledge_sources WHERE tenant_id = :tenant_id AND source = :source",
                {"source": source},
            )
        return row is not None

    def mark_synced(self, tenant_id: str, source: str) -> None:
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                "UPDATE knowledge_sources SET synced_at = :now WHERE tenant_id = :tenant_id AND source = :source",
                {"source": source, "now": self._clock.now()},
            )

    def document_hashes(self, tenant_id: str, source: str) -> dict[str, str]:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                "SELECT doc_id, content_hash FROM documents WHERE tenant_id = :tenant_id AND source = :source",
                {"source": source},
            )
        return {row["doc_id"]: row["content_hash"] for row in rows}

    def replace_document(
        self,
        tenant_id: str,
        source: str,
        document: Document,
        content_hash: str,
        chunks: list[tuple[str, int, str, Vector]],
    ) -> None:
        """Atomically replace one document and all of its chunks."""
        with self._db.tenant_tx(tenant_id) as tx:
            params = {"source": source, "doc_id": document.doc_id}
            tx.execute(
                "DELETE FROM chunks WHERE tenant_id = :tenant_id AND source = :source AND doc_id = :doc_id",
                params,
            )
            tx.execute(
                """
                INSERT INTO documents (tenant_id, source, doc_id, title, uri, acl, content_hash, updated_at)
                VALUES (:tenant_id, :source, :doc_id, :title, :uri, :acl, :content_hash, :now)
                ON CONFLICT (tenant_id, source, doc_id) DO UPDATE SET
                  title = excluded.title, uri = excluded.uri, acl = excluded.acl,
                  content_hash = excluded.content_hash, updated_at = excluded.updated_at
                """,
                {
                    **params,
                    "title": document.title,
                    "uri": document.uri,
                    "acl": dumps(list(document.acl)),
                    "content_hash": content_hash,
                    "now": self._clock.now(),
                },
            )
            for chunk_id, ordinal, text, vector in chunks:
                tx.execute(
                    """
                    INSERT INTO chunks (tenant_id, chunk_id, source, doc_id, ordinal, text, embedding)
                    VALUES (:tenant_id, :chunk_id, :source, :doc_id, :ordinal, :text, :embedding)
                    """,
                    {
                        **params,
                        "chunk_id": chunk_id,
                        "ordinal": ordinal,
                        "text": text,
                        "embedding": encode_vector(vector),
                    },
                )

    def delete_document(self, tenant_id: str, source: str, doc_id: str) -> None:
        params = {"source": source, "doc_id": doc_id}
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                "DELETE FROM chunks WHERE tenant_id = :tenant_id AND source = :source AND doc_id = :doc_id",
                params,
            )
            tx.execute(
                "DELETE FROM documents WHERE tenant_id = :tenant_id AND source = :source AND doc_id = :doc_id",
                params,
            )

    def load_chunks(self, tenant_id: str, source: str) -> list[StoredChunk]:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                """
                SELECT c.chunk_id, c.source, c.doc_id, c.ordinal, c.text, c.embedding, d.title, d.uri, d.acl
                FROM chunks c
                JOIN documents d ON d.tenant_id = c.tenant_id AND d.source = c.source AND d.doc_id = c.doc_id
                WHERE c.tenant_id = :tenant_id AND d.tenant_id = :tenant_id AND c.source = :source
                ORDER BY c.doc_id, c.ordinal
                """,
                {"source": source},
            )
        return [_chunk(row) for row in rows]


def _chunk(row: Row) -> StoredChunk:
    return StoredChunk(
        chunk_id=row["chunk_id"],
        source=row["source"],
        doc_id=row["doc_id"],
        ordinal=row["ordinal"],
        text=row["text"],
        title=row["title"],
        uri=row["uri"],
        acl=tuple(loads(row["acl"], [])),
        embedding=decode_vector(row["embedding"]),
    )

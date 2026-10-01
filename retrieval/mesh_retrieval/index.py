"""Hybrid search over one knowledge source of one tenant.

Dense (cosine) and sparse (BM25) rankings are fused with Reciprocal Rank Fusion. The ACL
filter runs before ranking, so a document the user may not read can never displace one
they can, and never shows up in scores, counts or snippets.

The in-memory cache is keyed by (tenant, source); it is rebuilt from the tenant-scoped
store after any write to that source.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mesh_retrieval.bm25 import BM25
from mesh_retrieval.embedding import Vector, cosine
from mesh_retrieval.store import KnowledgeStore, StoredChunk

RRF_K = 60
CANDIDATES = 50


@dataclass(frozen=True)
class ScoredChunk:
    chunk: StoredChunk
    score: float  # fused RRF score
    dense: float
    sparse: float


@dataclass
class _Entry:
    chunks: list[StoredChunk]
    matrix: Vector
    bm25: BM25


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank + 1)
    return fused


class LocalIndex:
    def __init__(self, store: KnowledgeStore, dim: int) -> None:
        self._store = store
        self._dim = dim
        self._cache: dict[tuple[str, str], _Entry] = {}
        self._lock = threading.Lock()

    def invalidate(self, tenant_id: str, source: str) -> None:
        with self._lock:
            self._cache.pop((tenant_id, source), None)

    def _entry(self, tenant_id: str, source: str) -> _Entry:
        key = (tenant_id, source)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        chunks = self._store.load_chunks(tenant_id, source)
        matrix = (
            np.stack([chunk.embedding for chunk in chunks])
            if chunks
            else np.zeros((0, self._dim), dtype=np.float32)
        )
        entry = _Entry(chunks, matrix, BM25([chunk.text for chunk in chunks]))
        with self._lock:
            self._cache[key] = entry
        return entry

    def search(
        self,
        tenant_id: str,
        source: str,
        query_text: str,
        query_vector: Vector,
        *,
        allowed: Callable[[tuple[str, ...]], bool],
        limit: int,
    ) -> tuple[list[ScoredChunk], int]:
        """Top `limit` chunks the caller may read, and how many candidates the ACL removed."""
        entry = self._entry(tenant_id, source)
        if not entry.chunks:
            return [], 0
        visible = [i for i, chunk in enumerate(entry.chunks) if allowed(chunk.acl)]
        hidden = len(entry.chunks) - len(visible)
        if not visible:
            return [], hidden
        dense = cosine(query_vector, entry.matrix)
        sparse = entry.bm25.scores(query_text)
        dense_rank = sorted(visible, key=lambda i: -float(dense[i]))[:CANDIDATES]
        sparse_rank = [i for i in sorted(visible, key=lambda i: -sparse[i]) if sparse[i] > 0][
            :CANDIDATES
        ]
        fused = rrf([dense_rank, sparse_rank])
        best = sorted(fused, key=lambda i: -fused[i])[:limit]
        return [
            ScoredChunk(entry.chunks[i], fused[i], float(dense[i]), float(sparse[i])) for i in best
        ], hidden

"""Per-tenant retrieval (FR-05): connectors, chunking, embeddings, ACL-aware hybrid search."""

from mesh_retrieval.connectors import Connector, DirectoryConnector, InlineConnector
from mesh_retrieval.embedding import Embedder, HashingEmbedder
from mesh_retrieval.service import (
    PUBLIC_NAMESPACE,
    Hit,
    IngestReport,
    RetrievalService,
    SearchResult,
    can_read,
)
from mesh_retrieval.store import EVERYONE, Document, KnowledgeStore

__all__ = [
    "EVERYONE",
    "PUBLIC_NAMESPACE",
    "Connector",
    "DirectoryConnector",
    "Document",
    "Embedder",
    "HashingEmbedder",
    "Hit",
    "IngestReport",
    "InlineConnector",
    "KnowledgeStore",
    "RetrievalService",
    "SearchResult",
    "can_read",
]

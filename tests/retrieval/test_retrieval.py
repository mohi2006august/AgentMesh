from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mesh_core.audit import AuditStore
from mesh_core.clock import FakeClock
from mesh_core.db import Database
from mesh_core.tenancy import Principal
from mesh_retrieval import (
    PUBLIC_NAMESPACE,
    DirectoryConnector,
    Document,
    HashingEmbedder,
    InlineConnector,
    KnowledgeStore,
    RetrievalService,
)
from mesh_retrieval.bm25 import BM25
from mesh_retrieval.chunking import chunk_text
from mesh_retrieval.connectors import parse_front_matter
from mesh_retrieval.index import rrf
from mesh_sdk.spec import KnowledgeSpec

KNOWLEDGE = Path(__file__).resolve().parents[2] / "examples" / "knowledge"
DRIVE = KnowledgeSpec(source="tenant_drive", acl="inherit")
DOCS = KnowledgeSpec(source="product_docs")
SUPPORT_AGENT = Principal("bob", frozenset({"support"}))
ALICE = Principal("alice", frozenset())
NOBODY = Principal("eve", frozenset())


@pytest.fixture
def service(db: Database, clock: FakeClock) -> RetrievalService:
    svc = RetrievalService(
        KnowledgeStore(db, clock), HashingEmbedder(), AuditStore(db, "us", clock)
    )
    svc.sync(PUBLIC_NAMESPACE, "product_docs", DirectoryConnector(KNOWLEDGE / "product_docs"))
    for tenant in ("acme", "globex"):
        svc.sync(tenant, "tenant_drive", DirectoryConnector(KNOWLEDGE / "tenant_drive" / tenant))
    return svc


def _titles(result: object) -> list[str]:
    return [hit.title for hit in result.hits]  # type: ignore[attr-defined]


def test_public_docs_answer_policy_questions(service: RetrievalService) -> None:
    result = service.search("acme", NOBODY, "What is your refund policy?", [DOCS], top_k=2)
    assert result.hits[0].title == "Refund Policy"
    assert "30 days" in result.hits[0].text
    assert result.all_public and result.searched == ["product_docs"]


def test_acl_hides_documents_the_user_cannot_read(service: RetrievalService) -> None:
    query = "outage escalation on-call"
    assert (
        _titles(service.search("acme", SUPPORT_AGENT, query, [DRIVE]))[0] == "Acme Support Playbook"
    )
    hidden = service.search("acme", NOBODY, query, [DRIVE])
    assert "Acme Support Playbook" not in _titles(hidden)
    assert hidden.hidden_by_acl == 2  # playbook + pricing exceptions

    secret = "Northwind discount"
    assert "Acme Pricing Exceptions" in _titles(service.search("acme", ALICE, secret, [DRIVE]))
    assert "Acme Pricing Exceptions" not in _titles(
        service.search("acme", SUPPORT_AGENT, secret, [DRIVE])
    )


def test_tenant_level_acl_opens_the_source_to_every_user(service: RetrievalService) -> None:
    open_drive = KnowledgeSpec(source="tenant_drive", acl="tenant")
    assert "Acme Pricing Exceptions" in _titles(
        service.search("acme", NOBODY, "Northwind discount", [open_drive])
    )


def test_tenants_never_see_each_others_drive(service: RetrievalService) -> None:
    globex = service.search(
        "globex", SUPPORT_AGENT, "Acme outage escalation incidents channel", [DRIVE], top_k=10
    )
    assert all(hit.title.startswith("Globex") for hit in globex.hits)
    acme = service.search(
        "acme", SUPPORT_AGENT, "Springfield depot hardware returns", [DRIVE], top_k=10
    )
    assert all(hit.title.startswith("Acme") for hit in acme.hits)


def test_mixed_sources_are_fused_and_marked(service: RetrievalService) -> None:
    result = service.search(
        "acme", SUPPORT_AGENT, "support hours and refunds", [DRIVE, DOCS], top_k=6
    )
    assert {hit.public for hit in result.hits} == {True, False}
    assert not result.all_public


def test_missing_sources_are_reported(service: RetrievalService) -> None:
    result = service.search("acme", NOBODY, "anything", [KnowledgeSpec(source="crm_notes")])
    assert result.hits == [] and result.missing == ["crm_notes"]


def test_search_is_audited_as_data_access(service: RetrievalService, db: Database) -> None:
    service.search("acme", SUPPORT_AGENT, "outage escalation", [DRIVE], run_id="run_9")
    (event,) = AuditStore(db, "us").events("acme", action="data.retrieve")
    assert event.actor == "user:bob" and event.run_id == "run_9"
    assert "tenant_drive/support-playbook.md" in event.detail["documents"]


def test_reingest_is_incremental_and_prunes(service: RetrievalService) -> None:
    connector = InlineConnector([Document("a", "A", "alpha text"), Document("b", "B", "beta text")])
    assert service.sync("acme", "notes", connector).added == 2
    again = service.sync("acme", "notes", connector)
    assert (again.added, again.updated, again.unchanged) == (0, 0, 2)

    changed = InlineConnector([Document("a", "A", "alpha text revised")])
    report = service.sync("acme", "notes", changed)
    assert (report.updated, report.removed) == (1, 1)
    hits = service.search(
        "acme", NOBODY, "beta", [KnowledgeSpec(source="notes", acl="tenant")]
    ).hits
    assert all(hit.doc_id != "b" for hit in hits)


def test_chunking_keeps_headings_and_respects_size() -> None:
    text = "# Intro\n\nFirst paragraph.\n\n# Details\n\n" + ("Sentence number one is here. " * 80)
    chunks = chunk_text(text, title="Guide", max_chars=300)
    assert chunks[0].text.startswith("Guide / Intro\n")
    assert all(len(chunk.text) <= 300 + len("Guide / Details\n") for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_front_matter_parsing() -> None:
    meta, body = parse_front_matter("---\ntitle: T\nacl: [group:a]\n---\nBody")
    assert meta == {"title": "T", "acl": ["group:a"]} and body == "Body"
    assert parse_front_matter("No front matter") == ({}, "No front matter")


def test_embeddings_are_normalised_and_lexically_meaningful() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.embed(
        ["refund within 30 days", "refunds are possible within thirty days", "GPU kernels"]
    )
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
    assert embedder.embed([]).shape == (0, 384)


def test_bm25_and_rrf() -> None:
    scores = BM25(["refund policy days", "export dashboards", "refund refund"]).scores("refunds")
    assert scores[1] == 0 and scores[2] > scores[0] > 0
    assert rrf([[0, 1], [1, 2]]) == pytest.approx({0: 1 / 61, 1: 1 / 62 + 1 / 61, 2: 1 / 62})

from __future__ import annotations

import pytest

from mesh_core.audit import AuditStore, redact
from mesh_core.clock import FakeClock
from mesh_core.db import CrossTenant, Database
from mesh_core.errors import IsolationViolation
from mesh_core.secret_store import DbSecretStore
from tests.conftest import ROOT_KEY


def test_audit_records_are_tenant_scoped_and_ordered(db: Database, clock: FakeClock) -> None:
    audit = AuditStore(db, "us", clock)
    audit.record(
        "acme",
        actor="user:u1",
        action="tool.call",
        resource="zendesk/create_ticket",
        decision="allow",
    )
    clock.advance(1)
    audit.record(
        "acme", actor="user:u1", action="tool.call", resource="zendesk/refund", decision="deny"
    )
    audit.record(
        "globex", actor="user:u9", action="tool.call", resource="zendesk/refund", decision="allow"
    )

    acme = audit.events("acme")
    assert [event.resource for event in acme] == ["zendesk/refund", "zendesk/create_ticket"]
    assert {event.tenant_id for event in acme} == {"acme"}
    assert [
        event.decision for event in audit.events("acme", action="tool.call", since=clock.now())
    ] == ["deny"]
    assert len(audit.list_all()) == 3


def test_audit_chain_detects_tampering(db: Database, clock: FakeClock) -> None:
    audit = AuditStore(db, "us", clock)
    for index in range(3):
        audit.record("acme", actor="system", action="a", resource=f"r{index}", decision="allow")
    assert audit.verify_chain("acme")

    with db.platform_tx(CrossTenant.OPERATOR_CONSOLE) as tx:
        tx.execute("UPDATE audit_log SET decision = 'deny' WHERE resource = 'r1'")
    assert not audit.verify_chain("acme")


def test_audit_chain_detects_deletion(db: Database, clock: FakeClock) -> None:
    audit = AuditStore(db, "us", clock)
    for index in range(3):
        audit.record("acme", actor="system", action="a", resource=f"r{index}", decision="allow")
    with db.platform_tx(CrossTenant.OPERATOR_CONSOLE) as tx:
        tx.execute("DELETE FROM audit_log WHERE resource = 'r1'")
    assert not audit.verify_chain("acme")


def test_audit_detail_is_redacted(db: Database) -> None:
    audit = AuditStore(db, "us")
    audit.record(
        "acme", actor="user:u1", action="tool.call", resource="crm/update", decision="allow",
        detail={"arguments": {"api_key": "sk-live-123", "note": "x" * 900, "id": 7}},
    )  # fmt: skip
    arguments = audit.events("acme")[0].detail["arguments"]
    assert arguments["api_key"] == "[redacted]"
    assert arguments["id"] == 7
    assert len(arguments["note"]) < 600


def test_redact_handles_nested_structures() -> None:
    assert redact({"a": [{"access_token": "t", "ok": 1}]}) == {
        "a": [{"access_token": "[redacted]", "ok": 1}]
    }


def test_secrets_round_trip_and_stay_per_tenant(db: Database) -> None:
    store = DbSecretStore(db, ROOT_KEY)
    store.put("acme", "zendesk/token", "acme-token")
    store.put("globex", "zendesk/token", "globex-token")
    store.put("acme", "zendesk/token", "acme-token-rotated")

    assert store.get("acme", "zendesk/token") == "acme-token-rotated"
    assert store.get("globex", "zendesk/token") == "globex-token"
    assert store.get("acme", "missing") is None
    assert store.names("acme") == ["zendesk/token"]

    store.delete("acme", "zendesk/token")
    assert store.get("acme", "zendesk/token") is None
    assert store.get("globex", "zendesk/token") == "globex-token"


def test_secret_is_encrypted_at_rest(db: Database) -> None:
    DbSecretStore(db, ROOT_KEY).put("acme", "k", "plain-value")
    with db.tenant_tx("acme") as tx:
        row = tx.query_one("SELECT ciphertext FROM secrets WHERE tenant_id = :tenant_id")
    assert row is not None and "plain-value" not in row["ciphertext"]


def test_ciphertext_copied_to_another_tenant_does_not_decrypt(db: Database) -> None:
    store = DbSecretStore(db, ROOT_KEY)
    store.put("acme", "k", "acme-secret")
    with db.platform_tx(CrossTenant.OPERATOR_CONSOLE) as tx:
        tx.execute(
            "INSERT INTO secrets (tenant_id, name, ciphertext, updated_at) "
            "SELECT 'globex', name, ciphertext, updated_at FROM secrets WHERE tenant_id = 'acme'"
        )
    with pytest.raises(IsolationViolation):
        store.get("globex", "k")


def test_ciphertext_renamed_within_a_tenant_does_not_decrypt(db: Database) -> None:
    store = DbSecretStore(db, ROOT_KEY)
    store.put("acme", "low-privilege", "value")
    with db.tenant_tx("acme") as tx:
        tx.execute("UPDATE secrets SET name = 'high-privilege' WHERE tenant_id = :tenant_id")
    with pytest.raises(IsolationViolation):
        store.get("acme", "high-privilege")

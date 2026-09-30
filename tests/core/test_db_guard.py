"""The statement guard is the engine-independent half of tenant isolation (FR-03)."""

from __future__ import annotations

import pytest

from mesh_core.db import CrossTenant, Database
from mesh_core.errors import InvalidRequest, IsolationViolation
from mesh_core.schema import PLATFORM_TABLES, TENANT_TABLES, rls_statements


def _put_secret(db: Database, tenant_id: str, name: str, value: str) -> None:
    with db.tenant_tx(tenant_id) as tx:
        tx.execute(
            "INSERT INTO secrets (tenant_id, name, ciphertext, updated_at) "
            "VALUES (:tenant_id, :name, :value, 0)",
            {"name": name, "value": value},
        )


def test_tenant_tx_binds_its_own_tenant(db: Database) -> None:
    _put_secret(db, "acme", "k", "acme-value")
    _put_secret(db, "globex", "k", "globex-value")
    with db.tenant_tx("acme") as tx:
        rows = tx.query("SELECT ciphertext FROM secrets WHERE tenant_id = :tenant_id")
    assert [row["ciphertext"] for row in rows] == ["acme-value"]


def test_unscoped_select_is_rejected(db: Database) -> None:
    with db.tenant_tx("acme") as tx, pytest.raises(IsolationViolation):
        tx.query("SELECT * FROM secrets")


def test_scoping_with_a_literal_instead_of_the_bound_tenant_is_rejected(db: Database) -> None:
    with db.tenant_tx("acme") as tx, pytest.raises(IsolationViolation):
        tx.query("SELECT * FROM secrets WHERE tenant_id = 'globex'")


def test_caller_cannot_bind_another_tenant(db: Database) -> None:
    with db.tenant_tx("acme") as tx, pytest.raises(IsolationViolation):
        tx.query("SELECT * FROM secrets WHERE tenant_id = :tenant_id", {"tenant_id": "globex"})


def test_join_needs_a_predicate_per_tenant_table(db: Database) -> None:
    unscoped_join = (
        "SELECT s.name FROM secrets s JOIN api_keys k ON k.tenant_id = s.tenant_id "
        "WHERE s.tenant_id = :tenant_id"
    )
    scoped_join = unscoped_join + " AND k.tenant_id = :tenant_id"
    with db.tenant_tx("acme") as tx:
        with pytest.raises(IsolationViolation):
            tx.query(unscoped_join)
        assert tx.query(scoped_join) == []


def test_insert_must_bind_tenant(db: Database) -> None:
    with db.tenant_tx("acme") as tx, pytest.raises(IsolationViolation):
        tx.execute(
            "INSERT INTO secrets (tenant_id, name, ciphertext, updated_at) VALUES ('globex', 'k', 'v', 0)"
        )


def test_update_and_delete_must_be_scoped(db: Database) -> None:
    _put_secret(db, "globex", "k", "v")
    with db.tenant_tx("acme") as tx:
        with pytest.raises(IsolationViolation):
            tx.execute("UPDATE secrets SET ciphertext = 'x'")
        with pytest.raises(IsolationViolation):
            tx.execute("DELETE FROM secrets")
        assert tx.execute("DELETE FROM secrets WHERE tenant_id = :tenant_id") == 0
    with db.tenant_tx("globex") as tx:
        assert len(tx.query("SELECT * FROM secrets WHERE tenant_id = :tenant_id")) == 1


def test_upsert_is_not_mistaken_for_an_update_of_another_table(db: Database) -> None:
    for value in ("v1", "v2"):
        with db.tenant_tx("acme") as tx:
            tx.execute(
                "INSERT INTO secrets (tenant_id, name, ciphertext, updated_at) VALUES (:tenant_id, 'k', :v, 0) "
                "ON CONFLICT (tenant_id, name) DO UPDATE SET ciphertext = excluded.ciphertext",
                {"v": value},
            )
    with db.tenant_tx("acme") as tx:
        row = tx.query_one("SELECT ciphertext FROM secrets WHERE tenant_id = :tenant_id")
    assert row == {"ciphertext": "v2"}


def test_tenant_tx_cannot_write_platform_tables(db: Database) -> None:
    with db.tenant_tx("acme") as tx, pytest.raises(IsolationViolation):
        tx.execute("DELETE FROM plans")


def test_tenant_tx_rejects_ddl(db: Database) -> None:
    with db.tenant_tx("acme") as tx, pytest.raises(IsolationViolation):
        tx.execute("DROP TABLE secrets")


def test_unclassified_table_is_rejected(db: Database) -> None:
    with db.tenant_tx("acme") as tx, pytest.raises(IsolationViolation):
        tx.query("SELECT * FROM sqlite_master WHERE tenant_id = :tenant_id")


def test_platform_tx_needs_a_reason_for_tenant_tables(db: Database) -> None:
    _put_secret(db, "acme", "k", "v")
    with db.platform_tx() as tx, pytest.raises(IsolationViolation):
        tx.query("SELECT * FROM secrets")
    with db.platform_tx(CrossTenant.OPERATOR_CONSOLE) as tx:
        assert len(tx.query("SELECT * FROM secrets")) == 1
    with db.platform_tx() as tx:
        assert tx.query("SELECT * FROM plans") == []


def test_failed_transaction_rolls_back(db: Database) -> None:
    with pytest.raises(RuntimeError), db.tenant_tx("acme") as tx:
        tx.execute(
            "INSERT INTO secrets (tenant_id, name, ciphertext, updated_at) VALUES (:tenant_id, 'k', 'v', 0)"
        )
        raise RuntimeError("boom")
    with db.tenant_tx("acme") as tx:
        assert tx.query("SELECT * FROM secrets WHERE tenant_id = :tenant_id") == []


@pytest.mark.parametrize("bad", ["", "A", "Acme", "a b", "a/../b", "x" * 64, "acme;drop"])
def test_invalid_tenant_ids_are_rejected(db: Database, bad: str) -> None:
    with pytest.raises(InvalidRequest), db.tenant_tx(bad):
        pass


def test_every_tenant_table_has_a_tenant_id_column(db: Database) -> None:
    assert not (TENANT_TABLES & PLATFORM_TABLES)
    with db._engine.transaction("platform", None) as execute:
        for table in TENANT_TABLES:
            columns = {row["name"] for row in execute(f"PRAGMA table_info({table})", {}).fetchall()}
            assert "tenant_id" in columns, table


def test_rls_covers_every_tenant_table() -> None:
    statements = " ".join(rls_statements())
    for table in TENANT_TABLES:
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in statements
        assert f"CREATE POLICY tenant_isolation ON {table}" in statements

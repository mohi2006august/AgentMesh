"""Append-only, tamper-evident audit trail (FR-02, SOC 2-ready logging).

Each tenant has its own hash chain: a record's hash covers the previous record's hash,
so deleting or editing a row is detectable with `verify_chain`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mesh_core.clock import Clock, SystemClock
from mesh_core.db import CrossTenant, Database, Row, dumps, filters, loads
from mesh_core.ids import content_hash, new_id
from mesh_core.tenancy import platform_scoped

GENESIS = "0" * 64
_SENSITIVE = ("secret", "token", "password", "api_key", "authorization", "credential")
_MAX_VALUE_CHARS = 500


def redact(value: Any) -> Any:
    """Make tool arguments safe to persist: drop credential-looking keys, clip long values."""
    if isinstance(value, dict):
        return {
            key: "[redacted]"
            if any(marker in str(key).lower() for marker in _SENSITIVE)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value[:50]]
    if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
        return value[:_MAX_VALUE_CHARS] + f"... [{len(value) - _MAX_VALUE_CHARS} more chars]"
    return value


@dataclass(frozen=True)
class AuditEvent:
    audit_id: str
    tenant_id: str
    seq: int
    ts: float
    actor: str
    action: str
    resource: str
    decision: str
    run_id: str | None
    region: str
    detail: dict[str, Any]
    hash: str

    @classmethod
    def from_row(cls, row: Row) -> AuditEvent:
        return cls(
            audit_id=row["audit_id"],
            tenant_id=row["tenant_id"],
            seq=row["seq"],
            ts=row["ts"],
            actor=row["actor"],
            action=row["action"],
            resource=row["resource"],
            decision=row["decision"],
            run_id=row["run_id"],
            region=row["region"],
            detail=loads(row["detail"], {}),
            hash=row["hash"],
        )


def _record_hash(prev_hash: str, row: dict[str, Any]) -> str:
    return content_hash(
        prev_hash, row["tenant_id"], row["seq"], repr(row["ts"]), row["actor"], row["action"],
        row["resource"], row["decision"], row["run_id"] or "", row["region"], row["detail"],
    )  # fmt: skip


class AuditStore:
    def __init__(self, db: Database, region: str, clock: Clock | None = None) -> None:
        self._db = db
        self._region = region
        self._clock = clock or SystemClock()

    def record(
        self,
        tenant_id: str,
        *,
        actor: str,
        action: str,
        resource: str,
        decision: str,
        run_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> str:
        audit_id = new_id("aud")
        with self._db.tenant_tx(tenant_id) as tx:
            if self._db.dialect == "postgres":
                # Serialise appends per tenant so the chain never forks.
                tx.query("SELECT pg_advisory_xact_lock(hashtext(:tenant_id))")
            last = tx.query_one(
                "SELECT seq, hash FROM audit_log WHERE tenant_id = :tenant_id ORDER BY seq DESC LIMIT 1"
            )
            row: dict[str, Any] = {
                "audit_id": audit_id,
                "tenant_id": tenant_id,
                "seq": (last["seq"] + 1) if last else 1,
                "ts": self._clock.now(),
                "actor": actor,
                "action": action,
                "resource": resource,
                "decision": decision,
                "run_id": run_id,
                "region": self._region,
                "detail": dumps(redact(detail or {})),
                "prev_hash": last["hash"] if last else GENESIS,
            }
            row["hash"] = _record_hash(row["prev_hash"], row)
            tx.execute(
                """
                INSERT INTO audit_log
                  (audit_id, tenant_id, seq, ts, actor, action, resource, decision, run_id,
                   region, detail, prev_hash, hash)
                VALUES
                  (:audit_id, :tenant_id, :seq, :ts, :actor, :action, :resource, :decision, :run_id,
                   :region, :detail, :prev_hash, :hash)
                """,
                row,
            )
        return audit_id

    def events(
        self,
        tenant_id: str,
        *,
        action: str | None = None,
        actor: str | None = None,
        run_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        where, params = filters(
            ("action = :action", "action", action),
            ("actor = :actor", "actor", actor),
            ("run_id = :run_id", "run_id", run_id),
            ("ts >= :since", "since", since),
        )
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                f"SELECT * FROM audit_log WHERE tenant_id = :tenant_id{where} "
                "ORDER BY seq DESC LIMIT :limit",
                {**params, "limit": limit},
            )
        return [AuditEvent.from_row(row) for row in rows]

    def verify_chain(self, tenant_id: str) -> bool:
        """Recompute the tenant's hash chain; False means a record was altered or removed."""
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query("SELECT * FROM audit_log WHERE tenant_id = :tenant_id ORDER BY seq ASC")
        prev_hash, expected_seq = GENESIS, 1
        for row in rows:
            if row["seq"] != expected_seq or row["prev_hash"] != prev_hash:
                return False
            if _record_hash(prev_hash, row) != row["hash"]:
                return False
            prev_hash, expected_seq = row["hash"], expected_seq + 1
        return True

    @platform_scoped("operator console: audit view across tenants")
    def list_all(self, *, since: float | None = None, limit: int = 200) -> list[AuditEvent]:
        where, params = filters(("ts >= :since", "since", since))
        with self._db.platform_tx(CrossTenant.OPERATOR_CONSOLE) as tx:
            rows = tx.query(
                f"SELECT * FROM audit_log WHERE 1 = 1{where} ORDER BY ts DESC LIMIT :limit",
                {**params, "limit": limit},
            )
        return [AuditEvent.from_row(row) for row in rows]

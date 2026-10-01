"""Human approval for sensitive tool calls (FR-01 human-approval steps, FR-02 policy).

An approval is bound to the exact call it was requested for: run, step and a hash of the
arguments. The gateway re-checks that binding before executing, so approving one refund
cannot be replayed to authorise a different amount.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from mesh_core.clock import Clock, SystemClock
from mesh_core.db import Database, Row, dumps, loads
from mesh_core.errors import Conflict, NotFound
from mesh_core.ids import content_hash, new_id

Status = Literal["pending", "approved", "denied", "expired"]
DEFAULT_TTL_S = 24 * 3600


def arguments_hash(arguments: dict[str, Any]) -> str:
    return content_hash(json.dumps(arguments, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class Approval:
    approval_id: str
    tenant_id: str
    run_id: str
    step_key: str
    agent: str
    user_id: str
    server: str
    tool: str
    arguments: dict[str, Any]
    arguments_hash: str
    status: Status
    requested_at: float
    decided_at: float | None
    decided_by: str | None
    reason: str | None

    @classmethod
    def from_row(cls, row: Row) -> Approval:
        return cls(**{**row, "arguments": loads(row["arguments"], {})})

    def matches(self, server: str, tool: str, arguments: dict[str, Any]) -> bool:
        return (self.server, self.tool, self.arguments_hash) == (
            server,
            tool,
            arguments_hash(arguments),
        )


class ApprovalStore:
    def __init__(
        self, db: Database, clock: Clock | None = None, ttl_s: float = DEFAULT_TTL_S
    ) -> None:
        self._db = db
        self._clock = clock or SystemClock()
        self._ttl = ttl_s

    def request(
        self,
        tenant_id: str,
        *,
        run_id: str,
        step_key: str,
        agent: str,
        user_id: str,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> Approval:
        """Idempotent per (run, step): a replayed step gets the same approval back."""
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                """
                INSERT INTO approvals
                  (approval_id, tenant_id, run_id, step_key, agent, user_id, server, tool,
                   arguments, arguments_hash, status, requested_at)
                VALUES
                  (:approval_id, :tenant_id, :run_id, :step_key, :agent, :user_id, :server, :tool,
                   :arguments, :arguments_hash, 'pending', :now)
                ON CONFLICT (tenant_id, run_id, step_key) DO NOTHING
                """,
                {
                    "approval_id": new_id("apr"),
                    "run_id": run_id,
                    "step_key": step_key,
                    "agent": agent,
                    "user_id": user_id,
                    "server": server,
                    "tool": tool,
                    "arguments": dumps(arguments),
                    "arguments_hash": arguments_hash(arguments),
                    "now": self._clock.now(),
                },
            )
            row = tx.query_one(
                "SELECT * FROM approvals WHERE tenant_id = :tenant_id AND run_id = :run_id AND step_key = :step_key",
                {"run_id": run_id, "step_key": step_key},
            )
        assert row is not None
        return self._expire(tenant_id, Approval.from_row(row))

    def get(self, tenant_id: str, approval_id: str) -> Approval:
        with self._db.tenant_tx(tenant_id) as tx:
            row = tx.query_one(
                "SELECT * FROM approvals WHERE tenant_id = :tenant_id AND approval_id = :approval_id",
                {"approval_id": approval_id},
            )
        if row is None:
            raise NotFound("approval not found", approval_id=approval_id)
        return self._expire(tenant_id, Approval.from_row(row))

    def decide(
        self, tenant_id: str, approval_id: str, *, approve: bool, decided_by: str, reason: str = ""
    ) -> Approval:
        current = self.get(tenant_id, approval_id)
        if current.status != "pending":
            raise Conflict(
                "approval was already decided", approval_id=approval_id, status=current.status
            )
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                """
                UPDATE approvals SET status = :status, decided_at = :now, decided_by = :decided_by, reason = :reason
                WHERE tenant_id = :tenant_id AND approval_id = :approval_id AND status = 'pending'
                """,
                {
                    "status": "approved" if approve else "denied",
                    "now": self._clock.now(),
                    "decided_by": decided_by,
                    "reason": reason,
                    "approval_id": approval_id,
                },
            )
        return self.get(tenant_id, approval_id)

    def pending(self, tenant_id: str, *, run_id: str | None = None) -> list[Approval]:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                "SELECT * FROM approvals WHERE tenant_id = :tenant_id AND status = 'pending' ORDER BY requested_at",
            )
        approvals = [self._expire(tenant_id, Approval.from_row(row)) for row in rows]
        return [
            a for a in approvals if a.status == "pending" and (run_id is None or a.run_id == run_id)
        ]

    def _expire(self, tenant_id: str, approval: Approval) -> Approval:
        if approval.status != "pending" or self._clock.now() - approval.requested_at < self._ttl:
            return approval
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                "UPDATE approvals SET status = 'expired', decided_at = :now "
                "WHERE tenant_id = :tenant_id AND approval_id = :approval_id AND status = 'pending'",
                {"now": self._clock.now(), "approval_id": approval.approval_id},
            )
        return Approval(
            **{**approval.__dict__, "status": "expired", "decided_at": self._clock.now()}
        )

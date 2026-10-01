"""Tool-call policy (FR-02): allow, approve (human in the loop) or deny.

Three inputs, combined so that a tenant can only make things stricter:

1. the agent's `policy` map in agent.yaml (`"*"` sets the agent's default),
2. the tool's nature when the agent says nothing: read-only tools are allowed, anything
   with side effects needs approval,
3. the tenant admin's overrides for that server (`"*"` covers every tool).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from mesh_core.db import Database
from mesh_core.errors import InvalidRequest

Action = Literal["allow", "approve", "deny"]
ACTIONS: tuple[Action, ...] = ("allow", "approve", "deny")
_RANK = {"allow": 0, "approve": 1, "deny": 2}
WILDCARD = "*"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


def _lookup(policy: Mapping[str, str], tool: str) -> str | None:
    return policy.get(tool) or policy.get(WILDCARD)


def decide(
    tool: str,
    *,
    read_only: bool,
    agent_policy: Mapping[str, str],
    tenant_policy: Mapping[str, str],
) -> Decision:
    configured = _lookup(agent_policy, tool)
    if configured is not None:
        base: Action = configured  # type: ignore[assignment]
        base_reason = "agent policy"
    elif read_only:
        base, base_reason = "allow", "read-only tool"
    else:
        base, base_reason = "approve", "tool has side effects and no policy allows it"

    override = _lookup(tenant_policy, tool)
    if override is not None and _RANK[override] > _RANK[base]:
        return Decision(override, "tenant policy")  # type: ignore[arg-type]
    return Decision(base, base_reason)


class PolicyStore:
    """Tenant admins' per-server overrides."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def set(self, tenant_id: str, server: str, tool: str, action: str) -> None:
        if action not in ACTIONS:
            raise InvalidRequest("action must be allow, approve or deny", action=action)
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                """
                INSERT INTO tool_policies (tenant_id, server, tool, action, updated_at)
                VALUES (:tenant_id, :server, :tool, :action, :now)
                ON CONFLICT (tenant_id, server, tool) DO UPDATE SET action = excluded.action, updated_at = excluded.updated_at
                """,
                {"server": server, "tool": tool, "action": action, "now": time.time()},
            )

    def clear(self, tenant_id: str, server: str, tool: str) -> None:
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                "DELETE FROM tool_policies WHERE tenant_id = :tenant_id AND server = :server AND tool = :tool",
                {"server": server, "tool": tool},
            )

    def for_server(self, tenant_id: str, server: str) -> dict[str, str]:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                "SELECT tool, action FROM tool_policies WHERE tenant_id = :tenant_id AND server = :server",
                {"server": server},
            )
        return {row["tool"]: row["action"] for row in rows}

    def all(self, tenant_id: str) -> list[dict[str, str]]:
        with self._db.tenant_tx(tenant_id) as tx:
            return tx.query(
                "SELECT server, tool, action FROM tool_policies WHERE tenant_id = :tenant_id ORDER BY server, tool"
            )

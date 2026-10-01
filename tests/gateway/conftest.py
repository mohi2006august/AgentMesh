from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from mesh_core.audit import AuditStore
from mesh_core.clock import FakeClock
from mesh_core.db import Database
from mesh_core.secret_store import DbSecretStore
from mesh_gateway import (
    ApprovalStore,
    CallContext,
    ConnectionStore,
    InProcessServers,
    McpConnector,
    McpGateway,
    OAuthFlow,
    PolicyStore,
    ServerRegistry,
)
from mesh_gateway.examples import dev_servers, dev_token
from mesh_metering import MemoryMeterSink, Meter, RunScope, SqlMeterSink
from mesh_sdk.spec import ToolSpec
from tests.conftest import ROOT_KEY

SUPPORT_TOOLS = (
    ToolSpec(
        mcp="zendesk",
        policy={"create_ticket": "allow", "search_tickets": "allow", "refund": "approve"},
    ),
    ToolSpec(mcp="stripe-readonly"),
)


@dataclass
class GatewayEnv:
    gateway: McpGateway
    audit: AuditStore
    sink: MemoryMeterSink
    secrets: DbSecretStore
    clock: FakeClock

    def connect(self, tenant_id: str, *servers: str) -> None:
        for server in servers:
            self.gateway.connections.connect_token(
                tenant_id,
                self.gateway.registry.get(server),
                dev_token(server, tenant_id),
                created_by="admin",
            )


def call_context(
    step_key: str = "tool:1", run_id: str = "run_1", user_id: str = "u1"
) -> CallContext:
    return CallContext(
        run_id=run_id, step_key=step_key, agent="support-agent", user_id=user_id,
        scope=RunScope(agent="support-agent", run_id=run_id),
    )  # fmt: skip


@pytest.fixture
async def env(db: Database, clock: FakeClock) -> AsyncIterator[GatewayEnv]:
    secrets = DbSecretStore(db, ROOT_KEY)
    registry = ServerRegistry(db)
    inprocess = InProcessServers()
    for definition, server in dev_servers():
        registry.upsert(definition)
        inprocess.register(definition.name, server)
    connections = ConnectionStore(db, secrets, clock)
    sink = MemoryMeterSink()
    audit = AuditStore(db, "us", clock)
    gateway = McpGateway(
        registry=registry,
        connections=connections,
        oauth=OAuthFlow(db, connections, ROOT_KEY, clock=clock),
        policies=PolicyStore(db),
        approvals=ApprovalStore(db, clock),
        audit=audit,
        meter=Meter([SqlMeterSink(db), sink], clock),
        connector=McpConnector(inprocess),
        clock=clock,
    )
    yield GatewayEnv(gateway, audit, sink, secrets, clock)
    await inprocess.aclose()

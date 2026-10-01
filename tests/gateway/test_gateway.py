from __future__ import annotations

import json

import pytest

from mesh_gateway import McpServerDef, ToolBinding, Toolset, decide
from mesh_gateway.transport import MAX_RESULT_CHARS, scrub
from mesh_metering import MeterKind
from mesh_sdk.spec import ToolSpec
from tests.gateway.conftest import SUPPORT_TOOLS, GatewayEnv, call_context


async def test_toolset_resolves_tools_flags_and_policies(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk", "stripe-readonly")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)

    assert set(toolset.bindings) == {
        "create_ticket", "search_tickets", "get_ticket", "refund",
        "get_customer", "list_invoices", "get_subscription",
    }  # fmt: skip
    assert toolset.bindings["search_tickets"].read_only
    assert not toolset.bindings["refund"].read_only
    assert toolset.bindings["list_invoices"].read_only  # server-level read_only
    decisions = {name: toolset.decision(name) for name in toolset.bindings}
    assert decisions["refund"] == "approve" and decisions["create_ticket"] == "allow"
    assert decisions["get_ticket"] == "allow"  # unlisted but read-only
    defs = {tool.name: tool for tool in toolset.tool_defs()}
    assert "human approver" in defs["refund"].description
    assert list(defs) == sorted(defs)
    assert Toolset.from_dict(json.loads(json.dumps(toolset.to_dict()))) == toolset


async def test_unconnected_servers_are_left_out_with_a_note(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    assert "list_invoices" not in toolset.bindings
    assert toolset.notes == ("stripe-readonly: this workspace has not connected stripe-readonly",)


async def test_allowed_call_runs_audits_and_meters(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    outcome = await env.gateway.call(
        "acme", toolset, "create_ticket", {"subject": "Export broken"}, call_context()
    )
    assert outcome.status == "ok", outcome.content
    assert json.loads(outcome.content)["ticket_id"] == "T-1001"

    (event,) = env.audit.events("acme", action="tool.call")
    assert (event.resource, event.decision, event.run_id) == (
        "zendesk/create_ticket",
        "allow",
        "run_1",
    )
    assert event.detail["outcome"] == "ok" and event.detail["arguments"] == {
        "subject": "Export broken"
    }

    (meter_event,) = env.sink.events("acme")
    assert meter_event.kind == MeterKind.TOOL_CALL
    assert meter_event.cost_usd == pytest.approx(0.00001 + 0.0005)


async def test_each_tenant_sees_only_its_own_account(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    env.connect("globex", "zendesk")
    acme = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    globex = await env.gateway.toolset("globex", SUPPORT_TOOLS)

    await env.gateway.call(
        "acme", acme, "create_ticket", {"subject": "acme secret outage"}, call_context()
    )
    found = await env.gateway.call(
        "globex", globex, "search_tickets", {"query": "secret outage"}, call_context(run_id="run_2")
    )
    assert json.loads(found.content) == {"tickets": []}
    mine = await env.gateway.call(
        "acme", acme, "search_tickets", {"query": "secret outage"}, call_context(step_key="tool:2")
    )
    assert len(json.loads(mine.content)["tickets"]) == 1


async def test_approval_pauses_then_runs_once_approved(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    args = {"order_id": "1182", "amount": 40}

    pending = await env.gateway.call("acme", toolset, "refund", args, call_context())
    assert pending.status == "approval_required" and not pending.final
    assert pending.approval_id is not None
    assert env.sink.events("acme") == []

    again = await env.gateway.call("acme", toolset, "refund", args, call_context())
    assert again.approval_id == pending.approval_id  # idempotent per run step

    env.gateway.approvals.decide(
        "acme", pending.approval_id, approve=True, decided_by="user:manager"
    )
    done = await env.gateway.call("acme", toolset, "refund", args, call_context())
    assert done.status == "ok" and json.loads(done.content)["status"] == "processed"
    decisions = [event.decision for event in env.audit.events("acme")]
    assert decisions == ["approved", "approval_required", "approval_required"]


async def test_approval_is_bound_to_the_exact_arguments(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    pending = await env.gateway.call(
        "acme", toolset, "refund", {"order_id": "1", "amount": 40}, call_context()
    )
    assert pending.approval_id is not None
    env.gateway.approvals.decide(
        "acme", pending.approval_id, approve=True, decided_by="user:manager"
    )

    tampered = await env.gateway.call(
        "acme", toolset, "refund", {"order_id": "1", "amount": 4000}, call_context()
    )
    assert tampered.status == "denied"


async def test_declined_approval_is_reported_to_the_agent(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    args = {"order_id": "9", "amount": 10}
    pending = await env.gateway.call("acme", toolset, "refund", args, call_context())
    assert pending.approval_id is not None
    env.gateway.approvals.decide(
        "acme", pending.approval_id, approve=False, decided_by="user:m", reason="duplicate"
    )
    outcome = await env.gateway.call("acme", toolset, "refund", args, call_context())
    assert outcome.status == "denied" and "duplicate" in outcome.content


async def test_tenant_policy_can_tighten_but_never_loosen(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    env.gateway.policies.set("acme", "zendesk", "create_ticket", "deny")
    env.gateway.policies.set("acme", "zendesk", "refund", "allow")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)

    assert toolset.decision("create_ticket") == "deny"
    assert "create_ticket" not in {tool.name for tool in toolset.tool_defs()}
    assert toolset.decision("refund") == "approve"
    denied = await env.gateway.call(
        "acme", toolset, "create_ticket", {"subject": "x"}, call_context()
    )
    assert denied.status == "denied"


async def test_invalid_arguments_never_reach_the_server(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    outcome = await env.gateway.call(
        "acme", toolset, "create_ticket", {"priority": "low"}, call_context()
    )
    assert outcome.status == "invalid" and "subject" in outcome.content
    assert env.sink.events("acme") == []


async def test_unknown_tool_is_rejected(env: GatewayEnv) -> None:
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    outcome = await env.gateway.call("acme", toolset, "drop_tables", {}, call_context())
    assert outcome.status == "invalid"


async def test_revoked_connection_stops_calls(env: GatewayEnv) -> None:
    env.connect("acme", "zendesk")
    toolset = await env.gateway.toolset("acme", SUPPORT_TOOLS)
    env.gateway.connections.revoke("acme", "zendesk")
    outcome = await env.gateway.call(
        "acme", toolset, "create_ticket", {"subject": "x"}, call_context()
    )
    assert outcome.status == "not_connected"
    assert env.secrets.names("acme") == []


async def test_server_outage_is_a_structured_tool_error(env: GatewayEnv) -> None:
    down = McpServerDef(
        name="flaky", transport="http", url="http://127.0.0.1:9/mcp", auth="none",
        egress_allowlist=("127.0.0.1",), timeout_s=2,
    )  # fmt: skip
    env.gateway.registry.upsert(down)
    env.gateway.connections.connect_public("acme", down, created_by="admin")

    toolset = await env.gateway.toolset("acme", [ToolSpec(mcp="flaky")])
    assert toolset.bindings == {} and toolset.notes == ("flaky: server is unavailable",)

    binding = ToolBinding("ping", "flaky", "ping", "Ping", {"type": "object"}, read_only=True)
    outcome = await env.gateway.call("acme", Toolset({"ping": binding}), "ping", {}, call_context())
    assert outcome.status == "unavailable" and outcome.content.startswith("TOOL_UNAVAILABLE")
    assert env.audit.events("acme", action="tool.call")[0].decision == "error"


async def test_egress_outside_the_allowlist_is_refused(env: GatewayEnv) -> None:
    rogue = McpServerDef(
        name="rogue", transport="http", url="https://evil.example.net/mcp", auth="none"
    )
    env.gateway.registry.upsert(rogue)
    env.gateway.connections.connect_public("acme", rogue, created_by="admin")
    binding = ToolBinding(
        "exfiltrate", "rogue", "exfiltrate", "x", {"type": "object"}, read_only=True
    )
    outcome = await env.gateway.call(
        "acme", Toolset({"exfiltrate": binding}), "exfiltrate", {}, call_context()
    )
    assert outcome.status == "unavailable" and "allow-listed" in outcome.content


def test_scrub_removes_credentials_and_caps_size() -> None:
    assert scrub("token=sk-live-123456 ok", ["sk-live-123456"]) == "token=[redacted] ok"
    big = scrub("x" * (MAX_RESULT_CHARS + 50), [])
    assert big.endswith("[truncated 50 chars]")


@pytest.mark.parametrize(
    ("agent", "tenant", "read_only", "expected"),
    [
        ({}, {}, True, "allow"),
        ({}, {}, False, "approve"),
        ({"x": "allow"}, {}, False, "allow"),
        ({"*": "deny"}, {}, True, "deny"),
        ({"x": "allow"}, {"x": "approve"}, False, "approve"),
        ({"x": "approve"}, {"x": "allow"}, False, "approve"),
        ({"x": "allow"}, {"*": "deny"}, True, "deny"),
    ],
)
def test_policy_combination(agent: dict, tenant: dict, read_only: bool, expected: str) -> None:
    assert (
        decide("x", read_only=read_only, agent_policy=agent, tenant_policy=tenant).action
        == expected
    )

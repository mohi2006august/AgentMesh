"""The MCP gateway (FR-02): every tool call an agent makes passes through `McpGateway.call`.

For each call, in order:

1. resolve the exposed tool name to (server, tool),
2. validate the arguments against the tool's JSON schema,
3. decide policy: allow, approve or deny (tenants can only tighten),
4. for approval, check the approval bound to this run step and these exact arguments,
5. fetch the tenant's credential and call the server with it,
6. scrub credentials from the result and cap its size,
7. audit the decision and outcome, and meter executed calls.

Server outages come back as a structured tool error so the agent can adapt instead of
failing the run.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import jsonschema

from mesh_core.audit import AuditStore
from mesh_core.clock import Clock, SystemClock
from mesh_core.errors import MeshError, NotFound
from mesh_core.log import get_logger
from mesh_gateway.approvals import ApprovalStore
from mesh_gateway.connections import ConnectionStore, OAuthFlow
from mesh_gateway.policy import PolicyStore, decide
from mesh_gateway.registry import McpServerDef, ServerRegistry
from mesh_gateway.transport import McpConnector, ServerUnavailable, scrub
from mesh_inference.types import ToolDef
from mesh_metering import Meter, RunScope
from mesh_sdk.spec import ToolSpec

log = get_logger(__name__)

OutcomeStatus = Literal[
    "ok", "error", "denied", "approval_required", "unavailable", "invalid", "not_connected"
]
APPROVAL_NOTE = " Calls to this tool wait for a human approver before they run."


@dataclass(frozen=True)
class ToolBinding:
    exposed_name: str
    server: str
    tool: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    policy: dict[str, str] = field(default_factory=dict)

    def tool_def(self, needs_approval: bool) -> ToolDef:
        description = self.description + (APPROVAL_NOTE if needs_approval else "")
        return ToolDef(self.exposed_name, description, self.input_schema, self.read_only)


@dataclass(frozen=True)
class Toolset:
    """The tools one agent may use for one tenant, resolved at the start of a run."""

    bindings: dict[str, ToolBinding]
    tenant_policies: dict[str, dict[str, str]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def decision(self, name: str) -> str:
        binding = self.bindings[name]
        return decide(
            binding.tool,
            read_only=binding.read_only,
            agent_policy=binding.policy,
            tenant_policy=self.tenant_policies.get(binding.server, {}),
        ).action

    def tool_defs(self) -> tuple[ToolDef, ...]:
        return tuple(
            binding.tool_def(self.decision(name) == "approve")
            for name, binding in sorted(self.bindings.items())
            if self.decision(name) != "deny"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bindings": {name: asdict(binding) for name, binding in self.bindings.items()},
            "tenant_policies": self.tenant_policies,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Toolset:
        return cls(
            bindings={name: ToolBinding(**raw) for name, raw in data["bindings"].items()},
            tenant_policies=data.get("tenant_policies", {}),
            notes=tuple(data.get("notes", [])),
        )


@dataclass(frozen=True)
class CallContext:
    run_id: str
    step_key: str
    agent: str
    user_id: str
    scope: RunScope
    timeout_s: float = 30.0


@dataclass(frozen=True)
class ToolOutcome:
    status: OutcomeStatus
    content: str
    server: str = ""
    tool: str = ""
    approval_id: str | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    @property
    def is_error(self) -> bool:
        return self.status != "ok"

    @property
    def final(self) -> bool:
        """False only while a human decision is pending."""
        return self.status != "approval_required"


class McpGateway:
    def __init__(
        self,
        *,
        registry: ServerRegistry,
        connections: ConnectionStore,
        oauth: OAuthFlow,
        policies: PolicyStore,
        approvals: ApprovalStore,
        audit: AuditStore,
        meter: Meter,
        connector: McpConnector,
        clock: Clock | None = None,
        list_ttl_s: float = 300.0,
    ) -> None:
        self.registry = registry
        self.connections = connections
        self.oauth = oauth
        self.policies = policies
        self.approvals = approvals
        self._audit = audit
        self._meter = meter
        self.connector = connector
        self._clock = clock or SystemClock()
        self._list_ttl = list_ttl_s
        self._tool_cache: dict[tuple[str, str], tuple[float, list[Any]]] = {}

    # --- credentials ----------------------------------------------------------------

    async def _credential(
        self, tenant_id: str, server: McpServerDef
    ) -> tuple[str | None, str | None]:
        """(credential, problem). `problem` is set when the tenant cannot use the server."""
        if server.auth == "none":
            return None, None
        connection = self.connections.get(tenant_id, server.name)
        if connection is None or connection.status != "active":
            return None, f"this workspace has not connected {server.name}"
        if server.auth == "bearer":
            token = self.connections.secret(tenant_id, server.name, "token")
        else:
            token = await self.oauth.access_token(tenant_id, server)
        if not token:
            return None, f"the {server.name} connection needs to be re-authorised"
        return token, None

    # --- tool discovery ---------------------------------------------------------------

    async def _server_tools(
        self, tenant_id: str, server: McpServerDef, credential: str | None
    ) -> list[Any]:
        key = (tenant_id, server.name)
        cached = self._tool_cache.get(key)
        if cached and cached[0] > self._clock.now():
            return cached[1]
        tools = await self.connector.list_tools(server, credential)
        self._tool_cache[key] = (self._clock.now() + self._list_ttl, tools)
        return tools

    async def toolset(self, tenant_id: str, specs: Sequence[ToolSpec]) -> Toolset:
        found: list[tuple[ToolSpec, McpServerDef, Any]] = []
        notes: list[str] = []
        policies: dict[str, dict[str, str]] = {}
        for spec in specs:
            try:
                server = self.registry.get(spec.mcp)
            except NotFound:
                notes.append(f"{spec.mcp}: server is not registered")
                continue
            credential, problem = await self._credential(tenant_id, server)
            if problem:
                notes.append(f"{spec.mcp}: {problem}")
                continue
            try:
                tools = await self._server_tools(tenant_id, server, credential)
            except ServerUnavailable as exc:
                log.warning(
                    "gateway.list_failed",
                    tenant_id=tenant_id,
                    server=server.name,
                    reason=exc.detail.get("reason"),
                )
                notes.append(f"{spec.mcp}: server is unavailable")
                continue
            policies[server.name] = self.policies.for_server(tenant_id, server.name)
            found.extend((spec, server, tool) for tool in tools)

        counts = Counter(tool.name for _, _, tool in found)
        bindings: dict[str, ToolBinding] = {}
        for spec, server, tool in found:
            exposed = (
                tool.name
                if counts[tool.name] == 1
                else f"{server.name.replace('-', '_')}__{tool.name}"
            )
            annotations = tool.annotations
            read_only = server.read_only or bool(annotations and annotations.read_only_hint)
            bindings[exposed] = ToolBinding(
                exposed_name=exposed,
                server=server.name,
                tool=tool.name,
                description=tool.description or tool.name,
                input_schema=dict(tool.input_schema),
                read_only=read_only,
                policy=dict(spec.policy),
            )
        return Toolset(bindings, policies, tuple(notes))

    # --- calls ------------------------------------------------------------------------

    def _record(
        self,
        tenant_id: str,
        ctx: CallContext,
        binding: ToolBinding | None,
        name: str,
        decision: str,
        **detail: Any,
    ) -> None:
        self._audit.record(
            tenant_id,
            actor=f"user:{ctx.user_id}",
            action="tool.call",
            resource=f"{binding.server}/{binding.tool}" if binding else name,
            decision=decision,
            run_id=ctx.run_id,
            detail={"agent": ctx.agent, "step": ctx.step_key, **detail},
        )

    async def call(
        self,
        tenant_id: str,
        toolset: Toolset,
        name: str,
        arguments: dict[str, Any],
        ctx: CallContext,
    ) -> ToolOutcome:
        binding = toolset.bindings.get(name)
        if binding is None:
            self._record(tenant_id, ctx, None, name, "invalid", reason="unknown tool")
            return ToolOutcome("invalid", f"Unknown tool {name!r}.")

        def outcome(
            status: OutcomeStatus,
            content: str,
            *,
            approval_id: str | None = None,
            latency_ms: float = 0.0,
            cost_usd: float = 0.0,
        ) -> ToolOutcome:
            assert binding is not None
            return ToolOutcome(
                status, content, binding.server, binding.tool, approval_id, latency_ms, cost_usd
            )

        try:
            jsonschema.validate(arguments, binding.input_schema)
        except jsonschema.ValidationError as exc:
            self._record(
                tenant_id, ctx, binding, name, "invalid", arguments=arguments, reason=exc.message
            )
            return outcome("invalid", f"INVALID_ARGUMENTS: {exc.message}")

        action = toolset.decision(name)
        if action == "deny":
            self._record(tenant_id, ctx, binding, name, "deny", arguments=arguments)
            return outcome("denied", "Policy does not allow this tool call.")

        approval_id = None
        decision = "allow"
        if action == "approve":
            approval = self.approvals.request(
                tenant_id, run_id=ctx.run_id, step_key=ctx.step_key, agent=ctx.agent,
                user_id=ctx.user_id, server=binding.server, tool=binding.tool, arguments=arguments,
            )  # fmt: skip
            approval_id = approval.approval_id
            if approval.status == "pending":
                self._record(
                    tenant_id,
                    ctx,
                    binding,
                    name,
                    "approval_required",
                    arguments=arguments,
                    approval_id=approval_id,
                )
                return outcome(
                    "approval_required",
                    "Waiting for a human to approve this call.",
                    approval_id=approval_id,
                )
            if approval.status != "approved" or not approval.matches(
                binding.server, binding.tool, arguments
            ):
                self._record(
                    tenant_id,
                    ctx,
                    binding,
                    name,
                    "deny",
                    approval_id=approval_id,
                    status=approval.status,
                )
                reason = f" Reason: {approval.reason}" if approval.reason else ""
                return outcome(
                    "denied",
                    f"An approver declined this call ({approval.status}).{reason}",
                    approval_id=approval_id,
                )
            decision = "approved"

        server = self.registry.get(binding.server)
        credential, problem = await self._credential(tenant_id, server)
        if problem:
            self._record(tenant_id, ctx, binding, name, "error", reason=problem)
            return outcome("not_connected", f"Tool unavailable: {problem}.")

        started = time.perf_counter()
        try:
            text, tool_failed = await self.connector.call_tool(
                server, credential, binding.tool, arguments, min(ctx.timeout_s, server.timeout_s)
            )
        except ServerUnavailable as exc:
            latency = (time.perf_counter() - started) * 1000
            self._record(
                tenant_id,
                ctx,
                binding,
                name,
                "error",
                reason=exc.detail.get("reason"),
                latency_ms=round(latency, 2),
            )
            return outcome(
                "unavailable",
                f"TOOL_UNAVAILABLE: {binding.server} is not responding. Try again later or continue without it.",
                approval_id=approval_id, latency_ms=latency
            )  # fmt: skip
        except MeshError as exc:
            self._record(tenant_id, ctx, binding, name, "error", reason=exc.message)
            return outcome(
                "unavailable", f"TOOL_UNAVAILABLE: {exc.message}", approval_id=approval_id
            )
        latency = (time.perf_counter() - started) * 1000

        content = scrub(text, [credential] if credential else [])
        cost = self._meter.tool_call(
            tenant_id, ctx.scope, step_key=ctx.step_key, server=binding.server, tool=binding.tool,
            server_cost_usd=server.tool_cost_usd,
        )  # fmt: skip
        self._record(
            tenant_id, ctx, binding, name, decision, arguments=arguments, approval_id=approval_id,
            outcome="error" if tool_failed else "ok", latency_ms=round(latency, 2),
        )  # fmt: skip
        return outcome(
            "error" if tool_failed else "ok", content, approval_id=approval_id,
            latency_ms=latency, cost_usd=cost
        )  # fmt: skip

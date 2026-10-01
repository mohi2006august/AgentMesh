"""Opening MCP sessions to servers, with the tenant's credential injected (FR-02).

The credential is attached here, at the transport, and nowhere else: as an
`Authorization: Bearer` header for HTTP servers, or `MCP_ACCESS_TOKEN` in the environment
of a stdio server. The model never sees it.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from mesh_core.errors import ConfigError, MeshError
from mesh_core.log import get_logger
from mesh_gateway.registry import INPROCESS_DOMAIN, McpServerDef, check_egress

log = get_logger(__name__)

MAX_RESULT_CHARS = 20_000
# Environment a stdio server inherits. Nothing else from the platform's environment leaks in.
_STDIO_ENV = ("PATH", "SYSTEMROOT", "HOME", "USERPROFILE", "TEMP", "TMP", "LANG")


class ServerUnavailable(MeshError):
    """The MCP server could not be reached or failed mid-call."""

    code = "tool_server_unavailable"
    status = 503


class InProcessServers:
    """MCP servers hosted inside this process, spoken to over real Streamable HTTP.

    Their session managers run in one background task, started on first use, so the
    anyio task groups are entered and exited by the same task.
    """

    def __init__(self) -> None:
        self._apps: dict[str, Any] = {}
        self._servers: dict[str, MCPServer] = {}
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()

    def register(self, name: str, server: MCPServer) -> None:
        if self._task is not None:
            raise ConfigError("register in-process servers before first use", server=name)
        host = f"{name}.{INPROCESS_DOMAIN}"
        self._servers[name] = server
        self._apps[name] = server.streamable_http_app(
            stateless_http=True,
            json_response=True,
            transport_security=TransportSecuritySettings(allowed_hosts=[host]),
        )

    def __contains__(self, name: str) -> bool:
        return name in self._apps

    async def _serve(self) -> None:
        async with AsyncExitStack() as stack:
            for server in self._servers.values():
                await stack.enter_async_context(server.session_manager.run())
            self._ready.set()
            await self._stop.wait()

    async def transport(self, name: str) -> httpx2.ASGITransport:
        if name not in self._apps:
            raise ConfigError("in-process MCP server is not registered", server=name)
        if self._task is None:
            self._task = asyncio.create_task(self._serve())
        await self._ready.wait()
        return httpx2.ASGITransport(app=self._apps[name])

    async def aclose(self) -> None:
        if self._task is not None:
            self._stop.set()
            await self._task
            self._task = None


def result_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into the text a model can read."""
    parts: list[str] = []
    for item in result.content or []:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(item.text)
        elif kind == "resource" and getattr(item.resource, "text", None):
            parts.append(item.resource.text)
        else:
            parts.append(f"[{kind or 'unknown'} content omitted]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, default=str))
    return "\n".join(parts)


def scrub(text: str, secrets: list[str]) -> str:
    """Output filtering: never hand a credential back to the model, and cap the size."""
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "[redacted]")
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n[truncated {len(text) - MAX_RESULT_CHARS} chars]"
    return text


class McpConnector:
    def __init__(
        self,
        inprocess: InProcessServers | None = None,
        *,
        allow_inprocess: bool = True,
        platform_allowlist: tuple[str, ...] = (),
    ) -> None:
        self._inprocess = inprocess or InProcessServers()
        self._allow_inprocess = allow_inprocess
        self._allowlist = platform_allowlist

    @property
    def inprocess(self) -> InProcessServers:
        return self._inprocess

    @asynccontextmanager
    async def session(self, server: McpServerDef, credential: str | None) -> AsyncIterator[Client]:
        headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        async with AsyncExitStack() as stack:
            if server.transport == "stdio":
                env = {key: os.environ[key] for key in _STDIO_ENV if key in os.environ}
                if credential:
                    env["MCP_ACCESS_TOKEN"] = credential
                target: Any = StdioServerParameters(
                    command=server.command[0], args=list(server.command[1:]), env=env
                )
            else:
                if server.transport == "inprocess":
                    if not self._allow_inprocess:
                        raise ConfigError("in-process MCP servers are disabled", server=server.name)
                    transport: Any = await self._inprocess.transport(server.name)
                else:
                    check_egress(server, server.endpoint, self._allowlist)
                    transport = None
                http = httpx2.AsyncClient(
                    transport=transport, headers=headers, timeout=httpx2.Timeout(server.timeout_s)
                )
                await stack.enter_async_context(http)
                target = streamable_http_client(server.endpoint, http_client=http)
            client = await stack.enter_async_context(
                Client(target, read_timeout_seconds=server.timeout_s)
            )
            yield client

    async def list_tools(self, server: McpServerDef, credential: str | None) -> list[Any]:
        try:
            async with self.session(server, credential) as client:
                tools: list[Any] = []
                cursor: str | None = None
                while True:
                    page = await client.list_tools(cursor=cursor)
                    tools.extend(page.tools)
                    cursor = getattr(page, "next_cursor", None)
                    if not cursor:
                        return tools
        except MeshError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport failures arrive as many types (incl. ExceptionGroup)
            raise ServerUnavailable(
                "could not list tools", server=server.name, reason=_describe(exc)
            ) from exc

    async def call_tool(
        self,
        server: McpServerDef,
        credential: str | None,
        tool: str,
        arguments: dict[str, Any],
        timeout_s: float,
    ) -> tuple[str, bool]:
        """Returns (text, is_error). `is_error` means the tool ran and reported failure."""
        try:
            async with self.session(server, credential) as client:
                result = await asyncio.wait_for(
                    client.call_tool(tool, arguments, read_timeout_seconds=timeout_s), timeout_s
                )
        except MeshError:
            raise
        except Exception as exc:  # noqa: BLE001 - see list_tools
            raise ServerUnavailable(
                "tool call failed", server=server.name, reason=_describe(exc)
            ) from exc
        return result_text(result), bool(result.is_error)


def _describe(exc: BaseException) -> str:
    """Unwrap anyio exception groups so the log names the real cause."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"[:300]

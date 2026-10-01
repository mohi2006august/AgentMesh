"""Registry of MCP servers the platform can connect agents to (FR-02).

A server definition is platform-level: it says how to reach the server and how tenants
authenticate to it. Credentials are per tenant and live with the tenant's connection.

Transports:
- `http`: Streamable HTTP at `url` (production default).
- `stdio`: a local command; the tenant credential is passed in `MCP_ACCESS_TOKEN`.
- `inprocess`: an MCP server object in this process, reached over the same Streamable
  HTTP protocol through an in-memory transport. Development and tests only.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mesh_core.db import Database, dumps, loads
from mesh_core.errors import InvalidRequest, NotFound, PolicyDenied
from mesh_core.tenancy import platform_scoped

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
INPROCESS_DOMAIN = "mcp.local"


class OAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorize_url: str
    token_url: str
    client_id: str
    scopes: tuple[str, ...] = ()


class McpServerDef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    transport: Literal["http", "stdio", "inprocess"]
    url: str | None = None
    command: tuple[str, ...] = ()
    auth: Literal["none", "bearer", "oauth"] = "bearer"
    oauth: OAuthConfig | None = None
    # Hosts this server's URL and OAuth endpoints may use. `.example.com` matches subdomains.
    egress_allowlist: tuple[str, ...] = ()
    # Every tool on the server is read-only (lets unlisted tools default to `allow`).
    read_only: bool = False
    # Extra COGS per call (third-party API fees), added to the gateway's own unit cost.
    tool_cost_usd: float = Field(default=0.0, ge=0)
    timeout_s: float = Field(default=30.0, gt=0)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError("server name must be lowercase letters, digits and hyphens")
        return value

    @model_validator(mode="after")
    def _transport_fields(self) -> McpServerDef:
        if self.transport == "http" and not self.url:
            raise ValueError("http servers need a url")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio servers need a command")
        if self.auth == "oauth" and self.oauth is None:
            raise ValueError("oauth servers need an oauth block")
        return self

    @property
    def endpoint(self) -> str:
        if self.transport == "inprocess":
            return f"http://{self.name}.{INPROCESS_DOMAIN}/mcp"
        return self.url or ""


def host_allowed(url: str, allowlist: tuple[str, ...]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    for entry in allowlist:
        entry = entry.lower().removeprefix("*")
        if host == entry or (entry.startswith(".") and host.endswith(entry)):
            return True
    return False


def check_egress(server: McpServerDef, url: str, platform_allowlist: tuple[str, ...] = ()) -> None:
    """Refuse to contact a host the server definition (or platform) has not allow-listed."""
    if server.transport == "inprocess":
        return
    if not host_allowed(url, server.egress_allowlist + platform_allowlist):
        raise PolicyDenied("egress to this host is not allow-listed", server=server.name, url=url)


def parse_server(data: Any) -> McpServerDef:
    try:
        return McpServerDef.model_validate(data)
    except ValidationError as exc:
        errors = [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()]
        raise InvalidRequest("invalid MCP server definition", errors=errors) from exc


class ServerRegistry:
    def __init__(self, db: Database) -> None:
        self._db = db

    @platform_scoped("MCP server catalog is shared by all tenants")
    def upsert(self, server: McpServerDef) -> None:
        now = time.time()
        with self._db.platform_tx() as tx:
            tx.execute(
                """
                INSERT INTO mcp_servers (name, definition, created_at, updated_at)
                VALUES (:name, :definition, :now, :now)
                ON CONFLICT (name) DO UPDATE SET definition = excluded.definition, updated_at = excluded.updated_at
                """,
                {
                    "name": server.name,
                    "definition": dumps(server.model_dump(mode="json")),
                    "now": now,
                },
            )

    @platform_scoped("MCP server catalog is shared by all tenants")
    def get(self, name: str) -> McpServerDef:
        with self._db.platform_tx() as tx:
            row = tx.query_one(
                "SELECT definition FROM mcp_servers WHERE name = :name", {"name": name}
            )
        if row is None:
            raise NotFound("MCP server is not registered", server=name)
        return McpServerDef.model_validate(loads(row["definition"]))

    @platform_scoped("MCP server catalog is shared by all tenants")
    def all(self) -> list[McpServerDef]:
        with self._db.platform_tx() as tx:
            rows = tx.query("SELECT definition FROM mcp_servers ORDER BY name")
        return [McpServerDef.model_validate(loads(row["definition"])) for row in rows]

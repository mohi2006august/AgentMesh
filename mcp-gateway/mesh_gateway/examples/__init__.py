"""Mock third-party MCP servers for local development, demos and tests.

They behave like real multi-account SaaS APIs: every call must carry a bearer token, and
the token decides which account's data the call sees. A dev token looks like
`<prefix>-<account>-<random>`; `dev_token()` mints one.
"""

from __future__ import annotations

import secrets

from mcp.server.mcpserver import MCPServer

from mesh_gateway.examples.stripe_readonly import build_server as build_stripe
from mesh_gateway.examples.zendesk import build_server as build_zendesk
from mesh_gateway.registry import McpServerDef

TOKEN_PREFIX = {"zendesk": "zd", "stripe-readonly": "sk"}


def dev_token(server: str, account: str) -> str:
    return f"{TOKEN_PREFIX[server]}-{account}-{secrets.token_hex(8)}"


def dev_servers() -> list[tuple[McpServerDef, MCPServer]]:
    return [
        (
            McpServerDef(
                name="zendesk",
                description="Support tickets (mock Zendesk)",
                transport="inprocess",
                auth="bearer",
                tool_cost_usd=0.0005,
            ),
            build_zendesk(),
        ),
        (
            McpServerDef(
                name="stripe-readonly",
                description="Read-only billing lookups (mock Stripe)",
                transport="inprocess",
                auth="bearer",
                read_only=True,
            ),
            build_stripe(),
        ),
    ]

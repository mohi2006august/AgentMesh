"""Bearer-token account resolution shared by the mock servers."""

from __future__ import annotations

import os
import re

from mcp.server.mcpserver import Context

_TOKEN = re.compile(r"^(?P<prefix>[a-z]+)-(?P<account>[a-z0-9][a-z0-9-]*)-[0-9a-f]{8,}$")


class Unauthorized(Exception):
    pass


def account_for(ctx: Context, prefix: str) -> str:
    """The account a request acts for, from `Authorization: Bearer` (or MCP_ACCESS_TOKEN on stdio)."""
    headers = ctx.headers
    header = headers.get("authorization", "") if headers is not None else ""
    token = (
        header.removeprefix("Bearer ").strip() if header else os.environ.get("MCP_ACCESS_TOKEN", "")
    )
    match = _TOKEN.match(token)
    if not match or match.group("prefix") != prefix:
        raise Unauthorized("missing or invalid access token")
    return match.group("account")

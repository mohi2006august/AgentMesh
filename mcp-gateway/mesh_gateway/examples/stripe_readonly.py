"""Mock read-only Stripe MCP server: the calling account's customer, invoices and plan.

Run standalone over Streamable HTTP:  python -m mesh_gateway.examples.stripe_readonly --port 9102
"""

from __future__ import annotations

import argparse
import hashlib
import json

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from mesh_gateway.examples._auth import account_for

READ_ONLY = ToolAnnotations(read_only_hint=True)
PLANS = ("starter", "pro", "enterprise")


def _seed(account: str) -> int:
    return int(hashlib.sha256(account.encode()).hexdigest()[:8], 16)


def build_server() -> MCPServer:
    server = MCPServer(name="stripe-readonly", instructions="Read-only billing data.")

    def _account(ctx: Context) -> str:
        return account_for(ctx, "sk")

    @server.tool(annotations=READ_ONLY)
    def get_customer(ctx: Context) -> str:
        """The billing customer record for this workspace."""
        account = _account(ctx)
        return json.dumps(
            {
                "customer_id": f"cus_{account}",
                "name": account.replace("-", " ").title(),
                "currency": "usd",
            }
        )

    @server.tool(annotations=READ_ONLY)
    def list_invoices(ctx: Context, limit: int = 3) -> str:
        """Recent invoices for this workspace, newest first."""
        account = _account(ctx)
        seed = _seed(account)
        invoices = [
            {
                "invoice_id": f"in_{account}_{index}",
                "month": f"2026-{9 - index:02d}",
                "amount_usd": round(49 + (seed % 400) + index * 7.5, 2),
                "status": "paid" if index else "open",
            }
            for index in range(max(1, min(limit, 12)))
        ]
        return json.dumps({"invoices": invoices})

    @server.tool(annotations=READ_ONLY)
    def get_subscription(ctx: Context) -> str:
        """The plan this workspace is subscribed to."""
        account = _account(ctx)
        return json.dumps({"plan": PLANS[_seed(account) % len(PLANS)], "status": "active"})

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9102)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(build_server().streamable_http_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

"""Mock Zendesk MCP server: tickets per account, and refunds (a destructive action).

Run standalone over Streamable HTTP:  python -m mesh_gateway.examples.zendesk --port 9101
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from mesh_gateway.examples._auth import account_for

READ_ONLY = ToolAnnotations(read_only_hint=True)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True)


def build_server() -> MCPServer:
    server = MCPServer(name="zendesk", instructions="Customer support ticketing.")
    tickets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    refunds: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _account(ctx: Context) -> str:
        return account_for(ctx, "zd")

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def create_ticket(
        ctx: Context,
        subject: str,
        description: str = "",
        priority: Literal["low", "normal", "high", "urgent"] = "normal",
    ) -> str:
        """Open a support ticket for the customer. Use when the customer reports a problem
        that cannot be solved in the conversation."""
        account = _account(ctx)
        ticket = {
            "ticket_id": f"T-{1000 + len(tickets[account]) + 1}",
            "subject": subject,
            "description": description,
            "priority": priority,
            "status": "open",
        }
        tickets[account].append(ticket)
        return json.dumps(ticket)

    @server.tool(annotations=READ_ONLY)
    def search_tickets(ctx: Context, query: str) -> str:
        """Search this customer's existing tickets by words in the subject or description."""
        account = _account(ctx)
        words = [word for word in query.lower().split() if len(word) > 2]
        hits = [
            ticket
            for ticket in tickets[account]
            if any(
                word in (ticket["subject"] + " " + ticket["description"]).lower() for word in words
            )
        ]
        return json.dumps({"tickets": hits})

    @server.tool(annotations=READ_ONLY)
    def get_ticket(ctx: Context, ticket_id: str) -> str:
        """Look up one ticket by id, e.g. T-1001."""
        account = _account(ctx)
        for ticket in tickets[account]:
            if ticket["ticket_id"] == ticket_id:
                return json.dumps(ticket)
        return json.dumps({"error": "ticket not found", "ticket_id": ticket_id})

    @server.tool(annotations=DESTRUCTIVE)
    def refund(ctx: Context, order_id: str, amount: float, reason: str = "") -> str:
        """Refund an order. Moves money: only use when the customer asks for a refund."""
        account = _account(ctx)
        if amount <= 0:
            raise ValueError("amount must be positive")
        record = {
            "refund_id": f"R-{len(refunds[account]) + 1:04d}",
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "status": "processed",
        }
        refunds[account].append(record)
        return json.dumps(record)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9101)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(build_server().streamable_http_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

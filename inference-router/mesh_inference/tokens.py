"""Local token estimation for budgeting.

This is an estimate, deliberately on the high side, used where a network round trip is
not affordable (context packing sits inside the 100 ms step-overhead budget). Billing
never uses it: metered usage always comes from the provider's reported token counts.
"""

from __future__ import annotations

import json
from typing import Any

# English prose averages ~4 characters per token; 3.2 keeps the estimate conservative
# for code, JSON and non-English text.
_CHARS_PER_TOKEN = 3.2
_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1 if text else 0


def estimate_block_tokens(block: dict[str, Any]) -> int:
    kind = block.get("type")
    if kind == "text":
        return estimate_tokens(block.get("text", ""))
    if kind == "tool_use":
        return estimate_tokens(block.get("name", "")) + estimate_tokens(
            json.dumps(block.get("input", {}))
        )
    if kind == "tool_result":
        content = block.get("content", "")
        return estimate_tokens(content if isinstance(content, str) else json.dumps(content))
    return estimate_tokens(json.dumps(block, default=str))


def estimate_message_tokens(message: dict[str, Any]) -> int:
    content = message.get("content", [])
    if isinstance(content, str):
        return estimate_tokens(content) + _MESSAGE_OVERHEAD
    return sum(estimate_block_tokens(block) for block in content) + _MESSAGE_OVERHEAD


def estimate_tools_tokens(tools: Any) -> int:
    return sum(
        estimate_tokens(tool.name)
        + estimate_tokens(tool.description)
        + estimate_tokens(json.dumps(tool.input_schema))
        for tool in tools
    )

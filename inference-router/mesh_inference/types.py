"""Provider-neutral request and response shapes.

Messages use content blocks in the Messages API layout because it is the richest of the
formats the router speaks:

    {"role": "user" | "assistant", "content": [block, ...]}
    {"type": "text", "text": ...}
    {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
    {"type": "tool_result", "tool_use_id": ..., "content": "...", "is_error": bool}

Providers may return additional block types (for example `thinking`). Those are kept
verbatim so they can be passed back unchanged to the model that produced them, and
dropped when a conversation moves to a provider that cannot read them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from mesh_metering import TokenCost, TokenUsage

Message = dict[str, Any]
Block = dict[str, Any]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal"]
DeltaHandler = Callable[[str], Awaitable[None]]

PORTABLE_BLOCKS = frozenset({"text", "tool_use", "tool_result"})


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False


@dataclass(frozen=True)
class SystemSegment:
    """One part of the system prompt. `cache=True` ends a stable, cacheable prefix."""

    text: str
    cache: bool = False


@dataclass(frozen=True)
class ModelRequest:
    model: str
    system: tuple[SystemSegment, ...]
    messages: tuple[Message, ...]
    tools: tuple[ToolDef, ...] = ()
    max_output_tokens: int = 16000
    effort: str | None = None
    fallback_model: str | None = None
    # LoRA adapter to apply; only models served by the vLLM fleet can honour it.
    adapter: str | None = None


@dataclass(frozen=True)
class Attempt:
    model: str
    provider: str
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class ModelResponse:
    content: tuple[Block, ...]
    stop_reason: StopReason
    usage: TokenUsage
    cost: TokenCost
    model: str  # catalog alias that served the response
    model_id: str  # provider model id that served the response
    provider: str
    adapter: str | None = None
    latency_ms: float = 0.0
    gpu_seconds: float = 0.0
    simulated: bool = False  # produced by the mock provider
    fallback_used: bool = False
    refusal_category: str | None = None
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return "".join(block["text"] for block in self.content if block.get("type") == "text")

    @property
    def tool_uses(self) -> list[Block]:
        return [block for block in self.content if block.get("type") == "tool_use"]


def text_block(text: str) -> Block:
    return {"type": "text", "text": text}


def user_message(text: str) -> Message:
    return {"role": "user", "content": [text_block(text)]}


def tool_result_block(tool_use_id: str, content: str, *, is_error: bool = False) -> Block:
    block: Block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def message_text(message: Message) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if block.get("type") == "text")


def portable(messages: tuple[Message, ...] | list[Message]) -> list[Message]:
    """Drop provider-specific blocks (e.g. thinking) so any provider can read the history."""
    result: list[Message] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            content = [text_block(content)]
        blocks = [block for block in content if block.get("type") in PORTABLE_BLOCKS]
        if blocks:
            result.append({"role": message["role"], "content": blocks})
    return result

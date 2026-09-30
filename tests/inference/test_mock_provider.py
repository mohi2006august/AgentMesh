from __future__ import annotations

import json

import pytest

from mesh_core.clock import FakeClock
from mesh_inference import ModelRequest, SystemSegment, ToolDef, default_catalog
from mesh_inference.providers.mock import MockProvider, extract_arguments, generate
from mesh_inference.types import tool_result_block, user_message

TICKET_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "description": {"type": "string"}},
    "required": ["subject"],
}
REFUND_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
    "required": ["order_id", "amount"],
}
QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
EMPTY_SCHEMA = {"type": "object", "properties": {}}

TOOLS = (
    ToolDef("create_ticket", "Open a support ticket", TICKET_SCHEMA),
    ToolDef("search_tickets", "Search tickets", QUERY_SCHEMA, read_only=True),
    ToolDef("refund", "Refund an order", REFUND_SCHEMA),
    ToolDef("list_invoices", "List invoices", EMPTY_SCHEMA, read_only=True),
    ToolDef("search_knowledge", "Search the knowledge base", QUERY_SCHEMA, read_only=True),
)


def _tools_for(text: str) -> list[str]:
    blocks, _ = generate([user_message(text)], TOOLS)
    return [block["name"] for block in blocks if block["type"] == "tool_use"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My export is broken, please open a ticket about it.", ["create_ticket"]),
        ("Can you list my invoices?", ["list_invoices"]),
        ("What is your refund policy?", ["search_knowledge"]),
        ("Please refund 40 dollars for order 1182.", ["refund"]),
        ("Find tickets about login failures", ["search_tickets"]),
        ("Thanks, that is all.", []),
    ],
)
def test_tool_choice(text: str, expected: list[str]) -> None:
    assert _tools_for(text) == expected


def test_side_effect_tools_need_an_explicit_request() -> None:
    assert _tools_for("I was wondering about refund options") == []


def test_arguments_come_from_the_schema_and_the_text() -> None:
    assert extract_arguments(REFUND_SCHEMA, "Please refund $40.50 for order A-1182 today") == {
        "order_id": "A-1182",
        "amount": 40.5,
    }
    args = extract_arguments(TICKET_SCHEMA, "Export fails on large files")
    assert args == {
        "subject": "Export fails on large files",
        "description": "Export fails on large files",
    }


def test_final_answer_is_written_from_tool_results() -> None:
    first, stop = generate([user_message("What is your refund policy?")], TOOLS)
    assert stop == "tool_use"
    passages = {"results": [{"title": "Refund Policy", "text": "Refunds within 30 days."}]}
    history = [
        user_message("What is your refund policy?"),
        {"role": "assistant", "content": first},
        {"role": "user", "content": [tool_result_block(first[0]["id"], json.dumps(passages))]},
    ]
    answer, stop = generate(history, TOOLS)
    assert stop == "end_turn"
    assert answer[0]["text"] == "According to Refund Policy: Refunds within 30 days."


def test_errors_are_reported_in_the_answer() -> None:
    first, _ = generate([user_message("Please refund 40 dollars for order 7")], TOOLS)
    history = [
        user_message("Please refund 40 dollars for order 7"),
        {"role": "assistant", "content": first},
        {
            "role": "user",
            "content": [tool_result_block(first[0]["id"], "declined by approver", is_error=True)],
        },
    ]
    answer, _ = generate(history, TOOLS)
    assert answer[0]["text"] == "I could not complete refund: declined by approver"


def test_generation_is_deterministic() -> None:
    assert generate([user_message("Please open a ticket")], TOOLS) == generate(
        [user_message("Please open a ticket")], TOOLS
    )


async def test_provider_streams_applies_adapter_and_prices_usage() -> None:
    model = default_catalog().resolve("claude-sonnet")
    deltas: list[str] = []

    async def collect(text: str) -> None:
        deltas.append(text)

    request = ModelRequest(
        model="claude-sonnet",
        system=(SystemSegment("Be helpful."),),
        messages=(user_message("Hello there"),),
        adapter="acme-tone",
    )
    result = await MockProvider().complete("acme", model, request, collect)
    assert result.simulated and result.stop_reason == "end_turn"
    assert result.content[0]["text"].endswith("[styled by acme-tone]")
    assert "".join(deltas) == result.content[0]["text"]
    assert result.usage.input_tokens > 0 and result.usage.output_tokens > 0
    assert result.cost.total == pytest.approx(model.cost(result.usage).total)


async def test_simulated_prompt_cache_writes_then_reads(clock: FakeClock) -> None:
    model = default_catalog().resolve("claude-sonnet")
    request = ModelRequest(
        model="claude-sonnet",
        system=(SystemSegment("Instructions. " * 400, cache=True),),
        messages=(user_message("Hi"),),
    )
    provider = MockProvider(clock)
    first = await provider.complete("acme", model, request, None)
    second = await provider.complete("globex", model, request, None)
    assert first.usage.cache_write_tokens > 0 and first.usage.cache_read_tokens == 0
    assert second.usage.cache_read_tokens == first.usage.cache_write_tokens
    assert second.cost.total < first.cost.total

    clock.advance(301)
    expired = await provider.complete("acme", model, request, None)
    assert expired.usage.cache_write_tokens > 0


async def test_short_prefixes_are_not_cached() -> None:
    model = default_catalog().resolve("claude-haiku")  # 4096-token minimum
    request = ModelRequest(
        model="claude-haiku",
        system=(SystemSegment("Short.", cache=True),),
        messages=(user_message("Hi"),),
    )
    result = await MockProvider().complete("acme", model, request, None)
    assert result.usage.cache_write_tokens == 0 and result.usage.cache_read_tokens == 0

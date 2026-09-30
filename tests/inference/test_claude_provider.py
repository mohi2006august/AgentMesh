"""AnthropicProvider against the real SDK, with HTTP served by a mock transport."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from mesh_inference import ModelRequest, SystemSegment, ToolDef, default_catalog
from mesh_inference.providers import ProviderError, ProviderUnavailable
from mesh_inference.providers.claude import AnthropicProvider
from mesh_inference.types import user_message

CATALOG = default_catalog()
SONNET = CATALOG.resolve("claude-sonnet")
HAIKU = CATALOG.resolve("claude-haiku")
TOOLS = (
    ToolDef("zeta_tool", "Z", {"type": "object", "properties": {}}),
    ToolDef(
        "create_ticket",
        "Open a ticket",
        {"type": "object", "properties": {"subject": {"type": "string"}}},
    ),
)


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def _stream_body(model_id: str = "claude-sonnet-5-5") -> bytes:
    return _sse(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 3000,
                        "cache_creation_input_tokens": 0,
                    },
                },
            },  # fmt: skip
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Opening "},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "a ticket."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "create_ticket",
                    "input": {},
                },
            },  # fmt: skip
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"subject": "Export broken"}',
                },
            },  # fmt: skip
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 40},
            },
            {"type": "message_stop"},
        ]
    )


class Recorder:
    def __init__(self, status: int = 200, body: bytes | None = None) -> None:
        self.status = status
        self.body = body if body is not None else _stream_body()
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            error = {"type": "error", "error": {"type": "api_error", "message": "nope"}}
            return httpx2.Response(self.status, json=error)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=self.body
        )

    @property
    def last_json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def _provider(
    recorder: Recorder, *, base_url: str = "https://api.anthropic.com", **kwargs: Any
) -> AnthropicProvider:
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url=base_url,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(recorder)),
    )
    return AnthropicProvider(CATALOG, client=client, **kwargs)


def _request(**overrides: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "model": "claude-sonnet",
        "system": (SystemSegment("Stable instructions.", cache=True), SystemSegment("")),
        "messages": (user_message("Please open a ticket"),),
        "tools": TOOLS,
        "effort": "medium",
    }
    fields.update(overrides)
    return ModelRequest(**fields)


async def test_streams_text_and_parses_tool_use_usage_and_cost() -> None:
    recorder = Recorder()
    deltas: list[str] = []

    async def collect(text: str) -> None:
        deltas.append(text)

    result = await _provider(recorder).complete("acme", SONNET, _request(), collect)

    assert deltas == ["Opening ", "a ticket."]
    assert result.stop_reason == "tool_use"
    assert result.content[0] == {"type": "text", "text": "Opening a ticket."}
    assert result.content[1]["type"] == "tool_use"
    assert result.content[1]["input"] == {"subject": "Export broken"}
    assert (
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.cache_read_tokens,
    ) == (120, 40, 3000)
    assert result.cost.total == pytest.approx(120 * 2 / 1e6 + 40 * 10 / 1e6 + 3000 * 0.2 / 1e6)
    assert not result.fallback_used


async def test_request_shape_first_party() -> None:
    recorder = Recorder()
    await _provider(recorder).complete("acme", SONNET, _request(), None)
    body = recorder.last_json
    request = recorder.requests[-1]

    assert body["model"] == "claude-sonnet-5-5" and body["stream"] is True
    assert body["system"] == [
        {"type": "text", "text": "Stable instructions.", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["cache_control"] == {"type": "ephemeral"}
    assert [tool["name"] for tool in body["tools"]] == ["create_ticket", "zeta_tool"]
    assert all(tool["eager_input_streaming"] is True for tool in body["tools"])
    assert body["output_config"] == {"effort": "medium"}
    assert "thinking" not in body
    # Server-side fallback on the first-party API.
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request.headers.get("anthropic-beta", "")


async def test_request_shape_behind_a_proxy_omits_newer_fields() -> None:
    recorder = Recorder()
    await _provider(recorder, base_url="https://llm-gateway.internal").complete(
        "acme", SONNET, _request(), None
    )
    body = recorder.last_json
    assert "fallbacks" not in body
    assert all("eager_input_streaming" not in tool for tool in body["tools"])


async def test_effort_is_only_sent_to_models_that_support_it() -> None:
    recorder = Recorder(body=_stream_body("claude-haiku-4-5"))
    await _provider(recorder).complete("acme", HAIKU, _request(model="claude-haiku"), None)
    body = recorder.last_json
    assert "output_config" not in body and "fallbacks" not in body
    assert body["max_tokens"] == 16000


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, ProviderUnavailable),
        (529, ProviderUnavailable),
        (500, ProviderUnavailable),
        (400, ProviderError),
    ],
)
async def test_http_errors_map_to_failover_semantics(status: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        await _provider(Recorder(status=status)).complete("acme", SONNET, _request(), None)


def test_server_side_fallback_is_billed_per_iteration_and_echoable() -> None:
    message = BetaMessage.model_validate(
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig-a"},
                {"type": "text", "text": "Partial. "},
                {
                    "type": "fallback",
                    "from": {"model": "claude-sonnet-5-5"},
                    "to": {"model": "claude-sonnet-5"},
                    "trigger": {"type": "refusal", "category": "cyber"},
                },
                {"type": "text", "text": "Answer from the fallback."},
            ],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 30,
                "iterations": [
                    {
                        "type": "message",
                        "model": "claude-sonnet-5-5",
                        "input_tokens": 1000,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                    {
                        "type": "fallback_message",
                        "model": "claude-sonnet-5",
                        "input_tokens": 1000,
                        "output_tokens": 30,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                ],
            },
        }  # fmt: skip
    )
    result = AnthropicProvider(CATALOG)._result("acme", SONNET, message)
    assert result.fallback_used
    assert [block["type"] for block in result.content] == ["text", "text"]
    assert result.usage.input_tokens == 2000 and result.usage.output_tokens == 35
    assert result.cost.total == pytest.approx(2000 * 2 / 1e6 + 35 * 10 / 1e6)


def test_refusal_category_is_reported() -> None:
    message = BetaMessage.model_validate(
        {
            "id": "msg_3",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "stop_reason": "refusal",
            "stop_sequence": None,
            "content": [],
            "stop_details": {"type": "refusal", "category": "bio", "explanation": None},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }  # fmt: skip
    )
    result = AnthropicProvider(CATALOG)._result("acme", SONNET, message)
    assert result.stop_reason == "refusal" and result.refusal_category == "bio"
    assert result.content == ()


def test_unconfigured_provider_is_unavailable() -> None:
    assert not AnthropicProvider(CATALOG).available()

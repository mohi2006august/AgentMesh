"""Claude models through the official Anthropic SDK.

Request shape:

- Always streamed, so long generations never hit HTTP timeouts.
- `thinking` is left at the model default and depth is steered with `effort`.
- Prompt caching: an explicit breakpoint on each stable system segment (tools render
  before system, so that one marker covers both) plus automatic caching for the growing
  conversation tail. Tools are sorted by name to keep the prefix byte-stable.
- Content blocks are returned verbatim, including thinking blocks, so the runtime can
  pass them back unchanged on the next turn.

Safety declines come back as `stop_reason: "refusal"`. On models that support it the
request opts into server-side fallback, which re-runs a declined request on the
provider's recommended substitute model inside the same call.
"""

from __future__ import annotations

from typing import Any

import anthropic

from mesh_core.log import get_logger
from mesh_inference.catalog import ModelCatalog, ModelInfo
from mesh_inference.providers.base import ProviderError, ProviderResult, ProviderUnavailable
from mesh_inference.types import Block, DeltaHandler, ModelRequest, StopReason
from mesh_metering import TokenCost, TokenUsage

log = get_logger(__name__)

_SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FIRST_PARTY_HOST = "api.anthropic.com"
_MAX_JSON_RETRIES = 2
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "pause_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}
_EPHEMERAL = {"type": "ephemeral"}


def _sum_costs(costs: list[TokenCost]) -> TokenCost:
    return TokenCost(
        input_usd=sum(cost.input_usd for cost in costs),
        output_usd=sum(cost.output_usd for cost in costs),
        cache_read_usd=sum(cost.cache_read_usd for cost in costs),
        cache_write_usd=sum(cost.cache_write_usd for cost in costs),
    )


def _usage_of(entry: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=entry.input_tokens or 0,
        output_tokens=entry.output_tokens or 0,
        cache_read_tokens=getattr(entry, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(entry, "cache_creation_input_tokens", None) or 0,
    )


def _echoable(blocks: list[Block]) -> tuple[list[Block], bool]:
    """Drop `fallback` markers and what cannot be echoed back after a mid-output fallback.

    Before the last switch point only text survives: thinking and tool_use blocks written
    by the model that declined must not be sent back.
    """
    boundary = max(
        (i for i, block in enumerate(blocks) if block.get("type") == "fallback"), default=-1
    )
    if boundary < 0:
        return blocks, False
    kept = [block for block in blocks[:boundary] if block.get("type") == "text"]
    kept += [block for block in blocks[boundary + 1 :] if block.get("type") != "fallback"]
    return kept, True


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        api_key: str = "",
        client: anthropic.AsyncAnthropic | None = None,
        server_fallback: bool | None = None,
        eager_tool_streaming: bool | None = None,
    ) -> None:
        self._catalog = catalog
        self._client = client or (anthropic.AsyncAnthropic(api_key=api_key) if api_key else None)
        # Newer request fields may be rejected by a proxy in front of the API, so they are
        # on by default only when talking to the first-party endpoint.
        first_party = self._client is not None and _FIRST_PARTY_HOST in str(self._client.base_url)
        self._server_fallback = first_party if server_fallback is None else server_fallback
        self._eager_tools = first_party if eager_tool_streaming is None else eager_tool_streaming

    def available(self) -> bool:
        return self._client is not None

    def build_request(self, model: ModelInfo, request: ModelRequest) -> dict[str, Any]:
        system = [
            {
                "type": "text",
                "text": segment.text,
                **({"cache_control": _EPHEMERAL} if segment.cache else {}),
            }
            for segment in request.system
            if segment.text
        ]
        tools = []
        for tool in sorted(request.tools, key=lambda item: item.name):
            definition: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            if self._eager_tools:
                definition["eager_input_streaming"] = True
            tools.append(definition)

        params: dict[str, Any] = {
            "model": model.model_id,
            "max_tokens": min(request.max_output_tokens, model.max_output_tokens),
            "messages": list(request.messages),
            "cache_control": _EPHEMERAL,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        if request.effort and model.supports_effort:
            params["output_config"] = {"effort": request.effort}
        return params

    async def _stream(
        self, model: ModelInfo, params: dict[str, Any], on_delta: DeltaHandler | None
    ) -> Any:
        assert self._client is not None
        manager: Any
        if self._server_fallback and model.server_fallback:
            manager = self._client.beta.messages.stream(
                betas=[_SERVER_FALLBACK_BETA], fallbacks="default", **params
            )
        else:
            manager = self._client.messages.stream(**params)
        async with manager as stream:
            async for event in stream:
                if event.type == "text" and on_delta is not None:
                    await on_delta(event.text)
            return await stream.get_final_message()

    async def complete(
        self, tenant_id: str, model: ModelInfo, request: ModelRequest, on_delta: DeltaHandler | None
    ) -> ProviderResult:
        if self._client is None:
            raise ProviderUnavailable("Anthropic provider is not configured", model=model.alias)
        params = self.build_request(model, request)
        message = None
        for attempt in range(_MAX_JSON_RETRIES + 1):
            try:
                message = await self._stream(model, params, on_delta)
                break
            except ValueError as exc:
                # With eager tool streaming the SDK raises ValueError for tool input it cannot
                # parse at all. No tool_use block exists to answer, so re-issue the turn.
                log.warning(
                    "anthropic.unparseable_tool_input", tenant_id=tenant_id, attempt=attempt
                )
                if attempt == _MAX_JSON_RETRIES:
                    raise ProviderError(
                        "model produced unparseable tool input", model=model.alias
                    ) from exc
            except anthropic.RateLimitError as exc:
                raise ProviderUnavailable("rate limited", model=model.alias) from exc
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    raise ProviderUnavailable(
                        "provider server error", model=model.alias, status_code=exc.status_code
                    ) from exc
                raise ProviderError(
                    exc.message, model=model.alias, status_code=exc.status_code
                ) from exc
            except anthropic.APIConnectionError as exc:
                raise ProviderUnavailable("could not reach provider", model=model.alias) from exc
        assert message is not None
        return self._result(tenant_id, model, message)

    def _result(self, tenant_id: str, model: ModelInfo, message: Any) -> ProviderResult:
        blocks = [
            block.model_dump(mode="json", by_alias=True, exclude_none=True)
            for block in message.content
        ]
        content, switched = _echoable(blocks)

        # With server-side fallback, top-level usage covers only the attempt that produced
        # the message; `usage.iterations` is the per-attempt record, each at its own model's rates.
        iterations = [
            entry
            for entry in (getattr(message.usage, "iterations", None) or [])
            if entry.type in ("message", "fallback_message")
        ]
        entries = iterations or [message.usage]
        usage, costs = TokenUsage(), []
        for entry in entries:
            served_by = getattr(entry, "model", None) or message.model
            info = self._catalog.by_model_id(served_by)
            if info is None:
                log.warning("anthropic.unpriced_model", tenant_id=tenant_id, model_id=served_by)
                info = model
            entry_usage = _usage_of(entry)
            usage += entry_usage
            costs.append(info.cost(entry_usage))

        stop_details = getattr(message, "stop_details", None)
        return ProviderResult(
            content=tuple(content),
            stop_reason=_STOP_REASONS.get(message.stop_reason or "end_turn", "end_turn"),
            usage=usage,
            cost=_sum_costs(costs),
            model_id=message.model,
            fallback_used=switched or any(entry.type == "fallback_message" for entry in iterations),
            refusal_category=getattr(stop_details, "category", None)
            if message.stop_reason == "refusal"
            else None,
        )

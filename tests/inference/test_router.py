from __future__ import annotations

import pytest

from mesh_core.clock import FakeClock
from mesh_core.errors import ConfigError, InvalidRequest, Unavailable
from mesh_inference import (
    CircuitBreaker,
    InferenceRouter,
    ModelInfo,
    ModelRequest,
    SystemSegment,
    default_catalog,
)
from mesh_inference.providers import ProviderError, ProviderResult, ProviderUnavailable
from mesh_inference.providers.mock import MockProvider
from mesh_inference.types import DeltaHandler, text_block, user_message
from mesh_metering import TokenUsage


class ScriptedProvider:
    """Anthropic stand-in whose outcome per model id is set by the test."""

    name = "anthropic"

    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    async def complete(
        self, tenant_id: str, model: ModelInfo, request: ModelRequest, on_delta: DeltaHandler | None
    ) -> ProviderResult:
        self.calls.append(model.model_id)
        if model.model_id in self.failures:
            raise self.failures[model.model_id]
        usage = TokenUsage(input_tokens=100, output_tokens=10)
        return ProviderResult(
            content=(text_block(f"from {model.model_id}"),),
            stop_reason="end_turn",
            usage=usage,
            cost=model.cost(usage),
            model_id=model.model_id,
        )


def _request(**overrides: object) -> ModelRequest:
    fields: dict[str, object] = {
        "model": "claude-sonnet",
        "fallback_model": "small-model",
        "system": (SystemSegment("Be helpful."),),
        "messages": (user_message("Hi"),),
    }
    fields.update(overrides)
    return ModelRequest(**fields)  # type: ignore[arg-type]


async def test_primary_model_serves_when_healthy() -> None:
    provider = ScriptedProvider()
    router = InferenceRouter(default_catalog(), [provider])
    response = await router.complete("acme", _request())
    assert response.text == "from claude-sonnet-5-5"
    assert response.model == "claude-sonnet" and not response.fallback_used
    assert response.cost.total == pytest.approx(100 * 2 / 1e6 + 10 * 10 / 1e6)


async def test_outage_fails_over_to_the_fallback_model() -> None:
    provider = ScriptedProvider({"claude-sonnet-5-5": ProviderUnavailable("overloaded")})
    router = InferenceRouter(default_catalog(), [provider])
    response = await router.complete("acme", _request())
    assert response.model == "claude-haiku" and response.fallback_used
    assert [(a.model, a.ok) for a in response.attempts] == [
        ("claude-sonnet", False),
        ("claude-haiku", True),
    ]


async def test_rejected_requests_do_not_fail_over() -> None:
    provider = ScriptedProvider({"claude-sonnet-5-5": ProviderError("bad request")})
    router = InferenceRouter(default_catalog(), [provider])
    with pytest.raises(ProviderError):
        await router.complete("acme", _request())
    assert provider.calls == ["claude-sonnet-5-5"]


async def test_all_routes_down_is_unavailable() -> None:
    down = ProviderUnavailable("down")
    provider = ScriptedProvider({"claude-sonnet-5-5": down, "claude-haiku-4-5": down})
    with pytest.raises(Unavailable):
        await InferenceRouter(default_catalog(), [provider]).complete("acme", _request())


async def test_open_circuit_skips_the_primary_until_cooldown() -> None:
    clock = FakeClock()
    provider = ScriptedProvider({"claude-sonnet-5-5": ProviderUnavailable("down")})
    router = InferenceRouter(
        default_catalog(),
        [provider],
        breaker=CircuitBreaker(failure_threshold=2, cooldown_s=30, clock=clock),
    )
    for _ in range(2):
        await router.complete("acme", _request())
    provider.calls.clear()

    response = await router.complete("acme", _request())
    assert provider.calls == ["claude-haiku-4-5"]
    assert response.attempts[0].error == "circuit open"

    clock.advance(31)
    provider.failures.clear()
    provider.calls.clear()
    response = await router.complete("acme", _request())
    assert provider.calls == ["claude-sonnet-5-5"] and not response.fallback_used


async def test_dev_substitutes_the_mock_when_no_provider_is_configured() -> None:
    router = InferenceRouter(default_catalog(), [MockProvider()], allow_mock=True)
    response = await router.complete("acme", _request())
    assert response.provider == "mock" and response.simulated
    assert response.model == "claude-sonnet"


async def test_prod_refuses_to_substitute_the_mock() -> None:
    router = InferenceRouter(default_catalog(), [MockProvider()], allow_mock=False)
    with pytest.raises(ConfigError):
        await router.complete("acme", _request())


async def test_unknown_model_is_rejected() -> None:
    router = InferenceRouter(default_catalog(), [MockProvider()], allow_mock=True)
    with pytest.raises(InvalidRequest):
        await router.complete("acme", _request(model="gpt-imaginary"))


def test_catalog_aliases_and_prices() -> None:
    catalog = default_catalog()
    assert catalog.resolve("small-model").model_id == "claude-haiku-4-5"
    assert catalog.resolve("claude-sonnet-5-5").alias == "claude-sonnet"
    cost = catalog.resolve("claude-opus").cost(
        TokenUsage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000,
        )
    )
    assert (cost.input_usd, cost.output_usd, cost.cache_read_usd, cost.cache_write_usd) == (
        4.0,
        20.0,
        0.2,
        5.0,
    )

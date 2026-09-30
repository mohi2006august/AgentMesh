"""Inference router: pick a provider for each model call and fail over on outages.

Order of attempts for a request:

1. the requested model on its provider,
2. the agent's `fallback` model.

A route is skipped while its circuit breaker is open. Only `ProviderUnavailable` (outage,
overload, rate limit) moves on to the next route; a rejected request is returned to the
caller as-is, because another model would not fix it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from mesh_core.errors import ConfigError, Unavailable
from mesh_core.log import get_logger
from mesh_inference.breaker import CircuitBreaker
from mesh_inference.catalog import ModelCatalog, ModelInfo
from mesh_inference.providers.base import Provider, ProviderUnavailable
from mesh_inference.types import Attempt, DeltaHandler, ModelRequest, ModelResponse

log = get_logger(__name__)

MOCK_PROVIDER = "mock"


@dataclass(frozen=True)
class Route:
    model: ModelInfo
    provider: Provider
    adapter: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider.name}:{self.model.model_id}:{self.adapter or '-'}"


class InferenceRouter:
    def __init__(
        self,
        catalog: ModelCatalog,
        providers: Sequence[Provider],
        *,
        allow_mock: bool = False,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.catalog = catalog
        self._providers = {provider.name: provider for provider in providers}
        self._allow_mock = allow_mock
        self._breaker = breaker or CircuitBreaker()

    def _provider_for(self, model: ModelInfo) -> Provider:
        provider = self._providers.get(model.provider)
        if provider is not None and provider.available():
            return provider
        mock = self._providers.get(MOCK_PROVIDER)
        if self._allow_mock and mock is not None:
            return mock
        raise ConfigError(
            "no provider is configured for this model",
            model=model.alias,
            provider=model.provider,
        )

    def plan(self, tenant_id: str, request: ModelRequest) -> list[Route]:
        routes: list[Route] = []
        for name in (request.model, request.fallback_model):
            if not name:
                continue
            model = self.catalog.resolve(name)
            route = Route(model, self._provider_for(model))
            if all(route.key != existing.key for existing in routes):
                routes.append(route)
        return routes

    async def complete(
        self, tenant_id: str, request: ModelRequest, on_delta: DeltaHandler | None = None
    ) -> ModelResponse:
        routes = self.plan(tenant_id, request)
        attempts: list[Attempt] = []
        for index, route in enumerate(routes):
            if not self._breaker.allows(route.key):
                attempts.append(
                    Attempt(route.model.alias, route.provider.name, False, "circuit open")
                )
                continue
            started = time.perf_counter()
            try:
                result = await route.provider.complete(
                    tenant_id, route.model, replace(request, adapter=route.adapter), on_delta
                )
            except ProviderUnavailable as exc:
                self._breaker.record_failure(route.key)
                attempts.append(Attempt(route.model.alias, route.provider.name, False, exc.message))
                log.warning(
                    "router.provider_unavailable",
                    tenant_id=tenant_id,
                    model=route.model.alias,
                    provider=route.provider.name,
                    reason=exc.message,
                )
                continue
            self._breaker.record_success(route.key)
            attempts.append(Attempt(route.model.alias, route.provider.name, True))
            return ModelResponse(
                content=result.content,
                stop_reason=result.stop_reason,
                usage=result.usage,
                cost=result.cost,
                model=route.model.alias,
                model_id=result.model_id,
                provider=route.provider.name,
                adapter=route.adapter,
                latency_ms=(time.perf_counter() - started) * 1000,
                gpu_seconds=result.gpu_seconds,
                simulated=result.simulated,
                fallback_used=index > 0 or result.fallback_used,
                refusal_category=result.refusal_category,
                attempts=tuple(attempts),
            )
        raise Unavailable(
            "no inference route is available",
            attempts=[f"{a.provider}/{a.model}: {a.error}" for a in attempts],
        )

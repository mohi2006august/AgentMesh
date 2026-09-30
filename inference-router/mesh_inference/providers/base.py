"""Provider interface. One implementation per inference backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mesh_core.errors import MeshError
from mesh_inference.catalog import ModelInfo
from mesh_inference.types import Block, DeltaHandler, ModelRequest, StopReason
from mesh_metering import TokenCost, TokenUsage


class ProviderUnavailable(MeshError):
    """Transient failure (outage, overload, rate limit). The router may fail over."""

    code = "provider_unavailable"
    status = 503


class ProviderError(MeshError):
    """The request itself was rejected. Failing over would not help."""

    code = "provider_error"
    status = 502


@dataclass(frozen=True)
class ProviderResult:
    content: tuple[Block, ...]
    stop_reason: StopReason
    usage: TokenUsage
    cost: TokenCost
    model_id: str
    gpu_seconds: float = 0.0
    simulated: bool = False
    fallback_used: bool = False
    refusal_category: str | None = None


class Provider(Protocol):
    name: str

    def available(self) -> bool:
        """Whether this provider is configured (credentials, endpoints) at all."""
        ...

    async def complete(
        self, tenant_id: str, model: ModelInfo, request: ModelRequest, on_delta: DeltaHandler | None
    ) -> ProviderResult: ...

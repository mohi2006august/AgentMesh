"""Model catalog: aliases used in agent.yaml, the provider model behind each, and prices.

Prices are USD per million tokens from the provider's published rate card (checked
2026-09). Cache writes are the 5-minute TTL rate. Update this table when the rate card
or the model line-up changes; nothing else in the platform hard-codes a price.
"""

from __future__ import annotations

from dataclasses import dataclass

from mesh_core.errors import InvalidRequest
from mesh_metering import TokenCost, TokenUsage

_MTOK = 1_000_000


@dataclass(frozen=True)
class ModelInfo:
    alias: str
    provider: str  # anthropic | vllm | mock
    model_id: str
    context_window: int
    max_output_tokens: int
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float
    # Prompts shorter than this are silently not cached by the provider.
    min_cache_tokens: int = 1024
    supports_effort: bool = False
    # The provider can re-run a safety-declined request on a substitute model server-side.
    server_fallback: bool = False
    # Served on the GPU fleet: COGS is GPU seconds, not tokens.
    gpu_priced: bool = False

    def cost(self, usage: TokenUsage) -> TokenCost:
        return TokenCost(
            input_usd=usage.input_tokens * self.input_per_mtok / _MTOK,
            output_usd=usage.output_tokens * self.output_per_mtok / _MTOK,
            cache_read_usd=usage.cache_read_tokens * self.cache_read_per_mtok / _MTOK,
            cache_write_usd=usage.cache_write_tokens * self.cache_write_per_mtok / _MTOK,
        )


_OPUS = ModelInfo(
    alias="claude-opus",
    provider="anthropic",
    model_id="claude-opus-5-5",
    context_window=1_000_000,
    max_output_tokens=128_000,
    input_per_mtok=4.00,
    output_per_mtok=20.00,
    cache_read_per_mtok=0.20,
    cache_write_per_mtok=5.00,
    min_cache_tokens=512,
    supports_effort=True,
    server_fallback=True,
)
_SONNET = ModelInfo(
    alias="claude-sonnet",
    provider="anthropic",
    model_id="claude-sonnet-5-5",
    context_window=1_000_000,
    max_output_tokens=128_000,
    input_per_mtok=2.00,
    output_per_mtok=10.00,
    cache_read_per_mtok=0.20,
    cache_write_per_mtok=2.50,
    min_cache_tokens=512,
    supports_effort=True,
    server_fallback=True,
)
_HAIKU = ModelInfo(
    alias="claude-haiku",
    provider="anthropic",
    model_id="claude-haiku-4-5",
    context_window=200_000,
    max_output_tokens=64_000,
    input_per_mtok=1.00,
    output_per_mtok=5.00,
    cache_read_per_mtok=0.10,
    cache_write_per_mtok=1.25,
    min_cache_tokens=4096,
)
# Shared open-weights base model on the vLLM fleet; tenant LoRA adapters attach to it.
_BASE = ModelInfo(
    alias="mesh-base",
    provider="vllm",
    model_id="mesh-base",
    context_window=32_000,
    max_output_tokens=4_096,
    input_per_mtok=0.0,
    output_per_mtok=0.0,
    cache_read_per_mtok=0.0,
    cache_write_per_mtok=0.0,
    gpu_priced=True,
)


class ModelCatalog:
    def __init__(self, models: list[ModelInfo], aliases: dict[str, str] | None = None) -> None:
        self._models = {model.alias: model for model in models}
        self._aliases = dict(aliases or {})
        for model in models:
            self._aliases.setdefault(model.model_id, model.alias)

    def resolve(self, name: str) -> ModelInfo:
        alias = self._aliases.get(name, name)
        try:
            return self._models[alias]
        except KeyError:
            raise InvalidRequest("unknown model", model=name, known=sorted(self.names())) from None

    def names(self) -> set[str]:
        return set(self._models) | set(self._aliases)

    def by_model_id(self, model_id: str) -> ModelInfo | None:
        alias = self._aliases.get(model_id)
        return self._models.get(alias) if alias else None


def default_catalog() -> ModelCatalog:
    return ModelCatalog(
        [_OPUS, _SONNET, _HAIKU, _BASE],
        aliases={"small-model": "claude-haiku", "large-model": "claude-opus"},
    )

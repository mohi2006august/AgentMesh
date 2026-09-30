"""Meter event schema. The same shape goes to SQL, Kafka and ClickHouse."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class MeterKind(StrEnum):
    RUN = "run"
    TOKENS_INPUT = "tokens.input"
    TOKENS_OUTPUT = "tokens.output"
    TOKENS_CACHE_READ = "tokens.cache_read"
    TOKENS_CACHE_WRITE = "tokens.cache_write"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    GPU_SECONDS = "gpu_seconds"
    CACHE_HIT = "cache_hit"


# Which event kinds roll up into each billable meter named in agent.yaml `billing.meter`.
BILLING_METERS: dict[str, tuple[MeterKind, ...]] = {
    "tokens": (
        MeterKind.TOKENS_INPUT,
        MeterKind.TOKENS_OUTPUT,
        MeterKind.TOKENS_CACHE_READ,
        MeterKind.TOKENS_CACHE_WRITE,
    ),
    "tool_calls": (MeterKind.TOOL_CALL,),
    "retrievals": (MeterKind.RETRIEVAL,),
}

UNITS: dict[MeterKind, str] = {
    MeterKind.RUN: "run",
    MeterKind.TOKENS_INPUT: "token",
    MeterKind.TOKENS_OUTPUT: "token",
    MeterKind.TOKENS_CACHE_READ: "token",
    MeterKind.TOKENS_CACHE_WRITE: "token",
    MeterKind.TOOL_CALL: "call",
    MeterKind.RETRIEVAL: "op",
    MeterKind.GPU_SECONDS: "second",
    MeterKind.CACHE_HIT: "hit",
}


def meter_for(kind: MeterKind | str) -> str | None:
    """The billable meter an event kind belongs to, if any."""
    for meter, kinds in BILLING_METERS.items():
        if kind in kinds:
            return meter
    return None


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one model call.

    `input_tokens` excludes cached tokens, matching how providers report and price them:
    total prompt size = input_tokens + cache_read_tokens + cache_write_tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class MeterEvent:
    """One priced unit of usage. `event_id` is deterministic, which makes emission idempotent."""

    event_id: str
    tenant_id: str
    ts: float
    month: str
    kind: MeterKind
    quantity: float
    unit: str
    cost_usd: float
    billable: bool
    agent: str | None = None
    run_id: str | None = None
    model: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

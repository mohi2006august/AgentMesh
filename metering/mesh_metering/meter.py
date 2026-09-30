"""The single metering helper. Runtime, gateway, retrieval and router all emit through here.

Event ids are derived from `(tenant, run, step, kind)`, so a step that is retried or
replayed after a crash writes the same ids again and the ledger ignores the duplicates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mesh_core.clock import Clock, SystemClock, month_of
from mesh_core.ids import stable_id
from mesh_core.log import get_logger
from mesh_metering.events import UNITS, MeterEvent, MeterKind, TokenUsage, meter_for
from mesh_metering.sinks import MeterSink

log = get_logger(__name__)


@dataclass(frozen=True)
class UnitCosts:
    """Platform COGS rates for non-token usage. Defaults are estimates; set them from your bills."""

    retrieval_op_usd: float = 0.00002  # query embedding + vector search
    tool_call_usd: float = 0.00001  # gateway compute per proxied call
    gpu_second_usd: float = 0.0006  # vLLM fleet, roughly $2.16 per GPU-hour


@dataclass(frozen=True)
class TokenCost:
    """COGS of one model call, split the way providers price it."""

    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_read_usd: float = 0.0
    cache_write_usd: float = 0.0

    @property
    def total(self) -> float:
        return self.input_usd + self.output_usd + self.cache_read_usd + self.cache_write_usd


@dataclass(frozen=True)
class RunScope:
    """What a unit of usage is attributed to."""

    agent: str | None = None
    run_id: str | None = None
    # Meters the agent bills for (`billing.meter` in agent.yaml). Unlisted usage is still
    # recorded as COGS, just not invoiced.
    billable_meters: frozenset[str] = field(
        default_factory=lambda: frozenset({"tokens", "tool_calls", "retrievals"})
    )


class Meter:
    def __init__(
        self,
        sinks: Sequence[MeterSink],
        clock: Clock | None = None,
        unit_costs: UnitCosts | None = None,
    ) -> None:
        if not sinks:
            raise ValueError("Meter needs at least one sink (the ledger)")
        self._ledger, self._mirrors = sinks[0], list(sinks[1:])
        self._clock = clock or SystemClock()
        self.unit_costs = unit_costs or UnitCosts()

    # --- emit helpers ---------------------------------------------------------------

    def run_started(self, tenant_id: str, scope: RunScope) -> None:
        self._emit(tenant_id, [self._event(tenant_id, scope, "run", MeterKind.RUN, 1, 0.0)])

    def model_call(
        self,
        tenant_id: str,
        scope: RunScope,
        *,
        step_key: str,
        model: str,
        usage: TokenUsage,
        cost: TokenCost,
        attrs: dict[str, Any] | None = None,
    ) -> float:
        parts = (
            (MeterKind.TOKENS_INPUT, usage.input_tokens, cost.input_usd),
            (MeterKind.TOKENS_OUTPUT, usage.output_tokens, cost.output_usd),
            (MeterKind.TOKENS_CACHE_READ, usage.cache_read_tokens, cost.cache_read_usd),
            (MeterKind.TOKENS_CACHE_WRITE, usage.cache_write_tokens, cost.cache_write_usd),
        )
        events = [
            self._event(tenant_id, scope, step_key, kind, tokens, usd, model=model, attrs=attrs)
            for kind, tokens, usd in parts
            if tokens
        ]
        self._emit(tenant_id, events)
        return cost.total

    def tool_call(
        self,
        tenant_id: str,
        scope: RunScope,
        *,
        step_key: str,
        server: str,
        tool: str,
        server_cost_usd: float = 0.0,
    ) -> float:
        cost = self.unit_costs.tool_call_usd + server_cost_usd
        event = self._event(
            tenant_id, scope, step_key, MeterKind.TOOL_CALL, 1, cost,
            attrs={"server": server, "tool": tool},
        )  # fmt: skip
        self._emit(tenant_id, [event])
        return cost

    def retrieval(
        self, tenant_id: str, scope: RunScope, *, step_key: str, ops: int = 1, source: str = ""
    ) -> float:
        cost = ops * self.unit_costs.retrieval_op_usd
        event = self._event(
            tenant_id, scope, step_key, MeterKind.RETRIEVAL, ops, cost, attrs={"source": source}
        )
        self._emit(tenant_id, [event])
        return cost

    def gpu_seconds(
        self, tenant_id: str, scope: RunScope, *, step_key: str, seconds: float, model: str
    ) -> float:
        cost = seconds * self.unit_costs.gpu_second_usd
        event = self._event(
            tenant_id, scope, step_key, MeterKind.GPU_SECONDS, seconds, cost, model=model
        )
        self._emit(tenant_id, [event])
        return cost

    def cache_hit(
        self, tenant_id: str, scope: RunScope, *, step_key: str, layer: str, saved_usd: float
    ) -> None:
        """Costs nothing; records what the hit saved so cache ROI is measurable."""
        event = self._event(
            tenant_id, scope, step_key, MeterKind.CACHE_HIT, 1, 0.0,
            attrs={"layer": layer, "saved_usd": round(saved_usd, 8)},
        )  # fmt: skip
        self._emit(tenant_id, [event])

    # --- internals ------------------------------------------------------------------

    def _event(
        self,
        tenant_id: str,
        scope: RunScope,
        step_key: str,
        kind: MeterKind,
        quantity: float,
        cost_usd: float,
        *,
        model: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> MeterEvent:
        now = self._clock.now()
        # Usage outside a run (e.g. indexing) has no natural replay key, so time makes it unique.
        anchor = scope.run_id or f"adhoc:{now!r}"
        meter = meter_for(kind)
        return MeterEvent(
            event_id=stable_id("evt", tenant_id, anchor, step_key, kind.value),
            tenant_id=tenant_id,
            ts=now,
            month=month_of(now),
            kind=kind,
            quantity=float(quantity),
            unit=UNITS[kind],
            cost_usd=float(cost_usd),
            billable=meter is not None and meter in scope.billable_meters,
            agent=scope.agent,
            run_id=scope.run_id,
            model=model,
            attrs={"step": step_key, **(attrs or {})},
        )

    def _emit(self, tenant_id: str, events: list[MeterEvent]) -> None:
        if not events:
            return
        # The ledger write must succeed: billing is derived from it, so failure propagates
        # and the calling step retries.
        self._ledger.write(tenant_id, events)
        for mirror in self._mirrors:
            try:
                mirror.write(tenant_id, events)
            except Exception:  # noqa: BLE001 - a mirror outage must not fail the agent step
                log.exception(
                    "meter.mirror_failed",
                    tenant_id=tenant_id,
                    sink=type(mirror).__name__,
                    events=len(events),
                )

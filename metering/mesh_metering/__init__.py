"""Metering (FR-04): every model call, tool call and retrieval becomes a priced event.

`Meter` is the single helper that builds and emits events. Nothing else constructs a
`MeterEvent` (enforced by `tools/lint_tenancy.py`, rule T004).
"""

from mesh_metering.events import MeterEvent, MeterKind, TokenUsage
from mesh_metering.meter import Meter, RunScope, TokenCost, UnitCosts
from mesh_metering.sinks import MemoryMeterSink, MeterSink, SqlMeterSink
from mesh_metering.usage import UsageStore

__all__ = [
    "MemoryMeterSink",
    "Meter",
    "MeterEvent",
    "MeterKind",
    "MeterSink",
    "RunScope",
    "SqlMeterSink",
    "TokenCost",
    "TokenUsage",
    "UnitCosts",
    "UsageStore",
]

"""Where meter events go. The first sink a `Meter` is given is the billing ledger."""

from __future__ import annotations

import threading
from typing import Protocol

from mesh_core.db import Database, dumps
from mesh_metering.events import MeterEvent


class MeterSink(Protocol):
    def write(self, tenant_id: str, events: list[MeterEvent]) -> None: ...


class SqlMeterSink:
    """Ledger of record. Duplicate event ids are ignored, which makes emission idempotent."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def write(self, tenant_id: str, events: list[MeterEvent]) -> None:
        with self._db.tenant_tx(tenant_id) as tx:
            for event in events:
                tx.execute(
                    """
                    INSERT INTO meter_events
                      (event_id, tenant_id, ts, month, kind, quantity, unit, cost_usd, billable,
                       agent, run_id, model, attrs)
                    VALUES
                      (:event_id, :tenant_id, :ts, :month, :kind, :quantity, :unit, :cost_usd,
                       :billable, :agent, :run_id, :model, :attrs)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    {
                        "event_id": event.event_id,
                        "tenant_id": event.tenant_id,
                        "ts": event.ts,
                        "month": event.month,
                        "kind": event.kind.value,
                        "quantity": event.quantity,
                        "unit": event.unit,
                        "cost_usd": event.cost_usd,
                        "billable": int(event.billable),
                        "agent": event.agent,
                        "run_id": event.run_id,
                        "model": event.model,
                        "attrs": dumps(event.attrs),
                    },
                )


class MemoryMeterSink:
    """In-process sink for tests and as a mirror stand-in."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, MeterEvent] = {}

    def write(self, tenant_id: str, events: list[MeterEvent]) -> None:
        with self._lock:
            for event in events:
                self._events.setdefault(event.event_id, event)

    def events(self, tenant_id: str) -> list[MeterEvent]:
        with self._lock:
            return [event for event in self._events.values() if event.tenant_id == tenant_id]

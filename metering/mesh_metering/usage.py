"""Read side of the meter ledger: quotas, per-run cost and run traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mesh_core.db import Database, loads
from mesh_metering.events import BILLING_METERS, MeterKind


@dataclass(frozen=True)
class KindTotal:
    quantity: float = 0.0
    cost_usd: float = 0.0
    events: int = 0


@dataclass(frozen=True)
class MonthUsage:
    month: str
    by_kind: dict[str, KindTotal] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return sum(total.cost_usd for total in self.by_kind.values())

    @property
    def runs(self) -> int:
        return int(self.by_kind.get(MeterKind.RUN.value, KindTotal()).quantity)

    def meter_quantity(self, meter: str) -> float:
        """Quantity for a billable meter (`tokens`, `tool_calls`, `retrievals`)."""
        return sum(
            self.by_kind.get(kind.value, KindTotal()).quantity for kind in BILLING_METERS[meter]
        )


class UsageStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def month_usage(self, tenant_id: str, month: str) -> MonthUsage:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                """
                SELECT kind, SUM(quantity) AS quantity, SUM(cost_usd) AS cost_usd, COUNT(*) AS events
                FROM meter_events
                WHERE tenant_id = :tenant_id AND month = :month
                GROUP BY kind
                """,
                {"month": month},
            )
        return MonthUsage(
            month=month,
            by_kind={
                row["kind"]: KindTotal(row["quantity"], row["cost_usd"], row["events"])
                for row in rows
            },
        )

    def run_cost(self, tenant_id: str, run_id: str) -> float:
        with self._db.tenant_tx(tenant_id) as tx:
            row = tx.query_one(
                "SELECT COALESCE(SUM(cost_usd), 0) AS cost FROM meter_events "
                "WHERE tenant_id = :tenant_id AND run_id = :run_id",
                {"run_id": run_id},
            )
        return float(row["cost"]) if row else 0.0

    def run_events(self, tenant_id: str, run_id: str) -> list[dict[str, Any]]:
        """Priced events for one run, in order, for the run trace's cost-per-step view."""
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                "SELECT event_id, ts, kind, quantity, unit, cost_usd, model, attrs FROM meter_events "
                "WHERE tenant_id = :tenant_id AND run_id = :run_id ORDER BY ts, event_id",
                {"run_id": run_id},
            )
        return [{**row, "attrs": loads(row["attrs"], {})} for row in rows]

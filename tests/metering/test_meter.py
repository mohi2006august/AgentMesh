from __future__ import annotations

import pytest

from mesh_core.clock import FakeClock, month_of
from mesh_core.db import Database
from mesh_metering import (
    MemoryMeterSink,
    Meter,
    MeterKind,
    RunScope,
    SqlMeterSink,
    TokenCost,
    TokenUsage,
    UnitCosts,
    UsageStore,
)
from mesh_metering.events import meter_for

SCOPE = RunScope(agent="support-agent", run_id="run_1")


@pytest.fixture
def meter(db: Database, clock: FakeClock) -> Meter:
    return Meter([SqlMeterSink(db)], clock)


def _model_call(meter: Meter, tenant_id: str = "acme", step_key: str = "model:1") -> float:
    return meter.model_call(
        tenant_id,
        SCOPE,
        step_key=step_key,
        model="claude-sonnet",
        usage=TokenUsage(input_tokens=1000, output_tokens=200, cache_read_tokens=4000),
        cost=TokenCost(input_usd=0.003, output_usd=0.003, cache_read_usd=0.0012),
    )


def test_model_call_emits_one_event_per_token_class(
    db: Database, meter: Meter, clock: FakeClock
) -> None:
    assert _model_call(meter) == pytest.approx(0.0072)
    usage = UsageStore(db).month_usage("acme", month_of(clock.now()))
    assert set(usage.by_kind) == {"tokens.input", "tokens.output", "tokens.cache_read"}
    assert usage.by_kind["tokens.cache_read"].quantity == 4000
    assert usage.meter_quantity("tokens") == 5200
    assert usage.cost_usd == pytest.approx(0.0072)


def test_replayed_steps_do_not_double_meter(db: Database, meter: Meter, clock: FakeClock) -> None:
    for _ in range(3):
        _model_call(meter)
        meter.tool_call("acme", SCOPE, step_key="tool:1:a", server="zendesk", tool="create_ticket")
        meter.run_started("acme", SCOPE)
        clock.advance(5)
    usage = UsageStore(db).month_usage("acme", month_of(clock.now()))
    assert usage.by_kind["tokens.input"].events == 1
    assert usage.by_kind["tool_call"].events == 1
    assert usage.runs == 1


def test_distinct_steps_are_distinct_events(db: Database, meter: Meter, clock: FakeClock) -> None:
    _model_call(meter, step_key="model:1")
    _model_call(meter, step_key="model:2")
    usage = UsageStore(db).month_usage("acme", month_of(clock.now()))
    assert usage.by_kind["tokens.input"].quantity == 2000


def test_unit_costs_apply_to_tools_retrieval_and_gpu(db: Database, clock: FakeClock) -> None:
    costs = UnitCosts(retrieval_op_usd=0.001, tool_call_usd=0.002, gpu_second_usd=0.01)
    meter = Meter([SqlMeterSink(db)], clock, costs)
    assert meter.tool_call(
        "acme", SCOPE, step_key="t", server="s", tool="x", server_cost_usd=0.05
    ) == pytest.approx(0.052)
    assert meter.retrieval("acme", SCOPE, step_key="r", ops=3, source="drive") == pytest.approx(
        0.003
    )
    assert meter.gpu_seconds(
        "acme", SCOPE, step_key="g", seconds=2.5, model="base"
    ) == pytest.approx(0.025)
    assert UsageStore(db).run_cost("acme", "run_1") == pytest.approx(0.08)


def test_cache_hit_records_savings_at_zero_cost(db: Database, meter: Meter) -> None:
    meter.cache_hit("acme", SCOPE, step_key="cache", layer="semantic", saved_usd=0.0123)
    (event,) = UsageStore(db).run_events("acme", "run_1")
    assert event["kind"] == "cache_hit" and event["cost_usd"] == 0
    assert event["attrs"] == {"step": "cache", "layer": "semantic", "saved_usd": 0.0123}


def test_billable_follows_the_agent_billing_meters(db: Database, clock: FakeClock) -> None:
    sink = MemoryMeterSink()
    meter = Meter([SqlMeterSink(db), sink], clock)
    scope = RunScope(agent="a", run_id="run_2", billable_meters=frozenset({"tokens"}))
    _model_call(meter)
    meter.model_call(
        "acme", scope, step_key="m", model="m", usage=TokenUsage(input_tokens=1), cost=TokenCost()
    )
    meter.tool_call("acme", scope, step_key="t", server="s", tool="x")
    meter.run_started("acme", scope)
    by_kind = {
        event.kind: event.billable for event in sink.events("acme") if event.run_id == "run_2"
    }
    assert by_kind == {
        MeterKind.TOKENS_INPUT: True,
        MeterKind.TOOL_CALL: False,
        MeterKind.RUN: False,
    }


def test_usage_is_tenant_scoped(db: Database, meter: Meter, clock: FakeClock) -> None:
    _model_call(meter, "acme")
    _model_call(meter, "globex")
    store = UsageStore(db)
    assert store.month_usage("acme", month_of(clock.now())).meter_quantity("tokens") == 5200
    assert store.run_cost("globex", "run_1") == pytest.approx(0.0072)
    assert store.month_usage("initech", month_of(clock.now())).cost_usd == 0


def test_mirror_failure_does_not_fail_the_step_but_ledger_failure_does(
    db: Database, clock: FakeClock
) -> None:
    class Broken:
        def write(self, tenant_id: str, events: list) -> None:
            raise ConnectionError("kafka down")

    Meter([SqlMeterSink(db), Broken()], clock).run_started("acme", SCOPE)
    assert UsageStore(db).month_usage("acme", month_of(clock.now())).runs == 1

    with pytest.raises(ConnectionError):
        Meter([Broken(), SqlMeterSink(db)], clock).run_started("acme", SCOPE)


def test_meter_for_maps_kinds_to_billable_meters() -> None:
    assert meter_for(MeterKind.TOKENS_CACHE_READ) == "tokens"
    assert meter_for(MeterKind.TOOL_CALL) == "tool_calls"
    assert meter_for(MeterKind.RUN) is None
    assert meter_for(MeterKind.GPU_SECONDS) is None

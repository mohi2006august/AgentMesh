from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from mesh_core.errors import InvalidRequest
from mesh_sdk.cli import main
from mesh_sdk.spec import agent_json_schema, load_agent_file, parse_agent, parse_duration

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "support-agent.yaml"


def _minimal(**overrides: Any) -> dict[str, Any]:
    return {"name": "demo", "model": "claude-sonnet", "instructions": "Be helpful.", **overrides}


def test_example_agent_loads_with_inlined_files() -> None:
    spec = load_agent_file(EXAMPLE)
    assert spec.name == "support-agent"
    assert spec.model.default == "claude-sonnet" and spec.model.fallback == "small-model"
    assert "customer support agent" in spec.instructions
    assert [tool.mcp for tool in spec.tools] == ["zendesk", "stripe-readonly"]
    assert spec.tools[0].policy == {
        "create_ticket": "allow",
        "search_tickets": "allow",
        "refund": "approve",
    }
    assert spec.context.budget_tokens == 24000
    assert spec.context.memory == {"session", "profile"}
    assert spec.cache.semantic is not None
    assert spec.cache.semantic.threshold == 0.93 and spec.cache.semantic.ttl == 86400
    assert spec.limits.max_steps == 12 and spec.limits.max_usd_per_run == 0.40
    assert spec.billing.meter == {"tokens", "tool_calls", "retrievals"}
    assert spec.evals is not None and len(spec.evals.cases) == 5


def test_adapter_template_expands_per_tenant() -> None:
    spec = parse_agent(_minimal(model={"default": "m", "adapter": "{tenant}-tone"}))
    assert spec.adapter_for("acme") == "acme-tone"
    assert parse_agent(_minimal()).adapter_for("acme") is None


def test_defaults_and_shorthand() -> None:
    spec = parse_agent(_minimal())
    assert spec.model.default == "claude-sonnet"
    assert spec.context.memory == frozenset()
    assert spec.cache.scope == "tenant" and spec.cache.semantic is None
    assert spec.canary.percent == 0


def test_spec_hash_tracks_behaviour() -> None:
    base = parse_agent(_minimal())
    same = parse_agent(_minimal())
    changed = parse_agent(_minimal(instructions="Be terse."))
    assert base.spec_hash == same.spec_hash
    assert base.spec_hash != changed.spec_hash


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"name": "Bad Name"}, "name"),
        ({"limitz": {}}, "limitz"),
        ({"limits": {"max_steps": 0}}, "limits.max_steps"),
        ({"limits": {"max_usd_per_run": -1}}, "limits.max_usd_per_run"),
        ({"tools": [{"mcp": "zendesk", "policy": {"refund": "maybe"}}]}, "tools.0.policy.refund"),
        ({"tools": [{"mcp": "a"}, {"mcp": "a"}]}, "tools"),
        ({"context": {"memory": "session+everything"}}, "context.memory"),
        ({"cache": {"semantic": {"threshold": 0.2}}}, "cache.semantic.threshold"),
        ({"cache": {"semantic": {"ttl": "soon"}}}, "cache.semantic.ttl"),
        ({"billing": {"meter": ["vibes"]}}, "billing.meter"),
        ({"canary": {"percent": 140}}, "canary.percent"),
    ],
)
def test_invalid_specs_are_rejected_with_a_path(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(InvalidRequest) as excinfo:
        parse_agent(_minimal(**overrides))
    assert any(fragment in error for error in excinfo.value.detail["errors"])


@pytest.mark.parametrize(
    ("value", "seconds"),
    [(45, 45), ("45", 45), ("45s", 45), ("30m", 1800), ("12h", 43200), ("1d", 86400)],
)
def test_parse_duration(value: Any, seconds: int) -> None:
    assert parse_duration(value) == seconds


@pytest.mark.parametrize("value", ["", "1w", "-5s", 0, True, "soon"])
def test_parse_duration_rejects(value: Any) -> None:
    with pytest.raises(ValueError):
        parse_duration(value)


def test_missing_instruction_file_is_an_error(tmp_path: Path) -> None:
    file = tmp_path / "agent.yaml"
    file.write_text("name: demo\nmodel: m\ninstructions: prompts/missing.md\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="instructions file not found"):
        load_agent_file(file)


def test_invalid_yaml_is_an_error(tmp_path: Path) -> None:
    file = tmp_path / "agent.yaml"
    file.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="not valid YAML"):
        load_agent_file(file)


def test_json_schema_describes_the_top_level_keys() -> None:
    properties = agent_json_schema()["properties"]
    assert {"name", "model", "instructions", "tools", "knowledge", "limits"} <= set(properties)


def test_cli_validate() -> None:
    ok = CliRunner().invoke(main, ["validate", str(EXAMPLE)])
    assert ok.exit_code == 0 and "ok: support-agent" in ok.output


def test_cli_validate_reports_errors(tmp_path: Path) -> None:
    file = tmp_path / "agent.yaml"
    file.write_text(
        "name: demo\nmodel: m\ninstructions: hi\nlimits: {max_steps: 0}\n", encoding="utf-8"
    )
    result = CliRunner().invoke(main, ["validate", str(file)])
    assert result.exit_code == 1
    assert "limits.max_steps" in result.output

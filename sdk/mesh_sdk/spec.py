"""The declarative agent definition (`agent.yaml`, FR-01).

    name: support-agent
    model: {default: claude-sonnet, fallback: small-model, adapter: "{tenant}-tone"}
    instructions: prompts/support.md
    knowledge:
      - source: tenant_drive
        acl: inherit
    tools:
      - mcp: zendesk
        policy: {create_ticket: allow, refund: approve}
      - mcp: stripe-readonly
    context: {budget_tokens: 24000, memory: session+profile}
    cache: {semantic: {threshold: 0.93, ttl: 1d}, scope: tenant}
    limits: {max_steps: 12, max_usd_per_run: 0.40}
    billing: {meter: [tokens, tool_calls, retrievals]}

Unknown keys are rejected so a typo never silently disables a limit or a policy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, field_validator

from mesh_core.errors import InvalidRequest
from mesh_core.ids import content_hash

PolicyAction = Literal["allow", "approve", "deny"]
Meter = Literal["tokens", "tool_calls", "retrievals"]
MemoryKind = Literal["session", "profile"]

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_DURATION = re.compile(r"^(\d+)\s*(s|m|h|d)?$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_TEXT_FILE_SUFFIXES = (".md", ".txt")


def parse_duration(value: Any) -> int:
    """`45`, `"45s"`, `"30m"`, `"12h"`, `"1d"` -> seconds."""
    if isinstance(value, bool):
        raise ValueError("duration must be a number of seconds or a string like '1d'")
    if isinstance(value, int):
        seconds = value
    else:
        match = _DURATION.match(str(value).strip().lower())
        if not match:
            raise ValueError(f"invalid duration {value!r}: use e.g. 45s, 30m, 12h, 1d")
        seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2) or "s"]
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return seconds


def _parse_memory(value: Any) -> Any:
    if isinstance(value, str):
        if value.strip().lower() in ("", "none"):
            return []
        return [part.strip() for part in value.split("+")]
    return value


Duration = Annotated[int, BeforeValidator(parse_duration)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(_Model):
    default: str
    fallback: str | None = None
    # LoRA adapter applied on the shared base model. `{tenant}` expands per tenant.
    adapter: str | None = None
    max_output_tokens: int = Field(default=1024, ge=1, le=64000)


class KnowledgeSpec(_Model):
    source: str
    # inherit: enforce each document's ACL against the end user; tenant: any user of the tenant.
    acl: Literal["inherit", "tenant"] = "inherit"
    top_k: int = Field(default=5, ge=1, le=50)


class ToolSpec(_Model):
    mcp: str
    # Per-tool action; `"*"` sets the default for unlisted tools. Without one, unlisted
    # read-only tools are allowed and everything else needs approval.
    policy: dict[str, PolicyAction] = Field(default_factory=dict)


class ContextSpec(_Model):
    budget_tokens: int = Field(default=24000, ge=1000)
    memory: Annotated[frozenset[MemoryKind], BeforeValidator(_parse_memory)] = frozenset()


class SemanticCacheSpec(_Model):
    threshold: float = Field(default=0.93, gt=0.5, le=1.0)
    ttl: Duration = 86400


class CacheSpec(_Model):
    semantic: SemanticCacheSpec | None = None
    exact: bool = True
    # tenant: never shared. global: answers built only from public sources may be shared.
    scope: Literal["tenant", "global", "none"] = "tenant"


class LimitsSpec(_Model):
    max_steps: int = Field(default=12, ge=1, le=100)
    max_usd_per_run: float = Field(default=0.40, gt=0)
    tool_timeout_s: float = Field(default=30.0, gt=0)


class BillingSpec(_Model):
    meter: frozenset[Meter] = frozenset({"tokens", "tool_calls", "retrievals"})


class EvalCase(_Model):
    name: str
    input: str
    expect_contains: tuple[str, ...] = ()
    expect_not_contains: tuple[str, ...] = ()
    expect_tools: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()
    max_usd: float | None = None


class EvalSpec(_Model):
    # Path to a YAML list of cases (inlined on load) or the cases themselves.
    suite: str | None = None
    cases: tuple[EvalCase, ...] = ()
    min_pass_rate: float = Field(default=0.9, ge=0, le=1)


class CanarySpec(_Model):
    percent: int = Field(default=0, ge=0, le=100)


def _model_shorthand(value: Any) -> Any:
    return {"default": value} if isinstance(value, str) else value


class AgentSpec(_Model):
    name: str
    description: str = ""
    model: Annotated[ModelSpec, BeforeValidator(_model_shorthand)]
    instructions: str
    knowledge: tuple[KnowledgeSpec, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    context: ContextSpec = ContextSpec()
    cache: CacheSpec = CacheSpec()
    limits: LimitsSpec = LimitsSpec()
    billing: BillingSpec = BillingSpec()
    evals: EvalSpec | None = None
    canary: CanarySpec = CanarySpec()

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError("name must be 2-63 chars of lowercase letters, digits and hyphens")
        return value

    @field_validator("tools")
    @classmethod
    def _unique_servers(cls, value: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        servers = [tool.mcp for tool in value]
        if len(servers) != len(set(servers)):
            raise ValueError("each MCP server may be listed once under tools")
        return value

    def adapter_for(self, tenant_id: str) -> str | None:
        return self.model.adapter.replace("{tenant}", tenant_id) if self.model.adapter else None

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        """Identifies this exact behaviour; part of every cache key as `agent_version`."""
        return content_hash(self.canonical_json())[:16]


def _format_errors(exc: ValidationError) -> list[str]:
    return [
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    ]


def parse_agent(data: Any) -> AgentSpec:
    if not isinstance(data, dict):
        raise InvalidRequest("agent spec must be a mapping")
    try:
        return AgentSpec.model_validate(data)
    except ValidationError as exc:
        raise InvalidRequest("invalid agent spec", errors=_format_errors(exc)) from exc


def _inline_file(base: Path, reference: str, what: str) -> str:
    path = (base / reference).resolve()
    if not path.is_file():
        raise InvalidRequest(f"{what} file not found", path=str(path))
    return path.read_text(encoding="utf-8")


def load_agent_file(path: str | Path) -> AgentSpec:
    """Load `agent.yaml`, inlining `instructions` and eval suites referenced by path.

    The deployed spec is self-contained: the control plane never reads the developer's
    filesystem.
    """
    file = Path(path)
    if not file.is_file():
        raise InvalidRequest("agent file not found", path=str(file))
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InvalidRequest(
            "agent file is not valid YAML", path=str(file), reason=str(exc)
        ) from exc
    if not isinstance(data, dict):
        raise InvalidRequest("agent file must contain a mapping", path=str(file))

    instructions = data.get("instructions")
    if isinstance(instructions, str) and instructions.strip().endswith(_TEXT_FILE_SUFFIXES):
        data["instructions"] = _inline_file(file.parent, instructions.strip(), "instructions")

    evals = data.get("evals")
    if isinstance(evals, dict) and evals.get("suite") and not evals.get("cases"):
        suite = yaml.safe_load(_inline_file(file.parent, evals["suite"], "eval suite"))
        evals["cases"] = suite.get("cases", suite) if isinstance(suite, dict) else suite

    return parse_agent(data)


def agent_json_schema() -> dict[str, Any]:
    return AgentSpec.model_json_schema()

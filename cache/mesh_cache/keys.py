"""Cache keys (TECHNICAL.md, "Tenant-scoped cache key").

    key = H(tenant_id, agent_version, model, adapter, tool_set_hash, normalized_messages)

Every component is required. Dropping `adapter` would let one tenant's styled answer be
served to another tenant; dropping `agent_version` would serve answers from an older
prompt. The global scope replaces the tenant id with a fixed marker and is only written
for answers built entirely from public sources.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mesh_core.ids import content_hash
from mesh_inference.types import Message, ToolDef, portable

# Cannot collide with a tenant id: tenant ids never start with an underscore.
GLOBAL_SCOPE = "_global"
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def normalize_messages(messages: Sequence[Message]) -> str:
    """Canonical form: portable blocks only, normalised text, stable key order."""
    canonical: list[dict[str, Any]] = []
    for message in portable(list(messages)):
        blocks = []
        for block in message["content"]:
            if block["type"] == "text":
                blocks.append({"type": "text", "text": normalize_text(block["text"])})
            else:
                blocks.append(block)
        canonical.append({"role": message["role"], "content": blocks})
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)


def tool_set_hash(tools: Iterable[ToolDef]) -> str:
    described = sorted(
        (tool.name, json.dumps(tool.input_schema, sort_keys=True), tool.description)
        for tool in tools
    )
    return content_hash(*(part for item in described for part in item))[:16]


@dataclass(frozen=True)
class KeyParts:
    agent_version: str
    model: str
    adapter: str | None
    tool_set_hash: str

    def scope(self, owner: str, history: Sequence[Message]) -> str:
        """Semantic-cache scope: everything except the final user question."""
        return content_hash(
            owner, self.agent_version, self.model, self.adapter or "-", self.tool_set_hash,
            normalize_messages(history),
        )  # fmt: skip

    def exact(self, owner: str, messages: Sequence[Message]) -> str:
        return content_hash(
            owner, self.agent_version, self.model, self.adapter or "-", self.tool_set_hash,
            normalize_messages(messages),
        )  # fmt: skip

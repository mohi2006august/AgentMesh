"""Deterministic stand-in model for local development, tests and evals.

It is a rule engine, not a language model: it picks tools by matching the user's words
against tool names, fills arguments from the JSON schema, and writes its answer from the
tool results. That is enough to exercise the whole platform (tool loop, policies,
approvals, retrieval, caching, metering) with no API key, and the same input always
produces the same output.

Usage is estimated and priced from the requested model's rate card, including a
simulated prompt cache, so the economics views have realistic numbers. Every response
is flagged `simulated`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mesh_core.clock import Clock, SystemClock
from mesh_core.ids import content_hash
from mesh_inference.catalog import ModelInfo
from mesh_inference.providers.base import ProviderResult
from mesh_inference.tokens import estimate_message_tokens, estimate_tokens, estimate_tools_tokens
from mesh_inference.types import (
    Block,
    DeltaHandler,
    Message,
    ModelRequest,
    StopReason,
    ToolDef,
    message_text,
    portable,
    text_block,
)
from mesh_metering import TokenUsage

KNOWLEDGE_TOOL = "search_knowledge"
_PROMPT_CACHE_TTL_S = 300.0
_DELTA_CHARS = 24

_WH_WORDS = {"what", "how", "why", "when", "where", "who", "which"}
_VERB_SYNONYMS: dict[str, set[str]] = {
    "create": {"create", "open", "file", "raise", "log", "make", "submit", "add"},
    "search": {"search", "find", "look", "lookup"},
    "list": {"list", "show", "see", "view"},
    "get": {"get", "fetch", "show", "lookup", "check"},
    "update": {"update", "change", "edit", "set"},
    "delete": {"delete", "remove", "cancel"},
    "send": {"send", "email", "notify"},
    "refund": {"refund", "reimburse"},
}
_FREE_TEXT_FIELDS = ("description", "body", "message", "reason", "content", "details", "text")
_SHORT_TEXT_FIELDS = ("query", "question", "subject", "title", "summary", "name")
_AMOUNT_FIELDS = ("amount", "price", "total", "value")


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words(text: str) -> list[str]:
    return [_stem(word) for word in re.findall(r"[a-z0-9]+", text.lower())]


def apply_adapter(text: str, adapter: str | None) -> str:
    """Make a LoRA adapter's effect visible: each adapter signs the text it styled."""
    return f"{text}\n\n[styled by {adapter}]" if adapter and text else text


# --- tool selection -----------------------------------------------------------------


def _choose_tools(text: str, tools: tuple[ToolDef, ...]) -> list[ToolDef]:
    words = _words(text)
    word_set = set(words)
    asks_wh_question = bool(words) and words[0] in _WH_WORDS
    has_number = bool(re.search(r"\d", text))

    best: dict[tuple[str, ...], tuple[int, ToolDef]] = {}
    for tool in tools:
        if tool.name == KNOWLEDGE_TOOL:
            continue
        tokens = [_stem(token) for token in tool.name.lower().split("_") if token]
        verb = tokens[0] if tokens and tokens[0] in _VERB_SYNONYMS else None
        nouns = [token for token in tokens if token not in _VERB_SYNONYMS] or tokens
        if not any(noun in word_set for noun in nouns):
            continue
        # A one-word tool name ("refund") is both verb and noun: naming it is not a request.
        verb_hit = (
            verb is not None
            and verb not in nouns
            and bool({_stem(v) for v in _VERB_SYNONYMS[verb]} & word_set)
        )
        explicit = verb_hit or has_number or "please" in word_set
        # Side effects need an explicit request, never a "what is your X policy?" question.
        if not tool.read_only and (asks_wh_question or not explicit):
            continue
        score = 2 + int(verb_hit)
        key = tuple(sorted(nouns))
        current = best.get(key)
        if current is None or score > current[0]:
            best[key] = (score, tool)

    chosen = sorted((tool for _, tool in best.values()), key=lambda tool: tool.name)
    if chosen:
        return chosen
    knowledge = next((tool for tool in tools if tool.name == KNOWLEDGE_TOOL), None)
    looks_like_question = "?" in text or asks_wh_question
    return [knowledge] if knowledge and looks_like_question else []


# --- argument extraction ------------------------------------------------------------


def _number_after(text: str, anchor: str) -> str | None:
    match = re.search(
        rf"{re.escape(anchor)}\s*(?:id|number|no\.?|#)?\s*[:#]?\s*([A-Za-z]*-?\d[\w-]*)", text, re.I
    )
    return match.group(1) if match else None


def _amount(text: str) -> float | None:
    match = re.search(r"\$\s*(\d+(?:\.\d+)?)", text) or re.search(
        r"(\d+(?:\.\d+)?)\s*(?:dollars?|usd|bucks)", text, re.I
    )
    return float(match.group(1)) if match else None


def _extract(name: str, prop: dict[str, Any], text: str) -> Any:
    kind = prop.get("type", "string")
    lowered = name.lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if "enum" in prop:
        return next(
            (option for option in prop["enum"] if str(option).lower() in text.lower()), None
        )

    if kind in ("number", "integer"):
        value: float | None = None
        if any(field in lowered for field in _AMOUNT_FIELDS):
            value = _amount(text)
        elif lowered.endswith(("id", "number")):
            anchor = lowered.removesuffix("_id").removesuffix("_number").removesuffix("id")
            found = _number_after(text, anchor) if anchor else None
            value = float(found) if found and found.replace(".", "", 1).isdigit() else None
        if value is None and numbers and lowered not in ("limit", "top_k", "page", "offset"):
            value = float(numbers[0])
        if value is None:
            return None
        return int(value) if kind == "integer" or value.is_integer() else value

    if kind == "string":
        if "email" in lowered:
            match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
            return match.group(0) if match else None
        if lowered.endswith("id") or lowered.endswith("number"):
            anchor = lowered.removesuffix("_id").removesuffix("_number").removesuffix("id")
            return (_number_after(text, anchor) if anchor else None) or (
                numbers[0] if numbers else None
            )
        if any(field in lowered for field in _SHORT_TEXT_FIELDS):
            return text.strip()[:120]
        if any(field in lowered for field in _FREE_TEXT_FIELDS):
            return text.strip()
    return None


def _default(prop: dict[str, Any], text: str) -> Any:
    if "enum" in prop:
        return prop["enum"][0]
    return {"number": 0, "integer": 0, "boolean": False, "array": [], "object": {}}.get(
        prop.get("type", "string"), text.strip()[:120]
    )


def extract_arguments(schema: dict[str, Any], text: str) -> dict[str, Any]:
    required = set(schema.get("required", []))
    arguments: dict[str, Any] = {}
    for name, prop in schema.get("properties", {}).items():
        value = _extract(name, prop, text)
        if value is not None:
            arguments[name] = value
        elif name in required:
            arguments[name] = _default(prop, text)
    return arguments


# --- answers ------------------------------------------------------------------------


def _clip(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _describe_result(tool: str, content: str, is_error: bool) -> str:
    if is_error:
        return f"I could not complete {tool}: {_clip(content, 200)}"
    if tool == KNOWLEDGE_TOOL:
        try:
            results = json.loads(content).get("results", [])
        except (ValueError, AttributeError):
            results = []
        if not results:
            return "I could not find anything about that in the knowledge base."
        return "\n".join(
            f"According to {hit.get('title', 'the knowledge base')}: {_clip(hit.get('text', ''))}"
            for hit in results[:2]
        )
    return f"{tool} returned: {_clip(content)}"


def _answer_from_results(messages: list[Message]) -> str:
    names: dict[str, str] = {}
    for message in messages:
        if message["role"] == "assistant":
            for block in message["content"]:
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
    lines = []
    for block in messages[-1]["content"]:
        if block.get("type") == "tool_result":
            content = block.get("content", "")
            content = content if isinstance(content, str) else json.dumps(content)
            tool = names.get(block["tool_use_id"], "tool")
            lines.append(_describe_result(tool, content, bool(block.get("is_error"))))
    return "\n".join(lines)


def _direct_answer(text: str, tools: tuple[ToolDef, ...]) -> str:
    capabilities = ", ".join(tool.name.replace("_", " ") for tool in tools) or "general questions"
    return (
        f'I can help with that. You said: "{_clip(text, 160)}". I can assist with: {capabilities}.'
    )


def generate(messages: list[Message], tools: tuple[ToolDef, ...]) -> tuple[list[Block], StopReason]:
    """Decide the next assistant turn: tool calls, or a final answer."""
    if not messages:
        return [text_block("Hello. How can I help?")], "end_turn"
    last = messages[-1]
    if any(block.get("type") == "tool_result" for block in last["content"]):
        return [text_block(_answer_from_results(messages))], "end_turn"

    text = message_text(last)
    chosen = _choose_tools(text, tools)
    if not chosen:
        return [text_block(_direct_answer(text, tools))], "end_turn"

    blocks: list[Block] = []
    for index, tool in enumerate(chosen):
        arguments = extract_arguments(tool.input_schema, text)
        blocks.append(
            {
                "type": "tool_use",
                "id": "toolu_"
                + content_hash(len(messages), index, tool.name, json.dumps(arguments))[:16],
                "name": tool.name,
                "input": arguments,
            }
        )
    return blocks, "tool_use"


# --- provider -----------------------------------------------------------------------


class MockProvider:
    name = "mock"

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._prompt_cache: dict[str, float] = {}

    def available(self) -> bool:
        return True

    def _usage(self, model: ModelInfo, request: ModelRequest, blocks: list[Block]) -> TokenUsage:
        cached_until = max((i for i, seg in enumerate(request.system) if seg.cache), default=-1)
        prefix_text = "".join(seg.text for seg in request.system[: cached_until + 1])
        prefix_tokens = 0
        if cached_until >= 0:
            prefix_tokens = estimate_tokens(prefix_text) + estimate_tools_tokens(request.tools)
        rest_tokens = (
            sum(estimate_tokens(seg.text) for seg in request.system[cached_until + 1 :])
            + sum(estimate_message_tokens(message) for message in request.messages)
            + (0 if cached_until >= 0 else estimate_tools_tokens(request.tools))
        )
        output_tokens = estimate_message_tokens({"role": "assistant", "content": blocks})

        if prefix_tokens < model.min_cache_tokens:
            return TokenUsage(input_tokens=prefix_tokens + rest_tokens, output_tokens=output_tokens)

        tool_names = ",".join(sorted(tool.name for tool in request.tools))
        key = content_hash(model.model_id, prefix_text, tool_names)
        now = self._clock.now()
        warm = self._prompt_cache.get(key, 0.0) > now
        self._prompt_cache[key] = now + _PROMPT_CACHE_TTL_S
        return TokenUsage(
            input_tokens=rest_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=prefix_tokens if warm else 0,
            cache_write_tokens=0 if warm else prefix_tokens,
        )

    async def complete(
        self, tenant_id: str, model: ModelInfo, request: ModelRequest, on_delta: DeltaHandler | None
    ) -> ProviderResult:
        blocks, stop_reason = generate(portable(request.messages), request.tools)
        for block in blocks:
            if block["type"] == "text":
                block["text"] = apply_adapter(block["text"], request.adapter)
                if on_delta is not None:
                    text = block["text"]
                    for start in range(0, len(text), _DELTA_CHARS):
                        await on_delta(text[start : start + _DELTA_CHARS])
        usage = self._usage(model, request, blocks)
        return ProviderResult(
            content=tuple(blocks),
            stop_reason=stop_reason,
            usage=usage,
            cost=model.cost(usage),
            model_id=model.model_id,
            simulated=True,
        )

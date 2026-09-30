"""Structured logging: one JSON object per line, fields instead of interpolated strings.

log = get_logger(__name__)
log.info("tool.called", tenant_id=ctx.tenant_id, tool="create_ticket", latency_ms=12)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_REDACT_KEYS = ("secret", "token", "password", "api_key", "authorization", "credential")


def _redact(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: "[redacted]" if any(marker in key.lower() for marker in _REDACT_KEYS) else value
        for key, value in fields.items()
    }


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class StructuredLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, exc_info: bool, fields: dict[str, Any]) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, event, exc_info=exc_info, extra={"fields": _redact(fields)})

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, False, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, False, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, False, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, False, fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, True, fields)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("mesh")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)
        root.propagate = False
    root.setLevel(level.upper())


def get_logger(name: str) -> StructuredLogger:
    suffix = name.removeprefix("mesh_")
    return StructuredLogger(logging.getLogger(f"mesh.{suffix}"))

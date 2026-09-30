"""Error hierarchy shared across components.

Every error carries a stable `code` and an HTTP status so the API edge can map
failures without string matching.
"""

from __future__ import annotations

from typing import Any


class MeshError(Exception):
    code = "internal_error"
    status = 500

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, **self.detail}}


class InvalidRequest(MeshError):
    code = "invalid_request"
    status = 400


class Unauthorized(MeshError):
    code = "unauthorized"
    status = 401


class Forbidden(MeshError):
    code = "forbidden"
    status = 403


class NotFound(MeshError):
    code = "not_found"
    status = 404


class Conflict(MeshError):
    code = "conflict"
    status = 409


class IsolationViolation(MeshError):
    """A statement or lookup tried to cross a tenant boundary. Always a bug."""

    code = "isolation_violation"
    status = 500


class PolicyDenied(Forbidden):
    code = "policy_denied"


class ResidencyViolation(Forbidden):
    """The tenant is pinned to a different region than this deployment serves."""

    code = "residency_violation"


class QuotaExceeded(MeshError):
    """Hard plan limit reached. Carries an upsell hint for the caller."""

    code = "quota_exceeded"
    status = 429


class RateLimited(MeshError):
    code = "rate_limited"
    status = 429


class Unavailable(MeshError):
    """A dependency (MCP server, provider) is down. Structured so agents can adapt."""

    code = "unavailable"
    status = 503


class ConfigError(MeshError):
    code = "config_error"
    status = 500

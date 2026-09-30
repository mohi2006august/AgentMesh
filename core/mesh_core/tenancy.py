"""Tenant identity carried explicitly through every call (FR-03).

There is no ambient "current tenant": functions that touch tenant data take
`tenant_id` as their first argument, and `tools/lint_tenancy.py` enforces it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from mesh_core.errors import InvalidRequest

_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")

F = TypeVar("F", bound=Callable[..., object])


def validate_tenant_id(tenant_id: str) -> str:
    """Tenant ids end up in cache keys, collection names and adapter names, so keep them tame."""
    if not isinstance(tenant_id, str) or not _TENANT_ID.match(tenant_id):
        raise InvalidRequest(
            "tenant_id must be 2-63 chars of lowercase letters, digits and hyphens",
            tenant_id=str(tenant_id),
        )
    return tenant_id


@dataclass(frozen=True)
class Principal:
    """The end user inside a tenant on whose behalf an agent acts."""

    user_id: str
    groups: frozenset[str] = field(default_factory=frozenset)

    @property
    def acl_principals(self) -> frozenset[str]:
        """Everything this user matches in a document ACL."""
        return frozenset({f"user:{self.user_id}", *(f"group:{g}" for g in self.groups)})


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    plan_id: str
    region: str
    status: str
    principal: Principal

    def __post_init__(self) -> None:
        validate_tenant_id(self.tenant_id)


def platform_scoped(reason: str) -> Callable[[F], F]:
    """Mark a data access function as deliberately not tenant-scoped.

    The tenancy lint rule requires `tenant_id` first on every data access function;
    this is the explicit, greppable opt-out for platform-level data (plans, the agent
    catalog) and audited cross-tenant jobs (billing sync, economics rollups).
    """

    def mark(fn: F) -> F:
        fn.__platform_scoped__ = reason  # type: ignore[attr-defined]
        return fn

    return mark

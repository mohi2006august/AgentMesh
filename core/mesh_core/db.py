"""Tenant-scoped SQL access (FR-03).

Two layers keep tenants apart:

1. **Statement guard (all engines).** `TenantTx` binds `:tenant_id` itself and refuses
   any statement that touches a tenant table without a `tenant_id = :tenant_id`
   predicate. `PlatformTx` refuses tenant tables unless opened with an explicit,
   enumerated `CrossTenant` reason.
2. **Row-level security (Postgres).** Every tenant table has a policy on
   `current_setting('app.tenant_id')`, set per transaction, so even a statement that
   slipped past layer 1 sees no foreign rows.

SQL is written once in the subset SQLite and Postgres share, with `:name` parameters.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from mesh_core.errors import ConfigError, IsolationViolation
from mesh_core.log import get_logger
from mesh_core.schema import MIGRATIONS, PLATFORM_TABLES, TENANT_TABLES, rls_statements
from mesh_core.tenancy import validate_tenant_id

log = get_logger(__name__)

Row = dict[str, Any]
Params = Mapping[str, Any] | None


class CrossTenant(StrEnum):
    """The only reasons a statement may read across tenants. Each use is logged."""

    AUTH_LOOKUP = "auth_lookup"  # resolve an API key to its tenant before a tenant is known
    BILLING_SYNC = "billing_sync"
    ECONOMICS_ROLLUP = "economics_rollup"
    SCHEDULER = "scheduler"
    OPERATOR_CONSOLE = "operator_console"  # platform operator listing tenants, audit, runs


# --- statement guard ----------------------------------------------------------------

_TABLE_REF = re.compile(r"\b(?:from|join|update|into)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_CTE_NAME = re.compile(r"\b([a-z_][a-z0-9_]*)\s+as\s*\(", re.IGNORECASE)
_UPSERT_UPDATE = re.compile(r"\bdo\s+update\b", re.IGNORECASE)
_SCOPED = re.compile(r"\btenant_id\s*=\s*:tenant_id\b", re.IGNORECASE)
_TENANT_PARAM = re.compile(r":tenant_id\b")


@dataclass(frozen=True)
class _Statement:
    verb: str
    tenant_tables: tuple[str, ...]
    platform_tables: tuple[str, ...]
    scoped_predicates: int
    binds_tenant: bool


@lru_cache(maxsize=2048)
def _analyze(sql: str) -> _Statement:
    text = _UPSERT_UPDATE.sub("do_update", sql)
    verb = text.lstrip().split(None, 1)[0].lower() if text.strip() else ""
    ctes = {name.lower() for name in _CTE_NAME.findall(text)}
    refs = [name.lower() for name in _TABLE_REF.findall(text) if name.lower() not in ctes]
    unknown = [r for r in refs if r not in TENANT_TABLES and r not in PLATFORM_TABLES]
    if unknown:
        raise IsolationViolation(
            "statement references a table that is not classified as tenant or platform scoped",
            tables=sorted(set(unknown)),
        )
    return _Statement(
        verb=verb,
        tenant_tables=tuple(r for r in refs if r in TENANT_TABLES),
        platform_tables=tuple(r for r in refs if r in PLATFORM_TABLES),
        scoped_predicates=len(_SCOPED.findall(text)),
        binds_tenant=bool(_TENANT_PARAM.search(text)),
    )


def _guard_tenant(sql: str) -> None:
    stmt = _analyze(sql)
    if stmt.verb not in ("select", "with", "insert", "update", "delete"):
        raise IsolationViolation("only DML is allowed in a tenant transaction", verb=stmt.verb)
    if stmt.verb != "select" and stmt.verb != "with" and not stmt.tenant_tables:
        raise IsolationViolation(
            "platform tables are read-only inside a tenant transaction",
            tables=list(stmt.platform_tables),
        )
    if stmt.verb == "insert":
        if not stmt.binds_tenant:
            raise IsolationViolation("INSERT into a tenant table must bind :tenant_id")
        # INSERT ... SELECT reads must be scoped too.
        if len(stmt.tenant_tables) - 1 > stmt.scoped_predicates:
            raise IsolationViolation("INSERT ... SELECT reads a tenant table without tenant scope")
        return
    if stmt.scoped_predicates < len(stmt.tenant_tables):
        raise IsolationViolation(
            "every tenant table in a statement needs a `tenant_id = :tenant_id` predicate",
            tables=list(stmt.tenant_tables),
        )


def _guard_platform(sql: str, reason: CrossTenant | None) -> None:
    stmt = _analyze(sql)
    if stmt.tenant_tables and reason is None:
        raise IsolationViolation(
            "platform transaction touched tenant tables without a CrossTenant reason",
            tables=list(stmt.tenant_tables),
        )


# --- engines ------------------------------------------------------------------------


class _Cursor(Protocol):
    @property
    def rowcount(self) -> int: ...

    def fetchall(self) -> list[Any]: ...


class _Engine(Protocol):
    dialect: str

    def transaction(
        self, role: str, tenant_id: str | None
    ) -> Any: ...  # context manager yielding an executor

    def close(self) -> None: ...


class _SqliteEngine:
    dialect = "sqlite"

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

    @contextmanager
    def transaction(self, role: str, tenant_id: str | None) -> Iterator[Any]:
        with self._lock:
            if self._depth:
                raise RuntimeError("nested database transactions are not supported")
            self._depth += 1
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._execute
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            finally:
                self._depth -= 1

    def _execute(self, sql: str, params: Mapping[str, Any]) -> _Cursor:
        return self._conn.execute(sql, params)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_NAMED_PARAM = re.compile(r"(?<![:\w]):([a-zA-Z_]\w*)")


@lru_cache(maxsize=2048)
def _to_pyformat(sql: str) -> str:
    return _NAMED_PARAM.sub(r"%(\1)s", sql.replace("%", "%%"))


class _PostgresEngine:
    """psycopg pools: `app` connects as an RLS-bound role, `platform` as a BYPASSRLS role."""

    dialect = "postgres"

    def __init__(self, url: str, platform_url: str) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError("Postgres needs `pip install agentmesh[postgres]`") from exc
        kwargs = {"row_factory": dict_row, "autocommit": False}
        self._pools = {
            "app": ConnectionPool(url, kwargs=kwargs, min_size=1, max_size=10, open=True),
            "platform": ConnectionPool(
                platform_url, kwargs=kwargs, min_size=1, max_size=4, open=True
            ),
        }

    @contextmanager
    def transaction(self, role: str, tenant_id: str | None) -> Iterator[Any]:
        with self._pools[role].connection() as conn, conn.transaction():
            if tenant_id is not None:
                # Transaction-local: cleared on commit/rollback, so pooled connections never leak scope.
                conn.execute("SELECT set_config('app.tenant_id', %s, true)", [tenant_id])

            def execute(sql: str, params: Mapping[str, Any]) -> _Cursor:
                cursor: _Cursor = conn.execute(_to_pyformat(sql), params)
                return cursor

            yield execute

    def close(self) -> None:
        for pool in self._pools.values():
            pool.close()


# --- transactions -------------------------------------------------------------------


class _Tx:
    def __init__(self, execute: Any) -> None:
        self._execute = execute

    def _prepare(self, sql: str, params: Params) -> dict[str, Any]:
        raise NotImplementedError

    def execute(self, sql: str, params: Params = None) -> int:
        cursor = self._execute(sql, self._prepare(sql, params))
        return int(cursor.rowcount)

    def query(self, sql: str, params: Params = None) -> list[Row]:
        cursor = self._execute(sql, self._prepare(sql, params))
        return [dict(row) for row in cursor.fetchall()]

    def query_one(self, sql: str, params: Params = None) -> Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None


class TenantTx(_Tx):
    """A transaction bound to exactly one tenant. `:tenant_id` is always bound from here."""

    def __init__(self, execute: Any, tenant_id: str) -> None:
        super().__init__(execute)
        self.tenant_id = tenant_id

    def _prepare(self, sql: str, params: Params) -> dict[str, Any]:
        _guard_tenant(sql)
        bound = dict(params or {})
        if bound.get("tenant_id", self.tenant_id) != self.tenant_id:
            raise IsolationViolation(
                "statement bound a different tenant_id than its transaction",
                tx_tenant=self.tenant_id,
            )
        bound["tenant_id"] = self.tenant_id
        return bound


class PlatformTx(_Tx):
    """Platform-scoped data (plans, agent catalog) and audited cross-tenant jobs."""

    def __init__(self, execute: Any, cross_tenant: CrossTenant | None) -> None:
        super().__init__(execute)
        self.cross_tenant = cross_tenant

    def _prepare(self, sql: str, params: Params) -> dict[str, Any]:
        _guard_platform(sql, self.cross_tenant)
        return dict(params or {})


class Database:
    def __init__(self, url: str, platform_url: str = "") -> None:
        self.url = url
        if url.startswith("sqlite:///"):
            self._engine: _Engine = _SqliteEngine(url.removeprefix("sqlite:///"))
        elif url.startswith(("postgresql://", "postgres://")):
            self._engine = _PostgresEngine(url, platform_url or url)
        else:
            raise ConfigError("DATABASE_URL must be sqlite:/// or postgresql://")

    @classmethod
    def in_memory(cls) -> Database:
        db = cls("sqlite:///:memory:")
        db.migrate()
        return db

    @property
    def dialect(self) -> str:
        return self._engine.dialect

    @contextmanager
    def tenant_tx(self, tenant_id: str) -> Iterator[TenantTx]:
        validate_tenant_id(tenant_id)
        with self._engine.transaction("app", tenant_id) as execute:
            yield TenantTx(execute, tenant_id)

    @contextmanager
    def platform_tx(self, cross_tenant: CrossTenant | None = None) -> Iterator[PlatformTx]:
        if cross_tenant is not None:
            log.debug("db.cross_tenant_tx", reason=cross_tenant.value)
        with self._engine.transaction("platform", None) as execute:
            yield PlatformTx(execute, cross_tenant)

    def migrate(self) -> None:
        """Apply pending migrations, then (Postgres) re-assert RLS on every tenant table."""
        with self._engine.transaction("platform", None) as execute:
            execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, name TEXT NOT NULL)",
                {},
            )
            applied = {
                row["version"]
                for row in execute("SELECT version FROM schema_migrations", {}).fetchall()
            }
            for migration in MIGRATIONS:
                if migration.version in applied:
                    continue
                for statement in migration.statements:
                    execute(statement, {})
                execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (:version, :name)",
                    {"version": migration.version, "name": migration.name},
                )
                log.info("db.migrated", version=migration.version, name=migration.name)
            if self.dialect == "postgres":
                for statement in rls_statements():
                    execute(statement, {})

    def close(self) -> None:
        self._engine.close()


def filters(*clauses: tuple[str, str, Any]) -> tuple[str, dict[str, Any]]:
    """Build optional `AND ...` filters from `(sql, param_name, value)`, skipping None values.

    Only the fixed SQL fragments given by the caller are concatenated; values are always
    bound. (`:p IS NULL OR col = :p` is avoided because Postgres cannot type a bare NULL.)
    """
    sql, params = "", {}
    for clause, name, value in clauses:
        if value is not None:
            sql += f" AND {clause}"
            params[name] = value
    return sql, params


def dumps(value: Any) -> str:
    """JSON columns are TEXT on both engines."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)

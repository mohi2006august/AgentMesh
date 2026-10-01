"""Schema migrations and the tenant/platform classification of every table.

DDL uses only what SQLite and Postgres share: TEXT ids, DOUBLE PRECISION epoch
timestamps, JSON stored as TEXT. Every table must be classified here; the statement
guard in `db.py` rejects SQL that references an unclassified table.

A tenant table always has a `tenant_id` column, which is what both the statement guard
and the Postgres RLS policy key on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Postgres role the data plane connects as. It has no BYPASSRLS, so RLS always applies.
APP_ROLE = "mesh_app"


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]
    tenant_tables: tuple[str, ...] = field(default_factory=tuple)
    platform_tables: tuple[str, ...] = field(default_factory=tuple)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="tenancy",
        tenant_tables=("tenants", "api_keys", "audit_log", "secrets"),
        platform_tables=("plans", "schema_migrations"),
        statements=(
            """
            CREATE TABLE plans (
              plan_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              price_usd_month DOUBLE PRECISION NOT NULL,
              weight INTEGER NOT NULL,
              tokens_per_minute INTEGER NOT NULL,
              monthly_run_quota INTEGER NOT NULL,
              monthly_cogs_cap_usd DOUBLE PRECISION NOT NULL,
              usage_prices TEXT NOT NULL DEFAULT '{}',
              created_at DOUBLE PRECISION NOT NULL
            )
            """,
            """
            CREATE TABLE tenants (
              tenant_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              plan_id TEXT NOT NULL REFERENCES plans(plan_id),
              region TEXT NOT NULL,
              status TEXT NOT NULL,
              stripe_customer_id TEXT,
              routing_policy TEXT NOT NULL DEFAULT '{}',
              quota_overrides TEXT NOT NULL DEFAULT '{}',
              created_at DOUBLE PRECISION NOT NULL,
              converted_at DOUBLE PRECISION
            )
            """,
            """
            CREATE TABLE api_keys (
              key_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
              key_hash TEXT NOT NULL UNIQUE,
              prefix TEXT NOT NULL,
              role TEXT NOT NULL,
              created_at DOUBLE PRECISION NOT NULL,
              revoked_at DOUBLE PRECISION
            )
            """,
            "CREATE INDEX api_keys_tenant ON api_keys (tenant_id)",
            """
            CREATE TABLE audit_log (
              audit_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              ts DOUBLE PRECISION NOT NULL,
              actor TEXT NOT NULL,
              action TEXT NOT NULL,
              resource TEXT NOT NULL,
              decision TEXT NOT NULL,
              run_id TEXT,
              region TEXT NOT NULL,
              detail TEXT NOT NULL DEFAULT '{}',
              prev_hash TEXT NOT NULL,
              hash TEXT NOT NULL,
              UNIQUE (tenant_id, seq)
            )
            """,
            "CREATE INDEX audit_log_tenant_ts ON audit_log (tenant_id, ts)",
            """
            CREATE TABLE secrets (
              tenant_id TEXT NOT NULL,
              name TEXT NOT NULL,
              ciphertext TEXT NOT NULL,
              updated_at DOUBLE PRECISION NOT NULL,
              PRIMARY KEY (tenant_id, name)
            )
            """,
        ),
    ),
    Migration(
        version=2,
        name="metering",
        tenant_tables=("meter_events",),
        statements=(
            """
            CREATE TABLE meter_events (
              event_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              ts DOUBLE PRECISION NOT NULL,
              month TEXT NOT NULL,
              kind TEXT NOT NULL,
              quantity DOUBLE PRECISION NOT NULL,
              unit TEXT NOT NULL,
              cost_usd DOUBLE PRECISION NOT NULL,
              billable INTEGER NOT NULL,
              agent TEXT,
              run_id TEXT,
              model TEXT,
              attrs TEXT NOT NULL DEFAULT '{}'
            )
            """,
            "CREATE INDEX meter_events_tenant_month ON meter_events (tenant_id, month, kind)",
            "CREATE INDEX meter_events_run ON meter_events (tenant_id, run_id)",
            "CREATE INDEX meter_events_ts ON meter_events (ts)",
        ),
    ),
    Migration(
        version=3,
        name="mcp_gateway",
        tenant_tables=("connections", "tool_policies", "approvals", "oauth_states"),
        platform_tables=("mcp_servers",),
        statements=(
            """
            CREATE TABLE mcp_servers (
              name TEXT PRIMARY KEY,
              definition TEXT NOT NULL,
              created_at DOUBLE PRECISION NOT NULL,
              updated_at DOUBLE PRECISION NOT NULL
            )
            """,
            """
            CREATE TABLE connections (
              tenant_id TEXT NOT NULL,
              server TEXT NOT NULL,
              status TEXT NOT NULL,
              auth_type TEXT NOT NULL,
              scopes TEXT NOT NULL DEFAULT '[]',
              expires_at DOUBLE PRECISION,
              created_by TEXT NOT NULL,
              created_at DOUBLE PRECISION NOT NULL,
              updated_at DOUBLE PRECISION NOT NULL,
              PRIMARY KEY (tenant_id, server)
            )
            """,
            """
            CREATE TABLE tool_policies (
              tenant_id TEXT NOT NULL,
              server TEXT NOT NULL,
              tool TEXT NOT NULL,
              action TEXT NOT NULL,
              updated_at DOUBLE PRECISION NOT NULL,
              PRIMARY KEY (tenant_id, server, tool)
            )
            """,
            """
            CREATE TABLE approvals (
              approval_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              run_id TEXT NOT NULL,
              step_key TEXT NOT NULL,
              agent TEXT NOT NULL,
              user_id TEXT NOT NULL,
              server TEXT NOT NULL,
              tool TEXT NOT NULL,
              arguments TEXT NOT NULL,
              arguments_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              requested_at DOUBLE PRECISION NOT NULL,
              decided_at DOUBLE PRECISION,
              decided_by TEXT,
              reason TEXT,
              UNIQUE (tenant_id, run_id, step_key)
            )
            """,
            "CREATE INDEX approvals_status ON approvals (tenant_id, status)",
            """
            CREATE TABLE oauth_states (
              tenant_id TEXT NOT NULL,
              state TEXT NOT NULL,
              server TEXT NOT NULL,
              code_verifier TEXT NOT NULL,
              redirect_uri TEXT NOT NULL,
              created_by TEXT NOT NULL,
              created_at DOUBLE PRECISION NOT NULL,
              PRIMARY KEY (tenant_id, state)
            )
            """,
        ),
    ),
    Migration(
        version=4,
        name="retrieval",
        tenant_tables=("knowledge_sources", "documents", "chunks"),
        statements=(
            """
            CREATE TABLE knowledge_sources (
              tenant_id TEXT NOT NULL,
              source TEXT NOT NULL,
              kind TEXT NOT NULL,
              config TEXT NOT NULL DEFAULT '{}',
              created_at DOUBLE PRECISION NOT NULL,
              synced_at DOUBLE PRECISION,
              PRIMARY KEY (tenant_id, source)
            )
            """,
            """
            CREATE TABLE documents (
              tenant_id TEXT NOT NULL,
              source TEXT NOT NULL,
              doc_id TEXT NOT NULL,
              title TEXT NOT NULL,
              uri TEXT NOT NULL DEFAULT '',
              acl TEXT NOT NULL DEFAULT '[]',
              content_hash TEXT NOT NULL,
              updated_at DOUBLE PRECISION NOT NULL,
              PRIMARY KEY (tenant_id, source, doc_id)
            )
            """,
            """
            CREATE TABLE chunks (
              tenant_id TEXT NOT NULL,
              chunk_id TEXT NOT NULL,
              source TEXT NOT NULL,
              doc_id TEXT NOT NULL,
              ordinal INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding TEXT NOT NULL,
              PRIMARY KEY (tenant_id, chunk_id)
            )
            """,
            "CREATE INDEX chunks_source ON chunks (tenant_id, source)",
        ),
    ),
)


def _collect(attr: str) -> frozenset[str]:
    return frozenset(name for migration in MIGRATIONS for name in getattr(migration, attr))


TENANT_TABLES: frozenset[str] = _collect("tenant_tables")
PLATFORM_TABLES: frozenset[str] = _collect("platform_tables")


def rls_statements() -> list[str]:
    """Postgres only: force row-level security on every tenant table.

    `FORCE` makes the policy apply to the table owner too. The policy reads the
    transaction-local `app.tenant_id` set by `Database.tenant_tx`; with it unset the
    comparison is against NULL and no row is visible.
    """
    statements: list[str] = []
    for table in sorted(TENANT_TABLES):
        statements += [
            f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
            f"DROP POLICY IF EXISTS tenant_isolation ON {table}",
            f"""
            CREATE POLICY tenant_isolation ON {table}
              USING (tenant_id = current_setting('app.tenant_id', true))
              WITH CHECK (tenant_id = current_setting('app.tenant_id', true))
            """,
            _grant(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}"),
        ]
    for table in sorted(PLATFORM_TABLES):
        statements.append(_grant(f"GRANT SELECT ON {table} TO {APP_ROLE}"))
    return statements


def _grant(grant: str) -> str:
    return (
        "DO $$ BEGIN "
        f"IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN {grant}; END IF; "
        "END $$"
    )

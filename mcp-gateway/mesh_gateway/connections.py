"""Tenant connections to MCP servers, and the OAuth 2.1 flow that creates them (FR-02, FR-03).

A connection records that a tenant installed a server; the credential itself lives in the
tenant's secret store and is only ever read by the gateway at call time.

OAuth uses authorization code + PKCE (S256) with a `resource` indicator naming the MCP
server. The `state` parameter carries the tenant id and an HMAC over it, so the callback
(which arrives from a browser, without tenant credentials) can be routed to the right
tenant without any cross-tenant lookup.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx2

from mesh_core.clock import Clock, SystemClock
from mesh_core.db import Database, Row, dumps, loads
from mesh_core.errors import InvalidRequest, Unauthorized, Unavailable
from mesh_core.log import get_logger
from mesh_core.secret_store import SecretStore
from mesh_core.tenancy import validate_tenant_id
from mesh_gateway.registry import McpServerDef, check_egress

log = get_logger(__name__)

STATE_TTL_S = 600
REFRESH_MARGIN_S = 60


def secret_name(server: str, kind: str) -> str:
    return f"conn/{server}/{kind}"


@dataclass(frozen=True)
class Connection:
    tenant_id: str
    server: str
    status: str  # pending | active | error | revoked
    auth_type: str
    scopes: tuple[str, ...]
    expires_at: float | None
    created_by: str
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row: Row) -> Connection:
        return cls(**{**row, "scopes": tuple(loads(row["scopes"], []))})


class ConnectionStore:
    def __init__(
        self, db: Database, secrets_store: SecretStore, clock: Clock | None = None
    ) -> None:
        self._db = db
        self._secrets = secrets_store
        self._clock = clock or SystemClock()

    def upsert(
        self,
        tenant_id: str,
        server: str,
        *,
        status: str,
        auth_type: str,
        created_by: str,
        scopes: tuple[str, ...] = (),
        expires_at: float | None = None,
    ) -> Connection:
        now = self._clock.now()
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                """
                INSERT INTO connections
                  (tenant_id, server, status, auth_type, scopes, expires_at, created_by, created_at, updated_at)
                VALUES (:tenant_id, :server, :status, :auth_type, :scopes, :expires_at, :created_by, :now, :now)
                ON CONFLICT (tenant_id, server) DO UPDATE SET
                  status = excluded.status, auth_type = excluded.auth_type, scopes = excluded.scopes,
                  expires_at = excluded.expires_at, updated_at = excluded.updated_at
                """,
                {
                    "server": server,
                    "status": status,
                    "auth_type": auth_type,
                    "scopes": dumps(list(scopes)),
                    "expires_at": expires_at,
                    "created_by": created_by,
                    "now": now,
                },
            )
        connection = self.get(tenant_id, server)
        assert connection is not None
        return connection

    def get(self, tenant_id: str, server: str) -> Connection | None:
        with self._db.tenant_tx(tenant_id) as tx:
            row = tx.query_one(
                "SELECT * FROM connections WHERE tenant_id = :tenant_id AND server = :server",
                {"server": server},
            )
        return Connection.from_row(row) if row else None

    def list(self, tenant_id: str) -> list[Connection]:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query(
                "SELECT * FROM connections WHERE tenant_id = :tenant_id ORDER BY server"
            )
        return [Connection.from_row(row) for row in rows]

    def set_status(self, tenant_id: str, server: str, status: str) -> None:
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                "UPDATE connections SET status = :status, updated_at = :now "
                "WHERE tenant_id = :tenant_id AND server = :server",
                {"status": status, "now": self._clock.now(), "server": server},
            )

    def connect_token(
        self, tenant_id: str, server: McpServerDef, token: str, *, created_by: str
    ) -> Connection:
        """Install a server with a static bearer token (API key style auth)."""
        if server.auth != "bearer":
            raise InvalidRequest(
                "this server does not take a static token", server=server.name, auth=server.auth
            )
        if not token:
            raise InvalidRequest("token must not be empty")
        self._secrets.put(tenant_id, secret_name(server.name, "token"), token)
        return self.upsert(
            tenant_id, server.name, status="active", auth_type="bearer", created_by=created_by
        )

    def connect_public(
        self, tenant_id: str, server: McpServerDef, *, created_by: str
    ) -> Connection:
        if server.auth != "none":
            raise InvalidRequest(
                "this server requires credentials", server=server.name, auth=server.auth
            )
        return self.upsert(
            tenant_id, server.name, status="active", auth_type="none", created_by=created_by
        )

    def save_oauth_tokens(
        self, tenant_id: str, server: str, tokens: dict[str, Any], *, created_by: str
    ) -> Connection:
        access = tokens.get("access_token")
        if not isinstance(access, str) or not access:
            raise Unauthorized("token endpoint returned no access_token", server=server)
        self._secrets.put(tenant_id, secret_name(server, "access_token"), access)
        if isinstance(tokens.get("refresh_token"), str):
            self._secrets.put(
                tenant_id, secret_name(server, "refresh_token"), tokens["refresh_token"]
            )
        expires_in = tokens.get("expires_in")
        expires_at = (
            self._clock.now() + float(expires_in) if isinstance(expires_in, (int, float)) else None
        )
        scopes = tuple(str(tokens.get("scope", "")).split()) if tokens.get("scope") else ()
        return self.upsert(
            tenant_id, server, status="active", auth_type="oauth", created_by=created_by,
            scopes=scopes, expires_at=expires_at,
        )  # fmt: skip

    def revoke(self, tenant_id: str, server: str) -> None:
        for kind in ("token", "access_token", "refresh_token"):
            self._secrets.delete(tenant_id, secret_name(server, kind))
        self.set_status(tenant_id, server, "revoked")

    def secret(self, tenant_id: str, server: str, kind: str) -> str | None:
        return self._secrets.get(tenant_id, secret_name(server, kind))


# --- OAuth ------------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    return verifier, _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _client_secret(server: str) -> str | None:
    """Confidential OAuth clients: MESH_OAUTH_CLIENT_SECRET_<SERVER> (env only, never stored)."""
    return os.environ.get(f"MESH_OAUTH_CLIENT_SECRET_{server.upper().replace('-', '_')}") or None


class OAuthFlow:
    def __init__(
        self,
        db: Database,
        connections: ConnectionStore,
        signing_key: str,
        *,
        http: httpx2.AsyncClient | None = None,
        clock: Clock | None = None,
        platform_allowlist: tuple[str, ...] = (),
    ) -> None:
        self._db = db
        self._connections = connections
        self._key = hashlib.sha256(b"mesh/oauth-state/v1" + signing_key.encode()).digest()
        self._http = http
        self._clock = clock or SystemClock()
        self._allowlist = platform_allowlist

    def _sign(self, tenant_id: str, nonce: str) -> str:
        return hmac.new(self._key, f"{tenant_id}.{nonce}".encode(), hashlib.sha256).hexdigest()[:32]

    def _parse_state(self, state: str) -> str:
        tenant_id, _, rest = state.partition(".")
        nonce, _, signature = rest.partition(".")
        if not (tenant_id and nonce and signature) or not hmac.compare_digest(
            signature, self._sign(tenant_id, nonce)
        ):
            raise Unauthorized("invalid OAuth state")
        return validate_tenant_id(tenant_id)

    def start(
        self, tenant_id: str, server: McpServerDef, *, redirect_uri: str, created_by: str
    ) -> str:
        if server.auth != "oauth" or server.oauth is None:
            raise InvalidRequest("server does not use OAuth", server=server.name)
        check_egress(server, server.oauth.authorize_url, self._allowlist)
        nonce = secrets.token_urlsafe(16)
        state = f"{tenant_id}.{nonce}.{self._sign(tenant_id, nonce)}"
        verifier, challenge = pkce_pair()
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                """
                INSERT INTO oauth_states
                  (tenant_id, state, server, code_verifier, redirect_uri, created_by, created_at)
                VALUES (:tenant_id, :state, :server, :verifier, :redirect_uri, :created_by, :now)
                """,
                {
                    "state": state,
                    "server": server.name,
                    "verifier": verifier,
                    "redirect_uri": redirect_uri,
                    "created_by": created_by,
                    "now": self._clock.now(),
                },
            )
        self._connections.upsert(
            tenant_id, server.name, status="pending", auth_type="oauth", created_by=created_by
        )
        query = {
            "response_type": "code",
            "client_id": server.oauth.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": server.endpoint,
        }
        if server.oauth.scopes:
            query["scope"] = " ".join(server.oauth.scopes)
        return f"{server.oauth.authorize_url}?{urlencode(query)}"

    async def complete(self, state: str, code: str, server_lookup: Any) -> Connection:
        """Exchange the authorization code. `server_lookup(name)` returns the McpServerDef."""
        tenant_id = self._parse_state(state)
        row = self._consume_state(tenant_id, state)
        if row is None or self._clock.now() - row["created_at"] > STATE_TTL_S:
            raise Unauthorized("OAuth state expired or already used")
        server: McpServerDef = server_lookup(row["server"])
        assert server.oauth is not None
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": row["redirect_uri"],
            "client_id": server.oauth.client_id,
            "code_verifier": row["code_verifier"],
            "resource": server.endpoint,
        }
        tokens = await self._token_request(server, form)
        connection = self._connections.save_oauth_tokens(
            tenant_id, server.name, tokens, created_by=row["created_by"]
        )
        log.info("oauth.connected", tenant_id=tenant_id, server=server.name)
        return connection

    def _consume_state(self, tenant_id: str, state: str) -> Row | None:
        """Single use: the state row is deleted in the same transaction that reads it."""
        with self._db.tenant_tx(tenant_id) as tx:
            row = tx.query_one(
                "SELECT * FROM oauth_states WHERE tenant_id = :tenant_id AND state = :state",
                {"state": state},
            )
            tx.execute(
                "DELETE FROM oauth_states WHERE tenant_id = :tenant_id AND state = :state",
                {"state": state},
            )
        return row

    async def access_token(self, tenant_id: str, server: McpServerDef) -> str | None:
        """Current access token, refreshed first when it is about to expire."""
        connection = self._connections.get(tenant_id, server.name)
        if connection is None or connection.status != "active":
            return None
        expiring = (
            connection.expires_at is not None
            and connection.expires_at - self._clock.now() < REFRESH_MARGIN_S
        )
        if expiring:
            refresh = self._connections.secret(tenant_id, server.name, "refresh_token")
            if refresh is None or server.oauth is None:
                self._connections.set_status(tenant_id, server.name, "error")
                return None
            form = {
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": server.oauth.client_id,
                "resource": server.endpoint,
            }
            try:
                tokens = await self._token_request(server, form)
            except (Unauthorized, Unavailable) as exc:
                log.warning(
                    "oauth.refresh_failed",
                    tenant_id=tenant_id,
                    server=server.name,
                    reason=exc.message,
                )
                self._connections.set_status(tenant_id, server.name, "error")
                return None
            self._connections.save_oauth_tokens(
                tenant_id, server.name, tokens, created_by=connection.created_by
            )
        return self._connections.secret(tenant_id, server.name, "access_token")

    async def _token_request(self, server: McpServerDef, form: dict[str, str]) -> dict[str, Any]:
        assert server.oauth is not None
        check_egress(server, server.oauth.token_url, self._allowlist)
        client_secret = _client_secret(server.name)
        if client_secret:
            form = {**form, "client_secret": client_secret}
        http = self._http or httpx2.AsyncClient(timeout=15.0)
        try:
            response = await http.post(
                server.oauth.token_url, data=form, headers={"Accept": "application/json"}
            )
        except httpx2.HTTPError as exc:
            raise Unavailable("token endpoint unreachable", server=server.name) from exc
        finally:
            if self._http is None:
                await http.aclose()
        if response.status_code >= 500:
            raise Unavailable(
                "token endpoint error", server=server.name, status=response.status_code
            )
        if response.status_code != 200:
            raise Unauthorized(
                "token request rejected", server=server.name, status=response.status_code
            )
        payload: dict[str, Any] = response.json()
        return payload


def connection_required(server: McpServerDef) -> bool:
    return server.auth != "none"

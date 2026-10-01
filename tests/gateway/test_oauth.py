from __future__ import annotations

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest

from mesh_core.clock import FakeClock
from mesh_core.db import Database
from mesh_core.errors import PolicyDenied, Unauthorized
from mesh_core.secret_store import DbSecretStore
from mesh_gateway import ConnectionStore, McpServerDef, OAuthFlow
from mesh_gateway.connections import secret_name
from mesh_gateway.registry import OAuthConfig
from tests.conftest import ROOT_KEY

DRIVE = McpServerDef(
    name="drive",
    transport="http",
    url="https://mcp.drive.example.com/mcp",
    auth="oauth",
    oauth=OAuthConfig(
        authorize_url="https://auth.example.com/authorize",
        token_url="https://auth.example.com/token",
        client_id="mesh-client",
        scopes=("files.read",),
    ),
    egress_allowlist=(".example.com",),
)


class AuthServer:
    """Token endpoint that checks PKCE and issues rotating tokens."""

    def __init__(self) -> None:
        self.challenges: dict[str, str] = {}
        self.issued = 0
        self.requests: list[dict[str, str]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        form = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
        self.requests.append(form)
        if form["grant_type"] == "authorization_code":
            expected = self.challenges.get(form["code"])
            digest = hashlib.sha256(form["code_verifier"].encode()).digest()
            actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            if expected != actual:
                return httpx2.Response(400, json={"error": "invalid_grant"})
        elif form["refresh_token"] != f"refresh-{self.issued}":
            return httpx2.Response(400, json={"error": "invalid_grant"})
        self.issued += 1
        return httpx2.Response(
            200,
            json={
                "access_token": f"access-{self.issued}",
                "refresh_token": f"refresh-{self.issued}",
                "expires_in": 3600,
                "scope": "files.read",
            },
        )


@pytest.fixture
def setup(db: Database, clock: FakeClock) -> tuple[OAuthFlow, ConnectionStore, AuthServer]:
    server = AuthServer()
    connections = ConnectionStore(db, DbSecretStore(db, ROOT_KEY), clock)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(server))
    flow = OAuthFlow(db, connections, ROOT_KEY, http=http, clock=clock)
    return flow, connections, server


def _authorize(flow: OAuthFlow, server: AuthServer, tenant_id: str = "acme") -> tuple[str, str]:
    url = flow.start(
        tenant_id, DRIVE, redirect_uri="https://console.example.com/cb", created_by="user:admin"
    )
    query = {key: values[0] for key, values in parse_qs(urlparse(url).query).items()}
    code = f"code-{len(server.challenges)}"
    server.challenges[code] = query["code_challenge"]
    return query["state"], code


async def test_authorization_code_flow_with_pkce(
    setup: tuple[OAuthFlow, ConnectionStore, AuthServer],
) -> None:
    flow, connections, server = setup
    url = flow.start(
        "acme", DRIVE, redirect_uri="https://console.example.com/cb", created_by="user:admin"
    )
    query = parse_qs(urlparse(url).query)
    assert url.startswith("https://auth.example.com/authorize?")
    assert query["code_challenge_method"] == ["S256"]
    assert query["resource"] == ["https://mcp.drive.example.com/mcp"]
    assert query["scope"] == ["files.read"]
    assert query["state"][0].startswith("acme.")
    assert connections.get("acme", "drive").status == "pending"  # type: ignore[union-attr]

    server.challenges["the-code"] = query["code_challenge"][0]
    connection = await flow.complete(query["state"][0], "the-code", lambda name: DRIVE)
    assert connection.status == "active" and connection.scopes == ("files.read",)
    assert await flow.access_token("acme", DRIVE) == "access-1"
    assert connections.secret("acme", "drive", "refresh_token") == "refresh-1"


async def test_state_is_single_use(setup: tuple[OAuthFlow, ConnectionStore, AuthServer]) -> None:
    flow, _, server = setup
    state, code = _authorize(flow, server)
    await flow.complete(state, code, lambda name: DRIVE)
    with pytest.raises(Unauthorized):
        await flow.complete(state, code, lambda name: DRIVE)


async def test_tampered_state_cannot_redirect_tokens_to_another_tenant(
    setup: tuple[OAuthFlow, ConnectionStore, AuthServer],
) -> None:
    flow, connections, server = setup
    state, code = _authorize(flow, server, "acme")
    forged = "globex" + state[len("acme") :]
    with pytest.raises(Unauthorized):
        await flow.complete(forged, code, lambda name: DRIVE)
    assert connections.get("globex", "drive") is None


async def test_expired_state_is_rejected(
    setup: tuple[OAuthFlow, ConnectionStore, AuthServer], clock: FakeClock
) -> None:
    flow, _, server = setup
    state, code = _authorize(flow, server)
    clock.advance(601)
    with pytest.raises(Unauthorized):
        await flow.complete(state, code, lambda name: DRIVE)


async def test_tokens_refresh_before_expiry(
    setup: tuple[OAuthFlow, ConnectionStore, AuthServer], clock: FakeClock
) -> None:
    flow, _, server = setup
    state, code = _authorize(flow, server)
    await flow.complete(state, code, lambda name: DRIVE)

    clock.advance(3600 - 30)
    assert await flow.access_token("acme", DRIVE) == "access-2"
    assert server.requests[-1]["grant_type"] == "refresh_token"
    assert await flow.access_token("acme", DRIVE) == "access-2"  # not refreshed again


async def test_failed_refresh_marks_the_connection_for_reauthorisation(
    setup: tuple[OAuthFlow, ConnectionStore, AuthServer], clock: FakeClock
) -> None:
    flow, connections, server = setup
    state, code = _authorize(flow, server)
    await flow.complete(state, code, lambda name: DRIVE)
    connections._secrets.put("acme", secret_name("drive", "refresh_token"), "revoked-upstream")
    clock.advance(3600)
    assert await flow.access_token("acme", DRIVE) is None
    assert connections.get("acme", "drive").status == "error"  # type: ignore[union-attr]


async def test_oauth_endpoints_must_be_allow_listed(
    setup: tuple[OAuthFlow, ConnectionStore, AuthServer],
) -> None:
    flow, _, _ = setup
    rogue = DRIVE.model_copy(update={"egress_allowlist": ("drive.example.com",)})
    with pytest.raises(PolicyDenied):
        flow.start(
            "acme", rogue, redirect_uri="https://console.example.com/cb", created_by="user:admin"
        )

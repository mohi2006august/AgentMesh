"""Per-tenant secret storage (FR-03): OAuth tokens and API keys for tenant connections.

`DbSecretStore` encrypts with AES-256-GCM under a key derived per tenant from the root
key, and binds `tenant_id:name` as associated data. A ciphertext copied into another
tenant's row (or renamed) fails authentication instead of decrypting.

Secrets are injected by the MCP gateway at call time and never enter model context.
"""

from __future__ import annotations

import base64
import os
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from mesh_core.clock import Clock, SystemClock
from mesh_core.db import Database
from mesh_core.errors import ConfigError, IsolationViolation

_NONCE_BYTES = 12


class SecretStore(Protocol):
    def put(self, tenant_id: str, name: str, value: str) -> None: ...

    def get(self, tenant_id: str, name: str) -> str | None: ...

    def delete(self, tenant_id: str, name: str) -> None: ...

    def names(self, tenant_id: str) -> list[str]: ...


class DbSecretStore:
    def __init__(self, db: Database, root_key: str, clock: Clock | None = None) -> None:
        if len(root_key) < 32:
            raise ConfigError("MESH_SECRET_KEY must be at least 32 characters")
        self._db = db
        self._root = root_key.encode("utf-8")
        self._clock = clock or SystemClock()

    def _cipher(self, tenant_id: str) -> AESGCM:
        key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"mesh/tenant-secrets/v1/" + tenant_id.encode("utf-8"),
        ).derive(self._root)
        return AESGCM(key)

    @staticmethod
    def _aad(tenant_id: str, name: str) -> bytes:
        return f"{tenant_id}:{name}".encode()

    def put(self, tenant_id: str, name: str, value: str) -> None:
        nonce = os.urandom(_NONCE_BYTES)
        sealed = self._cipher(tenant_id).encrypt(
            nonce, value.encode("utf-8"), self._aad(tenant_id, name)
        )
        ciphertext = base64.b64encode(nonce + sealed).decode("ascii")
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                """
                INSERT INTO secrets (tenant_id, name, ciphertext, updated_at)
                VALUES (:tenant_id, :name, :ciphertext, :now)
                ON CONFLICT (tenant_id, name)
                DO UPDATE SET ciphertext = excluded.ciphertext, updated_at = excluded.updated_at
                """,
                {"name": name, "ciphertext": ciphertext, "now": self._clock.now()},
            )

    def get(self, tenant_id: str, name: str) -> str | None:
        with self._db.tenant_tx(tenant_id) as tx:
            row = tx.query_one(
                "SELECT ciphertext FROM secrets WHERE tenant_id = :tenant_id AND name = :name",
                {"name": name},
            )
        if row is None:
            return None
        raw = base64.b64decode(row["ciphertext"])
        try:
            plain = self._cipher(tenant_id).decrypt(
                raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], self._aad(tenant_id, name)
            )
        except InvalidTag as exc:
            raise IsolationViolation(
                "secret failed authentication for this tenant", tenant_id=tenant_id, name=name
            ) from exc
        return plain.decode("utf-8")

    def delete(self, tenant_id: str, name: str) -> None:
        with self._db.tenant_tx(tenant_id) as tx:
            tx.execute(
                "DELETE FROM secrets WHERE tenant_id = :tenant_id AND name = :name", {"name": name}
            )

    def names(self, tenant_id: str) -> list[str]:
        with self._db.tenant_tx(tenant_id) as tx:
            rows = tx.query("SELECT name FROM secrets WHERE tenant_id = :tenant_id ORDER BY name")
        return [row["name"] for row in rows]

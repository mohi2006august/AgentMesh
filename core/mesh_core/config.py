"""Configuration. Secrets come only from environment variables (or a local `.env`).

With nothing set the local profile applies: SQLite under `MESH_DATA_DIR`, in-process
stores and the mock model. In dev, missing keys are generated once into
`MESH_DATA_DIR/dev-keys.json`; in prod a missing key is a hard `ConfigError`.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from mesh_core.errors import ConfigError

REGIONS = ("us", "eu")


@dataclass(frozen=True)
class Settings:
    env: str = "dev"
    region: str = "us"
    data_dir: Path = Path(".mesh")
    admin_key: str = ""
    secret_key: str = ""
    database_url: str = ""
    database_platform_url: str = ""
    redis_url: str = ""
    qdrant_url: str = ""
    kafka_brokers: str = ""
    clickhouse_url: str = ""
    vault_addr: str = ""
    vault_token: str = ""
    temporal_addr: str = ""
    anthropic_api_key: str = ""
    vllm_endpoints: tuple[str, ...] = field(default_factory=tuple)
    stripe_key: str = ""
    public_url: str = "http://localhost:8080"

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'mesh.db').as_posix()}"


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal `.env` reader: KEY=VALUE lines, `#` comments, optional quotes."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip("\"'")
        values[key.strip()] = value
    return values


def _dev_keys(data_dir: Path) -> dict[str, str]:
    path = data_dir / "dev-keys.json"
    if path.exists():
        loaded: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    keys = {
        "admin_key": f"mesh_admin_{secrets.token_urlsafe(24)}",
        "secret_key": secrets.token_urlsafe(32),
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keys, indent=2), encoding="utf-8")
    return keys


def load_settings(
    environ: Mapping[str, str] | None = None, *, dotenv: Path | None = Path(".env")
) -> Settings:
    env: dict[str, str] = {}
    if dotenv is not None:
        env.update(read_dotenv(dotenv))
    env.update(os.environ if environ is None else environ)

    def get(name: str, default: str = "") -> str:
        return env.get(name, default).strip()

    mesh_env = get("MESH_ENV", "dev")
    region = get("MESH_REGION", "us")
    if region not in REGIONS:
        raise ConfigError(f"MESH_REGION must be one of {REGIONS}", region=region)
    data_dir = Path(get("MESH_DATA_DIR", ".mesh"))
    admin_key = get("MESH_ADMIN_KEY")
    secret_key = get("MESH_SECRET_KEY")

    if not admin_key or not secret_key:
        if mesh_env == "prod":
            raise ConfigError("MESH_ADMIN_KEY and MESH_SECRET_KEY are required when MESH_ENV=prod")
        generated = _dev_keys(data_dir)
        admin_key = admin_key or generated["admin_key"]
        secret_key = secret_key or generated["secret_key"]

    return Settings(
        env=mesh_env,
        region=region,
        data_dir=data_dir,
        admin_key=admin_key,
        secret_key=secret_key,
        database_url=get("DATABASE_URL"),
        database_platform_url=get("DATABASE_PLATFORM_URL"),
        redis_url=get("REDIS_URL"),
        qdrant_url=get("QDRANT_URL"),
        kafka_brokers=get("KAFKA_BROKERS"),
        clickhouse_url=get("CLICKHOUSE_URL"),
        vault_addr=get("VAULT_ADDR"),
        vault_token=get("VAULT_TOKEN"),
        temporal_addr=get("TEMPORAL_ADDR"),
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        vllm_endpoints=tuple(
            url.strip() for url in get("VLLM_ENDPOINTS").split(",") if url.strip()
        ),
        stripe_key=get("STRIPE_KEY"),
        public_url=get("MESH_PUBLIC_URL", "http://localhost:8080"),
    )

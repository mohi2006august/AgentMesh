from __future__ import annotations

from pathlib import Path

import pytest

from mesh_core.config import load_settings, read_dotenv
from mesh_core.errors import ConfigError
from mesh_core.ids import content_hash, new_id, stable_id


def test_dev_generates_keys_once(tmp_path: Path) -> None:
    env = {"MESH_DATA_DIR": str(tmp_path)}
    first = load_settings(env, dotenv=None)
    second = load_settings(env, dotenv=None)
    assert first.admin_key.startswith("mesh_admin_")
    assert (first.admin_key, first.secret_key) == (second.admin_key, second.secret_key)
    assert first.resolved_database_url == f"sqlite:///{(tmp_path / 'mesh.db').as_posix()}"


def test_prod_requires_explicit_keys(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings({"MESH_ENV": "prod", "MESH_DATA_DIR": str(tmp_path)}, dotenv=None)
    assert not (tmp_path / "dev-keys.json").exists()


def test_region_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings({"MESH_REGION": "mars", "MESH_DATA_DIR": str(tmp_path)}, dotenv=None)


def test_environment_overrides_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "MESH_REGION=eu  # pinned\nVLLM_ENDPOINTS='http://a, http://b'\n", encoding="utf-8"
    )
    assert read_dotenv(dotenv) == {"MESH_REGION": "eu", "VLLM_ENDPOINTS": "http://a, http://b"}

    settings = load_settings({"MESH_DATA_DIR": str(tmp_path)}, dotenv=dotenv)
    assert settings.region == "eu"
    assert settings.vllm_endpoints == ("http://a", "http://b")
    assert (
        load_settings({"MESH_DATA_DIR": str(tmp_path), "MESH_REGION": "us"}, dotenv=dotenv).region
        == "us"
    )


def test_ids() -> None:
    first, second = new_id("run"), new_id("run")
    assert first != second and first.startswith("run_") and len(first) == 30
    assert stable_id("evt", "run_1", "model:1") == stable_id("evt", "run_1", "model:1")
    assert stable_id("evt", "run_1", "model:1") != stable_id("evt", "run_1", "model:2")
    assert content_hash("ab", "c") != content_hash("a", "bc")

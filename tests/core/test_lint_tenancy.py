from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import lint_tenancy  # noqa: E402


def _rules(source: str, path: str = "runtime/mesh_runtime/example.py") -> list[str]:
    return [finding.rule for finding in lint_tenancy.check_source(source, Path(path))]


def test_tenant_tx_requires_tenant_id_first() -> None:
    bad = """
def load(db, tenant_id):
    with db.tenant_tx(tenant_id) as tx:
        return tx.query("...")
"""
    good = """
def load(tenant_id, db):
    with db.tenant_tx(tenant_id) as tx:
        return tx.query("...")
"""
    assert _rules(bad) == ["T001"]
    assert _rules(good) == []


def test_tenant_tx_must_pass_the_argument_through_unchanged() -> None:
    source = """
class RunStore:
    def get(self, tenant_id, other):
        with self._db.tenant_tx(other) as tx:
            return tx.query("...")
"""
    assert _rules(source) == ["T001"]


def test_platform_tx_requires_explicit_marker() -> None:
    bad = """
def rollup(db):
    with db.platform_tx() as tx:
        return tx.query("...")
"""
    good = """
@platform_scoped("economics rollup")
def rollup(db):
    with db.platform_tx() as tx:
        return tx.query("...")
"""
    assert _rules(bad) == ["T002"]
    assert _rules(good) == []


def test_data_access_class_methods_take_tenant_id_first() -> None:
    source = """
class VectorIndex:
    def __init__(self, db): ...
    def _helper(self, x): ...
    def close(self): ...
    def search(self, query, tenant_id): ...
    def upsert(self, tenant_id, chunks): ...
    @platform_scoped("catalog")
    def list_all(self): ...
"""
    assert _rules(source) == ["T003"]


def test_meter_events_are_built_only_in_the_metering_package() -> None:
    source = "event = MeterEvent(kind='tokens')\n"
    assert _rules(source) == ["T004"]
    assert _rules(source, "metering/mesh_metering/events.py") == []


def test_repository_passes_its_own_lint() -> None:
    targets = [ROOT / name for name in lint_tenancy.COMPONENT_DIRS if (ROOT / name).exists()]
    findings = [
        finding
        for file in lint_tenancy.iter_files(targets)
        for finding in lint_tenancy.check_source(
            file.read_text(encoding="utf-8"), file.relative_to(ROOT)
        )
    ]
    assert findings == [], "\n".join(str(finding) for finding in findings)

"""Cross-platform task runner behind the Makefile targets.

Usage: python scripts/tasks.py <target>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
BIN = VENV / ("Scripts" if os.name == "nt" else "bin")
PYTHON = BIN / ("python.exe" if os.name == "nt" else "python")


def run(*cmd: str, check: bool = True) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check).returncode


def venv_python() -> str:
    return str(PYTHON) if PYTHON.exists() else sys.executable


def install() -> None:
    if not PYTHON.exists():
        run(sys.executable, "-m", "venv", str(VENV))
    run(str(PYTHON), "-m", "pip", "install", "-e", ".[dev]")
    if shutil.which("pnpm"):
        run("pnpm", "install")
    else:
        print("pnpm not found: skipping console install")


def dev() -> None:
    missing = [tool for tool in ("docker", "kind") if not shutil.which(tool)]
    if missing:
        sys.exit(
            f"`make dev` needs {', '.join(missing)}. "
            "Use `make dev-local` for the zero-dependency stack."
        )
    run("kind", "create", "cluster", "--config", "deploy/kind/cluster.yaml", check=False)
    run("docker", "compose", "-f", "deploy/docker-compose.yml", "up", "-d", "--wait")
    run(venv_python(), "-m", "mesh_sdk.cli", "dev", "--profile", "compose")


def dev_local() -> None:
    run(venv_python(), "-m", "mesh_sdk.cli", "dev")


def test() -> None:
    run(venv_python(), "-m", "pytest")


def test_isolation() -> None:
    run(venv_python(), "-m", "pytest", "-m", "isolation", "tests/isolation")


def lint() -> None:
    run(venv_python(), "-m", "ruff", "check", ".")
    run(venv_python(), "tools/lint_tenancy.py")


def typecheck() -> None:
    run(venv_python(), "-m", "mypy", ".")


def bench() -> None:
    run(venv_python(), "-m", "pytest", "tests/bench", "-s")


def seed() -> None:
    run(venv_python(), "-m", "mesh_sdk.cli", "seed")


def clean() -> None:
    for name in (".mesh", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"):
        shutil.rmtree(ROOT / name, ignore_errors=True)


TARGETS = {
    "install": install,
    "dev": dev,
    "dev-local": dev_local,
    "test": test,
    "test-isolation": test_isolation,
    "lint": lint,
    "typecheck": typecheck,
    "bench": bench,
    "seed": seed,
    "clean": clean,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        sys.exit(f"usage: python scripts/tasks.py [{' | '.join(TARGETS)}]")
    try:
        TARGETS[sys.argv[1]]()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

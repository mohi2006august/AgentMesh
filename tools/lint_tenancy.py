"""Project lint rules that ruff cannot express (see CLAUDE.md -> Code Conventions).

T001  A function that opens `tenant_tx(...)` must take `tenant_id` as its first
      argument and pass that same name to `tenant_tx`.
T002  A function that opens `platform_tx(...)` must be marked `@platform_scoped("why")`.
T003  Public methods of data access classes (`*Store`, `*Index`) must take `tenant_id`
      first, unless marked `@platform_scoped("why")`.
T004  `MeterEvent(...)` is constructed only inside `mesh_metering`; everything else goes
      through the single `emit` helper.

Usage: python tools/lint_tenancy.py [paths...]   (exit code 1 on findings)
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPONENT_DIRS = (
    "core", "sdk", "metering", "inference-router", "mcp-gateway",
    "runtime", "control-plane", "retrieval", "cache", "economics",
)  # fmt: skip
DATA_ACCESS_SUFFIXES = ("Store", "Index")
EXEMPT_METHODS = {"close", "migrate"}
# The module that defines the transactions themselves.
EXEMPT_FILES = {Path("core/mesh_core/db.py")}

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path.as_posix()}:{self.line}: {self.rule} {self.message}"


def _decorator_names(node: FunctionNode) -> set[str]:
    names = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _first_param(node: FunctionNode) -> str | None:
    params = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
    if params and params[0] in ("self", "cls"):
        params = params[1:]
    return params[0] if params else None


def _own_calls(node: FunctionNode) -> Iterator[ast.Call]:
    """Calls in this function's body, not descending into nested function definitions."""
    stack: list[ast.AST] = list(node.body)
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(current, ast.Call):
            yield current
        stack.extend(ast.iter_child_nodes(current))


def _call_attr(call: ast.Call) -> str | None:
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _check_function(node: FunctionNode, path: Path) -> Iterator[Finding]:
    decorators = _decorator_names(node)
    for call in _own_calls(node):
        attr = _call_attr(call)
        if attr == "tenant_tx":
            passes_tenant = (
                bool(call.args)
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "tenant_id"
            )
            if _first_param(node) != "tenant_id" or not passes_tenant:
                yield Finding(
                    path, call.lineno, "T001",
                    f"`{node.name}` opens tenant_tx: take `tenant_id` first and pass it through unchanged",
                )  # fmt: skip
        elif attr == "platform_tx" and "platform_scoped" not in decorators:
            yield Finding(
                path, call.lineno, "T002",
                f"`{node.name}` opens platform_tx: mark it @platform_scoped(\"reason\")",
            )  # fmt: skip


def _check_class(node: ast.ClassDef, path: Path) -> Iterator[Finding]:
    if not node.name.endswith(DATA_ACCESS_SUFFIXES):
        return
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = _decorator_names(item)
        if item.name.startswith("_") or item.name in EXEMPT_METHODS:
            continue
        if decorators & {"platform_scoped", "staticmethod", "property", "classmethod"}:
            continue
        if _first_param(item) != "tenant_id":
            yield Finding(
                path, item.lineno, "T003",
                f"`{node.name}.{item.name}` is data access: take `tenant_id` first or mark @platform_scoped",
            )  # fmt: skip


def _check_meter_events(tree: ast.AST, path: Path) -> Iterator[Finding]:
    if "mesh_metering" in path.parts:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else _call_attr(node)
            if name == "MeterEvent":
                yield Finding(
                    path, node.lineno, "T004",
                    "metering events are emitted only through mesh_metering.emit helpers",
                )  # fmt: skip


def check_source(source: str, path: Path) -> list[Finding]:
    tree = ast.parse(source, filename=str(path))
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings.extend(_check_function(node, path))
        elif isinstance(node, ast.ClassDef):
            findings.extend(_check_class(node, path))
    findings.extend(_check_meter_events(tree, path))
    return sorted(findings, key=lambda finding: (str(finding.path), finding.line))


def iter_files(paths: list[Path]) -> Iterator[Path]:
    for base in paths:
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for file in files:
            relative = (
                file.resolve().relative_to(ROOT) if file.resolve().is_relative_to(ROOT) else file
            )
            if relative not in EXEMPT_FILES:
                yield file


def main(argv: list[str]) -> int:
    targets = [Path(arg) for arg in argv] or [ROOT / name for name in COMPONENT_DIRS]
    findings: list[Finding] = []
    for file in iter_files([target for target in targets if target.exists()]):
        relative = file.resolve().relative_to(ROOT) if file.resolve().is_relative_to(ROOT) else file
        findings.extend(check_source(file.read_text(encoding="utf-8"), relative))
    for finding in findings:
        print(finding)
    print(f"tenancy lint: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

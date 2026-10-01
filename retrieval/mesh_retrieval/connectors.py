"""Knowledge source connectors: where a tenant's documents come from.

`DirectoryConnector` reads Markdown and text files, with optional front matter:

    ---
    title: Refund policy
    acl: [group:support, user:alice]
    ---
    Body text...

Files without an `acl` are readable by everyone in the tenant.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import yaml

from mesh_core.errors import InvalidRequest
from mesh_retrieval.store import EVERYONE, Document

_SUFFIXES = (".md", ".markdown", ".txt")


class Connector(Protocol):
    kind: str

    def fetch(self) -> list[Document]: ...


def parse_front_matter(raw: str) -> tuple[dict[str, object], str]:
    if not raw.startswith("---"):
        return {}, raw
    _, _, rest = raw.partition("---")
    header, separator, body = rest.partition("\n---")
    if not separator:
        return {}, raw
    meta = yaml.safe_load(header) or {}
    if not isinstance(meta, dict):
        raise InvalidRequest("front matter must be a mapping")
    return meta, body.lstrip("\n")


class DirectoryConnector:
    kind = "directory"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self) -> list[Document]:
        if not self.path.is_dir():
            raise InvalidRequest("knowledge directory not found", path=str(self.path))
        documents = []
        for file in sorted(self.path.rglob("*")):
            if file.suffix.lower() not in _SUFFIXES or not file.is_file():
                continue
            meta, body = parse_front_matter(file.read_text(encoding="utf-8"))
            raw_acl = meta.get("acl", [EVERYONE])
            acl = (
                [raw_acl]
                if isinstance(raw_acl, str)
                else list(raw_acl)
                if isinstance(raw_acl, list)
                else []
            )
            if not acl:
                raise InvalidRequest("acl must be a string or a list", file=str(file))
            documents.append(
                Document(
                    doc_id=file.relative_to(self.path).as_posix(),
                    title=str(
                        meta.get("title") or file.stem.replace("-", " ").replace("_", " ").title()
                    ),
                    text=body,
                    uri=str(meta.get("uri", "")),
                    acl=tuple(str(entry) for entry in acl),
                )
            )
        return documents


class InlineConnector:
    kind = "inline"

    def __init__(self, documents: Iterable[Document]) -> None:
        self._documents = list(documents)

    def fetch(self) -> list[Document]:
        return list(self._documents)

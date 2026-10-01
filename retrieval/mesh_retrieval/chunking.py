"""Structure-aware chunking.

Documents are split at headings and blank lines, then paragraphs are packed into chunks of
up to `max_chars`. Each chunk is prefixed with its nearest heading so it stays meaningful
on its own, and an oversized paragraph is split at sentence boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str


def _sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            sections[-1][1].append(" ".join(line.strip() for line in paragraph))
            paragraph.clear()

    for line in text.splitlines():
        heading = _HEADING.match(line.strip())
        if heading:
            flush()
            sections.append((heading.group(1).strip(), []))
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush()
    return [(title, paragraphs) for title, paragraphs in sections if paragraphs]


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(paragraph):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = ""
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, *, title: str = "", max_chars: int = 900) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, paragraphs in _sections(text):
        prefix = " / ".join(part for part in (title, heading) if part)
        prefix = f"{prefix}\n" if prefix else ""
        current = ""
        for paragraph in (piece for p in paragraphs for piece in _split_long(p, max_chars)):
            if current and len(current) + len(paragraph) + 2 > max_chars:
                chunks.append(Chunk(len(chunks), prefix + current))
                current = ""
            current = f"{current}\n\n{paragraph}".strip()
        if current:
            chunks.append(Chunk(len(chunks), prefix + current))
    return chunks

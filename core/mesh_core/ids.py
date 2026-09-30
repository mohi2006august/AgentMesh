"""Identifier helpers.

`new_id` gives sortable unique ids. `stable_id` derives the same id from the same
inputs, which is what makes replayed steps and billing retries idempotent.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid

_NAMESPACE = uuid.UUID("6f0e2c1e-5d0a-4a57-9a53-0e7f6d6b9a11")
_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"


def _base32(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_id(prefix: str) -> str:
    """Time-sortable id: `<prefix>_<48-bit ms timestamp><80 random bits>` (ULID layout)."""
    millis = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    return f"{prefix}_{_base32(millis, 10)}{_base32(rand, 16)}"


def stable_id(prefix: str, *parts: object) -> str:
    """Deterministic id for the given parts."""
    joined = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{uuid.uuid5(_NAMESPACE, joined).hex}"


def content_hash(*parts: object) -> str:
    """SHA-256 over the parts, unit-separator delimited so ('ab','c') != ('a','bc')."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()

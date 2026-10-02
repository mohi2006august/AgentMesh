"""Cache storage backends. Keys arrive already namespaced by owner (tenant id or global).

`MemoryBackend` is the local default (LRU + TTL). `RedisBackend` stores exact entries as
strings with TTL, and each semantic scope as a hash of entry id -> vector + payload, so a
lookup costs one HGETALL over a bounded scope.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

from mesh_core.clock import Clock, SystemClock

SEMANTIC_SCOPE_LIMIT = 200


class CacheBackend(Protocol):
    def get(self, owner: str, key: str) -> dict[str, Any] | None: ...

    def set(self, owner: str, key: str, value: dict[str, Any], ttl_s: int) -> None: ...

    def scope_entries(self, owner: str, scope: str) -> list[dict[str, Any]]: ...

    def scope_add(
        self, owner: str, scope: str, entry_id: str, value: dict[str, Any], ttl_s: int
    ) -> None: ...

    def clear(self, owner: str) -> int: ...


class MemoryBackend:
    def __init__(self, max_entries: int = 10_000, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._max = max_entries
        self._exact: OrderedDict[tuple[str, str], tuple[float, dict[str, Any]]] = OrderedDict()
        self._scopes: dict[tuple[str, str], OrderedDict[str, tuple[float, dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def get(self, owner: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._exact.get((owner, key))
            if item is None:
                return None
            expires, value = item
            if expires <= self._clock.now():
                del self._exact[(owner, key)]
                return None
            self._exact.move_to_end((owner, key))
            return value

    def set(self, owner: str, key: str, value: dict[str, Any], ttl_s: int) -> None:
        with self._lock:
            self._exact[(owner, key)] = (self._clock.now() + ttl_s, value)
            self._exact.move_to_end((owner, key))
            while len(self._exact) > self._max:
                self._exact.popitem(last=False)

    def scope_entries(self, owner: str, scope: str) -> list[dict[str, Any]]:
        now = self._clock.now()
        with self._lock:
            entries = self._scopes.get((owner, scope))
            if not entries:
                return []
            for entry_id in [eid for eid, (expires, _) in entries.items() if expires <= now]:
                del entries[entry_id]
            return [value for _, value in entries.values()]

    def scope_add(
        self, owner: str, scope: str, entry_id: str, value: dict[str, Any], ttl_s: int
    ) -> None:
        with self._lock:
            entries = self._scopes.setdefault((owner, scope), OrderedDict())
            entries[entry_id] = (self._clock.now() + ttl_s, value)
            entries.move_to_end(entry_id)
            while len(entries) > SEMANTIC_SCOPE_LIMIT:
                entries.popitem(last=False)

    def clear(self, owner: str) -> int:
        with self._lock:
            exact = [key for key in self._exact if key[0] == owner]
            scopes = [key for key in self._scopes if key[0] == owner]
            for key in exact:
                del self._exact[key]
            for scope_key in scopes:
                del self._scopes[scope_key]
            return len(exact) + len(scopes)


class RedisBackend:
    """Production backend. `client` is a `redis.Redis` (sync) instance."""

    def __init__(self, client: Any, prefix: str = "mesh:cache") -> None:
        self._redis = client
        self._prefix = prefix

    def _exact_key(self, owner: str, key: str) -> str:
        return f"{self._prefix}:{owner}:x:{key}"

    def _scope_key(self, owner: str, scope: str) -> str:
        return f"{self._prefix}:{owner}:s:{scope}"

    def get(self, owner: str, key: str) -> dict[str, Any] | None:
        raw = self._redis.get(self._exact_key(owner, key))
        return json.loads(raw) if raw else None

    def set(self, owner: str, key: str, value: dict[str, Any], ttl_s: int) -> None:
        self._redis.set(self._exact_key(owner, key), json.dumps(value), ex=ttl_s)

    def scope_entries(self, owner: str, scope: str) -> list[dict[str, Any]]:
        now = time.time()
        entries = []
        stale = []
        for entry_id, raw in self._redis.hgetall(self._scope_key(owner, scope)).items():
            value = json.loads(raw)
            if value.get("expires_at", 0) <= now:
                stale.append(entry_id)
            else:
                entries.append(value)
        if stale:
            self._redis.hdel(self._scope_key(owner, scope), *stale)
        return entries

    def scope_add(
        self, owner: str, scope: str, entry_id: str, value: dict[str, Any], ttl_s: int
    ) -> None:
        key = self._scope_key(owner, scope)
        pipe = self._redis.pipeline()
        pipe.hset(key, entry_id, json.dumps({**value, "expires_at": time.time() + ttl_s}))
        pipe.expire(key, ttl_s)
        pipe.execute()
        if self._redis.hlen(key) > SEMANTIC_SCOPE_LIMIT:
            oldest = sorted(
                self.scope_entries(owner, scope), key=lambda item: item.get("created_at", 0)
            )
            self._redis.hdel(key, *[item["entry_id"] for item in oldest[:-SEMANTIC_SCOPE_LIMIT]])

    def clear(self, owner: str) -> int:
        removed = 0
        for key in self._redis.scan_iter(match=f"{self._prefix}:{owner}:*"):
            removed += int(self._redis.delete(key))
        return removed

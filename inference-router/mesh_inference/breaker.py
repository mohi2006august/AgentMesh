"""Circuit breaker per (provider, model): detect an outage by error rate, stop hammering it."""

from __future__ import annotations

from dataclasses import dataclass

from mesh_core.clock import Clock, SystemClock


@dataclass
class _State:
    failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    def __init__(
        self, *, failure_threshold: int = 3, cooldown_s: float = 30.0, clock: Clock | None = None
    ) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._clock = clock or SystemClock()
        self._states: dict[str, _State] = {}

    def allows(self, key: str) -> bool:
        """Closed, or open long enough that one trial request may go through (half-open)."""
        state = self._states.get(key)
        if state is None or state.opened_at is None:
            return True
        return self._clock.now() - state.opened_at >= self._cooldown

    def record_success(self, key: str) -> None:
        self._states.pop(key, None)

    def record_failure(self, key: str) -> None:
        state = self._states.setdefault(key, _State())
        state.failures += 1
        if state.failures >= self._threshold:
            # Also re-arms the cooldown when a half-open trial fails.
            state.opened_at = self._clock.now()

    def is_open(self, key: str) -> bool:
        return not self.allows(key)

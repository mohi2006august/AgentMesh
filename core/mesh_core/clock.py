"""Injectable time source so quotas, TTLs and billing windows are testable."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    def now(self) -> float:
        return time.time()


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def set(self, value: float) -> None:
        self._now = value


def month_of(ts: float) -> str:
    """`YYYY-MM` bucket (UTC) used by quotas, billing and economics."""
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m")


def month_start(month: str) -> float:
    return datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC).timestamp()


def next_month(month: str) -> str:
    year, mon = (int(part) for part in month.split("-"))
    return f"{year + 1}-01" if mon == 12 else f"{year}-{mon + 1:02d}"


def add_months(month: str, count: int) -> str:
    year, mon = (int(part) for part in month.split("-"))
    index = year * 12 + (mon - 1) + count
    return f"{index // 12}-{index % 12 + 1:02d}"

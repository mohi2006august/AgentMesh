from __future__ import annotations

from collections.abc import Iterator

import pytest

from mesh_core.clock import FakeClock
from mesh_core.db import Database

ROOT_KEY = "test-root-key-0123456789abcdef0123456789abcdef"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db() -> Iterator[Database]:
    database = Database.in_memory()
    yield database
    database.close()

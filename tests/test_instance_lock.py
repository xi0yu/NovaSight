from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from novasight.instance_lock import InstanceLock


def test_second_instance_fails_without_erasing_lock_owner(tmp_path: Path) -> None:
    path = tmp_path / "instance.lock"
    first = InstanceLock(path)
    second = InstanceLock(path)
    first.acquire()
    try:
        owner = path.read_text(encoding="utf-8")
        with pytest.raises(RuntimeError, match="already held"):
            second.acquire()
        assert path.read_text(encoding="utf-8") == owner
    finally:
        first.release()


def test_non_regular_lock_path_fails_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "instance.lock"
    os.mkfifo(path)

    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match="cannot be opened|not a regular file"):
        InstanceLock(path).acquire()

    assert time.monotonic() - started_at < 0.2

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import TextIO


class InstanceLock:
    def __init__(self, path: str | Path | None = None) -> None:
        value = str(path or os.environ.get("NOVASIGHT_INSTANCE_LOCK", "")).strip()
        self.path = Path(value) if value else None
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"NovaSight instance lock is already held: {self.path}") from exc
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def status(self) -> dict[str, object]:
        return {
            "configured": self.path is not None,
            "path": str(self.path) if self.path is not None else "",
            "acquired": self._handle is not None,
            "pid": os.getpid() if self._handle is not None else None,
        }

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


__all__ = ["InstanceLock"]

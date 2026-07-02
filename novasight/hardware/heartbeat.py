from __future__ import annotations

import time


class HardwareHeartbeat:
    def __init__(self, *, timeout_s: float = 0.05) -> None:
        self.timeout_s = timeout_s
        self.last_seen_s: float | None = None
        self.connected = False

    def mark_seen(self, now_s: float | None = None) -> None:
        self.last_seen_s = time.monotonic() if now_s is None else now_s
        self.connected = True

    def should_suspend(self, now_s: float | None = None) -> bool:
        if self.last_seen_s is None:
            return True
        now = time.monotonic() if now_s is None else now_s
        self.connected = now - self.last_seen_s <= self.timeout_s
        return not self.connected

    def status(self) -> dict[str, float | bool | None]:
        return {
            "connected": self.connected,
            "last_seen_s": self.last_seen_s,
            "timeout_s": self.timeout_s,
        }

from __future__ import annotations

import os
import socket
import threading


class SystemdNotifier:
    def __init__(self, *, interval_s: float = 5.0) -> None:
        self.interval_s = max(0.1, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(os.environ.get("NOTIFY_SOCKET"))

    def start(self) -> None:
        if not self.enabled:
            return
        self.notify("READY=1")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="novasight-systemd-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if self.enabled:
            self.notify("STOPPING=1")

    def notify(self, message: str) -> bool:
        address = os.environ.get("NOTIFY_SOCKET", "")
        if not address:
            return False
        if address.startswith("@"):
            address = "\0" + address[1:]
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.connect(address)
                sock.sendall(message.encode("utf-8"))
        except OSError:
            return False
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.notify("WATCHDOG=1")


def watchdog_interval_from_env(default_s: float = 5.0) -> float:
    raw = os.environ.get("WATCHDOG_USEC", "")
    try:
        usec = float(raw)
    except ValueError:
        return default_s
    if usec <= 0:
        return default_s
    return max(0.1, usec / 2_000_000.0)


__all__ = ["SystemdNotifier", "watchdog_interval_from_env"]

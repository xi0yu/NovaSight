from __future__ import annotations

import copy
import threading

from novasight.config import RuntimeConfig


class RuntimeConfigStore:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = copy.deepcopy(config)
        self._lock = threading.RLock()
        self.version = 0

    def snapshot(self) -> RuntimeConfig:
        with self._lock:
            return copy.deepcopy(self._config)

    def replace(self, config: RuntimeConfig) -> RuntimeConfig:
        with self._lock:
            self._config = copy.deepcopy(config)
            self.version += 1
            return copy.deepcopy(self._config)

    def status(self) -> dict[str, int]:
        with self._lock:
            return {
                "version": self.version,
                "roi_size": self._config.roi.size,
            }

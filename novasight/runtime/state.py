from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeState:
    running: bool
    source: str
    active_model: dict | None
    executor: dict

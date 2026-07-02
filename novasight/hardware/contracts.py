from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class BoxInputState:
    left: bool = False
    right: bool = False
    side: bool = False
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.left or self.right or self.side


class IHardwareBox(Protocol):
    def connect(self) -> None:
        ...

    def send_move(self, dx: int, dy: int) -> None:
        ...

    def send_click(self, button: str) -> None:
        ...

    def get_input_state(self) -> BoxInputState:
        ...

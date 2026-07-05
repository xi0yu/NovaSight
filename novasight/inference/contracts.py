from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from novasight.contracts import BBox


@dataclass(frozen=True, slots=True, init=False)
class InferenceDetection:
    cls: int
    score: float
    box: BBox

    def __init__(
        self,
        cls: int,
        score: float,
        x: float | None = None,
        y: float | None = None,
        w: float | None = None,
        h: float | None = None,
        *,
        box: BBox | None = None,
        x1: float | None = None,
        y1: float | None = None,
        x2: float | None = None,
        y2: float | None = None,
    ) -> None:
        object.__setattr__(self, "cls", int(cls))
        object.__setattr__(self, "score", float(score))
        if box is not None:
            next_box = box
        elif x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            next_box = BBox.from_xyxy(x1, y1, x2, y2)
        elif x is not None and y is not None and w is not None and h is not None:
            next_box = BBox.from_xywh(x, y, w, h)
        else:
            raise TypeError("InferenceDetection requires either box, x/y/w/h, or x1/y1/x2/y2")
        object.__setattr__(self, "box", next_box)

    @classmethod
    def from_xyxy(cls, *, cls_id: int, score: float, x1: float, y1: float, x2: float, y2: float) -> InferenceDetection:
        return cls(cls=cls_id, score=score, box=BBox.from_xyxy(x1, y1, x2, y2))

    @classmethod
    def from_cxcywh(cls, *, cls_id: int, score: float, cx: float, cy: float, w: float, h: float) -> InferenceDetection:
        return cls(cls=cls_id, score=score, box=BBox.from_cxcywh(cx, cy, w, h))

    @property
    def x(self) -> float:
        return self.box.x1

    @property
    def y(self) -> float:
        return self.box.y1

    @property
    def w(self) -> float:
        return self.box.width

    @property
    def h(self) -> float:
        return self.box.height

    @property
    def x1(self) -> float:
        return self.box.x1

    @property
    def y1(self) -> float:
        return self.box.y1

    @property
    def x2(self) -> float:
        return self.box.x2

    @property
    def y2(self) -> float:
        return self.box.y2

    @property
    def cx(self) -> float:
        return self.box.center_x

    @property
    def cy(self) -> float:
        return self.box.center_y

    @property
    def area(self) -> float:
        return self.box.area


@dataclass(frozen=True)
class InferenceResult:
    available: bool
    detections: list[InferenceDetection] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    reason: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


class InferenceEngine(Protocol):
    engine_id: str

    def available(self) -> bool: ...
    def last_reason(self) -> str: ...
    def status(self) -> dict: ...
    def load(
        self,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> None: ...
    def infer(self, frame: Any) -> InferenceResult: ...

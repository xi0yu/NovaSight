from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        return cls(float(x1), float(y1), float(x2), float(y2))

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        return cls(float(x), float(y), float(x) + float(w), float(y) + float(h))

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        half_w = float(w) / 2.0
        half_h = float(h) / 2.0
        return cls(float(cx) - half_w, float(cy) - half_h, float(cx) + half_w, float(cy) + half_h)

    @property
    def x(self) -> float:
        return self.x1

    @property
    def y(self) -> float:
        return self.y1

    @property
    def w(self) -> float:
        return self.width

    @property
    def h(self) -> float:
        return self.height

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def center_y(self) -> float:
        return self.point_y()

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def point_y(self, ratio: float = 0.5) -> float:
        clamped = max(0.0, min(1.0, float(ratio)))
        return self.y1 + self.height * clamped


@dataclass(frozen=True, slots=True, init=False)
class Detection:
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
        object.__setattr__(self, "box", _coerce_box(x=x, y=y, w=w, h=h, box=box, x1=x1, y1=y1, x2=x2, y2=y2))

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

    def point_y(self, ratio: float = 0.5) -> float:
        return self.box.point_y(ratio)


DetectionCoordinateSpace = Literal["model", "roi", "capture", "control", "display"]


@dataclass(frozen=True)
class DetectionBatch:
    frame_id: int
    capture_ts_ns: int
    inference_start_ts_ns: int
    inference_end_ts_ns: int
    detections: list[Detection] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    coordinate_space: DetectionCoordinateSpace = "roi"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("DetectionBatch.frame_id must be >= 0")
        if int(self.capture_ts_ns) <= 0:
            raise ValueError("DetectionBatch.capture_ts_ns must be positive")
        if int(self.inference_start_ts_ns) <= 0:
            raise ValueError("DetectionBatch.inference_start_ts_ns must be positive")
        if int(self.inference_end_ts_ns) < int(self.inference_start_ts_ns):
            raise ValueError("DetectionBatch.inference_end_ts_ns must be >= inference_start_ts_ns")
        object.__setattr__(self, "detections", list(self.detections))
        object.__setattr__(self, "classes", list(self.classes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def inference_latency_ms(self) -> float:
        return (int(self.inference_end_ts_ns) - int(self.inference_start_ts_ns)) / 1e6


@dataclass(frozen=True, slots=True, init=False)
class Track:
    track_id: int
    cls: int
    score: float
    box: BBox

    def __init__(
        self,
        track_id: int,
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
        object.__setattr__(self, "track_id", int(track_id))
        object.__setattr__(self, "cls", int(cls))
        object.__setattr__(self, "score", float(score))
        object.__setattr__(self, "box", _coerce_box(x=x, y=y, w=w, h=h, box=box, x1=x1, y1=y1, x2=x2, y2=y2))

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

    def point_y(self, ratio: float = 0.5) -> float:
        return self.box.point_y(ratio)


@dataclass(frozen=True)
class FrameContext:
    frame_id: int
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    capture_ts_ns: int | None = None
    dequeue_ts_ns: int | None = None
    decode_ts_ns: int | None = None
    roi_ts_ns: int | None = None
    inference_start_ts_ns: int | None = None
    inference_end_ts_ns: int | None = None
    postprocess_ts_ns: int | None = None
    control_start_ts_ns: int | None = None


@dataclass(frozen=True)
class ControlIntent:
    dx: float
    dy: float
    action: str | None
    confidence: float
    reason: str
    source_id: str
    move_kind: str = "raw"
    move_ms: int = 0
    trace_ms: int = 0
    bezier_ctrl: tuple[int, int, int, int] | None = None
    source_frame_id: int | None = None
    source_track_id: int | None = None
    predicted_source: bool = False


def _coerce_box(
    *,
    x: float | None,
    y: float | None,
    w: float | None,
    h: float | None,
    box: BBox | None,
    x1: float | None,
    y1: float | None,
    x2: float | None,
    y2: float | None,
) -> BBox:
    if box is not None:
        return box
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return BBox.from_xyxy(x1, y1, x2, y2)
    if x is not None and y is not None and w is not None and h is not None:
        return BBox.from_xywh(x, y, w, h)
    raise TypeError("Detection/Track requires either box, x/y/w/h, or x1/y1/x2/y2")

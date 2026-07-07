from __future__ import annotations

from dataclasses import dataclass

from novasight.contracts import BBox


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    model_width: float
    model_height: float
    roi_x: float
    roi_y: float
    roi_width: float
    roi_height: float
    capture_width: float
    capture_height: float
    control_origin_x: float = 0.0
    control_origin_y: float = 0.0
    display_scale_x: float = 1.0
    display_scale_y: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "model_width",
            "model_height",
            "roi_width",
            "roi_height",
            "capture_width",
            "capture_height",
            "display_scale_x",
            "display_scale_y",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"CoordinateTransform.{name} must be positive")

    @classmethod
    def from_frame(
        cls,
        frame: object,
        *,
        model_width: float | None = None,
        model_height: float | None = None,
        control_origin_x: float = 0.0,
        control_origin_y: float = 0.0,
        display_scale_x: float = 1.0,
        display_scale_y: float = 1.0,
        pad_x: float = 0.0,
        pad_y: float = 0.0,
    ) -> CoordinateTransform:
        roi_width = _number_attr(frame, "roi_width", _number_attr(frame, "width", 1.0))
        roi_height = _number_attr(frame, "roi_height", _number_attr(frame, "height", 1.0))
        capture_width = _number_attr(
            frame,
            "capture_width",
            _number_attr(frame, "source_width", roi_width),
        )
        capture_height = _number_attr(
            frame,
            "capture_height",
            _number_attr(frame, "source_height", roi_height),
        )
        return cls(
            model_width=float(model_width if model_width is not None else roi_width),
            model_height=float(model_height if model_height is not None else roi_height),
            roi_x=_number_attr(frame, "roi_x", _number_attr(frame, "offset_x", 0.0)),
            roi_y=_number_attr(frame, "roi_y", _number_attr(frame, "offset_y", 0.0)),
            roi_width=roi_width,
            roi_height=roi_height,
            capture_width=capture_width,
            capture_height=capture_height,
            control_origin_x=float(control_origin_x),
            control_origin_y=float(control_origin_y),
            display_scale_x=float(display_scale_x),
            display_scale_y=float(display_scale_y),
            pad_x=float(pad_x),
            pad_y=float(pad_y),
        )

    @property
    def model_to_roi_scale_x(self) -> float:
        return self.roi_width / self.model_width

    @property
    def model_to_roi_scale_y(self) -> float:
        return self.roi_height / self.model_height

    def model_to_roi_point(self, x: float, y: float) -> Point:
        return Point(
            (float(x) - self.pad_x) * self.model_to_roi_scale_x,
            (float(y) - self.pad_y) * self.model_to_roi_scale_y,
        )

    def roi_to_model_point(self, x: float, y: float) -> Point:
        return Point(
            float(x) / self.model_to_roi_scale_x + self.pad_x,
            float(y) / self.model_to_roi_scale_y + self.pad_y,
        )

    def roi_to_capture_point(self, x: float, y: float) -> Point:
        return Point(float(x) + self.roi_x, float(y) + self.roi_y)

    def capture_to_roi_point(self, x: float, y: float) -> Point:
        return Point(float(x) - self.roi_x, float(y) - self.roi_y)

    def capture_to_control_point(self, x: float, y: float) -> Point:
        return Point(float(x) - self.control_origin_x, float(y) - self.control_origin_y)

    def control_to_capture_point(self, x: float, y: float) -> Point:
        return Point(float(x) + self.control_origin_x, float(y) + self.control_origin_y)

    def control_to_display_point(self, x: float, y: float) -> Point:
        return Point(float(x) * self.display_scale_x, float(y) * self.display_scale_y)

    def display_to_control_point(self, x: float, y: float) -> Point:
        return Point(float(x) / self.display_scale_x, float(y) / self.display_scale_y)

    def model_to_roi_box(self, box: BBox) -> BBox:
        top_left = self.model_to_roi_point(box.x1, box.y1)
        bottom_right = self.model_to_roi_point(box.x2, box.y2)
        return BBox.from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)

    def roi_to_capture_box(self, box: BBox) -> BBox:
        top_left = self.roi_to_capture_point(box.x1, box.y1)
        bottom_right = self.roi_to_capture_point(box.x2, box.y2)
        return BBox.from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)

    def capture_to_control_box(self, box: BBox) -> BBox:
        top_left = self.capture_to_control_point(box.x1, box.y1)
        bottom_right = self.capture_to_control_point(box.x2, box.y2)
        return BBox.from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)

    def control_to_display_box(self, box: BBox) -> BBox:
        top_left = self.control_to_display_point(box.x1, box.y1)
        bottom_right = self.control_to_display_point(box.x2, box.y2)
        return BBox.from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)

    def roi_to_control_box(self, box: BBox) -> BBox:
        return self.capture_to_control_box(self.roi_to_capture_box(box))

    def roi_to_display_box(self, box: BBox) -> BBox:
        return self.control_to_display_box(self.roi_to_control_box(box))

    def model_to_control_box(self, box: BBox) -> BBox:
        return self.capture_to_control_box(self.roi_to_capture_box(self.model_to_roi_box(box)))


def _number_attr(obj: object, name: str, default: float) -> float:
    value = getattr(obj, name, None)
    if value is None:
        return float(default)
    return float(value)

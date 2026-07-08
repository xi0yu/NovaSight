from __future__ import annotations

from dataclasses import dataclass

from novasight.contracts import BBox, Detection, Track
from novasight.coordinates import CoordinateTransform, Point


@dataclass(frozen=True)
class RoiTransformer:
    model_width: float
    model_height: float
    roi_x: float
    roi_y: float
    roi_width: float
    roi_height: float
    source_width: float
    source_height: float
    pad_x: float = 0.0
    pad_y: float = 0.0

    def __post_init__(self) -> None:
        if self.model_content_width <= 0 or self.model_content_height <= 0:
            raise ValueError("RoiTransformer model content size must be positive after padding")

    @property
    def model_content_width(self) -> float:
        return float(self.model_width) - 2.0 * float(self.pad_x)

    @property
    def model_content_height(self) -> float:
        return float(self.model_height) - 2.0 * float(self.pad_y)

    @property
    def coordinate_transform(self) -> CoordinateTransform:
        return CoordinateTransform(
            model_width=self.model_content_width,
            model_height=self.model_content_height,
            roi_x=float(self.roi_x),
            roi_y=float(self.roi_y),
            roi_width=float(self.roi_width),
            roi_height=float(self.roi_height),
            capture_width=float(self.source_width),
            capture_height=float(self.source_height),
            pad_x=float(self.pad_x),
            pad_y=float(self.pad_y),
        )

    def model_to_roi_box(self, box: BBox) -> BBox:
        return self.coordinate_transform.model_to_roi_box(box)

    def model_to_source_box(self, box: BBox) -> BBox:
        transform = self.coordinate_transform
        return transform.roi_to_capture_box(transform.model_to_roi_box(box))

    def roi_to_source_box(self, box: BBox) -> BBox:
        return self.coordinate_transform.roi_to_capture_box(box)

    def source_to_roi_point(self, x: float, y: float) -> Point:
        return self.coordinate_transform.capture_to_roi_point(x, y)

    def roi_to_source_point(self, x: float, y: float) -> Point:
        return self.coordinate_transform.roi_to_capture_point(x, y)

    def model_detection_to_source(self, detection: Detection) -> Detection:
        return Detection(
            cls=detection.cls,
            score=detection.score,
            box=self.model_to_source_box(detection.box),
        )

    def roi_detection_to_source(self, detection: Detection) -> Detection:
        return Detection(
            cls=detection.cls,
            score=detection.score,
            box=self.roi_to_source_box(detection.box),
        )

    def roi_track_to_source(self, track: Track) -> Track:
        return self._track_with_box(track, self.roi_to_source_box(track.box))

    def model_track_to_source(self, track: Track) -> Track:
        return self._track_with_box(
            track,
            self.model_to_source_box(track.box),
            velocity_scale=(
                self.coordinate_transform.model_to_roi_scale_x,
                self.coordinate_transform.model_to_roi_scale_y,
            ),
        )

    def _track_with_box(
        self,
        track: Track,
        box: BBox,
        *,
        velocity_scale: tuple[float, float] = (1.0, 1.0),
    ) -> Track:
        return Track(
            track_id=track.track_id,
            cls=track.cls,
            score=track.score,
            box=box,
            velocity_px_s=(
                track.velocity_px_s[0] * velocity_scale[0],
                track.velocity_px_s[1] * velocity_scale[1],
            ),
            quality_score=track.quality_score,
            missed_frames=track.missed_frames,
            last_seen_ns=track.last_seen_ns,
            is_predicted=track.is_predicted,
            is_stale=track.is_stale,
        )


__all__ = [
    "RoiTransformer",
]

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from novasight.contracts import BBox, Detection, FrameContext, Track
from novasight.control.observation import resolve_aim_y_ratio


@dataclass(frozen=True)
class TrackObservation:
    detection_index: int
    class_id: int
    confidence: float
    bbox: BBox
    aim_x: float
    aim_y: float
    capture_ts_ns: int


@dataclass(frozen=True)
class BasicCandidateFilterResult:
    observations: list[TrackObservation]
    rejected: list[dict]
    raw_count: int

    def debug_payload(self) -> dict:
        return {
            "raw_candidates": self.raw_count,
            "filtered_candidates": len(self.observations),
            "rejected_candidates": len(self.rejected),
            "rejected": list(self.rejected),
        }


@dataclass(frozen=True)
class AssociationCandidateFilterResult:
    observations: list[TrackObservation]
    rejected: list[dict]
    input_count: int

    def debug_payload(self) -> dict:
        return {
            "raw_candidates": self.input_count,
            "filtered_candidates": len(self.observations),
            "rejected_candidates": len(self.rejected),
            "rejected": list(self.rejected),
        }


class AssociationCandidateFilter:
    """Apply control-space geometry gates before building the Hungarian matrix."""

    def apply(
        self,
        context: FrameContext,
        observations: Iterable[TrackObservation],
        *,
        center_x_px: float,
        center_y_px: float,
        radius_px: float,
        max_aspect_ratio: float,
        min_area_ratio: float = 256.0 / (640.0 * 640.0),
    ) -> AssociationCandidateFilterResult:
        source = list(observations)
        accepted: list[TrackObservation] = []
        rejected: list[dict] = []
        radius = max(0.0, float(radius_px))
        max_aspect = max(1.0, float(max_aspect_ratio))
        frame_area = max(1.0, float(context.width) * float(context.height))
        min_area = max(0.0, float(min_area_ratio)) * frame_area
        for observation in source:
            width = float(observation.bbox.width)
            height = float(observation.bbox.height)
            distance = (
                (float(observation.aim_x) - float(center_x_px)) ** 2
                + (float(observation.aim_y) - float(center_y_px)) ** 2
            ) ** 0.5
            reason = ""
            if radius <= 0.0 or distance > radius:
                reason = "selection_fov"
            elif float(observation.bbox.area) < min_area:
                reason = "area_filter"
            elif min(width, height) <= 0.0 or max(width / height, height / width) > max_aspect:
                reason = "ratio_check"
            if reason:
                rejected.append(
                    {
                        "detection_index": observation.detection_index,
                        "class_id": observation.class_id,
                        "confidence": observation.confidence,
                        "aim_x": observation.aim_x,
                        "aim_y": observation.aim_y,
                        "distance_px": distance,
                        "area_px": observation.bbox.area,
                        "reason": reason,
                    }
                )
                continue
            accepted.append(observation)
        return AssociationCandidateFilterResult(
            observations=accepted,
            rejected=rejected,
            input_count=len(source),
        )


class BasicCandidateFilter:
    """Build tracker observations using only class, confidence, and bbox validity."""

    def apply(
        self,
        context: FrameContext,
        *,
        allowed_class_ids: set[int] | None,
        min_confidence: float,
        aim_y_ratio: float,
        class_aim_y_ratios: dict[int, float] | None = None,
    ) -> BasicCandidateFilterResult:
        observations: list[TrackObservation] = []
        rejected: list[dict] = []
        capture_ts_ns = int(context.capture_ts_ns or 0)
        for detection_index, detection in enumerate(context.detections):
            reason = ""
            if allowed_class_ids is not None and int(detection.cls) not in allowed_class_ids:
                reason = "class_filter"
            elif float(detection.score) < float(min_confidence):
                reason = "confidence_filter"
            elif not _bbox_valid(detection, context):
                reason = "invalid_bbox"
            if reason:
                rejected.append(
                    {
                        "detection_index": detection_index,
                        "class_id": int(detection.cls),
                        "confidence": float(detection.score),
                        "reason": reason,
                    }
                )
                continue
            observations.append(
                TrackObservation(
                    detection_index=detection_index,
                    class_id=int(detection.cls),
                    confidence=float(detection.score),
                    bbox=detection.box,
                    aim_x=float(detection.box.center_x),
                    aim_y=float(
                        detection.box.y1
                        + detection.box.height
                        * resolve_aim_y_ratio(
                            aim_y_ratio,
                            class_aim_y_ratios,
                            int(detection.cls),
                        )
                    ),
                    capture_ts_ns=capture_ts_ns,
                )
            )
        return BasicCandidateFilterResult(
            observations=observations,
            rejected=rejected,
            raw_count=len(context.detections),
        )


@dataclass(frozen=True)
class QualityScoreConfig:
    confidence_weight: float = 0.7
    area_weight: float = 0.3


@dataclass(frozen=True)
class CandidateQuality:
    conf_score: float
    area_score: float
    quality_score: float


@dataclass(frozen=True)
class ScoredTrack:
    frame_id: int
    capture_ts_ns: int | None
    track: Track
    quality: CandidateQuality
    distance_px: float
    aim_x: float
    aim_y: float
    class_score: float
    distance_score: float
    selection_score: float
    reason: str = ""


@dataclass(frozen=True)
class TrackScoreResult:
    tracks: list[ScoredTrack]
    raw_count: int

    @property
    def inside_fov_count(self) -> int:
        return len(self.tracks)

    def debug_payload(self, *, limit: int = 12) -> dict:
        return {
            "raw_candidates": self.raw_count,
            "filtered_candidates": len(self.tracks),
            "inside_fov": self.inside_fov_count,
            "rejected_candidates": 0,
            "candidates": [scored_track_debug(item) for item in self.tracks[:limit]],
            "rejected": [],
        }


class QualityScorer:
    def __init__(self, config: QualityScoreConfig | None = None) -> None:
        self.config = config or QualityScoreConfig()

    def score(
        self,
        track: Track,
        *,
        context: FrameContext,
        class_reference_area: float | None = None,
    ) -> CandidateQuality:
        conf_score = _clamp01(float(track.score))
        frame_area = max(1.0, float(context.width) * float(context.height))
        reference_area = (
            max(1.0, float(class_reference_area))
            if class_reference_area is not None
            else frame_area
        )
        area_score = _clamp01((float(track.area) / reference_area) ** 0.5)
        confidence_weight = max(0.0, float(self.config.confidence_weight))
        area_weight = max(0.0, float(self.config.area_weight))
        total_weight = confidence_weight + area_weight
        if total_weight <= 0:
            total_weight = 1.0
            confidence_weight = 1.0
        q = (conf_score * confidence_weight + area_score * area_weight) / total_weight
        return CandidateQuality(
            conf_score=conf_score,
            area_score=area_score,
            quality_score=_clamp01(q),
        )


def parse_allowed_class_ids(value: str) -> set[int] | None:
    selected = str(value or "all").strip()
    if selected == "all":
        return None
    try:
        return {int(selected)}
    except ValueError:
        return None


def scored_track_debug(candidate: ScoredTrack) -> dict:
    item = candidate.track
    return {
        "frame_id": candidate.frame_id,
        "capture_ts_ns": candidate.capture_ts_ns,
        "cls": int(item.cls),
        "score": float(item.score),
        "x": float(item.x),
        "y": float(item.y),
        "w": float(item.w),
        "h": float(item.h),
        "cx": float(item.cx),
        "cy": float(item.cy),
        "aim_x": float(candidate.aim_x),
        "aim_y": float(candidate.aim_y),
        "distance_px": float(candidate.distance_px),
        "selection_fov_pass": True,
        "ratio_valid": True,
        "conf_score": float(candidate.quality.conf_score),
        "area_score": float(candidate.quality.area_score),
        "quality_score": float(candidate.quality.quality_score),
        "class_score": float(candidate.class_score),
        "distance_score": float(candidate.distance_score),
        "selection_score": float(candidate.selection_score),
    }


def _bbox_valid(target: Detection, context: FrameContext) -> bool:
    values = (target.x, target.y, target.w, target.h, target.x2, target.y2)
    if not all(isfinite(float(value)) for value in values):
        return False
    if float(target.w) <= 0 or float(target.h) <= 0:
        return False
    if float(target.x2) <= 0 or float(target.y2) <= 0:
        return False
    if float(target.x) >= float(context.width) or float(target.y) >= float(context.height):
        return False
    return True


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

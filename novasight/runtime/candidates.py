from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

from novasight.contracts import BBox, Detection, FrameContext, Track


Target = Detection | Track


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


class BasicCandidateFilter:
    """Build tracker observations using only class, confidence, and bbox validity."""

    def apply(
        self,
        context: FrameContext,
        *,
        allowed_class_ids: set[int] | None,
        min_confidence: float,
        aim_y_ratio: float,
    ) -> BasicCandidateFilterResult:
        observations: list[TrackObservation] = []
        rejected: list[dict] = []
        ratio = max(0.0, min(1.0, float(aim_y_ratio)))
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
                    aim_y=float(detection.box.y1 + detection.box.height * ratio),
                    capture_ts_ns=capture_ts_ns,
                )
            )
        return BasicCandidateFilterResult(
            observations=observations,
            rejected=rejected,
            raw_count=len(context.detections),
        )


@dataclass(frozen=True)
class SelectionFovConfig:
    enabled: bool = True
    shape: str = "circle"
    center_horizontal_percent: float = 50.0
    center_vertical_percent_from_top: float = 50.0
    center_x_px: float | None = None
    center_y_px: float | None = None
    radius_x_percent: float = 28.0
    radius_y_percent: float = 28.0


@dataclass(frozen=True)
class RatioCheckConfig:
    max_aspect_ratio: float = 6.0


@dataclass(frozen=True)
class QualityScoreConfig:
    confidence_weight: float = 0.7
    area_weight: float = 0.3


@dataclass(frozen=True)
class CandidateFilterConfig:
    allowed_class_ids: set[int] | None = None
    min_confidence: float = 0.0
    selection_fov: SelectionFovConfig = field(default_factory=SelectionFovConfig)
    ratio_check: RatioCheckConfig = field(default_factory=RatioCheckConfig)


@dataclass(frozen=True)
class CandidateQuality:
    conf_score: float
    area_score: float
    quality_score: float


@dataclass(frozen=True)
class ScoredCandidate:
    frame_id: int
    capture_ts_ns: int | None
    detection: Target
    selection_fov_pass: bool
    ratio_valid: bool
    quality: CandidateQuality
    distance_px: float
    aim_x: float
    aim_y: float
    reason: str = ""


@dataclass(frozen=True)
class RejectedCandidate:
    detection: Target
    reason: str
    selection_fov_pass: bool
    ratio_valid: bool


@dataclass(frozen=True)
class CandidateFilterResult:
    candidates: list[ScoredCandidate]
    rejected: list[RejectedCandidate]
    raw_count: int

    @property
    def inside_fov_count(self) -> int:
        return sum(1 for item in self.candidates if item.selection_fov_pass)

    def debug_payload(self, *, limit: int = 12) -> dict:
        return {
            "raw_candidates": self.raw_count,
            "filtered_candidates": len(self.candidates),
            "inside_fov": self.inside_fov_count,
            "rejected_candidates": len(self.rejected),
            "candidates": [candidate_debug(item) for item in self.candidates[:limit]],
            "rejected": [rejected_debug(item) for item in self.rejected[:limit]],
        }


class QualityScorer:
    def __init__(self, config: QualityScoreConfig | None = None) -> None:
        self.config = config or QualityScoreConfig()

    def score(self, target: Target, *, context: FrameContext) -> CandidateQuality:
        conf_score = _clamp01(float(target.score))
        frame_area = max(1.0, float(context.width) * float(context.height))
        area_score = _clamp01(float(target.area) / frame_area)
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


class CandidateFilter:
    def __init__(
        self,
        config: CandidateFilterConfig | None = None,
        *,
        quality_scorer: QualityScorer | None = None,
    ) -> None:
        self.config = config or CandidateFilterConfig()
        self.quality_scorer = quality_scorer or QualityScorer()

    def apply(
        self,
        context: FrameContext,
        candidates: Iterable[Target] | None = None,
        *,
        aim_ratio: float = 50.0,
    ) -> CandidateFilterResult:
        raw = list(candidates if candidates is not None else (list(context.tracks) or list(context.detections)))
        accepted: list[ScoredCandidate] = []
        rejected: list[RejectedCandidate] = []
        center_x, center_y = selection_fov_center(context, self.config.selection_fov)

        for item in raw:
            if not self._class_allowed(item):
                rejected.append(RejectedCandidate(item, "class_filter", False, False))
                continue
            if float(item.score) < float(self.config.min_confidence):
                rejected.append(RejectedCandidate(item, "confidence_filter", False, False))
                continue
            selection_pass = selection_fov_pass(item, context, self.config.selection_fov, aim_ratio=aim_ratio)
            if not selection_pass:
                rejected.append(RejectedCandidate(item, "selection_fov", False, True))
                continue
            if not _bbox_valid(item, context):
                rejected.append(RejectedCandidate(item, "invalid_bbox", True, False))
                continue
            ratio_valid = ratio_check_pass(item, self.config.ratio_check)
            if not ratio_valid:
                rejected.append(RejectedCandidate(item, "ratio_check", True, False))
                continue
            aim_x, aim_y = aim_point(item, aim_ratio)
            accepted.append(
                ScoredCandidate(
                    frame_id=int(context.frame_id),
                    capture_ts_ns=context.capture_ts_ns,
                    detection=item,
                    selection_fov_pass=True,
                    ratio_valid=True,
                    quality=self.quality_scorer.score(item, context=context),
                    distance_px=((aim_x - center_x) ** 2 + (aim_y - center_y) ** 2) ** 0.5,
                    aim_x=aim_x,
                    aim_y=aim_y,
                )
            )
        return CandidateFilterResult(candidates=accepted, rejected=rejected, raw_count=len(raw))

    def _class_allowed(self, target: Target) -> bool:
        allowed = self.config.allowed_class_ids
        return allowed is None or int(target.cls) in allowed


def parse_allowed_class_ids(value: str) -> set[int] | None:
    selected = str(value or "all").strip()
    if selected == "all":
        return None
    try:
        return {int(selected)}
    except ValueError:
        return None


def aim_point(target: Target, aim_ratio: float) -> tuple[float, float]:
    filtered_aim = getattr(target, "filtered_aim_px", None)
    if isinstance(target, Track) and filtered_aim is not None:
        return float(filtered_aim[0]), float(filtered_aim[1])
    point_y = getattr(target, "point_y", None)
    ratio = max(0.0, min(1.0, float(aim_ratio)))
    aim_y = float(point_y(ratio)) if callable(point_y) else float(target.y) + float(target.h) * ratio
    return float(target.cx), aim_y


def selection_fov_center(
    context: FrameContext,
    config: SelectionFovConfig,
) -> tuple[float, float]:
    center_x = config.center_x_px
    center_y = config.center_y_px
    return (
        float(center_x)
        if center_x is not None and isfinite(float(center_x))
        else float(context.width) * _clamp_percent(config.center_horizontal_percent),
        float(center_y)
        if center_y is not None and isfinite(float(center_y))
        else float(context.height) * _clamp_percent(config.center_vertical_percent_from_top),
    )


def selection_fov_pass(
    target: Target,
    context: FrameContext,
    config: SelectionFovConfig,
    *,
    aim_ratio: float,
) -> bool:
    if not config.enabled:
        return True
    center_x, center_y = selection_fov_center(context, config)
    aim_x, aim_y = aim_point(target, aim_ratio)
    rx = max(1e-6, float(context.width) * _clamp_percent(config.radius_x_percent))
    ry = max(1e-6, float(context.height) * _clamp_percent(config.radius_y_percent))
    shape = str(config.shape or "circle").lower()
    if shape == "ellipse":
        return ((aim_x - center_x) / rx) ** 2 + ((aim_y - center_y) / ry) ** 2 <= 1.0
    radius = min(rx, ry)
    return ((aim_x - center_x) ** 2 + (aim_y - center_y) ** 2) ** 0.5 <= radius


def ratio_check_pass(target: Target, config: RatioCheckConfig) -> bool:
    width = float(target.w)
    height = float(target.h)
    if width <= 0 or height <= 0:
        return False
    ratio = max(width / height, height / width)
    return ratio <= max(1.0, float(config.max_aspect_ratio))


def candidate_debug(candidate: ScoredCandidate) -> dict:
    item = candidate.detection
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
        "selection_fov_pass": bool(candidate.selection_fov_pass),
        "ratio_valid": bool(candidate.ratio_valid),
        "conf_score": float(candidate.quality.conf_score),
        "area_score": float(candidate.quality.area_score),
        "quality_score": float(candidate.quality.quality_score),
    }


def rejected_debug(candidate: RejectedCandidate) -> dict:
    item = candidate.detection
    return {
        "cls": int(item.cls),
        "score": float(item.score),
        "x": float(item.x),
        "y": float(item.y),
        "w": float(item.w),
        "h": float(item.h),
        "reason": candidate.reason,
        "selection_fov_pass": bool(candidate.selection_fov_pass),
        "ratio_valid": bool(candidate.ratio_valid),
    }


def _bbox_valid(target: Target, context: FrameContext) -> bool:
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


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, float(value))) / 100.0

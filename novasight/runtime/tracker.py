from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import isfinite, log
from typing import Literal

from novasight.contracts import BBox, FrameContext, Track
from novasight.runtime.candidates import ScoredCandidate
from novasight.runtime.kalman import EstimatedState, KalmanConfig, KalmanEstimator


TrackState = Literal[
    "TENTATIVE",
    "CONFIRMED",
    "MISSING",
    "PREDICTING",
    "IDENTITY_UNCERTAIN",
    "OUT_OF_ROI",
    "LOST",
    "DELETED",
]

GlobalTrackerState = Literal[
    "IDLE",
    "ACQUIRING",
    "TRACKING",
    "PREDICTING",
    "IDENTITY_UNCERTAIN",
    "TARGET_UNAVAILABLE",
    "LOST",
    "COOLDOWN",
    "DISABLED",
]


@dataclass(frozen=True)
class TrackerConfig:
    confirm_frames: int = 2
    matching_distance_px: float = 140.0
    ambiguity_margin: float = 0.08
    missing_timeout_ms: float = 120.0
    delete_timeout_ms: float = 250.0
    max_size_ratio: float = 2.0
    position_weight: float = 0.55
    iou_weight: float = 0.30
    size_weight: float = 0.15
    match_threshold: float = 0.65
    mahalanobis_gate: float = 9.21
    kalman: KalmanConfig = field(default_factory=KalmanConfig)


@dataclass
class TrackRecord:
    track_id: int
    cls: int
    score: float
    box: BBox
    state: TrackState = "TENTATIVE"
    hits: int = 1
    misses: int = 0
    first_ts_ns: int = 0
    last_ts_ns: int = 0
    last_match_cost: float | None = None
    last_mahalanobis: float | None = None
    identity_confidence: float = 1.0
    switch_committed: bool = False
    estimator: KalmanEstimator | None = None
    estimate: EstimatedState | None = None

    def as_track(self) -> Track:
        box = self.box
        if self.state == "PREDICTING" and self.estimate is not None and self.estimate.valid:
            box = BBox.from_cxcywh(
                self.estimate.x,
                self.estimate.y,
                max(1.0, self.box.width),
                max(1.0, self.box.height),
            )
        return Track(
            track_id=self.track_id,
            cls=self.cls,
            score=self.score,
            box=box,
            velocity_px_s=(
                (float(self.estimate.vx), float(self.estimate.vy))
                if self.estimate is not None and self.estimate.valid
                else (0.0, 0.0)
            ),
            missed_frames=self.misses,
            last_seen_ns=self.last_ts_ns,
            is_predicted=self.state == "PREDICTING",
            is_stale=self.state in {"MISSING", "LOST", "OUT_OF_ROI"},
        )


@dataclass(frozen=True)
class TrackerUpdate:
    tracks: list[Track]
    state: str
    reason: str
    debug: dict


class RuntimeTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[int, TrackRecord] = {}
        self._next_track_id = 1
        self._last_frame_id: int | None = None
        self.last_debug: dict = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_frame_id = None
        self.last_debug = {}

    def update(
        self,
        context: FrameContext,
        candidates: list[ScoredCandidate],
        *,
        now_ns: int | None = None,
    ) -> TrackerUpdate:
        now = _frame_ts(context, now_ns)
        if self._last_frame_id == int(context.frame_id):
            return self._repeat_frame_update(context, candidates, now)
        self._last_frame_id = int(context.frame_id)
        self._expire_deleted(now)
        assignments: list[tuple[int, int, float]] = []
        uncertain_tracks: set[int] = set()
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()

        feasible_by_track = {
            track_id: self._feasible_candidates(track, candidates, now)
            for track_id, track in self._tracks.items()
            if track.state not in {"LOST", "DELETED", "OUT_OF_ROI"}
        }
        for track_id, feasible in feasible_by_track.items():
            if len(feasible) < 2:
                continue
            best_cost = feasible[0][1]
            second_cost = feasible[1][1]
            if second_cost - best_cost < max(0.0, self.config.ambiguity_margin):
                track = self._tracks[track_id]
                track.state = "IDENTITY_UNCERTAIN"
                track.identity_confidence = max(0.0, second_cost - best_cost)
                uncertain_tracks.add(track_id)

        pairs: list[tuple[float, int, int]] = []
        for track_id, feasible in feasible_by_track.items():
            if track_id in uncertain_tracks:
                continue
            for candidate_index, cost in feasible:
                if cost <= self.config.match_threshold:
                    pairs.append((cost, track_id, candidate_index))

        for cost, track_id, candidate_index in sorted(pairs, key=lambda item: item[0]):
            if track_id in used_tracks or candidate_index in used_candidates:
                continue
            self._update_track(self._tracks[track_id], candidates[candidate_index], cost, now)
            used_tracks.add(track_id)
            used_candidates.add(candidate_index)
            assignments.append((track_id, candidate_index, cost))

        for track_id, track in list(self._tracks.items()):
            if track_id in used_tracks or track.state in {"LOST", "DELETED", "OUT_OF_ROI"}:
                continue
            if track_id in uncertain_tracks:
                continue
            self._mark_missing(track, now, allow_prediction=not candidates)

        for index, candidate in enumerate(candidates):
            if index not in used_candidates:
                self._new_track(candidate, now)

        self._expire_deleted(now)
        available = [track.as_track() for track in self._tracks.values() if self._track_control_candidate(track)]
        state = self._summary_state()
        reason = self._summary_reason(state)
        self.last_debug = {
            "enabled": True,
            "state": state,
            "reason": reason,
            "tracks": [self._track_debug(track, now) for track in self._tracks.values()],
            "available_tracks": len(available),
            "assignments": [
                {"track_id": track_id, "candidate_index": candidate_index, "cost": cost}
                for track_id, candidate_index, cost in assignments
            ],
            "identity_uncertain_tracks": sorted(uncertain_tracks),
            "candidates": len(candidates),
            "config": self._config_debug(),
        }
        return TrackerUpdate(tracks=available, state=state, reason=reason, debug=self.last_debug)

    def _repeat_frame_update(
        self,
        context: FrameContext,
        candidates: list[ScoredCandidate],
        now_ns: int,
    ) -> TrackerUpdate:
        available = [track.as_track() for track in self._tracks.values() if self._track_control_candidate(track)]
        state = self._summary_state()
        reason = "repeat frame; tracker state unchanged"
        self.last_debug = {
            "enabled": True,
            "state": state,
            "reason": reason,
            "frame_id": int(context.frame_id),
            "repeat_frame": True,
            "tracks": [self._track_debug(track, now_ns) for track in self._tracks.values()],
            "available_tracks": len(available),
            "assignments": [],
            "identity_uncertain_tracks": [
                track.track_id for track in self._tracks.values() if track.state == "IDENTITY_UNCERTAIN"
            ],
            "candidates": len(candidates),
            "config": self._config_debug(),
        }
        return TrackerUpdate(tracks=available, state=state, reason=reason, debug=self.last_debug)

    def mark_unavailable(
        self,
        state: str,
        *,
        context: FrameContext,
        reason: str,
        now_ns: int | None = None,
    ) -> TrackerUpdate:
        now = _frame_ts(context, now_ns)
        global_state = _unavailable_global_state(state)
        track_state = _unavailable_track_state(state)
        for track in self._tracks.values():
            if track.state not in {"DELETED", "LOST"}:
                track.state = track_state
                track.misses += 1
                track.last_ts_ns = now
        self.last_debug = {
            "enabled": True,
            "state": global_state,
            "reason": reason,
            "unavailable_state": str(state),
            "tracks": [self._track_debug(track, now) for track in self._tracks.values()],
            "available_tracks": 0,
            "assignments": [],
            "identity_uncertain_tracks": [],
            "candidates": 0,
            "config": self._config_debug(),
        }
        return TrackerUpdate(tracks=[], state=global_state, reason=reason, debug=self.last_debug)

    def _feasible_candidates(
        self,
        track: TrackRecord,
        candidates: list[ScoredCandidate],
        now_ns: int,
    ) -> list[tuple[int, float]]:
        feasible: list[tuple[int, float]] = []
        for index, candidate in enumerate(candidates):
            detection = candidate.detection
            if int(detection.cls) != int(track.cls):
                continue
            if not _finite_box(detection.box) or not _finite_box(track.box):
                continue
            predicted_x, predicted_y = self._predicted_center(track)
            distance = _point_distance(predicted_x, predicted_y, detection.box.center_x, detection.box.center_y)
            if distance > self.config.matching_distance_px:
                continue
            nis = self._candidate_nis(track, detection.box, now_ns)
            if nis is not None and (
                not isfinite(nis) or nis > float(self.config.mahalanobis_gate)
            ):
                continue
            cost = self._cost(track, candidate, predicted_x, predicted_y, nis)
            if isfinite(cost):
                feasible.append((index, cost))
        return sorted(feasible, key=lambda item: item[1])

    def _cost(
        self,
        track: TrackRecord,
        candidate: ScoredCandidate,
        predicted_x: float,
        predicted_y: float,
        nis: float | None,
    ) -> float:
        det = candidate.detection
        distance = _point_distance(predicted_x, predicted_y, det.box.center_x, det.box.center_y)
        d_pred = _clamp01(distance / max(1.0, self.config.matching_distance_px))
        if nis is not None:
            d_pred = _clamp01(nis / max(1e-6, float(self.config.mahalanobis_gate)))
        d_iou = 1.0 - _iou(track.box, det.box)
        area_track = max(1e-6, track.box.area)
        area_det = max(1e-6, det.box.area)
        d_size = abs(log(area_det / area_track)) / log(max(1.01, self.config.max_size_ratio))
        d_size = _clamp01(d_size)
        wp = max(0.0, self.config.position_weight)
        wi = max(0.0, self.config.iou_weight)
        ws = max(0.0, self.config.size_weight)
        total = wp + wi + ws
        if total <= 0:
            return d_pred
        return (wp * d_pred + wi * d_iou + ws * d_size) / total

    def _new_track(self, candidate: ScoredCandidate, now_ns: int) -> TrackRecord:
        detection = candidate.detection
        hits = 1
        state: TrackState = "CONFIRMED" if hits >= self.config.confirm_frames else "TENTATIVE"
        track = TrackRecord(
            track_id=self._next_track_id,
            cls=int(detection.cls),
            score=float(detection.score),
            box=detection.box,
            state=state,
            hits=hits,
            first_ts_ns=now_ns,
            last_ts_ns=now_ns,
        )
        track.estimator = KalmanEstimator(
            track_id=track.track_id,
            x=detection.box.center_x,
            y=detection.box.center_y,
            ts_ns=now_ns,
            config=self.config.kalman,
            identity_confidence=track.identity_confidence,
        )
        track.estimate = track.estimator.last_estimate
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    def _update_track(
        self,
        track: TrackRecord,
        candidate: ScoredCandidate,
        cost: float,
        now_ns: int,
    ) -> None:
        detection = candidate.detection
        previous_state = track.state
        previous_center_x = track.box.center_x
        previous_center_y = track.box.center_y
        previous_ts_ns = int(track.last_ts_ns or now_ns)
        track.score = float(detection.score)
        track.box = detection.box
        track.hits += 1
        track.misses = 0
        track.last_ts_ns = now_ns
        track.last_match_cost = cost
        track.identity_confidence = _clamp01(1.0 - cost)
        if track.estimator is None:
            track.estimator = KalmanEstimator(
                track_id=track.track_id,
                x=detection.box.center_x,
                y=detection.box.center_y,
                ts_ns=now_ns,
                config=self.config.kalman,
                identity_confidence=track.identity_confidence,
            )
        else:
            track.estimator.update_config(self.config.kalman)
        track.last_mahalanobis = track.estimator.measurement_nis(
            detection.box.center_x,
            detection.box.center_y,
            now_ns,
        )
        track.estimate = track.estimator.update(
            measurement_x=detection.box.center_x,
            measurement_y=detection.box.center_y,
            ts_ns=now_ns,
            identity_confidence=track.identity_confidence,
        )
        dt_s = (int(now_ns) - previous_ts_ns) / 1e9
        if isfinite(dt_s) and dt_s > 0:
            observed_vx = (detection.box.center_x - previous_center_x) / dt_s
            observed_vy = (detection.box.center_y - previous_center_y) / dt_s
            velocity_weight = 0.60 if track.hits <= 2 else 0.35
            track.estimate = track.estimator.apply_velocity_hint(
                vx=observed_vx,
                vy=observed_vy,
                ts_ns=now_ns,
                weight=velocity_weight,
            )
        if track.hits >= self.config.confirm_frames:
            track.state = "CONFIRMED"
        else:
            track.state = "TENTATIVE"
        track.switch_committed = previous_state in {"TENTATIVE", "MISSING", "IDENTITY_UNCERTAIN"} and track.state == "CONFIRMED"

    def _mark_missing(self, track: TrackRecord, now_ns: int, *, allow_prediction: bool) -> None:
        track.misses += 1
        if track.estimator is not None:
            track.estimator.update_config(self.config.kalman)
            track.estimate = track.estimator.predict_only(
                ts_ns=now_ns,
                identity_confidence=track.identity_confidence,
            )
        missing_ms = (now_ns - int(track.last_ts_ns or now_ns)) / 1e6
        if missing_ms > self.config.missing_timeout_ms:
            track.state = "LOST"
        elif track.state == "TENTATIVE":
            track.state = "LOST"
        elif allow_prediction and track.estimate is not None and track.estimate.valid and track.estimate.predicted:
            track.state = "PREDICTING"
        else:
            track.state = "MISSING"

    def _expire_deleted(self, now_ns: int) -> None:
        for track_id, track in list(self._tracks.items()):
            if track.state not in {"LOST", "OUT_OF_ROI"}:
                continue
            age_ms = (now_ns - int(track.last_ts_ns or now_ns)) / 1e6
            if age_ms > self.config.delete_timeout_ms:
                track.state = "DELETED"
                self._tracks.pop(track_id, None)

    def _summary_state(self) -> GlobalTrackerState:
        states = {track.state for track in self._tracks.values()}
        if "IDENTITY_UNCERTAIN" in states:
            return "IDENTITY_UNCERTAIN"
        if "CONFIRMED" in states:
            return "TRACKING"
        if "PREDICTING" in states:
            return "PREDICTING"
        if "MISSING" in states:
            return "TARGET_UNAVAILABLE"
        if "TENTATIVE" in states:
            return "ACQUIRING"
        if "OUT_OF_ROI" in states:
            return "TARGET_UNAVAILABLE"
        if "LOST" in states:
            return "LOST"
        return "IDLE"

    @staticmethod
    def _summary_reason(state: str) -> str:
        return {
            "IDENTITY_UNCERTAIN": "association ambiguity margin violated",
            "TRACKING": "confirmed track available",
            "PREDICTING": "confirmed track temporarily missing; valid prediction available",
            "ACQUIRING": "tentative track awaiting confirmation",
            "TARGET_UNAVAILABLE": "track moved outside ROI/FOV",
            "LOST": "track lost",
            "COOLDOWN": "device output is cooling down",
            "DISABLED": "tracker disabled by runtime gate",
            "IDLE": "no active tracks",
        }.get(state, state)

    def _track_debug(self, track: TrackRecord, now_ns: int) -> dict:
        return {
            "track_id": track.track_id,
            "state": track.state,
            "cls": track.cls,
            "score": track.score,
            "hits": track.hits,
            "misses": track.misses,
            "missing_ms": max(0.0, (now_ns - int(track.last_ts_ns or now_ns)) / 1e6),
            "identity_confidence": track.identity_confidence,
            "last_match_cost": track.last_match_cost,
            "mahalanobis": track.last_mahalanobis,
            "switch_committed": track.switch_committed,
            "bbox": {
                "x1": track.box.x1,
                "y1": track.box.y1,
                "x2": track.box.x2,
                "y2": track.box.y2,
            },
            "estimate": _estimate_debug(track.estimate),
        }

    @staticmethod
    def _track_control_candidate(track: TrackRecord) -> bool:
        if track.state == "CONFIRMED":
            return True
        return (
            track.state == "PREDICTING"
            and track.estimate is not None
            and track.estimate.valid
            and track.estimate.predicted
        )

    def _config_debug(self) -> dict:
        return {
            "confirm_frames": self.config.confirm_frames,
            "matching_distance_px": self.config.matching_distance_px,
            "ambiguity_margin": self.config.ambiguity_margin,
            "missing_timeout_ms": self.config.missing_timeout_ms,
            "delete_timeout_ms": self.config.delete_timeout_ms,
            "max_size_ratio": self.config.max_size_ratio,
            "weights": {
                "position": self.config.position_weight,
                "iou": self.config.iou_weight,
                "size": self.config.size_weight,
            },
            "match_threshold": self.config.match_threshold,
            "mahalanobis_gate": self.config.mahalanobis_gate,
            "kalman": {
                "enabled": self.config.kalman.enabled,
                "acceleration_noise": self.config.kalman.acceleration_noise,
                "measurement_noise_x": self.config.kalman.measurement_noise_x,
                "measurement_noise_y": self.config.kalman.measurement_noise_y,
                "max_predict_missing_ms": self.config.kalman.max_predict_missing_ms,
                "max_predict_steps": self.config.kalman.max_predict_steps,
                "max_predict_dt_ms": self.config.kalman.max_predict_dt_ms,
                "max_position_sigma_px": self.config.kalman.max_position_sigma_px,
                "max_covariance_trace": self.config.kalman.max_covariance_trace,
                "nis_threshold": self.config.kalman.nis_threshold,
                "nis_hard_reject": self.config.kalman.nis_hard_reject,
                "min_identity_confidence": self.config.kalman.min_identity_confidence,
                "min_prediction_confidence": self.config.kalman.min_prediction_confidence,
                "prediction_decay_tau_ms": self.config.kalman.prediction_decay_tau_ms,
            },
        }

    def _predicted_center(self, track: TrackRecord) -> tuple[float, float]:
        estimate = track.estimate
        if estimate is None:
            return track.box.center_x, track.box.center_y
        return estimate.x, estimate.y

    def _candidate_nis(self, track: TrackRecord, box: BBox, now_ns: int) -> float | None:
        estimator = track.estimator
        if estimator is None or not self.config.kalman.enabled:
            return None
        estimator.update_config(self.config.kalman)
        return estimator.measurement_nis(box.center_x, box.center_y, now_ns)


def _frame_ts(context: FrameContext, now_ns: int | None) -> int:
    if now_ns is not None:
        return int(now_ns)
    if isinstance(context.capture_ts_ns, int) and context.capture_ts_ns > 0:
        return int(context.capture_ts_ns)
    return time.monotonic_ns()


def _finite_box(box: BBox) -> bool:
    return all(isfinite(float(value)) for value in (box.x1, box.y1, box.x2, box.y2)) and box.area > 0


def _center_distance(left: BBox, right: BBox) -> float:
    return ((left.center_x - right.center_x) ** 2 + (left.center_y - right.center_y) ** 2) ** 0.5


def _point_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2) ** 0.5


def _estimate_debug(estimate: EstimatedState | None) -> dict | None:
    if estimate is None:
        return None
    return {
        "track_id": estimate.track_id,
        "state_ts_ns": estimate.state_ts_ns,
        "x": estimate.x,
        "y": estimate.y,
        "vx": estimate.vx,
        "vy": estimate.vy,
        "cov_trace": estimate.cov_trace,
        "position_sigma_px": estimate.position_sigma_px,
        "nis": estimate.nis,
        "prediction_confidence": estimate.prediction_confidence,
        "predicted": estimate.predicted,
        "prediction_steps": estimate.prediction_steps,
        "valid": estimate.valid,
        "reason": estimate.reason,
    }


def _iou(left: BBox, right: BBox) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = left.area + right.area - intersection
    if union <= 0:
        return 0.0
    return _clamp01(intersection / union)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _unavailable_global_state(state: str) -> GlobalTrackerState:
    normalized = str(state or "").strip().upper()
    if normalized in {"COOLDOWN", "DISABLED", "LOST"}:
        return normalized  # type: ignore[return-value]
    return "TARGET_UNAVAILABLE"


def _unavailable_track_state(state: str) -> TrackState:
    normalized = str(state or "").strip().upper()
    if normalized == "OUT_OF_ROI":
        return "OUT_OF_ROI"
    if normalized == "LOST":
        return "LOST"
    return "LOST"

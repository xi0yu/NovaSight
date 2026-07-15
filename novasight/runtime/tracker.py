from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log, sqrt
from time import perf_counter_ns
from typing import Literal

from novasight.contracts import BBox, FrameContext, Track
from novasight.runtime.candidates import TrackObservation
from novasight.runtime.kalman import EstimatedState, KalmanConfig, KalmanEstimator


TrackState = Literal["TENTATIVE", "CONFIRMED", "LOST", "DELETED"]
GlobalTrackerState = Literal["IDLE", "ACQUIRING", "TRACKING", "LOST", "COOLDOWN", "DISABLED"]

_FORBIDDEN_COST = 1_000_000.0
MAX_ACTIVE_TRACKS = 16
MAX_DETECTIONS_FOR_ASSOCIATION = 16
TRACK_CONFIRM_HITS = 2
IMMEDIATE_CONFIRM_CONFIDENCE = 0.75


@dataclass(frozen=True)
class TrackerConfig:
    max_match_distance: float = 1.5
    position_cost_weight: float = 0.75
    iou_cost_weight: float = 0.25
    scale_cost_weight: float = 0.15
    max_size_ratio: float = 2.5
    max_association_dt_ms: float = 150.0
    max_missed_frames: int = 2
    max_lost_age_ms: float = 120.0
    kalman: KalmanConfig = field(default_factory=KalmanConfig)


@dataclass
class TrackRecord:
    track_id: int
    class_id: int
    confidence: float
    bbox: BBox
    observed_aim_x: float
    observed_aim_y: float
    filtered_x: float
    filtered_y: float
    velocity_x: float
    velocity_y: float
    last_capture_ts_ns: int
    status: TrackState = "TENTATIVE"
    confirmed: bool = False
    age_frames: int = 1
    hit_count: int = 1
    missed_count: int = 0
    velocity_valid: bool = False
    estimator: KalmanEstimator | None = None
    estimate: EstimatedState | None = None
    last_match_cost: float | None = None
    last_normalized_distance: float | None = None
    last_iou: float | None = None
    last_scale_cost: float | None = None
    last_mahalanobis_distance_sq: float | None = None
    identity_confidence: float = 1.0
    lost_since_ts_ns: int | None = None

    def to_output(self) -> Track:
        return Track(
            track_id=self.track_id,
            cls=self.class_id,
            score=self.confidence,
            box=self.bbox,
            velocity_px_s=(self.velocity_x, self.velocity_y),
            quality_score=_track_quality(self),
            observed_aim_px=(self.observed_aim_x, self.observed_aim_y),
            filtered_aim_px=(self.filtered_x, self.filtered_y),
            velocity_valid=self.velocity_valid,
            missed_frames=self.missed_count,
            last_seen_ns=self.last_capture_ts_ns,
            is_predicted=False,
            is_stale=False,
        )


@dataclass(frozen=True)
class TrackerResult:
    active_tracks: list[Track]
    lost_track_count: int
    created_track_ids: list[int]
    restored_track_ids: list[int]
    removed_track_ids: list[int]
    state: str
    reason: str
    debug: dict

    @property
    def tracks(self) -> list[Track]:
        return self.active_tracks


TrackerUpdate = TrackerResult


@dataclass(frozen=True)
class _AssociationEdge:
    cost: float
    normalized_distance: float
    iou: float
    scale_cost: float
    mahalanobis_distance_sq: float


class RuntimeTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[int, TrackRecord] = {}
        self._next_track_id = 1
        self._last_capture_ts_ns: int | None = None
        self._last_frame_id: int | None = None
        self.last_debug: dict = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_capture_ts_ns = None
        self._last_frame_id = None
        self.last_debug = {}

    def update(
        self,
        observations: list[TrackObservation],
        capture_ts_ns: int,
        *,
        frame_id: int | None = None,
    ) -> TrackerResult:
        tracker_start_ns = perf_counter_ns()
        input_candidate_count = len(observations)
        observations = self._limit_observations(observations)
        association_candidates_dropped = input_candidate_count - len(observations)
        capture_ts = int(capture_ts_ns)
        if capture_ts <= 0:
            return self._unchanged_result(
                capture_ts_ns=capture_ts,
                frame_id=frame_id,
                reason="capture_ts_ns must be positive",
            )
        if self._last_capture_ts_ns is not None and capture_ts <= self._last_capture_ts_ns:
            reason = (
                "repeat observation; tracker state unchanged"
                if capture_ts == self._last_capture_ts_ns
                else "non-monotonic capture_ts_ns; tracker state unchanged"
            )
            return self._unchanged_result(
                capture_ts_ns=capture_ts,
                frame_id=frame_id,
                reason=reason,
            )

        self._last_capture_ts_ns = capture_ts
        self._last_frame_id = int(frame_id) if frame_id is not None else None
        predict_start_ns = perf_counter_ns()
        self._predict_tracks(capture_ts)
        predict_end_ns = perf_counter_ns()
        track_ids = sorted(self._tracks)
        matrix_start_ns = perf_counter_ns()
        cost_matrix, edges = self._association_costs(track_ids, observations)
        matrix_end_ns = perf_counter_ns()
        hungarian_start_ns = perf_counter_ns()
        matches = self._hungarian_matches(track_ids, observations, cost_matrix, edges)
        hungarian_end_ns = perf_counter_ns()
        update_start_ns = perf_counter_ns()
        matched_track_ids = {track_id for track_id, _, _ in matches}
        matched_observation_indexes = {observation_index for _, observation_index, _ in matches}
        created_track_ids: list[int] = []
        restored_track_ids: list[int] = []
        removed_track_ids: list[int] = []

        for track_id, observation_index, edge in matches:
            track = self._tracks[track_id]
            was_lost = track.status == "LOST"
            self._update_matched_track(
                track,
                observations[observation_index],
                edge=edge,
                capture_ts_ns=capture_ts,
            )
            if was_lost:
                restored_track_ids.append(track_id)

        for track_id in track_ids:
            if track_id not in matched_track_ids:
                self._mark_lost(self._tracks[track_id])

        for track_id, track in list(self._tracks.items()):
            lost_age_ms = (
                max(0.0, (capture_ts - track.lost_since_ts_ns) / 1e6)
                if track.lost_since_ts_ns is not None
                else 0.0
            )
            if track.status == "LOST" and (
                track.missed_count > max(0, int(self.config.max_missed_frames))
                or lost_age_ms > max(0.0, float(self.config.max_lost_age_ms))
            ):
                track.status = "DELETED"
                removed_track_ids.append(track_id)
                self._tracks.pop(track_id, None)

        capacity_dropped_detection_ids: list[int] = []
        for observation_index, observation in enumerate(observations):
            if observation_index in matched_observation_indexes:
                continue
            if len(self._tracks) >= MAX_ACTIVE_TRACKS:
                capacity_dropped_detection_ids.append(observation.detection_index)
                continue
            track = self._create_track(
                observation,
                capture_ts_ns=capture_ts,
                confirm_immediately=(
                    len(observations) == 1
                    and float(observation.confidence) >= IMMEDIATE_CONFIRM_CONFIDENCE
                ),
            )
            created_track_ids.append(track.track_id)

        active_records = [
            track for track in self._tracks.values() if track.status == "CONFIRMED"
        ]
        active_records.sort(key=lambda item: item.track_id)
        active_tracks = [track.to_output() for track in active_records]
        lost_count = sum(track.status == "LOST" for track in self._tracks.values())
        tentative_count = sum(track.status == "TENTATIVE" for track in self._tracks.values())
        state: GlobalTrackerState = (
            "TRACKING"
            if active_tracks
            else "ACQUIRING"
            if tentative_count
            else "LOST"
            if lost_count
            else "IDLE"
        )
        reason = {
            "TRACKING": "confirmed tracks updated from current detections",
            "ACQUIRING": "tentative tracks require another matching observation",
            "LOST": "tracks retained for association but excluded from control",
            "IDLE": "no active or retained tracks",
        }[state]
        assignment_debug = [
            {
                "track_id": track_id,
                "detection_index": observations[observation_index].detection_index,
                "cost": edge.cost,
                "normalized_distance": edge.normalized_distance,
                "iou": edge.iou,
                "scale_cost": edge.scale_cost,
                "mahalanobis_distance_sq": edge.mahalanobis_distance_sq,
            }
            for track_id, observation_index, edge in matches
        ]
        update_end_ns = perf_counter_ns()
        timing = {
            "tracker_predict_us": _elapsed_us(predict_start_ns, predict_end_ns),
            "association_matrix_us": _elapsed_us(matrix_start_ns, matrix_end_ns),
            "hungarian_us": _elapsed_us(hungarian_start_ns, hungarian_end_ns),
            "tracker_update_us": _elapsed_us(update_start_ns, update_end_ns),
            "tracker_total_us": 0.0,
        }
        self.last_debug = self._debug_payload(
            state=state,
            reason=reason,
            capture_ts_ns=capture_ts,
            frame_id=frame_id,
            observations=observations,
            assignments=assignment_debug,
            created_track_ids=created_track_ids,
            restored_track_ids=restored_track_ids,
            removed_track_ids=removed_track_ids,
            input_candidate_count=input_candidate_count,
            association_candidates_dropped=association_candidates_dropped,
            capacity_dropped_detection_ids=capacity_dropped_detection_ids,
            timing=timing,
        )
        timing["tracker_total_us"] = _elapsed_us(tracker_start_ns, perf_counter_ns())
        return TrackerResult(
            active_tracks=active_tracks,
            lost_track_count=lost_count,
            created_track_ids=created_track_ids,
            restored_track_ids=restored_track_ids,
            removed_track_ids=removed_track_ids,
            state=state,
            reason=reason,
            debug=self.last_debug,
        )

    @staticmethod
    def _limit_observations(observations: list[TrackObservation]) -> list[TrackObservation]:
        if len(observations) <= MAX_DETECTIONS_FOR_ASSOCIATION:
            return list(observations)
        ranked = sorted(
            enumerate(observations),
            key=lambda item: (
                -float(item[1].confidence),
                -float(item[1].bbox.area),
                int(item[1].detection_index),
            ),
        )[:MAX_DETECTIONS_FOR_ASSOCIATION]
        ranked.sort(key=lambda item: item[0])
        return [observation for _, observation in ranked]

    def mark_unavailable(
        self,
        state: str,
        *,
        context: FrameContext,
        reason: str,
        now_ns: int | None = None,
    ) -> TrackerResult:
        capture_ts_ns = int(now_ns or context.capture_ts_ns or 0)
        result = self.update([], capture_ts_ns, frame_id=context.frame_id)
        global_state = str(state or "").strip().upper()
        if global_state not in {"COOLDOWN", "DISABLED"}:
            global_state = result.state
        self.last_debug = {
            **result.debug,
            "state": global_state,
            "reason": reason,
            "unavailable_state": str(state),
        }
        return TrackerResult(
            active_tracks=[],
            lost_track_count=result.lost_track_count,
            created_track_ids=result.created_track_ids,
            restored_track_ids=result.restored_track_ids,
            removed_track_ids=result.removed_track_ids,
            state=global_state,
            reason=reason,
            debug=self.last_debug,
        )

    def _predict_tracks(self, capture_ts_ns: int) -> None:
        for track in self._tracks.values():
            if track.status not in {"TENTATIVE", "CONFIRMED", "LOST"} or track.estimator is None:
                continue
            track.age_frames += 1
            track.estimator.update_config(self.config.kalman)
            estimate = track.estimator.predict_only(
                ts_ns=capture_ts_ns,
                identity_confidence=track.identity_confidence,
            )
            track.estimate = estimate
            if isfinite(estimate.x) and isfinite(estimate.y):
                track.filtered_x = float(estimate.x)
                track.filtered_y = float(estimate.y)
            if isfinite(estimate.vx) and isfinite(estimate.vy):
                track.velocity_x = float(estimate.vx)
                track.velocity_y = float(estimate.vy)

    def _association_costs(
        self,
        track_ids: list[int],
        observations: list[TrackObservation],
    ) -> tuple[list[list[float]], dict[tuple[int, int], _AssociationEdge]]:
        matrix: list[list[float]] = []
        edges: dict[tuple[int, int], _AssociationEdge] = {}
        max_match_distance = max(0.0, float(self.config.max_match_distance))
        prepared_observations = [
            (
                observation,
                int(observation.class_id),
                float(observation.aim_x),
                float(observation.aim_y),
                float(observation.bbox.height),
            )
            for observation in observations
        ]
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            track_class_id = int(track.class_id)
            track_x = float(track.filtered_x)
            track_y = float(track.filtered_y)
            track_height = float(track.bbox.height)
            costs = [_FORBIDDEN_COST] * len(prepared_observations)
            for column, prepared in enumerate(prepared_observations):
                observation, class_id, aim_x, aim_y, observation_height = prepared
                reference_height = max(track_height, observation_height)
                dx = track_x - aim_x
                dy = track_y - aim_y
                distance_limit = max_match_distance * reference_height
                if (
                    track_class_id != class_id
                    or not isfinite(reference_height)
                    or reference_height <= 0.0
                    or dx * dx + dy * dy > distance_limit * distance_limit
                ):
                    continue
                edge = self._association_edge(track, observation)
                if edge is None:
                    continue
                costs[column] = edge.cost
                edges[(row, column)] = edge
            matrix.append(costs)
        return matrix, edges

    def _association_edge(
        self,
        track: TrackRecord,
        observation: TrackObservation,
    ) -> _AssociationEdge | None:
        if track.status not in {"TENTATIVE", "CONFIRMED", "LOST"}:
            return None
        if track.class_id != observation.class_id:
            return None
        if not _finite_box(track.bbox) or not _finite_box(observation.bbox):
            return None
        association_dt_ms = max(
            0.0,
            (int(self._last_capture_ts_ns or 0) - int(track.last_capture_ts_ns)) / 1e6,
        )
        if association_dt_ms > max(0.0, float(self.config.max_association_dt_ms)):
            return None
        mahalanobis_distance_sq = 0.0
        if track.estimator is not None:
            mahalanobis_distance_sq = track.estimator.measurement_nis_current(
                observation.aim_x,
                observation.aim_y,
            )
            if (
                not isfinite(mahalanobis_distance_sq)
                or mahalanobis_distance_sq > float(self.config.kalman.nis_hard_reject)
            ):
                return None
        distance_px = _point_distance(
            track.filtered_x,
            track.filtered_y,
            observation.aim_x,
            observation.aim_y,
        )
        reference_height = max(track.bbox.height, observation.bbox.height)
        if not isfinite(reference_height) or reference_height <= 0:
            return None
        normalized_distance = distance_px / reference_height
        if (
            not isfinite(normalized_distance)
            or normalized_distance > max(0.0, float(self.config.max_match_distance))
        ):
            return None
        width_ratio = _symmetric_ratio(track.bbox.width, observation.bbox.width)
        height_ratio = _symmetric_ratio(track.bbox.height, observation.bbox.height)
        max_size_ratio = max(1.0, float(self.config.max_size_ratio))
        if width_ratio > max_size_ratio or height_ratio > max_size_ratio:
            return None
        iou = _iou(track.bbox, observation.bbox)
        scale_cost = abs(log(track.bbox.width / observation.bbox.width)) + abs(
            log(track.bbox.height / observation.bbox.height)
        )
        position_weight = max(0.0, float(self.config.position_cost_weight))
        iou_weight = max(0.0, float(self.config.iou_cost_weight))
        scale_weight = max(0.0, float(self.config.scale_cost_weight))
        total_weight = position_weight + iou_weight + scale_weight
        if total_weight <= 0:
            position_weight = 1.0
            total_weight = 1.0
        cost = (
            position_weight * normalized_distance
            + iou_weight * (1.0 - iou)
            + scale_weight * scale_cost
        ) / total_weight
        return _AssociationEdge(
            cost=float(cost),
            normalized_distance=float(normalized_distance),
            iou=float(iou),
            scale_cost=float(scale_cost),
            mahalanobis_distance_sq=float(mahalanobis_distance_sq),
        )

    @staticmethod
    def _hungarian_matches(
        track_ids: list[int],
        observations: list[TrackObservation],
        cost_matrix: list[list[float]],
        edges: dict[tuple[int, int], _AssociationEdge],
    ) -> list[tuple[int, int, _AssociationEdge]]:
        if not track_ids or not observations:
            return []
        matches: list[tuple[int, int, _AssociationEdge]] = []
        for row, column in _linear_sum_assignment(cost_matrix):
            edge = edges.get((row, column))
            if edge is not None:
                matches.append((track_ids[row], column, edge))
        return matches

    def _update_matched_track(
        self,
        track: TrackRecord,
        observation: TrackObservation,
        *,
        edge: _AssociationEdge,
        capture_ts_ns: int,
    ) -> None:
        previous_capture_ts_ns = track.last_capture_ts_ns
        track.identity_confidence = _clamp01(1.0 - edge.cost)
        if track.estimator is None:
            track.estimator = self._new_estimator(
                track_id=track.track_id,
                x=observation.aim_x,
                y=observation.aim_y,
                capture_ts_ns=capture_ts_ns,
                identity_confidence=track.identity_confidence,
            )
            estimate = track.estimator.last_estimate
        else:
            track.estimator.update_config(self.config.kalman)
            estimate = track.estimator.update(
                measurement_x=observation.aim_x,
                measurement_y=observation.aim_y,
                ts_ns=capture_ts_ns,
                identity_confidence=track.identity_confidence,
            )
        track.estimate = estimate
        track.class_id = observation.class_id
        track.confidence = observation.confidence
        track.bbox = observation.bbox
        track.observed_aim_x = observation.aim_x
        track.observed_aim_y = observation.aim_y
        track.filtered_x = estimate.x
        track.filtered_y = estimate.y
        track.velocity_x = estimate.vx
        track.velocity_y = estimate.vy
        track.last_capture_ts_ns = capture_ts_ns
        track.hit_count += 1
        track.missed_count = 0
        track.confirmed = track.confirmed or track.hit_count >= TRACK_CONFIRM_HITS
        track.status = "CONFIRMED" if track.confirmed else "TENTATIVE"
        track.lost_since_ts_ns = None
        track.velocity_valid = capture_ts_ns > previous_capture_ts_ns and track.hit_count >= 2
        track.last_match_cost = edge.cost
        track.last_normalized_distance = edge.normalized_distance
        track.last_iou = edge.iou
        track.last_scale_cost = edge.scale_cost
        track.last_mahalanobis_distance_sq = edge.mahalanobis_distance_sq

    def _mark_lost(self, track: TrackRecord) -> None:
        track.status = "LOST"
        track.missed_count += 1
        if track.lost_since_ts_ns is None:
            track.lost_since_ts_ns = int(self._last_capture_ts_ns or track.last_capture_ts_ns)

    def _create_track(
        self,
        observation: TrackObservation,
        *,
        capture_ts_ns: int,
        confirm_immediately: bool,
    ) -> TrackRecord:
        track_id = self._next_track_id
        self._next_track_id += 1
        estimator = self._new_estimator(
            track_id=track_id,
            x=observation.aim_x,
            y=observation.aim_y,
            capture_ts_ns=capture_ts_ns,
            identity_confidence=1.0,
        )
        track = TrackRecord(
            track_id=track_id,
            class_id=observation.class_id,
            confidence=observation.confidence,
            bbox=observation.bbox,
            observed_aim_x=observation.aim_x,
            observed_aim_y=observation.aim_y,
            filtered_x=observation.aim_x,
            filtered_y=observation.aim_y,
            velocity_x=0.0,
            velocity_y=0.0,
            last_capture_ts_ns=capture_ts_ns,
            status="CONFIRMED" if confirm_immediately else "TENTATIVE",
            confirmed=bool(confirm_immediately),
            estimator=estimator,
            estimate=estimator.last_estimate,
        )
        self._tracks[track_id] = track
        return track

    def _new_estimator(
        self,
        *,
        track_id: int,
        x: float,
        y: float,
        capture_ts_ns: int,
        identity_confidence: float,
    ) -> KalmanEstimator:
        return KalmanEstimator(
            track_id=track_id,
            x=x,
            y=y,
            ts_ns=capture_ts_ns,
            config=self.config.kalman,
            identity_confidence=identity_confidence,
        )

    def _unchanged_result(
        self,
        *,
        capture_ts_ns: int,
        frame_id: int | None,
        reason: str,
    ) -> TrackerResult:
        active_records = sorted(
            (track for track in self._tracks.values() if track.status == "CONFIRMED"),
            key=lambda item: item.track_id,
        )
        active_tracks = [track.to_output() for track in active_records]
        lost_count = sum(track.status == "LOST" for track in self._tracks.values())
        tentative_count = sum(track.status == "TENTATIVE" for track in self._tracks.values())
        state: GlobalTrackerState = (
            "TRACKING"
            if active_tracks
            else "ACQUIRING"
            if tentative_count
            else "LOST"
            if lost_count
            else "IDLE"
        )
        self.last_debug = self._debug_payload(
            state=state,
            reason=reason,
            capture_ts_ns=capture_ts_ns,
            frame_id=frame_id,
            observations=[],
            assignments=[],
            created_track_ids=[],
            restored_track_ids=[],
            removed_track_ids=[],
            repeat_observation=True,
        )
        return TrackerResult(
            active_tracks=active_tracks,
            lost_track_count=lost_count,
            created_track_ids=[],
            restored_track_ids=[],
            removed_track_ids=[],
            state=state,
            reason=reason,
            debug=self.last_debug,
        )

    def _debug_payload(
        self,
        *,
        state: str,
        reason: str,
        capture_ts_ns: int,
        frame_id: int | None,
        observations: list[TrackObservation],
        assignments: list[dict],
        created_track_ids: list[int],
        restored_track_ids: list[int],
        removed_track_ids: list[int],
        repeat_observation: bool = False,
        input_candidate_count: int | None = None,
        association_candidates_dropped: int = 0,
        capacity_dropped_detection_ids: list[int] | None = None,
        timing: dict[str, float] | None = None,
    ) -> dict:
        active_count = sum(track.status == "CONFIRMED" for track in self._tracks.values())
        tentative_count = sum(track.status == "TENTATIVE" for track in self._tracks.values())
        lost_count = sum(track.status == "LOST" for track in self._tracks.values())
        return {
            "enabled": True,
            "association_algorithm": "hungarian",
            "state": state,
            "reason": reason,
            "frame_id": frame_id,
            "capture_ts_ns": capture_ts_ns,
            "repeat_frame": repeat_observation,
            "tracks": [
                self._track_debug(track)
                for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
            ],
            "available_tracks": active_count,
            "active_tracks": active_count,
            "confirmed_tracks": active_count,
            "tentative_tracks": tentative_count,
            "lost_track_count": lost_count,
            "assignments": assignments,
            "created_track_ids": list(created_track_ids),
            "restored_track_ids": list(restored_track_ids),
            "removed_track_ids": list(removed_track_ids),
            "identity_uncertain_tracks": [],
            "candidates": len(observations),
            "input_candidates": len(observations) if input_candidate_count is None else input_candidate_count,
            "association_candidates": len(observations),
            "association_candidates_dropped": association_candidates_dropped,
            "capacity_dropped_detection_ids": list(capacity_dropped_detection_ids or []),
            "max_active_tracks": MAX_ACTIVE_TRACKS,
            "max_detections_for_association": MAX_DETECTIONS_FOR_ASSOCIATION,
            "timing": timing or {
                "tracker_predict_us": 0.0,
                "association_matrix_us": 0.0,
                "hungarian_us": 0.0,
                "tracker_update_us": 0.0,
                "tracker_total_us": 0.0,
            },
            "config": {
                "max_match_distance": self.config.max_match_distance,
                "position_cost_weight": self.config.position_cost_weight,
                "iou_cost_weight": self.config.iou_cost_weight,
                "scale_cost_weight": self.config.scale_cost_weight,
                "max_size_ratio": self.config.max_size_ratio,
                "max_association_dt_ms": self.config.max_association_dt_ms,
                "max_missed_frames": self.config.max_missed_frames,
                "max_lost_age_ms": self.config.max_lost_age_ms,
                "kalman": {
                    "acceleration_noise": self.config.kalman.acceleration_noise,
                    "measurement_noise_x": self.config.kalman.measurement_noise_x,
                    "measurement_noise_y": self.config.kalman.measurement_noise_y,
                },
            },
        }

    def _track_debug(self, track: TrackRecord) -> dict:
        current_capture_ts_ns = int(self._last_capture_ts_ns or track.last_capture_ts_ns)
        return {
            "track_id": track.track_id,
            "state": track.status,
            "status": track.status,
            "confirmed": track.confirmed,
            "age_frames": track.age_frames,
            "cls": track.class_id,
            "class_id": track.class_id,
            "score": track.confidence,
            "confidence": track.confidence,
            "hits": track.hit_count,
            "hit_count": track.hit_count,
            "misses": track.missed_count,
            "missed_count": track.missed_count,
            "missing_ms": max(
                0.0,
                (current_capture_ts_ns - track.last_capture_ts_ns) / 1e6,
            ),
            "last_capture_ts_ns": track.last_capture_ts_ns,
            "lost_since_ts_ns": track.lost_since_ts_ns,
            "observed_aim_x": track.observed_aim_x,
            "observed_aim_y": track.observed_aim_y,
            "filtered_x": track.filtered_x,
            "filtered_y": track.filtered_y,
            "velocity_x": track.velocity_x,
            "velocity_y": track.velocity_y,
            "velocity_valid": track.velocity_valid,
            "identity_confidence": track.identity_confidence,
            "track_quality": _track_quality(track),
            "last_match_cost": track.last_match_cost,
            "normalized_distance": track.last_normalized_distance,
            "iou": track.last_iou,
            "scale_cost": track.last_scale_cost,
            "mahalanobis_distance_sq": track.last_mahalanobis_distance_sq,
            "bbox": {
                "x1": track.bbox.x1,
                "y1": track.bbox.y1,
                "x2": track.bbox.x2,
                "y2": track.bbox.y2,
            },
            "estimate": _estimate_debug(track.estimate),
        }


def _linear_sum_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Return a deterministic minimum-cost rectangular assignment in O(n^3)."""
    if not cost_matrix or not cost_matrix[0]:
        return []
    row_count = len(cost_matrix)
    column_count = len(cost_matrix[0])
    if any(len(row) != column_count for row in cost_matrix):
        raise ValueError("cost matrix rows must have equal length")
    transposed = row_count > column_count
    matrix = (
        [[float(cost_matrix[row][column]) for row in range(row_count)] for column in range(column_count)]
        if transposed
        else [[float(value) for value in row] for row in cost_matrix]
    )
    rows = len(matrix)
    columns = len(matrix[0])
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        min_values = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1][column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment: list[tuple[int, int]] = []
    for column in range(1, columns + 1):
        if p[column] == 0:
            continue
        row_index = p[column] - 1
        column_index = column - 1
        assignment.append(
            (column_index, row_index) if transposed else (row_index, column_index)
        )
    return sorted(assignment)


def _finite_box(box: BBox) -> bool:
    return all(
        isfinite(float(value))
        for value in (box.x1, box.y1, box.x2, box.y2)
    ) and box.area > 0


def _point_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2) ** 0.5


def _symmetric_ratio(left: float, right: float) -> float:
    if not isfinite(left) or not isfinite(right) or left <= 0.0 or right <= 0.0:
        return float("inf")
    return max(float(left) / float(right), float(right) / float(left))


def _track_quality(track: TrackRecord) -> float:
    estimate = track.estimate
    if estimate is None or not isfinite(float(estimate.position_sigma_px)):
        stability = 1.0
    else:
        characteristic_size = max(1.0, sqrt(max(1.0, float(track.bbox.area))))
        relative_sigma = max(0.0, float(estimate.position_sigma_px)) / characteristic_size
        stability = 1.0 / (1.0 + relative_sigma)
    return _clamp01(
        float(track.confidence)
        * float(track.identity_confidence)
        * stability
    )


def _elapsed_us(start_ns: int, end_ns: int) -> float:
    return max(0.0, (int(end_ns) - int(start_ns)) / 1_000.0)


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
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = left.area + right.area - intersection
    if union <= 0:
        return 0.0
    return _clamp01(intersection / union)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "RuntimeTracker",
    "TrackerConfig",
    "TrackerResult",
    "TrackerUpdate",
]

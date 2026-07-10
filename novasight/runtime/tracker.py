from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Literal

from novasight.contracts import BBox, FrameContext, Track
from novasight.runtime.candidates import TrackObservation
from novasight.runtime.kalman import EstimatedState, KalmanConfig, KalmanEstimator


TrackState = Literal["ACTIVE", "LOST", "REMOVED"]
GlobalTrackerState = Literal["IDLE", "TRACKING", "LOST", "COOLDOWN", "DISABLED"]

_FORBIDDEN_COST = 1_000_000.0


@dataclass(frozen=True)
class TrackerConfig:
    max_match_distance: float = 1.5
    position_cost_weight: float = 0.75
    iou_cost_weight: float = 0.25
    max_missed_frames: int = 2
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
    status: TrackState = "ACTIVE"
    hit_count: int = 1
    missed_count: int = 0
    velocity_valid: bool = False
    estimator: KalmanEstimator | None = None
    estimate: EstimatedState | None = None
    last_match_cost: float | None = None
    last_normalized_distance: float | None = None
    last_iou: float | None = None
    identity_confidence: float = 1.0

    def to_output(self) -> Track:
        return Track(
            track_id=self.track_id,
            cls=self.class_id,
            score=self.confidence,
            box=self.bbox,
            velocity_px_s=(self.velocity_x, self.velocity_y),
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
        self._predict_tracks(capture_ts)
        track_ids = sorted(self._tracks)
        cost_matrix, edges = self._association_costs(track_ids, observations)
        matches = self._hungarian_matches(track_ids, observations, cost_matrix, edges)
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

        for observation_index, observation in enumerate(observations):
            if observation_index in matched_observation_indexes:
                continue
            track = self._create_track(observation, capture_ts_ns=capture_ts)
            created_track_ids.append(track.track_id)

        for track_id, track in list(self._tracks.items()):
            if track.status == "LOST" and track.missed_count > max(0, int(self.config.max_missed_frames)):
                track.status = "REMOVED"
                removed_track_ids.append(track_id)
                self._tracks.pop(track_id, None)

        active_records = [
            track for track in self._tracks.values() if track.status == "ACTIVE"
        ]
        active_records.sort(key=lambda item: item.track_id)
        active_tracks = [track.to_output() for track in active_records]
        lost_count = sum(track.status == "LOST" for track in self._tracks.values())
        state: GlobalTrackerState = "TRACKING" if active_tracks else "LOST" if lost_count else "IDLE"
        reason = {
            "TRACKING": "active tracks updated from current detections",
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
            }
            for track_id, observation_index, edge in matches
        ]
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
        )
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
            if track.status not in {"ACTIVE", "LOST"} or track.estimator is None:
                continue
            track.estimator.update_config(_association_kalman_config(self.config.kalman))
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
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            costs: list[float] = []
            for column, observation in enumerate(observations):
                edge = self._association_edge(track, observation)
                if edge is None:
                    costs.append(_FORBIDDEN_COST)
                    continue
                costs.append(edge.cost)
                edges[(row, column)] = edge
            matrix.append(costs)
        return matrix, edges

    def _association_edge(
        self,
        track: TrackRecord,
        observation: TrackObservation,
    ) -> _AssociationEdge | None:
        if track.status not in {"ACTIVE", "LOST"}:
            return None
        if track.class_id != observation.class_id:
            return None
        if not _finite_box(track.bbox) or not _finite_box(observation.bbox):
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
        iou = _iou(track.bbox, observation.bbox)
        position_weight = max(0.0, float(self.config.position_cost_weight))
        iou_weight = max(0.0, float(self.config.iou_cost_weight))
        total_weight = position_weight + iou_weight
        if total_weight <= 0:
            position_weight = 1.0
            total_weight = 1.0
        cost = (
            position_weight * normalized_distance
            + iou_weight * (1.0 - iou)
        ) / total_weight
        return _AssociationEdge(
            cost=float(cost),
            normalized_distance=float(normalized_distance),
            iou=float(iou),
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
            track.estimator.update_config(_association_kalman_config(self.config.kalman))
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
        track.status = "ACTIVE"
        track.velocity_valid = capture_ts_ns > previous_capture_ts_ns and track.hit_count >= 2
        track.last_match_cost = edge.cost
        track.last_normalized_distance = edge.normalized_distance
        track.last_iou = edge.iou

    def _mark_lost(self, track: TrackRecord) -> None:
        track.status = "LOST"
        track.missed_count += 1

    def _create_track(
        self,
        observation: TrackObservation,
        *,
        capture_ts_ns: int,
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
            config=_association_kalman_config(self.config.kalman),
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
            (track for track in self._tracks.values() if track.status == "ACTIVE"),
            key=lambda item: item.track_id,
        )
        active_tracks = [track.to_output() for track in active_records]
        lost_count = sum(track.status == "LOST" for track in self._tracks.values())
        state: GlobalTrackerState = "TRACKING" if active_tracks else "LOST" if lost_count else "IDLE"
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
    ) -> dict:
        active_count = sum(track.status == "ACTIVE" for track in self._tracks.values())
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
            "lost_track_count": lost_count,
            "assignments": assignments,
            "created_track_ids": list(created_track_ids),
            "restored_track_ids": list(restored_track_ids),
            "removed_track_ids": list(removed_track_ids),
            "identity_uncertain_tracks": [],
            "candidates": len(observations),
            "config": {
                "max_match_distance": self.config.max_match_distance,
                "position_cost_weight": self.config.position_cost_weight,
                "iou_cost_weight": self.config.iou_cost_weight,
                "max_missed_frames": self.config.max_missed_frames,
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
            "observed_aim_x": track.observed_aim_x,
            "observed_aim_y": track.observed_aim_y,
            "filtered_x": track.filtered_x,
            "filtered_y": track.filtered_y,
            "velocity_x": track.velocity_x,
            "velocity_y": track.velocity_y,
            "velocity_valid": track.velocity_valid,
            "identity_confidence": track.identity_confidence,
            "last_match_cost": track.last_match_cost,
            "normalized_distance": track.last_normalized_distance,
            "iou": track.last_iou,
            "bbox": {
                "x1": track.bbox.x1,
                "y1": track.bbox.y1,
                "x2": track.bbox.x2,
                "y2": track.bbox.y2,
            },
            "estimate": _estimate_debug(track.estimate),
        }


def _association_kalman_config(config: KalmanConfig) -> KalmanConfig:
    return replace(
        config,
        max_predict_missing_ms=1_000_000_000.0,
        max_predict_steps=1_000_000_000,
        max_predict_dt_ms=1_000_000_000.0,
        max_position_sigma_px=1_000_000_000.0,
        max_covariance_trace=1_000_000_000_000.0,
        nis_threshold=1_000_000_000_000.0,
        nis_hard_reject=1_000_000_000_000.0,
        min_identity_confidence=0.0,
        min_prediction_confidence=0.0,
    )


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

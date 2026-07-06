from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from novasight.hardware import BoxInputState
from novasight.contracts import ControlIntent, Detection, Track


Target = Detection | Track
MOVE_KINDS = {"raw", "enc_raw", "auto", "enc_auto", "bezier", "enc_bezier"}


def aim_point(target: Target, aim_ratio: float) -> tuple[float, float]:
    ratio = max(0.0, min(100.0, aim_ratio)) / 100.0
    point_y = getattr(target, "point_y", None)
    y = point_y(ratio) if callable(point_y) else float(target.y) + float(target.h) * ratio
    return (float(target.cx), float(y))


def aim_point_y_ratio(target: Target, aim_y_ratio: float) -> tuple[float, float]:
    return aim_point(target, aim_y_ratio)


@dataclass(frozen=True)
class MoveCommand:
    dx: float
    dy: float
    confidence: float
    reason: str
    move_kind: str = "raw"
    move_ms: int = 0
    trace_ms: int = 0
    bezier_ctrl: tuple[int, int, int, int] | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class IControlStrategy(Protocol):
    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        ...


@dataclass(slots=True)
class _AnglePidAxis:
    kp: float = 0.35
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.0

    integral: float = 0.0
    prev_error: float | None = None
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0

    def update(self, error_rad: float, dt: float) -> float:
        dt = max(1e-6, float(dt))
        self.p_term = self.kp * error_rad
        self.integral += error_rad * dt
        if self.integral_limit > 0:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        self.i_term = self.ki * self.integral
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error_rad - self.prev_error) / dt
        self.d_term = self.kd * derivative
        self.prev_error = error_rad
        return self.p_term + self.i_term + self.d_term

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0


@dataclass(slots=True)
class _KalmanAxis:
    position: float
    velocity: float = 0.0
    p00: float = 20.0
    p01: float = 0.0
    p10: float = 0.0
    p11: float = 20.0

    def predict(self, dt: float, process_noise: float) -> float:
        dt = max(1e-6, float(dt))
        q = max(1e-6, float(process_noise))
        self.position += self.velocity * dt
        p00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + q
        p01 = self.p01 + dt * self.p11
        p10 = self.p10 + dt * self.p11
        p11 = self.p11 + q
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11
        return self.position

    def update(self, measurement: float, measurement_noise: float) -> float:
        r = max(1e-6, float(measurement_noise))
        innovation = float(measurement) - self.position
        s = self.p00 + r
        k0 = self.p00 / s
        k1 = self.p10 / s
        self.position += k0 * innovation
        self.velocity += k1 * innovation
        p00 = (1.0 - k0) * self.p00
        p01 = (1.0 - k0) * self.p01
        p10 = self.p10 - k1 * self.p00
        p11 = self.p11 - k1 * self.p01
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11
        return self.position


@dataclass(slots=True)
class _KalmanCenterTrack:
    track_id: int
    cls: int
    source_index: int | None
    kx: _KalmanAxis
    ky: _KalmanAxis
    w: float
    h: float
    score: float
    missed: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return self.kx.position, self.ky.position

    def predict(self, dt: float, process_noise: float) -> None:
        self.kx.predict(dt, process_noise)
        self.ky.predict(dt, process_noise)
        self.missed += 1

    def update(self, detection: dict[str, float], measurement_noise: float) -> None:
        self.kx.update(float(detection["cx"]), measurement_noise)
        self.ky.update(float(detection["cy"]), measurement_noise)
        self.w = float(detection["w"])
        self.h = float(detection["h"])
        self.score = float(detection["score"])
        self.cls = int(detection["cls"])
        raw_index = detection.get("index")
        self.source_index = int(raw_index) if isinstance(raw_index, (int, float)) else None
        self.missed = 0


class ExperimentalAnglePidStrategy:
    def __init__(
        self,
        *,
        kp_x: float = 0.35,
        kp_y: float = 0.24,
        ki: float = 0.0,
        kd: float = 0.0,
        integral_limit: float = 0.0,
        fov_x_deg: float = 105.0,
        counts_per_360: float = 9980.0,
        max_step_counts: float = 80.0,
        control_hz: float = 60.0,
        sign_x: float = 1.0,
        sign_y: float = 1.0,
        kalman_enabled: bool = True,
        kalman_process_noise: float = 2.0,
        kalman_measurement_noise: float = 16.0,
        hungarian_enabled: bool = True,
        matching_distance_px: float = 140.0,
        max_extrapolate_frames: int = 3,
        target_filter_enabled: bool = True,
        target_filter_min_score: float = 0.0,
        target_filter_fov_ratio: float = 1.0,
        target_filter_same_class: bool = False,
        prediction_lead_ms: float = 0.0,
        extrapolate_confidence_decay: float = 1.0,
        magnet_enabled: bool = False,
        magnet_radius_px: float = 120.0,
        magnet_strength: float = 0.25,
        magnet_curve: float = 1.0,
        magnet_deadzone_px: float = 0.0,
        magnet_max_counts: float = 20.0,
        capture_width: float = 0.0,
        capture_height: float = 0.0,
        move_kind: str = "raw",
        move_ms: int = 0,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.pid_x = _AnglePidAxis(kp=max(0.0, kp_x), ki=max(0.0, ki), kd=kd, integral_limit=max(0.0, integral_limit))
        self.pid_y = _AnglePidAxis(kp=max(0.0, kp_y), ki=max(0.0, ki), kd=kd, integral_limit=max(0.0, integral_limit))
        self.fov_x_deg = max(1.0, min(179.0, fov_x_deg))
        self.counts_per_360 = max(1.0, counts_per_360)
        self.max_step_counts = max(1.0, max_step_counts)
        self.control_hz = max(1.0, control_hz)
        self.sign_x = -1.0 if sign_x < 0 else 1.0
        self.sign_y = -1.0 if sign_y < 0 else 1.0
        self.kalman_enabled = bool(kalman_enabled)
        self.kalman_process_noise = max(1e-6, kalman_process_noise)
        self.kalman_measurement_noise = max(1e-6, kalman_measurement_noise)
        self.hungarian_enabled = bool(hungarian_enabled)
        self.matching_distance_px = max(1.0, matching_distance_px)
        self.max_extrapolate_frames = max(0, int(max_extrapolate_frames))
        self.target_filter_enabled = bool(target_filter_enabled)
        self.target_filter_min_score = max(0.0, target_filter_min_score)
        self.target_filter_fov_ratio = max(0.0, min(1.0, target_filter_fov_ratio))
        self.target_filter_same_class = bool(target_filter_same_class)
        self.prediction_lead_ms = max(0.0, prediction_lead_ms)
        self.extrapolate_confidence_decay = max(0.0, min(1.0, extrapolate_confidence_decay))
        self.magnet_enabled = bool(magnet_enabled)
        self.magnet_radius_px = max(1.0, magnet_radius_px)
        self.magnet_strength = max(0.0, magnet_strength)
        self.magnet_curve = max(0.1, magnet_curve)
        self.magnet_deadzone_px = max(0.0, magnet_deadzone_px)
        self.magnet_max_counts = max(0.0, magnet_max_counts)
        self.capture_width = max(0.0, capture_width)
        self.capture_height = max(0.0, capture_height)
        self.move_kind = move_kind if move_kind in MOVE_KINDS else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self._tracks: dict[int, _KalmanCenterTrack] = {}
        self._next_track_id = 1
        self._last_observation_frame_id: int | None = None
        self._last_observation_ts_ns: int | None = None
        self._last_tracker_tick_ns: int | None = None

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        raw = box_input.raw or {}
        roi_width = _positive_number(raw.get("roi_width"), current_pos[0] * 2.0)
        roi_height = _positive_number(raw.get("roi_height"), current_pos[1] * 2.0)
        capture_width, capture_height, capture_source = self._capture_dimensions(raw, roi_width, roi_height)
        raw_aim_x = (float(target.x1) + float(target.x2)) * 0.5
        raw_aim_y = (float(target.y1) + float(target.y2)) * 0.5
        aim_x, aim_y, tracker_debug = self._stable_aim_point(target, raw, raw_aim_x, raw_aim_y)
        roi_center_x = roi_width * 0.5
        roi_center_y = roi_height * 0.5
        error_x_px = aim_x - roi_center_x
        error_y_image_down_px = aim_y - roi_center_y
        error_y_px = roi_center_y - aim_y
        fov_x_rad = math.radians(self.fov_x_deg)
        focal_x = (capture_width * 0.5) / math.tan(fov_x_rad * 0.5)
        fov_y_rad = 2.0 * math.atan((capture_height / capture_width) * math.tan(fov_x_rad * 0.5))
        focal_y = (capture_height * 0.5) / math.tan(fov_y_rad * 0.5)
        error_x_rad = math.atan(error_x_px / focal_x)
        error_y_rad = math.atan(error_y_px / focal_y)
        dt = 1.0 / self.control_hz
        out_x_rad = self.pid_x.update(error_x_rad, dt)
        out_y_rad = self.pid_y.update(error_y_rad, dt)
        counts_per_rad = self.counts_per_360 / (2.0 * math.pi)
        pid_dx_counts = out_x_rad * counts_per_rad
        pid_dy_counts = out_y_rad * counts_per_rad
        magnet_dx_counts, magnet_dy_counts, magnet_debug = self._magnet_counts(
            error_x_px=error_x_px,
            error_y_px=error_y_px,
            error_x_rad=error_x_rad,
            error_y_rad=error_y_rad,
            counts_per_rad=counts_per_rad,
        )
        raw_tracker_decay = tracker_debug.get("extrapolate_confidence_decay")
        tracker_confidence_decay = (
            max(0.0, min(1.0, float(raw_tracker_decay)))
            if isinstance(raw_tracker_decay, (int, float)) and not isinstance(raw_tracker_decay, bool)
            else 1.0
        )
        raw_dx_counts = (pid_dx_counts + magnet_dx_counts) * tracker_confidence_decay
        raw_dy_counts = (pid_dy_counts + magnet_dy_counts) * tracker_confidence_decay
        dx = self._clamp(raw_dx_counts * self.sign_x, self.max_step_counts)
        dy = self._clamp(raw_dy_counts * self.sign_y, self.max_step_counts)
        dx_i = int(round(dx))
        dy_i = int(round(dy))
        bezier_ctrl = _bezier_ctrl(dx_i, dy_i, self.bezier_curvature) if self.move_kind in {"bezier", "enc_bezier"} else None
        return MoveCommand(
            dx=dx_i,
            dy=dy_i,
            confidence=float(target.score),
            reason=f"experimental angle pid px={math.hypot(error_x_px, error_y_px):.1f}",
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug={
                "stage": "experimental_angle_pid",
                "algorithm": "experimental_angle_pid",
                "unit_pipeline": "bbox_center_px_to_angle_rad_to_counts",
                "coordinate_y": "cartesian_up_positive_before_executor_flip",
                "raw_aim_x": raw_aim_x,
                "raw_aim_y": raw_aim_y,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "tracker": tracker_debug,
                "roi_center_x": roi_center_x,
                "roi_center_y": roi_center_y,
                "roi_width": roi_width,
                "roi_height": roi_height,
                "capture_width": capture_width,
                "capture_height": capture_height,
                "capture_size_source": capture_source,
                "error_x_px": error_x_px,
                "error_y_px": error_y_px,
                "error_y_image_down_px": error_y_image_down_px,
                "fov_x_deg": self.fov_x_deg,
                "fov_x_rad": fov_x_rad,
                "fov_y_rad": fov_y_rad,
                "focal_x": focal_x,
                "focal_y": focal_y,
                "error_x_rad": error_x_rad,
                "error_y_rad": error_y_rad,
                "error_x_deg": math.degrees(error_x_rad),
                "error_y_deg": math.degrees(error_y_rad),
                "out_x_rad": out_x_rad,
                "out_y_rad": out_y_rad,
                "counts_per_360": self.counts_per_360,
                "counts_per_rad": counts_per_rad,
                "pid_dx_counts": pid_dx_counts,
                "pid_dy_counts": pid_dy_counts,
                "magnet": magnet_debug,
                "magnet_dx_counts": magnet_dx_counts,
                "magnet_dy_counts": magnet_dy_counts,
                "tracker_confidence_decay": tracker_confidence_decay,
                "raw_dx_counts": raw_dx_counts,
                "raw_dy_counts": raw_dy_counts,
                "sign_x": self.sign_x,
                "sign_y": self.sign_y,
                "max_step_counts": self.max_step_counts,
                "dt": dt,
                "control_hz": self.control_hz,
                "p_x": self.pid_x.p_term,
                "p_y": self.pid_y.p_term,
                "i_x": self.pid_x.i_term,
                "i_y": self.pid_y.i_term,
                "d_x": self.pid_x.d_term,
                "d_y": self.pid_y.d_term,
                "final_dx": dx_i,
                "final_dy": dy_i,
            },
        )

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self._tracks.clear()
        self._last_observation_frame_id = None
        self._last_observation_ts_ns = None
        self._last_tracker_tick_ns = None

    def _capture_dimensions(self, raw: dict[str, object], roi_width: float, roi_height: float) -> tuple[float, float, str]:
        raw_capture_width = _positive_number(raw.get("capture_width"), 0.0)
        raw_capture_height = _positive_number(raw.get("capture_height"), 0.0)
        if raw_capture_width > 0 and raw_capture_height > 0:
            return raw_capture_width, raw_capture_height, "frame_metadata"
        if self.capture_width > 0 and self.capture_height > 0:
            return self.capture_width, self.capture_height, "runtime_config"
        return roi_width, roi_height, "roi_fallback"

    def _stable_aim_point(
        self,
        target: Target,
        raw: dict[str, object],
        fallback_x: float,
        fallback_y: float,
    ) -> tuple[float, float, dict[str, Any]]:
        if not self.kalman_enabled:
            return fallback_x, fallback_y, {"enabled": False, "reason": "disabled"}
        detections = _coerce_detection_items(raw.get("detections"))
        if not detections:
            detections = [_target_detection_item(target, raw)]
        raw_detection_count = len(detections)
        detections, filter_debug = self._filter_detections_for_tracking(detections, target, raw)
        frame_id = self._raw_frame_id(raw)
        is_new_observation = frame_id is None or frame_id != self._last_observation_frame_id
        dt, dt_source = self._tracker_dt(raw, is_new_observation)
        for track in self._tracks.values():
            track.predict(dt, self.kalman_process_noise)

        if not detections:
            stale = [track_id for track_id, track in self._tracks.items() if track.missed > self.max_extrapolate_frames]
            for track_id in stale:
                self._tracks.pop(track_id, None)
            return fallback_x, fallback_y, {
                "enabled": True,
                "used": False,
                "reason": "no detection after target filter",
                "tracks": len(self._tracks),
                "raw_detections": raw_detection_count,
                "detections": 0,
                "frame_id": frame_id,
                "new_observation": is_new_observation,
                "dt": dt,
                "dt_source": dt_source,
                "filter": filter_debug,
            }

        assignments = self._assign_tracks(detections) if is_new_observation and self.hungarian_enabled else self._greedy_assign_tracks(detections) if is_new_observation else []
        assigned_detection_indexes: set[int] = set()
        if is_new_observation:
            for track_id, detection_index, _cost in assignments:
                track = self._tracks.get(track_id)
                if track is None:
                    continue
                track.update(detections[detection_index], self.kalman_measurement_noise)
                assigned_detection_indexes.add(detection_index)

            for index, detection in enumerate(detections):
                if index in assigned_detection_indexes:
                    continue
                track = self._new_track(detection)
                self._tracks[track.track_id] = track
            self._remember_observation(raw, frame_id)
        else:
            for track in self._tracks.values():
                if track.source_index is not None:
                    assigned_detection_indexes.add(track.source_index)
                continue

        stale = [track_id for track_id, track in self._tracks.items() if track.missed > self.max_extrapolate_frames]
        for track_id in stale:
            self._tracks.pop(track_id, None)

        selected = self._select_track_for_target(raw, target, detections)
        if selected is None:
            return fallback_x, fallback_y, {
                "enabled": True,
                "used": False,
                "reason": "no matched track",
                "tracks": len(self._tracks),
                "raw_detections": raw_detection_count,
                "detections": len(detections),
                "assignments": len(assignments),
                "frame_id": frame_id,
                "new_observation": is_new_observation,
                "dt": dt,
                "dt_source": dt_source,
                "filter": filter_debug,
            }
        x, y = self._predicted_track_center(selected, dt)
        confidence_decay = self.extrapolate_confidence_decay ** max(0, selected.missed)
        return x, y, {
            "enabled": True,
            "used": True,
            "track_id": selected.track_id,
            "source_index": selected.source_index,
            "missed": selected.missed,
            "tracks": len(self._tracks),
            "raw_detections": raw_detection_count,
            "detections": len(detections),
            "assignments": len(assignments),
            "frame_id": frame_id,
            "new_observation": is_new_observation,
            "dt": dt,
            "dt_source": dt_source,
            "filter": filter_debug,
            "hungarian": self.hungarian_enabled,
            "prediction_lead_ms": self.prediction_lead_ms,
            "predicted_x": x,
            "predicted_y": y,
            "velocity_x_px_s": selected.kx.velocity,
            "velocity_y_px_s": selected.ky.velocity,
            "extrapolate_confidence_decay": confidence_decay,
            "kalman_process_noise": self.kalman_process_noise,
            "kalman_measurement_noise": self.kalman_measurement_noise,
            "matching_distance_px": self.matching_distance_px,
        }

    def _filter_detections_for_tracking(
        self,
        detections: list[dict[str, float]],
        target: Target,
        raw: dict[str, object],
    ) -> tuple[list[dict[str, float]], dict[str, Any]]:
        if not self.target_filter_enabled:
            return detections, {"enabled": False, "input": len(detections), "output": len(detections)}
        roi_width = _positive_number(raw.get("roi_width"), 0.0)
        roi_height = _positive_number(raw.get("roi_height"), 0.0)
        center_x = roi_width * 0.5
        center_y = roi_height * 0.5
        radius = min(roi_width, roi_height) * self.target_filter_fov_ratio if roi_width > 0 and roi_height > 0 else 0.0
        filtered: list[dict[str, float]] = []
        rejected_score = 0
        rejected_class = 0
        rejected_fov = 0
        for detection in detections:
            if float(detection.get("score", 0.0)) < self.target_filter_min_score:
                rejected_score += 1
                continue
            if self.target_filter_same_class and int(detection.get("cls", -1)) != int(target.cls):
                rejected_class += 1
                continue
            if radius > 0:
                distance = _center_distance((float(detection["cx"]), float(detection["cy"])), (center_x, center_y))
                if distance > radius:
                    rejected_fov += 1
                    continue
            filtered.append(detection)
        return filtered, {
            "enabled": True,
            "input": len(detections),
            "output": len(filtered),
            "min_score": self.target_filter_min_score,
            "fov_ratio": self.target_filter_fov_ratio,
            "same_class": self.target_filter_same_class,
            "rejected_score": rejected_score,
            "rejected_class": rejected_class,
            "rejected_fov": rejected_fov,
        }

    def _predicted_track_center(self, track: _KalmanCenterTrack, dt: float) -> tuple[float, float]:
        lead_s = self.prediction_lead_ms / 1000.0
        if lead_s <= 0:
            return track.center
        lead_s = min(lead_s, max(dt, 1.0 / self.control_hz) * (1 + self.max_extrapolate_frames))
        return track.kx.position + track.kx.velocity * lead_s, track.ky.position + track.ky.velocity * lead_s

    def _raw_frame_id(self, raw: dict[str, object]) -> int | None:
        value = raw.get("frame_id")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        return None

    def _raw_capture_ts_ns(self, raw: dict[str, object]) -> int | None:
        value = raw.get("capture_ts_ns")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
        return None

    def _tracker_dt(self, raw: dict[str, object], is_new_observation: bool) -> tuple[float, str]:
        now_ns = time.monotonic_ns()
        default_dt = 1.0 / self.control_hz
        capture_ts_ns = self._raw_capture_ts_ns(raw)
        if is_new_observation and capture_ts_ns is not None and self._last_observation_ts_ns is not None:
            delta_ns = capture_ts_ns - self._last_observation_ts_ns
            if delta_ns > 0:
                self._last_tracker_tick_ns = now_ns
                return self._clamp_dt(delta_ns / 1e9), "capture_ts_ns"
        if self._last_tracker_tick_ns is not None:
            delta_ns = now_ns - self._last_tracker_tick_ns
            if delta_ns > 0:
                self._last_tracker_tick_ns = now_ns
                return self._clamp_dt(delta_ns / 1e9), "control_tick"
        self._last_tracker_tick_ns = now_ns
        return default_dt, "default_control_hz"

    def _remember_observation(self, raw: dict[str, object], frame_id: int | None) -> None:
        if frame_id is not None:
            self._last_observation_frame_id = frame_id
        capture_ts_ns = self._raw_capture_ts_ns(raw)
        if capture_ts_ns is not None:
            self._last_observation_ts_ns = capture_ts_ns

    @staticmethod
    def _clamp_dt(dt: float) -> float:
        return max(1.0 / 240.0, min(1.0 / 15.0, float(dt)))

    def _magnet_counts(
        self,
        *,
        error_x_px: float,
        error_y_px: float,
        error_x_rad: float,
        error_y_rad: float,
        counts_per_rad: float,
    ) -> tuple[float, float, dict[str, Any]]:
        distance_px = math.hypot(error_x_px, error_y_px)
        if (
            not self.magnet_enabled
            or self.magnet_strength <= 0
            or self.magnet_max_counts <= 0
            or distance_px <= self.magnet_deadzone_px
            or distance_px >= self.magnet_radius_px
        ):
            return 0.0, 0.0, {
                "enabled": self.magnet_enabled,
                "used": False,
                "distance_px": distance_px,
                "radius_px": self.magnet_radius_px,
                "deadzone_px": self.magnet_deadzone_px,
            }
        proximity = max(0.0, min(1.0, 1.0 - distance_px / self.magnet_radius_px))
        gain = self.magnet_strength * (proximity ** self.magnet_curve)
        dx = self._clamp(error_x_rad * counts_per_rad * gain, self.magnet_max_counts)
        dy = self._clamp(error_y_rad * counts_per_rad * gain, self.magnet_max_counts)
        return dx, dy, {
            "enabled": True,
            "used": True,
            "distance_px": distance_px,
            "radius_px": self.magnet_radius_px,
            "deadzone_px": self.magnet_deadzone_px,
            "strength": self.magnet_strength,
            "curve": self.magnet_curve,
            "gain": gain,
            "max_counts": self.magnet_max_counts,
            "dx_counts": dx,
            "dy_counts": dy,
        }

    def _assign_tracks(self, detections: list[dict[str, float]]) -> list[tuple[int, int, float]]:
        tracks = list(self._tracks.values())
        if not tracks or not detections:
            return []
        if len(tracks) > 10 or len(detections) > 10:
            return self._greedy_assign_tracks(detections)
        det_count = len(detections)
        memo: dict[tuple[int, int], tuple[float, list[tuple[int, int, float]]]] = {}

        def solve(track_pos: int, used_mask: int) -> tuple[float, list[tuple[int, int, float]]]:
            key = (track_pos, used_mask)
            cached = memo.get(key)
            if cached is not None:
                return cached
            if track_pos >= len(tracks):
                return 0.0, []
            best_cost, best_pairs = solve(track_pos + 1, used_mask)
            track = tracks[track_pos]
            for det_index, detection in enumerate(detections):
                if used_mask & (1 << det_index):
                    continue
                cost = _center_distance(track.center, (float(detection["cx"]), float(detection["cy"])))
                if cost > self.matching_distance_px:
                    continue
                rest_cost, rest_pairs = solve(track_pos + 1, used_mask | (1 << det_index))
                candidate_cost = cost + rest_cost
                if not best_pairs or candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_pairs = [(track.track_id, det_index, cost), *rest_pairs]
            memo[key] = (best_cost, best_pairs)
            return memo[key]

        return solve(0, 0)[1]

    def _greedy_assign_tracks(self, detections: list[dict[str, float]]) -> list[tuple[int, int, float]]:
        pairs: list[tuple[float, int, int]] = []
        for track in self._tracks.values():
            for det_index, detection in enumerate(detections):
                cost = _center_distance(track.center, (float(detection["cx"]), float(detection["cy"])))
                if cost <= self.matching_distance_px:
                    pairs.append((cost, track.track_id, det_index))
        result: list[tuple[int, int, float]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for cost, track_id, det_index in sorted(pairs, key=lambda item: item[0]):
            if track_id in used_tracks or det_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(det_index)
            result.append((track_id, det_index, cost))
        return result

    def _new_track(self, detection: dict[str, float]) -> _KalmanCenterTrack:
        track = _KalmanCenterTrack(
            track_id=self._next_track_id,
            cls=int(detection["cls"]),
            source_index=int(detection["index"]) if isinstance(detection.get("index"), (int, float)) else None,
            kx=_KalmanAxis(float(detection["cx"])),
            ky=_KalmanAxis(float(detection["cy"])),
            w=float(detection["w"]),
            h=float(detection["h"]),
            score=float(detection["score"]),
        )
        self._next_track_id += 1
        return track

    def _select_track_for_target(self, raw: dict[str, object], target: Target, detections: list[dict[str, float]]) -> _KalmanCenterTrack | None:
        target_key = str(raw.get("target_key") or "")
        target_index: int | None = None
        if target_key.startswith("det:"):
            parts = target_key.split(":")
            if len(parts) > 1:
                try:
                    target_index = int(parts[1])
                except ValueError:
                    target_index = None
        if target_index is not None:
            for track in self._tracks.values():
                if track.source_index == target_index:
                    return track
        target_center = ((float(target.x1) + float(target.x2)) * 0.5, (float(target.y1) + float(target.y2)) * 0.5)
        compatible = [
            track for track in self._tracks.values()
            if int(track.cls) == int(target.cls) and track.missed <= self.max_extrapolate_frames
        ]
        if not compatible:
            return None
        return min(compatible, key=lambda track: _center_distance(track.center, target_center))

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))


class PIDStrategy:
    def __init__(
        self,
        *,
        kp: float = 0.35,
        ki: float = 0.0,
        kd: float = 0.04,
        kp_x: float | None = None,
        kp_y: float | None = None,
        integral_limit: float = 250.0,
        move_limit: float | None = None,
        move_limit_x: float | None = None,
        move_limit_y: float | None = None,
        derivative_alpha: float = 0.35,
        prediction_enabled: bool = True,
        prediction_factor: float = 0.1,
        derivative_enabled: bool = True,
        prediction_stationary_px: float = 1.5,
        prediction_moving_px: float = 12.0,
        aim_ratio: float = 40.0,
        near_px: float = 24.0,
        near_speed: float = 0.16,
        far_speed: float = 0.42,
        deadzone_counts: int = 1,
        counts_per_revolution_x: float = 9980.0,
        counts_per_revolution_y: float = 9980.0,
        fov_deg: float = 105.0,
        move_kind: str = "raw",
        move_ms: int = 0,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.kp_x = kp if kp_x is None else kp_x
        self.kp_y = kp if kp_y is None else kp_y
        self.ki = 0.0
        self.kd = kd
        self.integral_limit = abs(integral_limit)
        self.move_limit = None if move_limit is None else abs(move_limit)
        self.move_limit_x = (
            self.move_limit if move_limit_x is None else abs(move_limit_x)
        )
        self.move_limit_y = (
            self.move_limit if move_limit_y is None else abs(move_limit_y)
        )
        self.derivative_alpha = max(0.0, min(1.0, derivative_alpha))
        self.prediction_enabled = bool(prediction_enabled)
        self.prediction_factor = max(0.0, prediction_factor)
        self.derivative_enabled = bool(derivative_enabled)
        self.prediction_stationary_px = max(0.0, prediction_stationary_px)
        self.prediction_moving_px = max(
            self.prediction_stationary_px + 1.0,
            prediction_moving_px,
        )
        self.aim_ratio = max(0.0, min(100.0, aim_ratio))
        self.near_px = max(0.0, near_px)
        self.near_speed = max(0.0, near_speed)
        self.far_speed = max(0.0, far_speed)
        self.deadzone_counts = max(0, int(deadzone_counts))
        self.counts_per_revolution_x = max(1.0, counts_per_revolution_x)
        self.counts_per_revolution_y = max(1.0, counts_per_revolution_y)
        self.fov_deg = max(1.0, min(179.0, fov_deg))
        self.move_kind = move_kind if move_kind in MOVE_KINDS else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self._ix = 0.0
        self._iy = 0.0
        self._last_error: tuple[float, float] | None = None
        self._last_smoothed_error: tuple[float, float] | None = None
        self._last_derivative = (0.0, 0.0)
        self._last_center: tuple[float, float] | None = None
        self._last_center_s: float | None = None
        self._last_target_key: str | None = None
        self._last_s: float | None = None
        self._last_prediction_debug: dict[str, float] = {}

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        target_key = str(box_input.raw.get("target_key") or "")
        if target_key and self._last_target_key is not None and target_key != self._last_target_key:
            self._reset_motion_state()
        if target_key:
            self._last_target_key = target_key
        raw_frame_age_ms = box_input.raw.get("frame_age_ms")
        frame_age_ms = float(raw_frame_age_ms) if isinstance(raw_frame_age_ms, (int, float)) else 0.0
        aim_center = aim_point(target, self.aim_ratio)
        predicted_center, prediction_weight = self._predict_center(aim_center, frame_age_ms=frame_age_ms)
        ex_px = predicted_center[0] - current_pos[0]
        ey_px = current_pos[1] - predicted_center[1]
        err_px = math.hypot(ex_px, ey_px)
        projection = self._pixel_error_to_counts(ex_px, ey_px, current_pos[0] * 2.0)
        fov_counts_x = projection["err_cx"]
        fov_counts_y = projection["err_cy"]
        speed = 1.0
        ex = fov_counts_x
        ey = fov_counts_y
        now_s = time.monotonic()
        dt = now_s - self._last_s if self._last_s is not None else 1.0 / 60.0
        self._last_s = now_s
        if dt <= 0 or dt > 0.1:
            dt = 1.0 / 60.0
        dt = max(1.0 / 240.0, min(1.0 / 20.0, dt))
        self._ix = max(-self.integral_limit, min(self.integral_limit, self._ix + ex * dt))
        self._iy = max(-self.integral_limit, min(self.integral_limit, self._iy + ey * dt))
        if self._last_smoothed_error is None:
            smooth_ex = ex
            smooth_ey = ey
        else:
            alpha = self.derivative_alpha
            smooth_ex = alpha * ex + (1 - alpha) * self._last_smoothed_error[0]
            smooth_ey = alpha * ey + (1 - alpha) * self._last_smoothed_error[1]
        if self._last_error is None:
            raw_dx = raw_dy = 0.0
        else:
            raw_dx = (smooth_ex - self._last_error[0]) / dt
            raw_dy = (smooth_ey - self._last_error[1]) / dt
        dx_d = self.derivative_alpha * raw_dx + (1 - self.derivative_alpha) * self._last_derivative[0]
        dy_d = self.derivative_alpha * raw_dy + (1 - self.derivative_alpha) * self._last_derivative[1]
        self._last_error = (smooth_ex, smooth_ey)
        self._last_smoothed_error = (smooth_ex, smooth_ey)
        self._last_derivative = (dx_d, dy_d)
        px = self.kp_x * ex
        py = self.kp_y * ey
        ix = 0.0
        iy = 0.0
        raw_dx_term = self.kd * dx_d if self.derivative_enabled else 0.0
        raw_dy_term = self.kd * dy_d if self.derivative_enabled else 0.0
        dx_term = self._clamp_derivative_term(raw_dx_term, self.move_limit_x)
        dy_term = self._clamp_derivative_term(raw_dy_term, self.move_limit_y)
        dx = px + ix + dx_term
        dy = py + iy + dy_term
        debug = {
            **projection,
            "stage": "pid_counts_pipeline",
            "coordinate_y": "cartesian_up_positive",
            "prediction_enabled": self.prediction_enabled,
            "derivative_enabled": self.derivative_enabled,
            "raw_px_x": ex_px,
            "raw_px_y": ey_px,
            "aim_ratio": self.aim_ratio,
            "aim_x": aim_center[0],
            "aim_y": aim_center[1],
            "predicted_x": predicted_center[0],
            "predicted_y": predicted_center[1],
            "fov_counts_x": fov_counts_x,
            "fov_counts_y": fov_counts_y,
            "speed": speed,
            "speed_note": "disabled: PID gain is the only intentional movement scale",
            "work_counts_x": ex,
            "work_counts_y": ey,
            "smoothed_counts_x": smooth_ex,
            "smoothed_counts_y": smooth_ey,
            "c360_x": self.counts_per_revolution_x,
            "c360_y": self.counts_per_revolution_y,
            "fov_deg": self.fov_deg,
            "dt": dt,
            "target_key": target_key,
            "frame_age_ms": frame_age_ms,
            "prediction_lead_ms": self._last_prediction_debug.get("lead_ms", 0.0),
            "prediction_velocity_x_px_s": self._last_prediction_debug.get("velocity_x_px_s", 0.0),
            "prediction_velocity_y_px_s": self._last_prediction_debug.get("velocity_y_px_s", 0.0),
            "prediction_lead_x_px": self._last_prediction_debug.get("lead_x_px", 0.0),
            "prediction_lead_y_px": self._last_prediction_debug.get("lead_y_px", 0.0),
            "p_x": px,
            "p_y": py,
            "i_x": ix,
            "i_y": iy,
            "raw_d_x": raw_dx_term,
            "raw_d_y": raw_dy_term,
            "d_x": dx_term,
            "d_y": dy_term,
            "d_limited_x": raw_dx_term != dx_term,
            "d_limited_y": raw_dy_term != dy_term,
        }
        if abs(ex) <= self.deadzone_counts and abs(ey) <= self.deadzone_counts:
            return MoveCommand(
                0,
                0,
                target.score,
                "pid deadzone",
                debug={**debug, "final_dx": 0.0, "final_dy": 0.0},
            )
        dx = self._minimum_effective_step(dx, ex)
        dy = self._minimum_effective_step(dy, ey)
        dx = self._clamp_axis(dx, self.move_limit_x)
        dy = self._clamp_axis(dy, self.move_limit_y)
        bezier_ctrl = _bezier_ctrl(int(round(dx)), int(round(dy)), self.bezier_curvature) if self.move_kind in {"bezier", "enc_bezier"} else None
        debug["final_dx"] = dx
        debug["final_dy"] = dy
        return MoveCommand(
            dx=dx,
            dy=dy,
            confidence=target.score,
            reason=(
                f"pid counts strategy px={err_px:.1f}"
                if prediction_weight <= 0
                else f"pid counts strategy px={err_px:.1f} prediction={prediction_weight:.2f}"
            ),
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug=debug,
        )

    @staticmethod
    def _clamp_axis(value: float, limit: float | None) -> float:
        if limit is None:
            return value
        return max(-limit, min(limit, value))

    @staticmethod
    def _clamp_derivative_term(value: float, move_limit: float | None) -> float:
        # D is only damping. If it can consume the full move budget, one noisy
        # bbox can reverse the view and look like a sudden vertical jump.
        limit = 24.0 if move_limit is None else max(2.0, abs(move_limit) * 0.35)
        return max(-limit, min(limit, value))

    def _predict_center(
        self,
        center: tuple[float, float],
        *,
        frame_age_ms: float,
    ) -> tuple[tuple[float, float], float]:
        now_s = time.monotonic()
        if self._last_center is None or not self.prediction_enabled or self.prediction_factor <= 0:
            self._last_center = center
            self._last_center_s = now_s
            self._last_prediction_debug = {
                "lead_ms": 0.0,
                "velocity_x_px_s": 0.0,
                "velocity_y_px_s": 0.0,
                "lead_x_px": 0.0,
                "lead_y_px": 0.0,
            }
            return center, 0.0

        dt_s = now_s - self._last_center_s if self._last_center_s is not None else 1.0 / 60.0
        if dt_s <= 0 or dt_s > 0.1:
            dt_s = 1.0 / 60.0
        dt_s = max(1.0 / 240.0, min(1.0 / 20.0, dt_s))
        delta_x = center[0] - self._last_center[0]
        delta_y = center[1] - self._last_center[1]
        self._last_center = center
        self._last_center_s = now_s
        if abs(delta_x) <= self.prediction_stationary_px and abs(delta_y) <= self.prediction_stationary_px:
            self._last_prediction_debug = {
                "lead_ms": 0.0,
                "velocity_x_px_s": 0.0,
                "velocity_y_px_s": 0.0,
                "lead_x_px": 0.0,
                "lead_y_px": 0.0,
            }
            return center, 0.0

        vx_px_s = delta_x / dt_s
        vy_px_s = delta_y / dt_s
        # The prediction factor is a lead multiplier over measured frame age.
        # Clamp the lead window so a stalled frame cannot create a wild jump.
        lead_s = max(0.0, min(0.08, frame_age_ms / 1000.0)) * self.prediction_factor
        lead_x = vx_px_s * lead_s
        lead_y = vy_px_s * lead_s
        self._last_prediction_debug = {
            "lead_ms": lead_s * 1000.0,
            "velocity_x_px_s": vx_px_s,
            "velocity_y_px_s": vy_px_s,
            "lead_x_px": lead_x,
            "lead_y_px": lead_y,
        }
        return (center[0] + lead_x, center[1] + lead_y), self.prediction_factor

    def _reset_motion_state(self) -> None:
        self._ix = 0.0
        self._iy = 0.0
        self._last_error = None
        self._last_smoothed_error = None
        self._last_derivative = (0.0, 0.0)
        self._last_center = None
        self._last_center_s = None
        self._last_target_key = None
        self._last_s = None
        self._last_prediction_debug = {}

    def _pixel_error_to_counts(self, ex_px: float, ey_px: float, frame_width: float) -> dict[str, float]:
        focal = (frame_width * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5) if frame_width > 0 else 0.0
        counts_per_degree_x = self.counts_per_revolution_x / 360.0
        counts_per_degree_y = self.counts_per_revolution_y / 360.0
        yaw_deg = math.degrees(math.atan(ex_px / focal)) if focal > 0 else 0.0
        pitch_deg = math.degrees(math.atan(ey_px / focal)) if focal > 0 else 0.0
        return {
            "focal": focal,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "counts_per_degree_x": counts_per_degree_x,
            "counts_per_degree_y": counts_per_degree_y,
            "err_cx": yaw_deg * counts_per_degree_x,
            "err_cy": pitch_deg * counts_per_degree_y,
        }

    def _speed_for(self, err_px: float, fov_radius: float) -> float:
        if self.near_px <= 0:
            return self.far_speed
        if err_px <= self.near_px:
            return self.near_speed
        span = max(1.0, fov_radius - self.near_px)
        ratio = min(1.0, (err_px - self.near_px) / span)
        return self.near_speed + (self.far_speed - self.near_speed) * ratio

    @staticmethod
    def _minimum_effective_step(value: float, error_counts: float) -> float:
        if round(value) != 0 or abs(error_counts) < 1.0:
            return value
        return 1.0 if error_counts > 0 else -1.0


class StraightStrategy:
    VELOCITY_CLAMP_COUNTS_PER_SEC = 3000.0
    OVERSHOOT_GUARD_COUNTS = 2.0
    OVERSHOOT_VELOCITY_DAMP = 0.1
    MOTION_FLOOR_COUNTS_PER_SEC = 60.0
    FRAME_DT_FALLBACK = 1.0 / 60.0
    FRAME_DT_MIN = 0.001
    FRAME_DT_MAX = 0.35
    PREDICT_DT_MIN = 0.001
    PREDICT_DT_MAX = 0.05

    def __init__(
        self,
        *,
        aim_y_ratio: float = 40.0,
        fov_deg: float = 105.0,
        counts_per_revolution: float = 9980.0,
        kp_x: float = 0.3,
        kp_y: float = 0.3,
        first_frame_gain: float = 1.0,
        first_frame_max_step: float = 200.0,
        refine_max_step: float = 80.0,
        in_deadzone_px: float = 8.0,
        jump_threshold_px: float = 40.0,
        pred_gain: float = 0.0,
        pred_consistency_frames: int = 3,
        gain_y: float = 1.0,
        move_kind: str = "raw",
        move_ms: int = 12,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.aim_y_ratio = max(0.0, min(100.0, aim_y_ratio))
        self.fov_deg = max(1.0, min(179.0, fov_deg))
        self.counts_per_revolution = max(100.0, counts_per_revolution)
        self.kp_x = max(0.0, kp_x)
        self.kp_y = max(0.0, kp_y)
        self.first_frame_gain = max(0.0, first_frame_gain)
        self.first_frame_max_step = max(1.0, first_frame_max_step)
        self.refine_max_step = max(1.0, refine_max_step)
        self.in_deadzone_px = max(0.0, in_deadzone_px)
        self.jump_threshold_px = max(1.0, jump_threshold_px)
        self.pred_gain = max(0.0, pred_gain)
        self.pred_consistency_frames = max(1, int(pred_consistency_frames))
        self.gain_y = max(0.0, gain_y)
        self.move_kind = move_kind if move_kind in MOVE_KINDS else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self._last_error_px: tuple[float, float] | None = None
        self._last_error_counts = (0.0, 0.0)
        self._last_output_counts = (0.0, 0.0)
        self._last_s: float | None = None
        self._has_previous_sample = False
        self._consistency_x = 0
        self._consistency_y = 0
        self._prev_vel_sign_x = 0
        self._prev_vel_sign_y = 0

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            self._reset()
            return MoveCommand(
                0,
                0,
                0,
                "hardware trigger inactive",
                debug={"stage": "straight_trigger", "coordinate_y": "cartesian_up_positive"},
            )

        aim_x, aim_y = aim_point_y_ratio(target, self.aim_y_ratio)
        dx_px = aim_x - current_pos[0]
        dy_px = current_pos[1] - aim_y
        now_s = time.monotonic()
        dt_s = self._frame_dt(now_s)
        acquire = self._is_acquire(dx_px, dy_px)
        projection = self._project_pixels_to_counts(dx_px, dy_px, current_pos[0] * 2.0)
        err_cx = projection["err_cx"]
        err_cy = projection["err_cy"]
        velocity_x, velocity_y = self._estimate_velocity_counts(err_cx, err_cy, dt_s)
        motion_coef_x, motion_coef_y = self._motion_coef(velocity_x, velocity_y)
        pred_dt = max(self.PREDICT_DT_MIN, min(self.PREDICT_DT_MAX, dt_s))
        pred_scale_x = max(0.0, min(1.0, abs(dx_px) / max(1.0, float(target.w) * 0.5)))
        pred_scale_y = max(0.0, min(1.0, abs(dy_px) / max(1.0, float(target.h) * 0.5)))
        lead_x = velocity_x * pred_dt * pred_scale_x * motion_coef_x * self.pred_gain
        lead_y = velocity_y * pred_dt * pred_scale_y * motion_coef_y * self.pred_gain
        fused_x = err_cx + lead_x
        fused_y = err_cy + lead_y
        if acquire:
            out_x = self._clamp(fused_x * self.first_frame_gain, self.first_frame_max_step)
            out_y = self._clamp(fused_y * self.first_frame_gain * self.gain_y, self.first_frame_max_step)
            stage = "straight_acquire"
        else:
            out_x = self._clamp(self.kp_x * err_cx + lead_x, self.refine_max_step)
            out_y = self._clamp(self.kp_y * err_cy + lead_y, self.refine_max_step)
            stage = "straight_refine"
        if abs(dx_px) < self.in_deadzone_px:
            out_x = 0.0
        if abs(dy_px) < self.in_deadzone_px:
            out_y = 0.0
        if out_x == 0.0 and out_y == 0.0:
            stage = "straight_hold"
        self._last_error_px = (dx_px, dy_px)
        if self._was_sent(box_input):
            self._last_output_counts = (out_x, out_y)
        else:
            self._last_output_counts = (0.0, 0.0)
        dx = int(round(out_x))
        dy = int(round(out_y))
        bezier_ctrl = _bezier_ctrl(dx, dy, self.bezier_curvature) if self.move_kind in {"bezier", "enc_bezier"} else None
        return MoveCommand(
            dx=dx,
            dy=dy,
            confidence=target.score,
            reason=f"straight atan {stage} px={math.hypot(dx_px, dy_px):.1f}",
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug={
                **projection,
                "stage": stage,
                "coordinate_y": "cartesian_up_positive",
                "raw_px_x": dx_px,
                "raw_px_y": dy_px,
                "aim_ratio": self.aim_y_ratio,
                "aim_y_ratio": self.aim_y_ratio,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "dt": dt_s,
                "vel_x": velocity_x,
                "vel_y": velocity_y,
                "motion_coef_x": motion_coef_x,
                "motion_coef_y": motion_coef_y,
                "pred_scale_x": pred_scale_x,
                "pred_scale_y": pred_scale_y,
                "lead_x": lead_x,
                "lead_y": lead_y,
                "fused_x": fused_x,
                "fused_y": fused_y,
                "p_x": self.kp_x * err_cx,
                "p_y": self.kp_y * err_cy,
                "i_x": 0.0,
                "i_y": 0.0,
                "d_x": 0.0,
                "d_y": 0.0,
                "fov_counts_x": err_cx,
                "fov_counts_y": err_cy,
                "c360": self.counts_per_revolution,
                "fov_deg": self.fov_deg,
                "final_dx": dx,
                "final_dy": dy,
            },
        )

    def _frame_dt(self, now_s: float) -> float:
        dt_s = now_s - self._last_s if self._last_s is not None else self.FRAME_DT_FALLBACK
        self._last_s = now_s
        if not (self.FRAME_DT_MIN < dt_s < self.FRAME_DT_MAX):
            self._has_previous_sample = False
            return self.FRAME_DT_FALLBACK
        return dt_s

    def _is_acquire(self, dx_px: float, dy_px: float) -> bool:
        if self._last_error_px is None:
            self._has_previous_sample = False
            return True
        jumped = math.hypot(dx_px - self._last_error_px[0], dy_px - self._last_error_px[1])
        if jumped > self.jump_threshold_px:
            self._has_previous_sample = False
            return True
        return False

    def _project_pixels_to_counts(self, error_pixels_x: float, error_pixels_y: float, frame_width: float) -> dict[str, float]:
        focal = (frame_width * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5) if frame_width > 0 else 0.0
        counts_per_degree = self.counts_per_revolution / 360.0
        yaw_deg = math.degrees(math.atan(error_pixels_x / focal)) if focal > 0 else 0.0
        pitch_deg = math.degrees(math.atan(error_pixels_y / focal)) if focal > 0 else 0.0
        return {
            "focal": focal,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "counts_per_degree": counts_per_degree,
            "err_cx": yaw_deg * counts_per_degree,
            "err_cy": pitch_deg * counts_per_degree,
        }

    def _estimate_velocity_counts(self, error_counts_x: float, error_counts_y: float, dt_s: float) -> tuple[float, float]:
        if not self._has_previous_sample:
            self._last_error_counts = (error_counts_x, error_counts_y)
            self._has_previous_sample = True
            return 0.0, 0.0
        dt = max(self.PREDICT_DT_MIN, min(self.PREDICT_DT_MAX, dt_s))
        velocity_x = self._clamp(
            (error_counts_x - self._last_error_counts[0] + self._last_output_counts[0]) / dt,
            self.VELOCITY_CLAMP_COUNTS_PER_SEC,
        )
        velocity_y = self._clamp(
            (error_counts_y - self._last_error_counts[1] + self._last_output_counts[1]) / dt,
            self.VELOCITY_CLAMP_COUNTS_PER_SEC,
        )
        if abs(error_counts_x) > self.OVERSHOOT_GUARD_COUNTS and velocity_x * error_counts_x < 0.0:
            velocity_x *= self.OVERSHOOT_VELOCITY_DAMP
        if abs(error_counts_y) > self.OVERSHOOT_GUARD_COUNTS and velocity_y * error_counts_y < 0.0:
            velocity_y *= self.OVERSHOOT_VELOCITY_DAMP
        self._last_error_counts = (error_counts_x, error_counts_y)
        return velocity_x, velocity_y

    def _motion_coef(self, velocity_x: float, velocity_y: float) -> tuple[float, float]:
        self._consistency_x, self._prev_vel_sign_x, coef_x = self._axis_motion_coef(
            velocity_x,
            self._consistency_x,
            self._prev_vel_sign_x,
        )
        self._consistency_y, self._prev_vel_sign_y, coef_y = self._axis_motion_coef(
            velocity_y,
            self._consistency_y,
            self._prev_vel_sign_y,
        )
        return coef_x, coef_y

    def _axis_motion_coef(self, velocity: float, count: int, previous_sign: int) -> tuple[int, int, float]:
        if velocity > self.MOTION_FLOOR_COUNTS_PER_SEC:
            sign = 1
        elif velocity < -self.MOTION_FLOOR_COUNTS_PER_SEC:
            sign = -1
        else:
            sign = 0
        if sign == 0:
            count = 0
        elif sign == previous_sign:
            count += 1
        else:
            count = 1
        coef = max(0.0, min(1.0, count / float(self.pred_consistency_frames)))
        return count, sign, coef

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _was_sent(box_input: BoxInputState) -> bool:
        return str(box_input.raw.get("mode", "")) != "telemetry_control_calculation"

    def _reset(self) -> None:
        self._last_error_px = None
        self._last_error_counts = (0.0, 0.0)
        self._last_output_counts = (0.0, 0.0)
        self._last_s = None
        self._has_previous_sample = False
        self._consistency_x = 0
        self._consistency_y = 0
        self._prev_vel_sign_x = 0
        self._prev_vel_sign_y = 0

    def _reset_motion_state(self) -> None:
        self._reset()


class PredictiveStrategy:
    def __init__(
        self,
        *,
        lead_factor: float = 0.25,
        aim_ratio: float = 40.0,
        fov_deg: float = 105.0,
        counts_per_revolution: float = 9980.0,
    ) -> None:
        self.lead_factor = lead_factor
        self.aim_ratio = max(0.0, min(100.0, aim_ratio))
        self.fov_deg = max(1.0, min(179.0, fov_deg))
        self.counts_per_revolution = max(100.0, counts_per_revolution)
        self._last_center: tuple[float, float] | None = None

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            return MoveCommand(0, 0, 0, "hardware trigger inactive")
        center = aim_point(target, self.aim_ratio)
        if self._last_center is None:
            vx = vy = 0.0
        else:
            vx = center[0] - self._last_center[0]
            vy = center[1] - self._last_center[1]
        self._last_center = center
        predicted = (center[0] + vx * self.lead_factor, center[1] + vy * self.lead_factor)
        ex = predicted[0] - current_pos[0]
        ey = current_pos[1] - predicted[1]
        projection = _project_pixels_to_counts(
            ex,
            ey,
            current_pos[0] * 2.0,
            fov_deg=self.fov_deg,
            counts_per_revolution=self.counts_per_revolution,
        )
        dx = projection["err_cx"]
        dy = projection["err_cy"]
        return MoveCommand(
            dx=dx,
            dy=dy,
            confidence=target.score,
            reason="predictive fov counts strategy",
            debug={
                **projection,
                "stage": "predictive",
                "coordinate_y": "cartesian_up_positive",
                "raw_px_x": ex,
                "raw_px_y": ey,
                "aim_ratio": self.aim_ratio,
                "aim_x": center[0],
                "aim_y": center[1],
                "predicted_x": predicted[0],
                "predicted_y": predicted[1],
                "fov_counts_x": dx,
                "fov_counts_y": dy,
                "c360": self.counts_per_revolution,
                "fov_deg": self.fov_deg,
                "final_dx": dx,
                "final_dy": dy,
            },
        )


class ProportionalStrategy:
    def __init__(
        self,
        *,
        fov_ratio: float = 0.28,
        aim_ratio: float = 40.0,
        near_px: float = 24.0,
        near_speed: float = 0.16,
        far_speed: float = 0.42,
        ema_alpha: float = 0.45,
        deadzone_counts: int = 1,
        counts_per_revolution_x: float = 4096.0,
        counts_per_revolution_y: float = 4096.0,
        fov_deg: float = 105.0,
        move_kind: str = "raw",
        move_ms: int = 12,
        trace_ms: int = 0,
        bezier_curvature: float = 0.18,
    ) -> None:
        self.fov_ratio = max(0.0, min(1.0, fov_ratio))
        self.aim_ratio = max(0.0, min(100.0, aim_ratio))
        self.near_px = max(0.0, near_px)
        self.near_speed = max(0.0, near_speed)
        self.far_speed = max(0.0, far_speed)
        self.ema_alpha = max(0.0, min(1.0, ema_alpha))
        self.deadzone_counts = max(0, int(deadzone_counts))
        self.counts_per_revolution_x = max(1.0, counts_per_revolution_x)
        self.counts_per_revolution_y = max(1.0, counts_per_revolution_y)
        self.fov_deg = max(1.0, min(179.0, fov_deg))
        self.move_kind = move_kind if move_kind in MOVE_KINDS else "raw"
        self.move_ms = max(0, int(move_ms))
        self.trace_ms = max(0, int(trace_ms))
        self.bezier_curvature = max(0.0, bezier_curvature)
        self._ema_x = 0.0
        self._ema_y = 0.0

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        if not box_input.active:
            self._ema_x = 0.0
            self._ema_y = 0.0
            return MoveCommand(0, 0, 0, "hardware trigger inactive")

        aim_x, aim_y = aim_point(target, self.aim_ratio)
        ex = aim_x - current_pos[0]
        ey = current_pos[1] - aim_y
        err = math.hypot(ex, ey)
        radius = min(current_pos[0], current_pos[1]) * 2 * self.fov_ratio
        if radius > 0 and err > radius:
            self._ema_x = 0.0
            self._ema_y = 0.0
            return MoveCommand(
                0,
                0,
                target.score,
                "target outside proportional fov",
                debug={
                    "stage": "proportional",
                    "coordinate_y": "cartesian_up_positive",
                    "raw_px_x": ex,
                    "raw_px_y": ey,
                    "aim_ratio": self.aim_ratio,
                    "aim_x": aim_x,
                    "aim_y": aim_y,
                    "final_dx": 0,
                    "final_dy": 0,
                },
            )

        speed = self._speed_for(err, radius)
        projection_x = _project_pixels_to_counts(
            ex,
            ey,
            current_pos[0] * 2.0,
            fov_deg=self.fov_deg,
            counts_per_revolution=self.counts_per_revolution_x,
        )
        projection_y = _project_pixels_to_counts(
            ex,
            ey,
            current_pos[0] * 2.0,
            fov_deg=self.fov_deg,
            counts_per_revolution=self.counts_per_revolution_y,
        )
        counts_x = projection_x["err_cx"] * speed
        counts_y = projection_y["err_cy"] * speed
        self._ema_x = self.ema_alpha * counts_x + (1 - self.ema_alpha) * self._ema_x
        self._ema_y = self.ema_alpha * counts_y + (1 - self.ema_alpha) * self._ema_y
        dx = int(round(self._ema_x))
        dy = int(round(self._ema_y))
        if abs(dx) < self.deadzone_counts and abs(dy) < self.deadzone_counts:
            return MoveCommand(
                0,
                0,
                target.score,
                "proportional deadzone",
                debug={
                    "stage": "proportional",
                    "coordinate_y": "cartesian_up_positive",
                    "raw_px_x": ex,
                    "raw_px_y": ey,
                    "aim_ratio": self.aim_ratio,
                    "aim_x": aim_x,
                    "aim_y": aim_y,
                    "fov_counts_x": counts_x,
                    "fov_counts_y": counts_y,
                    "fov_deg": self.fov_deg,
                    "speed": speed,
                    "final_dx": 0,
                    "final_dy": 0,
                },
            )

        bezier_ctrl = _bezier_ctrl(dx, dy, self.bezier_curvature) if self.move_kind in {"bezier", "enc_bezier"} else None
        return MoveCommand(
            dx=dx,
            dy=dy,
            confidence=target.score,
            reason="proportional strategy",
            move_kind=self.move_kind,
            move_ms=self.move_ms,
            trace_ms=self.trace_ms,
            bezier_ctrl=bezier_ctrl,
            debug={
                **projection_x,
                "stage": "proportional",
                "coordinate_y": "cartesian_up_positive",
                "raw_px_x": ex,
                "raw_px_y": ey,
                "aim_ratio": self.aim_ratio,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "fov_counts_x": counts_x,
                "fov_counts_y": counts_y,
                "c360_x": self.counts_per_revolution_x,
                "c360_y": self.counts_per_revolution_y,
                "fov_deg": self.fov_deg,
                "speed": speed,
                "final_dx": dx,
                "final_dy": dy,
            },
        )

    def _speed_for(self, err_px: float, fov_radius: float) -> float:
        if err_px <= self.near_px:
            return self.near_speed
        ref = fov_radius if fov_radius > 0 else max(self.near_px * 5.0, self.near_px + 1.0)
        span = max(1.0, ref - self.near_px)
        ratio = min(1.0, (err_px - self.near_px) / span)
        return self.near_speed + (self.far_speed - self.near_speed) * ratio


def _bezier_ctrl(dx: int, dy: int, curvature: float) -> tuple[int, int, int, int]:
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (0, 0, 0, 0)
    nx = -dy / dist
    ny = dx / dist
    bow = curvature * dist
    return (
        int(round(dx / 3.0 + nx * bow)),
        int(round(dy / 3.0 + ny * bow)),
        int(round(dx * 2.0 / 3.0 + nx * bow)),
        int(round(dy * 2.0 / 3.0 + ny * bow)),
    )


def _positive_number(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return max(0.0, float(fallback))


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _target_detection_item(target: Target, raw: dict[str, object]) -> dict[str, float]:
    target_key = str(raw.get("target_key") or "")
    index = -1
    if target_key.startswith("det:"):
        parts = target_key.split(":")
        if len(parts) > 1:
            try:
                index = int(parts[1])
            except ValueError:
                index = -1
    return {
        "index": float(index),
        "cls": float(target.cls),
        "score": float(target.score),
        "x1": float(target.x1),
        "y1": float(target.y1),
        "x2": float(target.x2),
        "y2": float(target.y2),
        "cx": (float(target.x1) + float(target.x2)) * 0.5,
        "cy": (float(target.y1) + float(target.y2)) * 0.5,
        "w": max(1.0, float(target.w)),
        "h": max(1.0, float(target.h)),
    }


def _coerce_detection_items(value: object) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, float]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        x1 = _number_or_none(item.get("x1"))
        y1 = _number_or_none(item.get("y1"))
        x2 = _number_or_none(item.get("x2"))
        y2 = _number_or_none(item.get("y2"))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        result.append({
            "index": float(_number_or_none(item.get("index")) if _number_or_none(item.get("index")) is not None else index),
            "cls": float(_number_or_none(item.get("cls")) or 0.0),
            "score": float(_number_or_none(item.get("score")) or 0.0),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) * 0.5,
            "cy": (y1 + y2) * 0.5,
            "w": w,
            "h": h,
        })
    return result


def _number_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _project_pixels_to_counts(
    error_pixels_x: float,
    error_pixels_y: float,
    frame_width: float,
    *,
    fov_deg: float,
    counts_per_revolution: float,
) -> dict[str, float]:
    focal = (frame_width * 0.5) / math.tan(math.radians(fov_deg) * 0.5) if frame_width > 0 else 0.0
    counts_per_degree = counts_per_revolution / 360.0
    yaw_deg = math.degrees(math.atan(error_pixels_x / focal)) if focal > 0 else 0.0
    pitch_deg = math.degrees(math.atan(error_pixels_y / focal)) if focal > 0 else 0.0
    return {
        "focal": focal,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "counts_per_degree": counts_per_degree,
        "err_cx": yaw_deg * counts_per_degree,
        "err_cy": pitch_deg * counts_per_degree,
    }


class ControlCommandCoalescer:
    def __init__(self, *, min_interval_s: float = 0.001) -> None:
        self.min_interval_s = min_interval_s
        self._last_emit_s = 0.0
        self._pending_intent: ControlIntent | None = None
        self._dropped_since_emit = 0

    def push(self, intent: ControlIntent, now_s: float | None = None) -> ControlIntent | None:
        now = time.monotonic() if now_s is None else now_s
        if now - self._last_emit_s < self.min_interval_s:
            self._pending_intent = intent
            self._dropped_since_emit += 1
            return None
        latest = self._pending_intent or intent
        result = ControlIntent(
            dx=latest.dx,
            dy=latest.dy,
            action=latest.action,
            confidence=latest.confidence,
            reason=(
                f"latest-frame throttled control command; dropped={self._dropped_since_emit}"
                if self._dropped_since_emit
                else latest.reason
            ),
            source_id=latest.source_id,
            move_kind=latest.move_kind,
            move_ms=latest.move_ms,
            trace_ms=latest.trace_ms,
            bezier_ctrl=latest.bezier_ctrl,
        )
        self._pending_intent = None
        self._dropped_since_emit = 0
        self._last_emit_s = now
        return result

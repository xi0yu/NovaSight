from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    fov_x_deg: float = 105.0
    counts_per_360_x: float = 9980.0
    counts_per_360_y: float = 9980.0
    axis_sign_x: float = 1.0
    axis_sign_y: float = 1.0

    def normalized(self) -> CalibrationProfile:
        sign_x = -1.0 if self.axis_sign_x < 0 else 1.0
        sign_y = -1.0 if self.axis_sign_y < 0 else 1.0
        return CalibrationProfile(
            fov_x_deg=max(1.0, min(179.0, float(self.fov_x_deg))),
            counts_per_360_x=max(1.0, float(self.counts_per_360_x)),
            counts_per_360_y=max(1.0, float(self.counts_per_360_y)),
            axis_sign_x=sign_x,
            axis_sign_y=sign_y,
        )


@dataclass(frozen=True, slots=True)
class AngularErrorState:
    source_frame_id: int | None
    track_id: int | None
    control_width_px: float
    control_height_px: float
    center_x_px: float
    center_y_px: float
    comp_x: float
    comp_y: float
    error_x_px: float
    error_y_px: float
    fov_x_rad: float
    fov_y_rad: float
    focal_x_px: float
    focal_y_px: float
    error_x_rad: float
    error_y_rad: float
    dt_s: float
    predicted_source: bool = False
    prediction_confidence: float = 1.0
    control_allowed: bool = True
    invalid_reason: str = ""

    @property
    def error_norm_px(self) -> float:
        return math.hypot(self.error_x_px, self.error_y_px)


@dataclass(frozen=True, slots=True)
class AngularPDConfig:
    kp_x: float = 0.35
    kp_y: float = 0.24
    kd_x_s: float = 0.0
    kd_y_s: float = 0.0
    deadzone_rad: float = 0.0
    derivative_ema_alpha: float = 1.0
    max_step_counts: float = 80.0
    dt_min_s: float = 1.0 / 240.0
    dt_max_s: float = 1.0 / 15.0


@dataclass(slots=True)
class ControllerMemory:
    prev_error_x_rad: float = 0.0
    prev_error_y_rad: float = 0.0
    derivative_x_ema: float = 0.0
    derivative_y_ema: float = 0.0
    residual_x_counts: float = 0.0
    residual_y_counts: float = 0.0
    initialized: bool = False

    def reset(self) -> None:
        self.prev_error_x_rad = 0.0
        self.prev_error_y_rad = 0.0
        self.derivative_x_ema = 0.0
        self.derivative_y_ema = 0.0
        self.residual_x_counts = 0.0
        self.residual_y_counts = 0.0
        self.initialized = False


@dataclass(frozen=True, slots=True)
class AngularControlOutput:
    dx: int
    dy: int
    raw_x_counts: float
    raw_y_counts: float
    accum_x_counts: float
    accum_y_counts: float
    residual_x_counts: float
    residual_y_counts: float
    p_x_rad: float
    p_y_rad: float
    d_x_rad: float
    d_y_rad: float
    out_x_rad: float
    out_y_rad: float
    counts_per_rad_x: float
    counts_per_rad_y: float
    debug: dict[str, float | bool | str]


class AngularErrorMapper:
    def __init__(
        self,
        calibration: CalibrationProfile,
        *,
        center_horizontal_percent: float = 50.0,
        center_vertical_percent: float = 50.0,
        dt_hard_reject_s: float = 0.100,
    ) -> None:
        self.calibration = calibration.normalized()
        self.center_horizontal_percent = max(0.0, min(100.0, float(center_horizontal_percent)))
        self.center_vertical_percent = max(0.0, min(100.0, float(center_vertical_percent)))
        self.dt_hard_reject_s = max(1e-6, float(dt_hard_reject_s))

    def map(
        self,
        *,
        comp_x: float,
        comp_y: float,
        control_width_px: float,
        control_height_px: float,
        dt_s: float,
        source_frame_id: int | None = None,
        track_id: int | None = None,
        predicted_source: bool = False,
        prediction_confidence: float = 1.0,
    ) -> AngularErrorState:
        width = float(control_width_px)
        height = float(control_height_px)
        if width <= 0.0 or height <= 0.0:
            return self._invalid("CONTROL_GEOMETRY_INVALID", source_frame_id, track_id, dt_s)
        if not math.isfinite(width) or not math.isfinite(height):
            return self._invalid("CONTROL_GEOMETRY_INVALID", source_frame_id, track_id, dt_s)

        fov_x_rad = math.radians(self.calibration.fov_x_deg)
        if not (0.0 < fov_x_rad < math.pi):
            return self._invalid("FOVX_INVALID", source_frame_id, track_id, dt_s)

        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0 or dt > self.dt_hard_reject_s:
            return self._invalid("CONTROL_DT_INVALID", source_frame_id, track_id, dt_s)

        tan_half = math.tan(fov_x_rad * 0.5)
        fov_y_rad = 2.0 * math.atan(tan_half * height / width)
        focal_x = width / (2.0 * tan_half)
        focal_y = height / (2.0 * math.tan(fov_y_rad * 0.5))
        if focal_x <= 0.0 or focal_y <= 0.0 or not math.isfinite(focal_x) or not math.isfinite(focal_y):
            return self._invalid("FOCAL_INVALID", source_frame_id, track_id, dt_s)

        center_x = width * self.center_horizontal_percent / 100.0
        center_y = height * self.center_vertical_percent / 100.0
        x = float(comp_x)
        y = float(comp_y)
        error_x_px = x - center_x
        error_y_px = y - center_y
        return AngularErrorState(
            source_frame_id=source_frame_id,
            track_id=track_id,
            control_width_px=width,
            control_height_px=height,
            center_x_px=center_x,
            center_y_px=center_y,
            comp_x=x,
            comp_y=y,
            error_x_px=error_x_px,
            error_y_px=error_y_px,
            fov_x_rad=fov_x_rad,
            fov_y_rad=fov_y_rad,
            focal_x_px=focal_x,
            focal_y_px=focal_y,
            error_x_rad=math.atan(error_x_px / focal_x),
            error_y_rad=math.atan(error_y_px / focal_y),
            dt_s=dt,
            predicted_source=bool(predicted_source),
            prediction_confidence=max(0.0, min(1.0, float(prediction_confidence))),
            control_allowed=True,
        )

    @staticmethod
    def _invalid(
        reason: str,
        source_frame_id: int | None,
        track_id: int | None,
        dt_s: float,
    ) -> AngularErrorState:
        return AngularErrorState(
            source_frame_id=source_frame_id,
            track_id=track_id,
            control_width_px=0.0,
            control_height_px=0.0,
            center_x_px=0.0,
            center_y_px=0.0,
            comp_x=0.0,
            comp_y=0.0,
            error_x_px=0.0,
            error_y_px=0.0,
            fov_x_rad=0.0,
            fov_y_rad=0.0,
            focal_x_px=0.0,
            focal_y_px=0.0,
            error_x_rad=0.0,
            error_y_rad=0.0,
            dt_s=max(0.0, float(dt_s)) if math.isfinite(float(dt_s)) else 0.0,
            control_allowed=False,
            invalid_reason=reason,
        )


class AngularPDController:
    def __init__(
        self,
        config: AngularPDConfig,
        calibration: CalibrationProfile,
        memory: ControllerMemory | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration.normalized()
        self.memory = memory or ControllerMemory()

    def reset(self) -> None:
        self.memory.reset()

    def update(self, err: AngularErrorState) -> AngularControlOutput:
        if not err.control_allowed:
            self.reset()
            return self._zero(err, err.invalid_reason or "CONTROL_NOT_ALLOWED")

        cfg = self.config
        dt = max(float(cfg.dt_min_s), min(float(cfg.dt_max_s), float(err.dt_s)))
        ex = 0.0 if abs(err.error_x_rad) <= cfg.deadzone_rad else err.error_x_rad
        ey = 0.0 if abs(err.error_y_rad) <= cfg.deadzone_rad else err.error_y_rad

        if not self.memory.initialized:
            raw_dx = 0.0
            raw_dy = 0.0
            self.memory.initialized = True
        else:
            raw_dx = (ex - self.memory.prev_error_x_rad) / dt
            raw_dy = (ey - self.memory.prev_error_y_rad) / dt

        alpha = max(0.0, min(1.0, float(cfg.derivative_ema_alpha)))
        self.memory.derivative_x_ema = alpha * raw_dx + (1.0 - alpha) * self.memory.derivative_x_ema
        self.memory.derivative_y_ema = alpha * raw_dy + (1.0 - alpha) * self.memory.derivative_y_ema

        prediction_gain = err.prediction_confidence if err.predicted_source else 1.0
        prediction_gain = max(0.0, min(1.0, float(prediction_gain)))
        p_x = max(0.0, cfg.kp_x) * ex * prediction_gain
        p_y = max(0.0, cfg.kp_y) * ey * prediction_gain
        d_x = cfg.kd_x_s * self.memory.derivative_x_ema * prediction_gain
        d_y = cfg.kd_y_s * self.memory.derivative_y_ema * prediction_gain
        out_x_rad = p_x + d_x
        out_y_rad = p_y + d_y

        counts_per_rad_x = self.calibration.counts_per_360_x / (2.0 * math.pi)
        counts_per_rad_y = self.calibration.counts_per_360_y / (2.0 * math.pi)
        raw_x = out_x_rad * counts_per_rad_x * self.calibration.axis_sign_x
        raw_y = out_y_rad * counts_per_rad_y * self.calibration.axis_sign_y
        raw_x = _clamp(raw_x, cfg.max_step_counts)
        raw_y = _clamp(raw_y, cfg.max_step_counts)

        accum_x = raw_x + self.memory.residual_x_counts
        accum_y = raw_y + self.memory.residual_y_counts
        emit_x = math.trunc(accum_x)
        emit_y = math.trunc(accum_y)
        self.memory.residual_x_counts = accum_x - emit_x
        self.memory.residual_y_counts = accum_y - emit_y
        emit_x = int(_clamp(float(emit_x), cfg.max_step_counts))
        emit_y = int(_clamp(float(emit_y), cfg.max_step_counts))

        self.memory.prev_error_x_rad = ex
        self.memory.prev_error_y_rad = ey

        return AngularControlOutput(
            dx=emit_x,
            dy=emit_y,
            raw_x_counts=raw_x,
            raw_y_counts=raw_y,
            accum_x_counts=accum_x,
            accum_y_counts=accum_y,
            residual_x_counts=self.memory.residual_x_counts,
            residual_y_counts=self.memory.residual_y_counts,
            p_x_rad=p_x,
            p_y_rad=p_y,
            d_x_rad=d_x,
            d_y_rad=d_y,
            out_x_rad=out_x_rad,
            out_y_rad=out_y_rad,
            counts_per_rad_x=counts_per_rad_x,
            counts_per_rad_y=counts_per_rad_y,
            debug={
                "control_allowed": True,
                "dt": dt,
                "prediction_gain": prediction_gain,
                "derivative_x_rad_s": self.memory.derivative_x_ema,
                "derivative_y_rad_s": self.memory.derivative_y_ema,
            },
        )

    def _zero(self, err: AngularErrorState, reason: str) -> AngularControlOutput:
        return AngularControlOutput(
            dx=0,
            dy=0,
            raw_x_counts=0.0,
            raw_y_counts=0.0,
            accum_x_counts=0.0,
            accum_y_counts=0.0,
            residual_x_counts=0.0,
            residual_y_counts=0.0,
            p_x_rad=0.0,
            p_y_rad=0.0,
            d_x_rad=0.0,
            d_y_rad=0.0,
            out_x_rad=0.0,
            out_y_rad=0.0,
            counts_per_rad_x=self.calibration.counts_per_360_x / (2.0 * math.pi),
            counts_per_rad_y=self.calibration.counts_per_360_y / (2.0 * math.pi),
            debug={
                "control_allowed": False,
                "invalid_reason": reason,
                "source_frame_id": err.source_frame_id or 0,
            },
        )


def _clamp(value: float, limit: float) -> float:
    lim = abs(float(limit))
    return max(-lim, min(lim, float(value)))

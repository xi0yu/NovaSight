from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt


@dataclass(frozen=True)
class KalmanConfig:
    enabled: bool = True
    acceleration_noise: float = 1200.0
    measurement_noise_x: float = 16.0
    measurement_noise_y: float = 16.0
    max_predict_missing_ms: float = 80.0
    max_predict_steps: int = 5
    max_predict_dt_ms: float = 35.0
    max_position_sigma_px: float = 45.0
    max_covariance_trace: float = 5000.0
    nis_threshold: float = 9.21
    nis_hard_reject: float = 16.0
    min_identity_confidence: float = 0.70
    min_prediction_confidence: float = 0.35
    prediction_decay_tau_ms: float = 45.0


@dataclass(frozen=True)
class EstimatedState:
    track_id: int
    state_ts_ns: int
    x: float
    y: float
    vx: float
    vy: float
    cov_trace: float
    position_sigma_px: float
    nis: float
    prediction_confidence: float
    predicted: bool
    prediction_steps: int
    valid: bool
    reason: str = ""


class KalmanEstimator:
    def __init__(
        self,
        *,
        track_id: int,
        x: float,
        y: float,
        ts_ns: int,
        config: KalmanConfig | None = None,
        identity_confidence: float = 1.0,
    ) -> None:
        self.track_id = int(track_id)
        self.config = config or KalmanConfig()
        self.x = [float(x), float(y), 0.0, 0.0]
        pos_noise_x = max(1.0, float(self.config.measurement_noise_x))
        pos_noise_y = max(1.0, float(self.config.measurement_noise_y))
        self.p = [
            [pos_noise_x, 0.0, 0.0, 0.0],
            [0.0, pos_noise_y, 0.0, 0.0],
            [0.0, 0.0, 1000.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0],
        ]
        self.last_state_ts_ns = int(ts_ns)
        self.last_measurement_ts_ns = int(ts_ns)
        self.last_nis = 0.0
        self.prediction_steps = 0
        self.identity_confidence = _clamp01(identity_confidence)
        self.last_estimate = self._estimate(
            ts_ns=int(ts_ns),
            predicted=False,
            valid=True,
            reason="initialized",
        )

    def update_config(self, config: KalmanConfig) -> None:
        self.config = config

    def measurement_nis(self, x: float, y: float, ts_ns: int) -> float:
        if not self.config.enabled:
            return 0.0
        dt = self._dt(ts_ns)
        x_pred, p_pred = self._predict_matrices(dt)
        innovation = [float(x) - x_pred[0], float(y) - x_pred[1]]
        s00 = p_pred[0][0] + max(1e-6, float(self.config.measurement_noise_x))
        s01 = p_pred[0][1]
        s10 = p_pred[1][0]
        s11 = p_pred[1][1] + max(1e-6, float(self.config.measurement_noise_y))
        return _mahalanobis_2d(innovation, s00, s01, s10, s11)

    def predict_only(
        self,
        *,
        ts_ns: int,
        identity_confidence: float | None = None,
    ) -> EstimatedState:
        if not self.config.enabled:
            return self._estimate(
                ts_ns=int(ts_ns),
                predicted=True,
                valid=True,
                reason="kalman disabled",
            )
        if identity_confidence is not None:
            self.identity_confidence = _clamp01(identity_confidence)
        dt = self._dt(ts_ns)
        self.x, self.p = self._predict_matrices(dt)
        self.last_state_ts_ns = int(ts_ns)
        self.prediction_steps += 1
        estimate = self._estimate(
            ts_ns=int(ts_ns),
            predicted=True,
            valid=True,
            reason="predicted",
        )
        self.last_estimate = estimate
        return estimate

    def update(
        self,
        *,
        measurement_x: float,
        measurement_y: float,
        ts_ns: int,
        identity_confidence: float,
    ) -> EstimatedState:
        self.identity_confidence = _clamp01(identity_confidence)
        if not self.config.enabled:
            self.x[0] = float(measurement_x)
            self.x[1] = float(measurement_y)
            self.last_state_ts_ns = int(ts_ns)
            self.last_measurement_ts_ns = int(ts_ns)
            self.prediction_steps = 0
            estimate = self._estimate(
                ts_ns=int(ts_ns),
                predicted=False,
                valid=True,
                reason="kalman disabled",
            )
            self.last_estimate = estimate
            return estimate

        dt = self._dt(ts_ns)
        x_pred, p_pred = self._predict_matrices(dt)
        innovation = [float(measurement_x) - x_pred[0], float(measurement_y) - x_pred[1]]
        r00 = max(1e-6, float(self.config.measurement_noise_x))
        r11 = max(1e-6, float(self.config.measurement_noise_y))
        s00 = p_pred[0][0] + r00
        s01 = p_pred[0][1]
        s10 = p_pred[1][0]
        s11 = p_pred[1][1] + r11
        nis = _mahalanobis_2d(innovation, s00, s01, s10, s11)
        if not isfinite(nis) or nis > float(self.config.nis_hard_reject):
            self.x = x_pred
            self.p = p_pred
            self.last_state_ts_ns = int(ts_ns)
            self.prediction_steps += 1
            estimate = self._estimate(
                ts_ns=int(ts_ns),
                predicted=True,
                valid=False,
                reason="NIS_REJECT",
                nis=nis,
            )
            self.last_estimate = estimate
            return estimate

        inv_s = _inverse_2x2(s00, s01, s10, s11)
        if inv_s is None:
            estimate = self._estimate(
                ts_ns=int(ts_ns),
                predicted=True,
                valid=False,
                reason="SINGULAR_INNOVATION",
                nis=nis,
            )
            self.last_estimate = estimate
            return estimate

        k = [
            [
                p_pred[row][0] * inv_s[0][col] + p_pred[row][1] * inv_s[1][col]
                for col in range(2)
            ]
            for row in range(4)
        ]
        self.x = [
            x_pred[row] + k[row][0] * innovation[0] + k[row][1] * innovation[1]
            for row in range(4)
        ]
        kh = [
            [k[row][0], k[row][1], 0.0, 0.0]
            for row in range(4)
        ]
        i_minus_kh = [
            [1.0 if row == col else 0.0 for col in range(4)]
            for row in range(4)
        ]
        for row in range(4):
            for col in range(4):
                i_minus_kh[row][col] -= kh[row][col]
        self.p = _matmul(i_minus_kh, p_pred)
        self.last_nis = float(nis)
        self.last_state_ts_ns = int(ts_ns)
        self.last_measurement_ts_ns = int(ts_ns)
        self.prediction_steps = 0
        estimate = self._estimate(
            ts_ns=int(ts_ns),
            predicted=False,
            valid=True,
            reason="updated",
            nis=nis,
        )
        self.last_estimate = estimate
        return estimate

    def apply_velocity_hint(
        self,
        *,
        vx: float,
        vy: float,
        ts_ns: int,
        weight: float = 1.0,
    ) -> EstimatedState:
        """Blend a timestamp-derived measurement velocity into the state.

        A two-position constant-velocity Kalman filter can take several frames
        to infer velocity when dt is small. The tracker already has reliable
        matched bbox centers and capture timestamps, so use that finite
        difference as a velocity observation without changing position state.
        """
        if not self.config.enabled:
            return self.last_estimate
        if not all(isfinite(float(value)) for value in (vx, vy)):
            return self.last_estimate
        alpha = _clamp01(weight)
        self.x[2] = self.x[2] * (1.0 - alpha) + float(vx) * alpha
        self.x[3] = self.x[3] * (1.0 - alpha) + float(vy) * alpha
        estimate = self._estimate(
            ts_ns=int(ts_ns),
            predicted=False,
            valid=True,
            reason="updated",
        )
        self.last_estimate = estimate
        return estimate

    def _dt(self, ts_ns: int) -> float:
        raw = (int(ts_ns) - int(self.last_state_ts_ns)) / 1e9
        if not isfinite(raw) or raw < 0:
            return 0.0
        max_dt = max(0.001, float(self.config.max_predict_dt_ms) / 1000.0)
        return min(raw, max_dt)

    def _predict_matrices(self, dt: float) -> tuple[list[float], list[list[float]]]:
        f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        q = _process_noise(dt, max(1e-6, float(self.config.acceleration_noise)))
        x_pred = [
            f[row][0] * self.x[0]
            + f[row][1] * self.x[1]
            + f[row][2] * self.x[2]
            + f[row][3] * self.x[3]
            for row in range(4)
        ]
        p_pred = _matadd(_matmul(_matmul(f, self.p), _transpose(f)), q)
        return x_pred, p_pred

    def _estimate(
        self,
        *,
        ts_ns: int,
        predicted: bool,
        valid: bool,
        reason: str,
        nis: float | None = None,
    ) -> EstimatedState:
        cov_trace = sum(float(self.p[i][i]) for i in range(4))
        sigma = sqrt(max(0.0, float(self.p[0][0]) + float(self.p[1][1])) / 2.0)
        current_nis = self.last_nis if nis is None else float(nis)
        missing_ms = max(0.0, (int(ts_ns) - int(self.last_measurement_ts_ns)) / 1e6)
        confidence = self._prediction_confidence(
            missing_ms=missing_ms,
            sigma=sigma,
            nis=current_nis,
        )
        computed_valid = (
            bool(valid)
            and all(isfinite(float(value)) for value in self.x)
            and isfinite(cov_trace)
            and isfinite(sigma)
            and missing_ms <= float(self.config.max_predict_missing_ms)
            and self.prediction_steps <= int(self.config.max_predict_steps)
            and sigma <= float(self.config.max_position_sigma_px)
            and cov_trace <= float(self.config.max_covariance_trace)
            and self.identity_confidence >= float(self.config.min_identity_confidence)
            and current_nis <= float(self.config.nis_threshold)
            and confidence >= float(self.config.min_prediction_confidence)
        )
        return EstimatedState(
            track_id=self.track_id,
            state_ts_ns=int(ts_ns),
            x=float(self.x[0]),
            y=float(self.x[1]),
            vx=float(self.x[2]),
            vy=float(self.x[3]),
            cov_trace=float(cov_trace),
            position_sigma_px=float(sigma),
            nis=float(current_nis),
            prediction_confidence=float(confidence),
            predicted=bool(predicted),
            prediction_steps=int(self.prediction_steps),
            valid=bool(computed_valid),
            reason=reason,
        )

    def _prediction_confidence(self, *, missing_ms: float, sigma: float, nis: float) -> float:
        max_sigma = max(1e-6, float(self.config.max_position_sigma_px))
        nis_threshold = max(1e-6, float(self.config.nis_threshold))
        tau_ms = max(1e-6, float(self.config.prediction_decay_tau_ms))
        covariance_confidence = _clamp01(1.0 - sigma / max_sigma)
        residual_confidence = _clamp01(1.0 - max(0.0, nis) / nis_threshold)
        missing_decay = exp(-max(0.0, missing_ms) / tau_ms)
        return _clamp01(
            self.identity_confidence
            * covariance_confidence
            * residual_confidence
            * missing_decay
        )


def _process_noise(dt: float, acceleration_noise: float) -> list[list[float]]:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    scale = acceleration_noise
    return [
        [scale * dt4 / 4.0, 0.0, scale * dt3 / 2.0, 0.0],
        [0.0, scale * dt4 / 4.0, 0.0, scale * dt3 / 2.0],
        [scale * dt3 / 2.0, 0.0, scale * dt2, 0.0],
        [0.0, scale * dt3 / 2.0, 0.0, scale * dt2],
    ]


def _mahalanobis_2d(
    innovation: list[float],
    s00: float,
    s01: float,
    s10: float,
    s11: float,
) -> float:
    inv = _inverse_2x2(s00, s01, s10, s11)
    if inv is None:
        return float("inf")
    left = innovation[0] * inv[0][0] + innovation[1] * inv[1][0]
    right = innovation[0] * inv[0][1] + innovation[1] * inv[1][1]
    return float(left * innovation[0] + right * innovation[1])


def _inverse_2x2(
    a: float,
    b: float,
    c: float,
    d: float,
) -> list[list[float]] | None:
    det = a * d - b * c
    if not isfinite(det) or abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    return [[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    rows = len(left)
    cols = len(right[0])
    inner = len(right)
    return [
        [
            sum(left[row][idx] * right[idx][col] for idx in range(inner))
            for col in range(cols)
        ]
        for row in range(rows)
    ]


def _matadd(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [left[row][col] + right[row][col] for col in range(len(left[row]))]
        for row in range(len(left))
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix, strict=True)]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

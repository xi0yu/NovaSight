//! Allocation-free constant-velocity Kalman state used by target association.
//!
//! The filtered position is deliberately limited to identity association.
//! Target scoring and mouse control consume the latest observed aim point so
//! filtering cannot add visible selection or control lag.

use serde::{Deserialize, Serialize};

// Runtime-tunable values and internal safety limits share one canonical
// default here. Persistence reads these defaults through `KalmanConfig`
// instead of copying numeric literals into another crate.
const DEFAULT_ACCELERATION_NOISE: f64 = 1_200.0;
const DEFAULT_MEASUREMENT_NOISE_X: f64 = 16.0;
const DEFAULT_MEASUREMENT_NOISE_Y: f64 = 16.0;
const DEFAULT_MAX_PREDICT_DT_MS: f64 = 35.0;
const DEFAULT_MAX_PREDICT_MISSING_MS: f64 = 80.0;
const DEFAULT_MAX_PREDICT_STEPS: u32 = 5;
const DEFAULT_NIS_THRESHOLD: f64 = 9.21;
const DEFAULT_NIS_HARD_REJECT: f64 = 16.0;
const DEFAULT_MAX_POSITION_SIGMA_PX: f64 = 45.0;
const DEFAULT_MAX_COVARIANCE_TRACE: f64 = 5_000.0;
const DEFAULT_MIN_IDENTITY_CONFIDENCE: f64 = 0.70;
const DEFAULT_MIN_PREDICTION_CONFIDENCE: f64 = 0.35;
const DEFAULT_PREDICTION_DECAY_TAU_MS: f64 = 45.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct KalmanConfig {
    pub acceleration_noise: f64,
    pub measurement_noise_x: f64,
    pub measurement_noise_y: f64,
    pub max_predict_dt_ms: f64,
    pub max_predict_missing_ms: f64,
    pub max_predict_steps: u32,
    pub nis_threshold: f64,
    pub nis_hard_reject: f64,
    pub max_position_sigma_px: f64,
    pub max_covariance_trace: f64,
    pub min_identity_confidence: f64,
    pub min_prediction_confidence: f64,
    pub prediction_decay_tau_ms: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            acceleration_noise: DEFAULT_ACCELERATION_NOISE,
            measurement_noise_x: DEFAULT_MEASUREMENT_NOISE_X,
            measurement_noise_y: DEFAULT_MEASUREMENT_NOISE_Y,
            max_predict_dt_ms: DEFAULT_MAX_PREDICT_DT_MS,
            max_predict_missing_ms: DEFAULT_MAX_PREDICT_MISSING_MS,
            max_predict_steps: DEFAULT_MAX_PREDICT_STEPS,
            nis_threshold: DEFAULT_NIS_THRESHOLD,
            nis_hard_reject: DEFAULT_NIS_HARD_REJECT,
            max_position_sigma_px: DEFAULT_MAX_POSITION_SIGMA_PX,
            max_covariance_trace: DEFAULT_MAX_COVARIANCE_TRACE,
            min_identity_confidence: DEFAULT_MIN_IDENTITY_CONFIDENCE,
            min_prediction_confidence: DEFAULT_MIN_PREDICTION_CONFIDENCE,
            prediction_decay_tau_ms: DEFAULT_PREDICTION_DECAY_TAU_MS,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct KalmanState {
    state: [f64; 4],
    covariance: [[f64; 4]; 4],
    state_ts_ns: u64,
    last_measurement_ts_ns: u64,
    last_nis: f64,
    prediction_steps: u32,
    estimate_valid: bool,
    initialized: bool,
}

impl Default for KalmanState {
    fn default() -> Self {
        Self {
            state: [0.0; 4],
            covariance: [[0.0; 4]; 4],
            state_ts_ns: 0,
            last_measurement_ts_ns: 0,
            last_nis: 0.0,
            prediction_steps: 0,
            estimate_valid: false,
            initialized: false,
        }
    }
}

impl KalmanState {
    pub(super) fn new(
        x: f64,
        y: f64,
        ts_ns: u64,
        config: KalmanConfig,
        identity_confidence: f64,
    ) -> Self {
        let mut covariance = [[0.0; 4]; 4];
        covariance[0][0] = config.measurement_noise_x.max(1.0);
        covariance[1][1] = config.measurement_noise_y.max(1.0);
        covariance[2][2] = 1_000.0;
        covariance[3][3] = 1_000.0;
        let mut state = Self {
            state: [x, y, 0.0, 0.0],
            covariance,
            state_ts_ns: ts_ns,
            last_measurement_ts_ns: ts_ns,
            last_nis: 0.0,
            prediction_steps: 0,
            estimate_valid: false,
            initialized: true,
        };
        state.estimate_valid = state.computed_valid(config, identity_confidence);
        state
    }

    pub(super) fn position(&self) -> (f64, f64) {
        (self.state[0], self.state[1])
    }

    pub(super) fn prediction_valid(&self) -> bool {
        self.estimate_valid
    }

    pub(super) fn predict(
        &mut self,
        ts_ns: u64,
        config: KalmanConfig,
        identity_confidence: f64,
    ) -> bool {
        if !self.initialized || ts_ns <= self.state_ts_ns {
            return self.estimate_valid;
        }
        let raw_dt = ts_ns.saturating_sub(self.state_ts_ns) as f64 / 1_000_000_000.0;
        let dt = raw_dt.min(config.max_predict_dt_ms.max(1.0) / 1_000.0);
        let (state, covariance) = self.predicted(dt, config);
        self.state = state;
        self.covariance = covariance;
        self.state_ts_ns = ts_ns;
        self.prediction_steps = self.prediction_steps.saturating_add(1);
        self.estimate_valid = self.computed_valid(config, identity_confidence);
        self.estimate_valid
    }

    pub(super) fn measurement_nis(&self, x: f64, y: f64, config: KalmanConfig) -> f64 {
        let s00 = self.covariance[0][0] + config.measurement_noise_x.max(1e-6);
        let s01 = self.covariance[0][1];
        let s10 = self.covariance[1][0];
        let s11 = self.covariance[1][1] + config.measurement_noise_y.max(1e-6);
        mahalanobis_2d(x - self.state[0], y - self.state[1], s00, s01, s10, s11)
    }

    pub(super) fn measurement_nis_from_position(
        &self,
        x: f64,
        y: f64,
        reference_x: f64,
        reference_y: f64,
        config: KalmanConfig,
    ) -> f64 {
        let s00 = self.covariance[0][0] + config.measurement_noise_x.max(1e-6);
        let s01 = self.covariance[0][1];
        let s10 = self.covariance[1][0];
        let s11 = self.covariance[1][1] + config.measurement_noise_y.max(1e-6);
        mahalanobis_2d(x - reference_x, y - reference_y, s00, s01, s10, s11)
    }

    pub(super) fn update(
        &mut self,
        x: f64,
        y: f64,
        ts_ns: u64,
        config: KalmanConfig,
        identity_confidence: f64,
    ) -> bool {
        let r00 = config.measurement_noise_x.max(1e-6);
        let r11 = config.measurement_noise_y.max(1e-6);
        let s00 = self.covariance[0][0] + r00;
        let s01 = self.covariance[0][1];
        let s10 = self.covariance[1][0];
        let s11 = self.covariance[1][1] + r11;
        let nis = mahalanobis_2d(x - self.state[0], y - self.state[1], s00, s01, s10, s11);
        if !nis.is_finite() || nis > config.nis_hard_reject {
            *self = Self::new(x, y, ts_ns, config, identity_confidence);
            return self.estimate_valid;
        }
        let Some(inverse) = inverse_2x2(s00, s01, s10, s11) else {
            *self = Self::new(x, y, ts_ns, config, identity_confidence);
            return self.estimate_valid;
        };
        let innovation = [x - self.state[0], y - self.state[1]];
        let mut gain = [[0.0; 2]; 4];
        for (row, gain_row) in gain.iter_mut().enumerate() {
            gain_row[0] =
                self.covariance[row][0] * inverse[0][0] + self.covariance[row][1] * inverse[1][0];
            gain_row[1] =
                self.covariance[row][0] * inverse[0][1] + self.covariance[row][1] * inverse[1][1];
        }
        for (row, value) in self.state.iter_mut().enumerate() {
            *value += gain[row][0] * innovation[0] + gain[row][1] * innovation[1];
        }
        let previous = self.covariance;
        self.covariance = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                previous[row][column]
                    - gain[row][0] * previous[0][column]
                    - gain[row][1] * previous[1][column]
            })
        });
        self.state_ts_ns = ts_ns;
        self.last_measurement_ts_ns = ts_ns;
        self.last_nis = nis;
        self.prediction_steps = 0;
        self.estimate_valid = self.computed_valid(config, identity_confidence);
        self.estimate_valid
    }

    fn predicted(&self, dt: f64, config: KalmanConfig) -> ([f64; 4], [[f64; 4]; 4]) {
        let state = [
            self.state[0] + dt * self.state[2],
            self.state[1] + dt * self.state[3],
            self.state[2],
            self.state[3],
        ];
        let fp: [[f64; 4]; 4] = std::array::from_fn(|row| {
            std::array::from_fn(|column| match row {
                0 => self.covariance[0][column] + dt * self.covariance[2][column],
                1 => self.covariance[1][column] + dt * self.covariance[3][column],
                _ => self.covariance[row][column],
            })
        });
        let mut covariance = std::array::from_fn(|row| {
            [
                fp[row][0] + dt * fp[row][2],
                fp[row][1] + dt * fp[row][3],
                fp[row][2],
                fp[row][3],
            ]
        });
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;
        let q = config.acceleration_noise.max(1e-6);
        covariance[0][0] += q * dt4 / 4.0;
        covariance[0][2] += q * dt3 / 2.0;
        covariance[1][1] += q * dt4 / 4.0;
        covariance[1][3] += q * dt3 / 2.0;
        covariance[2][0] += q * dt3 / 2.0;
        covariance[2][2] += q * dt2;
        covariance[3][1] += q * dt3 / 2.0;
        covariance[3][3] += q * dt2;
        (state, covariance)
    }

    fn computed_valid(&self, config: KalmanConfig, identity_confidence: f64) -> bool {
        let covariance_trace = (0..4)
            .map(|index| self.covariance[index][index])
            .sum::<f64>();
        let sigma = ((self.covariance[0][0] + self.covariance[1][1]).max(0.0) * 0.5).sqrt();
        let missing_ms = self.state_ts_ns.saturating_sub(self.last_measurement_ts_ns) as f64 / 1e6;
        let covariance_confidence =
            (1.0 - sigma / config.max_position_sigma_px.max(1e-6)).clamp(0.0, 1.0);
        let residual_confidence =
            (1.0 - self.last_nis.max(0.0) / config.nis_threshold.max(1e-6)).clamp(0.0, 1.0);
        let missing_decay = (-missing_ms / config.prediction_decay_tau_ms.max(1e-6)).exp();
        let prediction_confidence = identity_confidence.clamp(0.0, 1.0)
            * covariance_confidence
            * residual_confidence
            * missing_decay;
        self.state.iter().all(|value| value.is_finite())
            && covariance_trace.is_finite()
            && covariance_trace <= config.max_covariance_trace
            && sigma.is_finite()
            && sigma <= config.max_position_sigma_px
            && missing_ms <= config.max_predict_missing_ms
            && self.prediction_steps <= config.max_predict_steps
            && identity_confidence >= config.min_identity_confidence
            && self.last_nis <= config.nis_threshold
            && prediction_confidence >= config.min_prediction_confidence
    }
}

fn inverse_2x2(a: f64, b: f64, c: f64, d: f64) -> Option<[[f64; 2]; 2]> {
    let determinant = a * d - b * c;
    (determinant.is_finite() && determinant.abs() >= 1e-12).then_some([
        [d / determinant, -b / determinant],
        [-c / determinant, a / determinant],
    ])
}

fn mahalanobis_2d(dx: f64, dy: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    let Some(inverse) = inverse_2x2(a, b, c, d) else {
        return f64::INFINITY;
    };
    dx * (inverse[0][0] * dx + inverse[0][1] * dy) + dy * (inverse[1][0] * dx + inverse[1][1] * dy)
}

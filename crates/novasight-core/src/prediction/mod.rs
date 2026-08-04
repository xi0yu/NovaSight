//! Single-focus-target motion prediction.
//!
//! Tracking may retain multiple identities for association, but this module
//! accepts observations for only the selected `track_id`. It owns the bounded
//! velocity history and returns one prediction result for the controller.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

const VELOCITY_POSITION_COUNT: usize = 4;
const PREDICTION_FULL_STRENGTH_CONFIDENCE: f64 = 0.55;
const PREDICTION_STABLE_MEAN_CONFIDENCE: f64 = 0.35;
const PREDICTION_REACTIVE_CONFIDENCE: f64 = 0.25;
const PREDICTION_PEEK_MAX_STRENGTH: f64 = 0.35;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionMotionState {
    Mean,
    Continuous,
    AbruptStopOrReverse,
    AlternatingPeek,
    Stationary,
    #[default]
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleTargetPredictionConfig {
    pub enabled: bool,
    pub history_reset_gap_ms: f64,
    pub spread_base_px_ms: f64,
    pub spread_relative: f64,
    /// Command-to-visible-response delay included in the capture-to-actuation
    /// prediction horizon before the optional time lead.
    pub actuation_delay_ms: f64,
    pub lead_ms: f64,
    pub far_absolute_cap_px: f64,
    pub far_base_cap_px: f64,
    pub far_relative_cap: f64,
    pub near_absolute_cap_px: f64,
    pub near_base_cap_px: f64,
    pub near_relative_cap: f64,
}

impl SingleTargetPredictionConfig {
    pub fn is_valid(self) -> bool {
        if !self.enabled {
            return true;
        }
        self.history_reset_gap_ms.is_finite()
            && self.history_reset_gap_ms > 0.0
            && self.spread_base_px_ms.is_finite()
            && self.spread_base_px_ms > 0.0
            && self.spread_relative.is_finite()
            && self.spread_relative >= 0.0
            && self.actuation_delay_ms.is_finite()
            && self.actuation_delay_ms >= 0.0
            && self.lead_ms.is_finite()
            && (0.0..=1_000.0).contains(&self.lead_ms)
            && [
                self.far_absolute_cap_px,
                self.far_base_cap_px,
                self.far_relative_cap,
                self.near_absolute_cap_px,
                self.near_base_cap_px,
                self.near_relative_cap,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocusTargetObservation {
    pub track_id: u64,
    pub aim_x: f64,
    pub aim_y: f64,
    pub measured_error_x: f64,
    pub measured_error_y: f64,
    pub capture_ts_ns: u64,
    /// Time already elapsed from this capture to the current control
    /// calculation. Prediction starts at capture time, so this measured age
    /// must be covered before any configured extra time lead is added.
    pub observation_age_ms: f64,
    pub detection_confidence: f64,
    pub identity_confidence: f64,
    /// Continuous FAR response weight shared with the controller response
    /// curve. Zero is fully NEAR and one is fully FAR.
    pub far_weight: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AxisPrediction {
    pub velocity: f64,
    pub motion_state: PredictionMotionState,
    pub trend_consistency: f64,
    pub acceleration_px_ms2: f64,
    pub motion_confidence: f64,
    pub velocity_samples: [Option<f64>; 3],
    pub mean_velocity: Option<f64>,
    pub median_velocity: Option<f64>,
    pub velocity_spread: Option<f64>,
    pub measurement_dt_ms: Option<f64>,
    pub reference_dt_ms: f64,
    pub horizon_ms: f64,
    pub raw_offset: f64,
    pub weighted_offset: f64,
    pub allowed_cap: f64,
    pub safe_offset: f64,
    pub allowed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SingleTargetPrediction {
    pub x: AxisPrediction,
    pub y: AxisPrediction,
    pub history_position_count: usize,
    pub actuation_delay_ms: f64,
    pub lead_ms: f64,
}

/// Stateful, bounded predictor for the one target selected upstream.
#[derive(Clone, Debug)]
pub struct SingleTargetPredictor {
    config: SingleTargetPredictionConfig,
    velocity: RobustAimVelocityEstimator,
}

impl SingleTargetPredictor {
    pub fn new(config: SingleTargetPredictionConfig) -> Self {
        Self {
            velocity: RobustAimVelocityEstimator::new(config),
            config,
        }
    }

    pub fn set_config(&mut self, config: SingleTargetPredictionConfig) {
        self.config = config;
        self.velocity.set_config(config);
    }

    pub fn config_valid(&self) -> bool {
        self.config.is_valid()
    }

    pub fn reset(&mut self, track_id: Option<u64>) {
        self.velocity.reset(track_id);
    }

    /// Return the configured prediction envelope without advancing history.
    /// Used when the current observation cannot safely join the time series.
    pub fn unavailable(
        &self,
        measured_error_x: f64,
        measured_error_y: f64,
        far_weight: f64,
    ) -> SingleTargetPrediction {
        if !self.config.enabled {
            return SingleTargetPrediction::default();
        }
        let measured_error_radius = Vector2::new(measured_error_x, measured_error_y).magnitude();
        let allowed_cap = self.allowed_cap(measured_error_radius, far_weight);
        SingleTargetPrediction {
            x: AxisPrediction {
                allowed_cap,
                ..AxisPrediction::default()
            },
            y: AxisPrediction {
                allowed_cap,
                ..AxisPrediction::default()
            },
            history_position_count: self.velocity.history_position_count(),
            actuation_delay_ms: self.config.actuation_delay_ms,
            lead_ms: self.config.lead_ms,
        }
    }

    pub fn predict(&mut self, observation: FocusTargetObservation) -> SingleTargetPrediction {
        if !self.config.enabled {
            return SingleTargetPrediction::default();
        }
        if !observation.aim_x.is_finite()
            || !observation.aim_y.is_finite()
            || observation.capture_ts_ns == 0
            || !observation.observation_age_ms.is_finite()
            || observation.observation_age_ms < 0.0
        {
            return self.unavailable(
                observation.measured_error_x,
                observation.measured_error_y,
                observation.far_weight,
            );
        }

        let Some(estimate) = self.velocity.update(
            observation.track_id,
            observation.aim_x,
            observation.aim_y,
            observation.capture_ts_ns,
            observation.detection_confidence,
            observation.identity_confidence,
        ) else {
            return self.unavailable(
                observation.measured_error_x,
                observation.measured_error_y,
                observation.far_weight,
            );
        };

        let allowed = estimate.reference_dt_ms.is_finite() && estimate.reference_dt_ms > 0.0;
        let horizon_ms = if allowed {
            observation.observation_age_ms
                + self.config.actuation_delay_ms
                + self.config.lead_ms.max(0.0)
        } else {
            0.0
        };
        let prediction_strength = if allowed {
            estimate.motion_confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let raw_offset = estimate.velocity.scale(horizon_ms);
        let weighted_offset = raw_offset.scale(prediction_strength);
        let measured_error_radius =
            Vector2::new(observation.measured_error_x, observation.measured_error_y).magnitude();
        let allowed_cap = self.allowed_cap(measured_error_radius, observation.far_weight);
        let safe_offset = clamp_vector_magnitude(weighted_offset, allowed_cap);

        SingleTargetPrediction {
            x: axis_prediction(
                estimate,
                Axis::X,
                horizon_ms,
                raw_offset,
                weighted_offset,
                safe_offset,
                allowed_cap,
                allowed,
            ),
            y: axis_prediction(
                estimate,
                Axis::Y,
                horizon_ms,
                raw_offset,
                weighted_offset,
                safe_offset,
                allowed_cap,
                allowed,
            ),
            history_position_count: self.velocity.history_position_count(),
            actuation_delay_ms: self.config.actuation_delay_ms,
            lead_ms: self.config.lead_ms,
        }
    }

    fn allowed_cap(&self, measured_error_radius: f64, far_weight: f64) -> f64 {
        let far_weight = far_weight.clamp(0.0, 1.0);
        let absolute_cap = lerp(
            self.config.near_absolute_cap_px,
            self.config.far_absolute_cap_px,
            far_weight,
        );
        let base_cap = lerp(
            self.config.near_base_cap_px,
            self.config.far_base_cap_px,
            far_weight,
        );
        let relative_cap = lerp(
            self.config.near_relative_cap,
            self.config.far_relative_cap,
            far_weight,
        );
        absolute_cap
            .min(base_cap + relative_cap * measured_error_radius.max(0.0))
            .max(0.0)
    }
}

pub(crate) fn prediction_gate_strength(
    motion_state: PredictionMotionState,
    trend_consistency: f64,
    confidence: f64,
) -> f64 {
    let confidence = confidence.clamp(0.0, 1.0);
    if confidence <= f64::EPSILON {
        return 0.0;
    }
    let trend_consistency = trend_consistency.clamp(0.0, 1.0);
    let (open_at, max_strength) = match motion_state {
        PredictionMotionState::Unavailable | PredictionMotionState::Stationary => {
            return 0.0;
        }
        PredictionMotionState::Continuous => {
            let threshold = PREDICTION_FULL_STRENGTH_CONFIDENCE - 0.15 * trend_consistency;
            (threshold.max(PREDICTION_STABLE_MEAN_CONFIDENCE), 1.0)
        }
        PredictionMotionState::Mean => {
            let threshold = if trend_consistency >= 0.70 {
                PREDICTION_STABLE_MEAN_CONFIDENCE
            } else {
                PREDICTION_FULL_STRENGTH_CONFIDENCE
            };
            (threshold, 1.0)
        }
        PredictionMotionState::AbruptStopOrReverse => (PREDICTION_REACTIVE_CONFIDENCE, 0.80),
        PredictionMotionState::AlternatingPeek => (
            PREDICTION_FULL_STRENGTH_CONFIDENCE,
            PREDICTION_PEEK_MAX_STRENGTH,
        ),
    };
    if confidence >= open_at {
        return max_strength;
    }
    let ratio = (confidence / open_at.max(1e-9)).clamp(0.0, 1.0);
    max_strength * ratio * ratio
}

fn lerp(start: f64, end: f64, weight: f64) -> f64 {
    start + (end - start) * weight
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PositionSample {
    aim_x: f64,
    aim_y: f64,
    capture_ts_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VectorVelocityEstimate {
    raw_velocities: [Vector2; 3],
    mean_velocity: Vector2,
    medoid_velocity: Vector2,
    velocity: Vector2,
    motion_state: PredictionMotionState,
    trend_consistency: f64,
    acceleration_px_ms2: Vector2,
    speed_spread: f64,
    motion_confidence: f64,
    measurement_dt_ms: f64,
    reference_dt_ms: f64,
}

#[derive(Clone, Debug)]
struct RobustAimVelocityEstimator {
    config: SingleTargetPredictionConfig,
    target_id: Option<u64>,
    samples: VecDeque<PositionSample>,
    complete_window_updates: u64,
}

impl RobustAimVelocityEstimator {
    fn new(config: SingleTargetPredictionConfig) -> Self {
        Self {
            config,
            target_id: None,
            samples: VecDeque::with_capacity(VELOCITY_POSITION_COUNT),
            complete_window_updates: 0,
        }
    }

    fn set_config(&mut self, config: SingleTargetPredictionConfig) {
        self.config = config;
    }

    fn history_position_count(&self) -> usize {
        self.samples.len()
    }

    fn reset(&mut self, target_id: Option<u64>) {
        self.target_id = target_id;
        self.samples.clear();
        self.complete_window_updates = 0;
    }

    fn update(
        &mut self,
        target_id: u64,
        aim_x: f64,
        aim_y: f64,
        capture_ts_ns: u64,
        detection_confidence: f64,
        track_confidence: f64,
    ) -> Option<VectorVelocityEstimate> {
        if !aim_x.is_finite() || !aim_y.is_finite() || capture_ts_ns == 0 {
            return None;
        }
        if self.target_id != Some(target_id) {
            self.reset(Some(target_id));
        }
        if let Some(previous) = self.samples.back() {
            if capture_ts_ns <= previous.capture_ts_ns {
                self.reset(Some(target_id));
                return None;
            }
            let dt_ms = (capture_ts_ns - previous.capture_ts_ns) as f64 / 1_000_000.0;
            if dt_ms > self.config.history_reset_gap_ms {
                self.reset(Some(target_id));
            }
        }
        if self.samples.len() == VELOCITY_POSITION_COUNT {
            self.samples.pop_front();
        }
        self.samples.push_back(PositionSample {
            aim_x,
            aim_y,
            capture_ts_ns,
        });
        if self.samples.len() < VELOCITY_POSITION_COUNT {
            return None;
        }

        let points = [
            self.samples[0],
            self.samples[1],
            self.samples[2],
            self.samples[3],
        ];
        let mut velocities = [Vector2::zero(); 3];
        let mut displacements = [Vector2::zero(); 3];
        let mut intervals_ms = [0.0; 3];
        for (index, pair) in points.windows(2).enumerate() {
            let dt_ms = (pair[1].capture_ts_ns - pair[0].capture_ts_ns) as f64 / 1_000_000.0;
            if dt_ms <= 0.0 {
                self.reset(Some(target_id));
                return None;
            }
            let displacement =
                Vector2::new(pair[1].aim_x - pair[0].aim_x, pair[1].aim_y - pair[0].aim_y);
            displacements[index] = displacement;
            velocities[index] = displacement.scale(1.0 / dt_ms);
            intervals_ms[index] = dt_ms;
        }
        let medoid_velocity = medoid_vector(velocities);
        let mean_velocity = mean_vector(velocities);
        let mean_speed = mean_three(velocities.map(Vector2::magnitude));
        let speed_spread =
            mean_three(velocities.map(|velocity| (velocity.magnitude() - mean_speed).abs()));
        let latest_dt_ms = intervals_ms[2];
        let window_dt_ms = intervals_ms.iter().sum::<f64>();
        let reference_dt_ms = window_dt_ms / 3.0;
        let straightness = trajectory_straightness(displacements);
        let direction_consistency =
            direction_consistency(velocities, self.config.spread_base_px_ms);
        let speed_stability = speed_stability(
            speed_spread,
            mean_speed,
            self.config.spread_base_px_ms,
            self.config.spread_relative,
        );
        let motion_consistency =
            (straightness.min(direction_consistency) * speed_stability).clamp(0.0, 1.0);
        let motion_state = diagnostic_motion_state(
            medoid_velocity,
            mean_speed,
            motion_consistency,
            self.config.spread_base_px_ms,
        );

        self.complete_window_updates += 1;
        let history_quality = (self.complete_window_updates as f64 / 2.0).min(1.0);
        let gate_open = detection_confidence.clamp(0.0, 1.0) > f64::EPSILON
            && track_confidence.clamp(0.0, 1.0) > f64::EPSILON;
        let motion_confidence = if gate_open {
            history_quality * motion_consistency
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        let acceleration_px_ms2 = mean_acceleration(velocities, intervals_ms);

        Some(VectorVelocityEstimate {
            raw_velocities: velocities,
            mean_velocity,
            medoid_velocity,
            velocity: medoid_velocity,
            motion_state,
            trend_consistency: direction_consistency,
            acceleration_px_ms2,
            speed_spread,
            motion_confidence,
            measurement_dt_ms: latest_dt_ms,
            reference_dt_ms,
        })
    }
}

fn axis_prediction(
    estimate: VectorVelocityEstimate,
    axis: Axis,
    horizon_ms: f64,
    raw_offset: Vector2,
    weighted_offset: Vector2,
    safe_offset: Vector2,
    allowed_cap: f64,
    allowed: bool,
) -> AxisPrediction {
    AxisPrediction {
        velocity: axis.component(estimate.velocity),
        motion_state: estimate.motion_state,
        trend_consistency: estimate.trend_consistency,
        acceleration_px_ms2: axis.component(estimate.acceleration_px_ms2),
        motion_confidence: estimate.motion_confidence,
        velocity_samples: estimate
            .raw_velocities
            .map(|velocity| Some(axis.component(velocity))),
        mean_velocity: Some(axis.component(estimate.mean_velocity)),
        median_velocity: Some(axis.component(estimate.medoid_velocity)),
        velocity_spread: Some(estimate.speed_spread),
        measurement_dt_ms: Some(estimate.measurement_dt_ms),
        reference_dt_ms: estimate.reference_dt_ms,
        horizon_ms,
        raw_offset: axis.component(raw_offset),
        weighted_offset: axis.component(weighted_offset),
        allowed_cap,
        safe_offset: axis.component(safe_offset),
        allowed,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Axis {
    X,
    Y,
}

impl Axis {
    fn component(self, value: Vector2) -> f64 {
        match self {
            Axis::X => value.x,
            Axis::Y => value.y,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Vector2 {
    x: f64,
    y: f64,
}

impl Vector2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    fn scale(self, scalar: f64) -> Self {
        Self::new(self.x * scalar, self.y * scalar)
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn magnitude(self) -> f64 {
        self.x.hypot(self.y)
    }

    fn distance(self, other: Self) -> f64 {
        self.sub(other).magnitude()
    }
}

fn mean_acceleration(velocities: [Vector2; 3], intervals_ms: [f64; 3]) -> Vector2 {
    let dt_1 = ((intervals_ms[0] + intervals_ms[1]) * 0.5).max(1e-9);
    let dt_2 = ((intervals_ms[1] + intervals_ms[2]) * 0.5).max(1e-9);
    let acceleration_1 = velocities[1].sub(velocities[0]).scale(1.0 / dt_1);
    let acceleration_2 = velocities[2].sub(velocities[1]).scale(1.0 / dt_2);
    acceleration_1.add(acceleration_2).scale(0.5)
}

fn trajectory_straightness(displacements: [Vector2; 3]) -> f64 {
    let path_length = displacements
        .into_iter()
        .map(Vector2::magnitude)
        .sum::<f64>();
    if path_length <= f64::EPSILON {
        return 1.0;
    }
    let net = displacements
        .into_iter()
        .fold(Vector2::zero(), Vector2::add)
        .magnitude();
    (net / path_length).clamp(0.0, 1.0)
}

fn direction_consistency(velocities: [Vector2; 3], base_deadband: f64) -> f64 {
    let deadband = (base_deadband.abs() * 0.25).max(1e-9);
    let pairs = [(0, 1), (1, 2), (0, 2)];
    let mut total = 0.0;
    let mut count = 0_u32;
    for (left, right) in pairs {
        let left_speed = velocities[left].magnitude();
        let right_speed = velocities[right].magnitude();
        if left_speed <= deadband || right_speed <= deadband {
            continue;
        }
        let cosine =
            (velocities[left].dot(velocities[right]) / (left_speed * right_speed)).clamp(-1.0, 1.0);
        total += cosine.max(0.0);
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    (total / f64::from(count)).clamp(0.0, 1.0)
}

fn speed_stability(
    speed_spread: f64,
    mean_speed: f64,
    spread_base_px_ms: f64,
    spread_relative: f64,
) -> f64 {
    let spread_scale = spread_base_px_ms + spread_relative * mean_speed;
    (1.0 / (1.0 + speed_spread / spread_scale.max(1e-9))).clamp(0.0, 1.0)
}

fn diagnostic_motion_state(
    velocity: Vector2,
    mean_speed: f64,
    motion_consistency: f64,
    base_deadband: f64,
) -> PredictionMotionState {
    let deadband = (base_deadband.abs() * 0.25).max(1e-9);
    if mean_speed <= deadband && velocity.magnitude() <= deadband {
        PredictionMotionState::Stationary
    } else if motion_consistency >= 0.75 {
        PredictionMotionState::Continuous
    } else {
        PredictionMotionState::Mean
    }
}

fn mean_three(values: [f64; 3]) -> f64 {
    values.into_iter().sum::<f64>() / 3.0
}

fn mean_vector(values: [Vector2; 3]) -> Vector2 {
    values
        .into_iter()
        .fold(Vector2::zero(), Vector2::add)
        .scale(1.0 / 3.0)
}

fn medoid_vector(values: [Vector2; 3]) -> Vector2 {
    let distances = values.map(|candidate| {
        values
            .into_iter()
            .map(|value| candidate.distance(value))
            .sum::<f64>()
    });
    if distances[0] <= distances[1] && distances[0] <= distances[2] {
        values[0]
    } else if distances[1] <= distances[2] {
        values[1]
    } else {
        values[2]
    }
}

fn clamp_vector_magnitude(value: Vector2, cap: f64) -> Vector2 {
    let cap = cap.max(0.0);
    let magnitude = value.magnitude();
    if magnitude <= cap || magnitude <= f64::EPSILON {
        return value;
    }
    value.scale(cap / magnitude)
}

#[cfg(test)]
mod tests {
    use super::{FocusTargetObservation, SingleTargetPredictionConfig, SingleTargetPredictor};

    fn config() -> SingleTargetPredictionConfig {
        SingleTargetPredictionConfig {
            enabled: true,
            history_reset_gap_ms: 80.0,
            spread_base_px_ms: 0.12,
            spread_relative: 0.50,
            actuation_delay_ms: 4.0,
            lead_ms: 10.0,
            far_absolute_cap_px: 10.0,
            far_base_cap_px: 1.25,
            far_relative_cap: 0.30,
            near_absolute_cap_px: 3.0,
            near_base_cap_px: 0.75,
            near_relative_cap: 0.20,
        }
    }

    fn observation_at(elapsed_ms: u64, aim_x: f64, aim_y: f64) -> FocusTargetObservation {
        FocusTargetObservation {
            track_id: 1,
            aim_x,
            aim_y,
            measured_error_x: aim_x - 100.0,
            measured_error_y: aim_y - 100.0,
            capture_ts_ns: 1_000_000_000 + elapsed_ms * 1_000_000,
            observation_age_ms: 8.0,
            detection_confidence: 1.0,
            identity_confidence: 1.0,
            far_weight: 1.0,
        }
    }

    fn high_cap_config() -> SingleTargetPredictionConfig {
        SingleTargetPredictionConfig {
            far_absolute_cap_px: 100.0,
            far_base_cap_px: 100.0,
            near_absolute_cap_px: 100.0,
            near_base_cap_px: 100.0,
            ..config()
        }
    }

    fn feed_points(
        cfg: SingleTargetPredictionConfig,
        points: &[(u64, f64, f64)],
    ) -> super::SingleTargetPrediction {
        let mut predictor = SingleTargetPredictor::new(cfg);
        let mut prediction = Default::default();
        for &(elapsed_ms, aim_x, aim_y) in points {
            prediction = predictor.predict(observation_at(elapsed_ms, aim_x, aim_y));
        }
        prediction
    }

    #[test]
    fn prediction_waits_for_four_points_on_one_track() {
        let prediction = feed_points(
            config(),
            &[(0, 100.0, 100.0), (10, 104.0, 102.0), (20, 108.0, 104.0)],
        );

        assert_eq!(prediction.history_position_count, 3);
        assert!(!prediction.x.allowed);
        assert!(!prediction.y.allowed);
        assert_eq!(prediction.x.safe_offset, 0.0);
        assert_eq!(prediction.y.safe_offset, 0.0);
    }

    #[test]
    fn stable_motion_uses_real_capture_intervals_and_time_horizon() {
        let prediction = feed_points(
            high_cap_config(),
            &[
                (0, 100.0, 100.0),
                (8, 104.0, 98.0),
                (20, 110.0, 95.0),
                (29, 114.5, 92.75),
                (41, 120.5, 89.75),
            ],
        );

        assert_eq!(prediction.history_position_count, 4);
        assert_eq!(prediction.x.velocity_samples, [Some(0.5); 3]);
        assert_eq!(prediction.y.velocity_samples, [Some(-0.25); 3]);
        assert!((prediction.x.velocity - 0.5).abs() < 1e-12);
        assert!((prediction.y.velocity + 0.25).abs() < 1e-12);
        assert!((prediction.x.reference_dt_ms - 11.0).abs() < 1e-12);
        assert!((prediction.x.horizon_ms - 22.0).abs() < 1e-12);
        assert!((prediction.x.raw_offset - 11.0).abs() < 1e-12);
        assert!((prediction.y.raw_offset + 5.5).abs() < 1e-12);
        assert!(prediction.x.motion_confidence > 0.99);
        assert!((prediction.x.weighted_offset - prediction.x.raw_offset).abs() < 1e-12);
        assert!((prediction.y.safe_offset - prediction.y.raw_offset).abs() < 1e-12);
    }

    #[test]
    fn medoid_velocity_rejects_single_segment_outlier_as_a_vector() {
        let prediction = feed_points(
            high_cap_config(),
            &[
                (0, 100.0, 100.0),
                (10, 110.0, 103.0),
                (20, 160.0, 63.0),
                (30, 170.0, 66.0),
            ],
        );

        assert!((prediction.x.velocity - 1.0).abs() < 1e-12);
        assert!((prediction.y.velocity - 0.3).abs() < 1e-12);
        assert!((prediction.x.mean_velocity.expect("mean") - (7.0 / 3.0)).abs() < 1e-12);
        assert!((prediction.y.mean_velocity.expect("mean") + (3.4 / 3.0)).abs() < 1e-12);
        assert!((prediction.x.median_velocity.expect("medoid") - 1.0).abs() < 1e-12);
        assert!((prediction.y.median_velocity.expect("medoid") - 0.3).abs() < 1e-12);
        assert!(prediction.x.motion_confidence < 0.25);
    }

    #[test]
    fn vector_cap_preserves_prediction_direction() {
        let cfg = SingleTargetPredictionConfig {
            far_absolute_cap_px: 5.0,
            far_base_cap_px: 5.0,
            far_relative_cap: 0.0,
            near_absolute_cap_px: 5.0,
            near_base_cap_px: 5.0,
            near_relative_cap: 0.0,
            ..config()
        };
        let prediction = feed_points(
            cfg,
            &[
                (0, 100.0, 100.0),
                (10, 110.0, 105.0),
                (20, 120.0, 110.0),
                (30, 130.0, 115.0),
                (40, 140.0, 120.0),
            ],
        );

        let safe_magnitude = prediction.x.safe_offset.hypot(prediction.y.safe_offset);
        assert!((prediction.x.allowed_cap - 5.0).abs() < 1e-12);
        assert!((prediction.y.allowed_cap - 5.0).abs() < 1e-12);
        assert!((safe_magnitude - 5.0).abs() < 1e-12);
        assert!((prediction.x.safe_offset / prediction.y.safe_offset - 2.0).abs() < 1e-12);
    }

    #[test]
    fn identity_gate_suppresses_offsets_without_hiding_velocity() {
        let mut predictor = SingleTargetPredictor::new(high_cap_config());
        let mut prediction = Default::default();
        for (elapsed_ms, aim_x, aim_y) in [
            (0, 100.0, 100.0),
            (10, 104.0, 102.0),
            (20, 108.0, 104.0),
            (30, 112.0, 106.0),
            (40, 116.0, 108.0),
        ] {
            let mut sample = observation_at(elapsed_ms, aim_x, aim_y);
            sample.identity_confidence = 0.0;
            prediction = predictor.predict(sample);
        }

        assert!((prediction.x.velocity - 0.4).abs() < 1e-12);
        assert!((prediction.y.velocity - 0.2).abs() < 1e-12);
        assert_eq!(prediction.x.motion_confidence, 0.0);
        assert_eq!(prediction.x.weighted_offset, 0.0);
        assert_eq!(prediction.y.safe_offset, 0.0);
    }

    #[test]
    fn zero_extra_lead_still_compensates_observation_and_actuation_age() {
        let mut zero_lead = config();
        zero_lead.lead_ms = 0.0;
        zero_lead.far_absolute_cap_px = 100.0;
        zero_lead.far_base_cap_px = 100.0;
        zero_lead.near_absolute_cap_px = 100.0;
        zero_lead.near_base_cap_px = 100.0;
        let prediction = feed_points(
            zero_lead,
            &[
                (0, 100.0, 100.0),
                (10, 102.0, 101.0),
                (20, 104.0, 102.0),
                (30, 106.0, 103.0),
                (40, 108.0, 104.0),
            ],
        );

        assert!(prediction.x.allowed);
        assert!((prediction.x.horizon_ms - 12.0).abs() < 1e-12);
        assert!((prediction.x.raw_offset - 2.4).abs() < 1e-12);
        assert!(prediction.x.safe_offset > 0.0);
    }
}

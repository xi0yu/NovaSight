//! Single-focus-target motion prediction.
//!
//! Tracking may retain multiple identities for association, but this module
//! accepts observations for only the selected `track_id`. It owns the bounded
//! velocity history and returns one prediction result for the controller.

use std::collections::VecDeque;

const VELOCITY_POSITION_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleTargetPredictionConfig {
    pub enabled: bool,
    pub smoothing_frames: f64,
    pub history_reset_gap_ms: f64,
    pub spread_base_px_ms: f64,
    pub spread_relative: f64,
    pub change_base_px_ms: f64,
    pub change_relative: f64,
    /// Command-to-visible-response delay included in the capture-to-actuation
    /// prediction horizon before the optional frame lead.
    pub actuation_delay_ms: f64,
    pub lead_frames: f64,
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
        self.smoothing_frames.is_finite()
            && self.smoothing_frames > 0.0
            && self.history_reset_gap_ms.is_finite()
            && self.history_reset_gap_ms > 0.0
            && self.spread_base_px_ms.is_finite()
            && self.spread_base_px_ms > 0.0
            && self.spread_relative.is_finite()
            && self.spread_relative >= 0.0
            && self.change_base_px_ms.is_finite()
            && self.change_base_px_ms > 0.0
            && self.change_relative.is_finite()
            && self.change_relative >= 0.0
            && self.actuation_delay_ms.is_finite()
            && self.actuation_delay_ms >= 0.0
            && self.lead_frames.is_finite()
            && (0.0..=10.0).contains(&self.lead_frames)
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
    /// must be covered before any configured frame lead is added.
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
    pub motion_confidence: f64,
    pub velocity_samples: [Option<f64>; 3],
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
    pub lead_frames: f64,
}

/// Stateful, bounded predictor for the one target selected upstream.
#[derive(Clone, Debug)]
pub struct SingleTargetPredictor {
    config: SingleTargetPredictionConfig,
    velocity_x: RobustVelocityEstimator,
    velocity_y: RobustVelocityEstimator,
}

impl SingleTargetPredictor {
    pub fn new(config: SingleTargetPredictionConfig) -> Self {
        Self {
            velocity_x: RobustVelocityEstimator::new(config),
            velocity_y: RobustVelocityEstimator::new(config),
            config,
        }
    }

    pub fn config_valid(&self) -> bool {
        self.config.is_valid()
    }

    pub fn reset(&mut self, track_id: Option<u64>) {
        self.velocity_x.reset(track_id);
        self.velocity_y.reset(track_id);
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
        SingleTargetPrediction {
            x: AxisPrediction {
                allowed_cap: self.allowed_cap(measured_error_x, far_weight),
                ..AxisPrediction::default()
            },
            y: AxisPrediction {
                allowed_cap: self.allowed_cap(measured_error_y, far_weight),
                ..AxisPrediction::default()
            },
            history_position_count: self.velocity_x.history_position_count(),
            actuation_delay_ms: self.config.actuation_delay_ms,
            lead_frames: self.config.lead_frames,
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

        let estimate_x = self.velocity_x.update(
            observation.track_id,
            observation.aim_x,
            observation.capture_ts_ns,
            observation.detection_confidence,
            observation.identity_confidence,
        );
        let estimate_y = self.velocity_y.update(
            observation.track_id,
            observation.aim_y,
            observation.capture_ts_ns,
            observation.detection_confidence,
            observation.identity_confidence,
        );
        let (Some(estimate_x), Some(estimate_y)) = (estimate_x, estimate_y) else {
            return self.unavailable(
                observation.measured_error_x,
                observation.measured_error_y,
                observation.far_weight,
            );
        };
        debug_assert_eq!(
            self.velocity_x.history_position_count(),
            self.velocity_y.history_position_count()
        );

        SingleTargetPrediction {
            x: self.axis_prediction(
                estimate_x,
                observation.measured_error_x,
                observation.far_weight,
                observation.observation_age_ms,
            ),
            y: self.axis_prediction(
                estimate_y,
                observation.measured_error_y,
                observation.far_weight,
                observation.observation_age_ms,
            ),
            history_position_count: self.velocity_x.history_position_count(),
            actuation_delay_ms: self.config.actuation_delay_ms,
            lead_frames: self.config.lead_frames,
        }
    }

    fn axis_prediction(
        &self,
        estimate: VelocityEstimate,
        measured_error: f64,
        far_weight: f64,
        observation_age_ms: f64,
    ) -> AxisPrediction {
        let allowed = estimate.reference_dt_ms.is_finite() && estimate.reference_dt_ms > 0.0;
        let confidence = if allowed {
            estimate.motion_confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let horizon_ms = if allowed {
            observation_age_ms
                + self.config.actuation_delay_ms
                + estimate.reference_dt_ms.max(0.0) * self.config.lead_frames
        } else {
            0.0
        };
        let raw_offset = estimate.filtered_velocity * horizon_ms;
        let allowed_cap = self.allowed_cap(measured_error, far_weight);
        let weighted_offset = raw_offset * confidence;
        AxisPrediction {
            velocity: estimate.filtered_velocity,
            motion_confidence: estimate.motion_confidence,
            velocity_samples: estimate.raw_velocities.map(Some),
            median_velocity: Some(estimate.median_velocity),
            velocity_spread: Some(estimate.spread),
            measurement_dt_ms: Some(estimate.measurement_dt_ms),
            reference_dt_ms: estimate.reference_dt_ms,
            horizon_ms,
            raw_offset,
            weighted_offset,
            allowed_cap,
            safe_offset: weighted_offset.clamp(-allowed_cap, allowed_cap),
            allowed,
        }
    }

    fn allowed_cap(&self, measured_error_x: f64, far_weight: f64) -> f64 {
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
        absolute_cap.min(base_cap + relative_cap * measured_error_x.abs())
    }
}

fn lerp(start: f64, end: f64, weight: f64) -> f64 {
    start + (end - start) * weight
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PositionSample {
    aim_x: f64,
    capture_ts_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VelocityEstimate {
    raw_velocities: [f64; 3],
    median_velocity: f64,
    filtered_velocity: f64,
    spread: f64,
    motion_confidence: f64,
    measurement_dt_ms: f64,
    reference_dt_ms: f64,
}

#[derive(Clone, Debug)]
struct RobustVelocityEstimator {
    config: SingleTargetPredictionConfig,
    target_id: Option<u64>,
    samples: VecDeque<PositionSample>,
    filtered_velocity: f64,
    initialized_velocity: bool,
    complete_window_updates: u64,
}

impl RobustVelocityEstimator {
    fn new(config: SingleTargetPredictionConfig) -> Self {
        Self {
            config,
            target_id: None,
            samples: VecDeque::with_capacity(VELOCITY_POSITION_COUNT),
            filtered_velocity: 0.0,
            initialized_velocity: false,
            complete_window_updates: 0,
        }
    }

    fn history_position_count(&self) -> usize {
        self.samples.len()
    }

    fn reset(&mut self, target_id: Option<u64>) {
        self.target_id = target_id;
        self.samples.clear();
        self.filtered_velocity = 0.0;
        self.initialized_velocity = false;
        self.complete_window_updates = 0;
    }

    fn update(
        &mut self,
        target_id: u64,
        aim_x: f64,
        capture_ts_ns: u64,
        detection_confidence: f64,
        track_confidence: f64,
    ) -> Option<VelocityEstimate> {
        if !aim_x.is_finite() || capture_ts_ns == 0 {
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
        let mut velocities = [0.0; 3];
        let mut intervals_ms = [0.0; 3];
        for (index, pair) in points.windows(2).enumerate() {
            let dt_ms = (pair[1].capture_ts_ns - pair[0].capture_ts_ns) as f64 / 1_000_000.0;
            if dt_ms <= 0.0 {
                self.reset(Some(target_id));
                return None;
            }
            velocities[index] = (pair[1].aim_x - pair[0].aim_x) / dt_ms;
            intervals_ms[index] = dt_ms;
        }
        let median_velocity = median_three(velocities);
        let spread = median_three(velocities.map(|value| (value - median_velocity).abs()));
        let latest_dt_ms = intervals_ms[2];
        let window_dt_ms = intervals_ms.iter().sum::<f64>();
        let reference_dt_ms = window_dt_ms / 3.0;
        let window_velocity = (points[3].aim_x - points[0].aim_x) / window_dt_ms;
        let previous_filtered = if self.initialized_velocity {
            self.filtered_velocity
        } else {
            median_velocity
        };
        if self.initialized_velocity {
            let smoothing_window_ms = (reference_dt_ms * self.config.smoothing_frames).max(1e-9);
            let alpha = 1.0 - (-latest_dt_ms / smoothing_window_ms).exp();
            self.filtered_velocity = previous_filtered * (1.0 - alpha) + median_velocity * alpha;
        } else {
            self.filtered_velocity = median_velocity;
            self.initialized_velocity = true;
        }
        self.complete_window_updates += 1;

        let history_quality = (self.complete_window_updates as f64 / 2.0).min(1.0);
        let spread_scale =
            self.config.spread_base_px_ms + self.config.spread_relative * median_velocity.abs();
        let spread_quality = 1.0 / (1.0 + spread / spread_scale.max(1e-9));
        let trend_delta = (median_velocity - previous_filtered).abs();
        let trend_scale =
            self.config.change_base_px_ms + self.config.change_relative * previous_filtered.abs();
        let trend_quality = 1.0 / (1.0 + trend_delta / trend_scale.max(1e-9));
        // Adjacent slopes may claim motion when detector jitter alternates
        // around a stationary point. The net displacement over the same
        // bounded window must corroborate that direction before it can
        // contribute prediction confidence. This also suppresses stale EMA
        // velocity after the current robust estimate reaches zero or reverses.
        let median_speed = median_velocity.abs();
        let window_speed = window_velocity.abs();
        let filtered_speed = self.filtered_velocity.abs();
        let displacement_quality = if median_speed <= f64::EPSILON
            || window_speed <= f64::EPSILON
            || filtered_speed <= f64::EPSILON
            || median_velocity.signum() != window_velocity.signum()
            || median_velocity.signum() != self.filtered_velocity.signum()
        {
            0.0
        } else {
            let window_agreement = median_speed.min(window_speed) / median_speed.max(window_speed);
            let current_velocity_support = (median_speed / filtered_speed).min(1.0);
            window_agreement * current_velocity_support
        };
        let detection_quality = detection_confidence.clamp(0.0, 1.0);
        let track_quality = track_confidence.clamp(0.0, 1.0);
        let motion_confidence = (history_quality
            * spread_quality
            * trend_quality
            * displacement_quality
            * detection_quality
            * track_quality)
            .clamp(0.0, 1.0);

        Some(VelocityEstimate {
            raw_velocities: velocities,
            median_velocity,
            filtered_velocity: self.filtered_velocity,
            spread,
            motion_confidence,
            measurement_dt_ms: latest_dt_ms,
            reference_dt_ms,
        })
    }
}

fn median_three(mut values: [f64; 3]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[1]
}

#[cfg(test)]
mod tests {
    use super::{FocusTargetObservation, SingleTargetPredictionConfig, SingleTargetPredictor};

    fn config() -> SingleTargetPredictionConfig {
        SingleTargetPredictionConfig {
            enabled: true,
            smoothing_frames: 3.0,
            history_reset_gap_ms: 80.0,
            spread_base_px_ms: 0.12,
            spread_relative: 0.50,
            change_base_px_ms: 0.20,
            change_relative: 0.75,
            actuation_delay_ms: 4.0,
            lead_frames: 1.0,
            far_absolute_cap_px: 10.0,
            far_base_cap_px: 1.25,
            far_relative_cap: 0.30,
            near_absolute_cap_px: 3.0,
            near_base_cap_px: 0.75,
            near_relative_cap: 0.20,
        }
    }

    fn observation(index: u64, aim_x: f64, aim_y: f64) -> FocusTargetObservation {
        FocusTargetObservation {
            track_id: 1,
            aim_x,
            aim_y,
            measured_error_x: aim_x - 100.0,
            measured_error_y: aim_y - 100.0,
            capture_ts_ns: 1_000_000_000 + index * 10_000_000,
            observation_age_ms: 8.0,
            detection_confidence: 1.0,
            identity_confidence: 1.0,
            far_weight: 1.0,
        }
    }

    #[test]
    fn robust_velocity_rejects_a_single_segment_outlier() {
        let mut predictor = SingleTargetPredictor::new(config());
        let mut prediction = Default::default();
        for (index, position) in [100.0, 102.0, 160.0, 106.0].into_iter().enumerate() {
            prediction = predictor.predict(observation(
                index as u64,
                position,
                200.0 + index as f64 * 3.0,
            ));
        }
        assert_eq!(
            prediction.x.velocity_samples,
            [Some(0.2), Some(5.8), Some(-5.4)]
        );
        assert!((prediction.x.median_velocity.expect("median") - 0.2).abs() < 1e-12);
        assert!((prediction.x.velocity - 0.2).abs() < 1e-12);
        assert!((prediction.x.velocity_spread.expect("spread") - 5.6).abs() < 1e-12);
        assert!(prediction.x.motion_confidence < 0.05);
        assert!((prediction.y.velocity - 0.3).abs() < 1e-12);
        assert!(prediction.y.safe_offset > 0.0);
    }

    #[test]
    fn robust_velocity_uses_real_capture_intervals() {
        let mut predictor = SingleTargetPredictor::new(config());
        let mut prediction = Default::default();
        for elapsed_ms in [0_u64, 8, 20, 29] {
            let mut sample = observation(
                0,
                100.0 + 0.5 * elapsed_ms as f64,
                100.0 - 0.25 * elapsed_ms as f64,
            );
            sample.capture_ts_ns = 1_000_000_000 + elapsed_ms * 1_000_000;
            prediction = predictor.predict(sample);
        }
        assert_eq!(prediction.x.velocity_samples, [Some(0.5); 3]);
        assert_eq!(prediction.y.velocity_samples, [Some(-0.25); 3]);
        assert!((prediction.x.reference_dt_ms - 29.0 / 3.0).abs() < 1e-12);
        assert!((prediction.y.reference_dt_ms - 29.0 / 3.0).abs() < 1e-12);
        assert!((prediction.x.horizon_ms - (8.0 + 4.0 + 29.0 / 3.0)).abs() < 1e-12);
        assert!((prediction.y.horizon_ms - (8.0 + 4.0 + 29.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn zero_extra_lead_still_compensates_observation_and_actuation_age() {
        let mut zero_lead = config();
        zero_lead.lead_frames = 0.0;
        let mut predictor = SingleTargetPredictor::new(zero_lead);
        let mut prediction = Default::default();
        for index in 0..4 {
            prediction = predictor.predict(observation(
                index,
                100.0 + index as f64 * 2.0,
                100.0 + index as f64,
            ));
        }

        assert!(prediction.x.allowed);
        assert!((prediction.x.horizon_ms - 12.0).abs() < 1e-12);
        assert!((prediction.x.raw_offset - 2.4).abs() < 1e-12);
        assert!(prediction.x.safe_offset > 0.0);
    }
}

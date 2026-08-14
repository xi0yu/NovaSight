//! Stateful aim algorithm for one selected target trajectory.
//!
//! This module owns sequencing, bounded temporal state, and output policy.
//! Target prediction is delegated to `prediction`, pure projection and
//! nonlinear response math to `control_law`, and integer conversion to
//! `quantizer`.
//!
//! * `dx`/`dy` are integer tracking demand. Recoil composition and the single
//!   fixed X/Y output limit are applied later at the device boundary.
//! * `emit_allowed` is `true` only when the algorithm produced an
//!   emit-eligible decision. Triggers and target validity gate the
//!   state machine; stale or non-monotonic observations are rejected
//!   with a typed `BlockReason`.
//! * `quantizer_residual` reports fractional demand retained only for integer
//!   device conversion. It does not stop tracking or alter the target position.

use serde::{Deserialize, Serialize};

use super::control_law::{AimControlInput, AimControlLaw, AimControlParameters, AxisPair};
use crate::prediction::{
    FocusTargetObservation, PredictionMotionState, SingleTargetPredictionConfig,
    SingleTargetPredictor,
};
use crate::quantizer::DeviceCountQuantizer;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ControlMode {
    Continuous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BlockReason {
    /// Frame timestamp is in the past relative to `control_now_ns`.
    TimestampDomainInvalid,
    /// Frame age exceeded the freshness threshold.
    StaleObservation,
    /// Observation generation is not strictly increasing.
    NonMonotonicObservation,
    /// Target no longer valid (lost track).
    TargetInvalid,
    /// Runtime projection geometry or controller constants are invalid.
    GeometryInvalid,
    /// Trigger not held; no command is emitted this step.
    TriggerInactive,
    /// No device count is actionable at the current position.
    DeadZone,
    /// Demand converted to a count out of signed 32-bit range.
    DemandOutOfRange,
    /// The algorithm produced an emit-eligible decision; no block.
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AimAlgorithmConfig {
    pub freshness_threshold_ms: f64,
    pub projection_fov_x_deg: f64,
    pub projection_counts_per_360: f64,
    pub response_scale: f64,
    pub response_boost: f64,
    pub response_curve_shape: f64,
    pub max_output_x_counts: f64,
    pub max_output_y_counts: f64,
    pub velocity_history_reset_gap_ms: f64,
    pub prediction_enabled: bool,
    /// Command-to-visible-response delay used by target prediction.
    pub prediction_actuation_delay_ms: f64,
    pub prediction_lead_ms: f64,
    pub prediction_cap_px: f64,
    pub source_width: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub observation_width: u32,
    pub observation_height: u32,
}

impl Default for AimAlgorithmConfig {
    fn default() -> Self {
        Self {
            freshness_threshold_ms: 55.0,
            projection_fov_x_deg: 105.0,
            projection_counts_per_360: 9_980.0,
            response_scale: 0.20,
            response_boost: 0.50,
            response_curve_shape: 1.0,
            max_output_x_counts: 127.0,
            max_output_y_counts: 127.0,
            velocity_history_reset_gap_ms: 80.0,
            prediction_enabled: true,
            prediction_actuation_delay_ms: 4.0,
            prediction_lead_ms: 16.0,
            prediction_cap_px: 10.0,
            source_width: 640,
            roi_width: 640,
            roi_height: 640,
            observation_width: 640,
            observation_height: 640,
        }
    }
}

impl AimAlgorithmConfig {
    fn control_parameters(self) -> AimControlParameters {
        AimControlParameters {
            source_width: self.source_width,
            roi_width: self.roi_width,
            roi_height: self.roi_height,
            observation_width: self.observation_width,
            observation_height: self.observation_height,
            projection_fov_x_deg: self.projection_fov_x_deg,
            projection_counts_per_360: self.projection_counts_per_360,
            response_scale: self.response_scale,
            response_boost: self.response_boost,
            response_curve_shape: self.response_curve_shape,
        }
    }

    fn prediction_config(self) -> SingleTargetPredictionConfig {
        SingleTargetPredictionConfig {
            enabled: self.prediction_enabled,
            history_reset_gap_ms: self.velocity_history_reset_gap_ms,
            actuation_delay_ms: self.prediction_actuation_delay_ms,
            lead_ms: self.prediction_lead_ms,
            cap_px: self.prediction_cap_px,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AimSample {
    pub generation: u64,
    pub target_id: u64,
    pub capture_ts_ns: u64,
    pub control_now_ns: u64,
    pub aim_x: f64,
    pub aim_y: f64,
    pub crosshair_x: f64,
    pub crosshair_y: f64,
    pub detection_confidence: f64,
    pub track_confidence: f64,
    pub target_valid: bool,
    pub trigger_active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AimResult {
    pub sample_available: bool,
    pub generation: u64,
    pub target_id: u64,
    pub capture_ts_ns: u64,
    pub control_now_ns: u64,
    pub frame_age_ms: f64,
    pub aim_x: f64,
    pub aim_y: f64,
    pub crosshair_x: f64,
    pub crosshair_y: f64,
    pub observation_width: u32,
    pub observation_height: u32,
    pub trigger_active: bool,
    pub dx: i32,
    pub dy: i32,
    pub emit_allowed: bool,
    pub block_reason: BlockReason,
    pub quantizer_residual_x: f64,
    pub quantizer_residual_y: f64,
    pub mode: ControlMode,
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub motion_state: PredictionMotionState,
    pub motion_state_y: PredictionMotionState,
    pub trend_consistency: f64,
    pub trend_consistency_y: f64,
    pub acceleration_px_ms2: f64,
    pub acceleration_y_px_ms2: f64,
    pub predicted_offset_x: f64,
    pub predicted_offset_y: f64,
    pub motion_confidence: f64,
    pub history_position_count: usize,
    pub velocity_samples: [Option<f64>; 3],
    pub mean_velocity: Option<f64>,
    pub medoid_velocity: Option<f64>,
    pub velocity_spread: Option<f64>,
    pub measurement_dt_ms: Option<f64>,
    pub reference_dt_ms: f64,
    pub prediction_actuation_delay_ms: f64,
    pub prediction_lead_ms: f64,
    pub prediction_horizon_ms: f64,
    pub prediction_raw_offset_x: f64,
    pub prediction_weighted_offset_x: f64,
    pub prediction_allowed_cap_x: f64,
    pub prediction_allowed: bool,
    pub motion_confidence_y: f64,
    pub velocity_samples_y: [Option<f64>; 3],
    pub mean_velocity_y: Option<f64>,
    pub medoid_velocity_y: Option<f64>,
    pub velocity_spread_y: Option<f64>,
    pub measurement_dt_ms_y: Option<f64>,
    pub reference_dt_ms_y: f64,
    pub prediction_raw_offset_y: f64,
    pub prediction_weighted_offset_y: f64,
    pub prediction_allowed_cap_y: f64,
    pub prediction_allowed_y: bool,
    pub observed_error_x: f64,
    pub observed_error_y: f64,
    pub filtered_error_x: f64,
    pub filtered_error_y: f64,
    pub full_error_counts_x: f64,
    pub full_error_counts_y: f64,
    pub float_demand_x: f64,
    pub float_demand_y: f64,
}

impl AimResult {
    pub fn blocked(reason: BlockReason) -> Self {
        Self {
            sample_available: false,
            generation: 0,
            target_id: 0,
            capture_ts_ns: 0,
            control_now_ns: 0,
            frame_age_ms: 0.0,
            aim_x: 0.0,
            aim_y: 0.0,
            crosshair_x: 0.0,
            crosshair_y: 0.0,
            observation_width: 0,
            observation_height: 0,
            trigger_active: false,
            dx: 0,
            dy: 0,
            emit_allowed: false,
            block_reason: reason,
            quantizer_residual_x: 0.0,
            quantizer_residual_y: 0.0,
            mode: ControlMode::Continuous,
            velocity_x: 0.0,
            velocity_y: 0.0,
            motion_state: PredictionMotionState::Unavailable,
            motion_state_y: PredictionMotionState::Unavailable,
            trend_consistency: 0.0,
            trend_consistency_y: 0.0,
            acceleration_px_ms2: 0.0,
            acceleration_y_px_ms2: 0.0,
            predicted_offset_x: 0.0,
            predicted_offset_y: 0.0,
            motion_confidence: 0.0,
            history_position_count: 0,
            velocity_samples: [None; 3],
            mean_velocity: None,
            medoid_velocity: None,
            velocity_spread: None,
            measurement_dt_ms: None,
            reference_dt_ms: 0.0,
            prediction_actuation_delay_ms: 0.0,
            prediction_lead_ms: 0.0,
            prediction_horizon_ms: 0.0,
            prediction_raw_offset_x: 0.0,
            prediction_weighted_offset_x: 0.0,
            prediction_allowed_cap_x: 0.0,
            prediction_allowed: false,
            motion_confidence_y: 0.0,
            velocity_samples_y: [None; 3],
            mean_velocity_y: None,
            medoid_velocity_y: None,
            velocity_spread_y: None,
            measurement_dt_ms_y: None,
            reference_dt_ms_y: 0.0,
            prediction_raw_offset_y: 0.0,
            prediction_weighted_offset_y: 0.0,
            prediction_allowed_cap_y: 0.0,
            prediction_allowed_y: false,
            observed_error_x: 0.0,
            observed_error_y: 0.0,
            filtered_error_x: 0.0,
            filtered_error_y: 0.0,
            full_error_counts_x: 0.0,
            full_error_counts_y: 0.0,
            float_demand_x: 0.0,
            float_demand_y: 0.0,
        }
    }
}

impl Default for AimResult {
    fn default() -> Self {
        Self::blocked(BlockReason::TriggerInactive)
    }
}

/// Deep stateful interface for one target trajectory.
#[derive(Clone, Debug)]
pub struct AimAlgorithm {
    config: AimAlgorithmConfig,
    control_law: Option<AimControlLaw>,
    quantizer: DeviceCountQuantizer,
    last_generation: Option<u64>,
    last_capture_ts_ns: Option<u64>,
    target_id: Option<u64>,
    prediction: SingleTargetPredictor,
}

impl AimAlgorithm {
    pub fn new(config: AimAlgorithmConfig) -> Self {
        Self {
            config,
            control_law: AimControlLaw::new(config.control_parameters()),
            quantizer: DeviceCountQuantizer::new(),
            last_generation: None,
            last_capture_ts_ns: None,
            target_id: None,
            prediction: SingleTargetPredictor::new(config.prediction_config()),
        }
    }

    pub fn set_config(&mut self, config: AimAlgorithmConfig) {
        self.config = config;
        self.control_law = AimControlLaw::new(config.control_parameters());
        self.prediction.set_config(config.prediction_config());
        self.quantizer.reset();
    }

    pub fn reset(&mut self) {
        self.quantizer.reset();
        self.last_generation = None;
        self.last_capture_ts_ns = None;
        self.target_id = None;
        self.prediction.reset(None);
    }

    pub fn release_trigger(&mut self) {
        self.quantizer.reset();
    }

    /// Clear target-relative state while preserving observation sequence
    /// guards. Used when a tracker restores or rebuilds an identity.
    pub fn reset_target_state(&mut self) {
        self.release_trigger();
        self.prediction.reset(None);
        self.target_id = None;
    }

    pub fn step(&mut self, sample: AimSample) -> AimResult {
        self.step_internal(sample)
    }

    fn step_internal(&mut self, observation: AimSample) -> AimResult {
        let frame_age_ns = observation.control_now_ns as i128 - observation.capture_ts_ns as i128;
        if frame_age_ns < 0 {
            self.release_trigger();
            return AimResult::blocked(BlockReason::TimestampDomainInvalid);
        }
        let frame_age_ms = frame_age_ns as f64 / 1_000_000.0;
        if frame_age_ms > self.config.freshness_threshold_ms {
            self.release_trigger();
            return AimResult::blocked(BlockReason::StaleObservation);
        }
        if let Some(prev_gen) = self.last_generation
            && observation.generation <= prev_gen
        {
            self.release_trigger();
            return AimResult::blocked(BlockReason::NonMonotonicObservation);
        }
        let capture_timestamp_discontinuity = self
            .last_capture_ts_ns
            .is_some_and(|prev| observation.capture_ts_ns <= prev);
        if capture_timestamp_discontinuity {
            self.prediction.reset(self.target_id);
        }

        if self
            .target_id
            .is_some_and(|target| target != observation.target_id)
        {
            self.reset_target_state();
        }

        if !observation.target_valid {
            self.last_generation = Some(observation.generation);
            self.last_capture_ts_ns = Some(observation.capture_ts_ns);
            self.target_id = None;
            self.prediction.reset(None);
            self.release_trigger();
            return AimResult::blocked(BlockReason::TargetInvalid);
        }
        if !self.prediction.config_valid() {
            self.release_trigger();
            return AimResult::blocked(BlockReason::GeometryInvalid);
        }
        let aim_x = observation.aim_x;
        let aim_y = observation.aim_y;
        let error_x = aim_x - observation.crosshair_x;
        let error_y = aim_y - observation.crosshair_y;
        let prediction = if capture_timestamp_discontinuity {
            self.prediction.unavailable()
        } else {
            self.prediction.predict(FocusTargetObservation {
                track_id: observation.target_id,
                aim_x,
                aim_y,
                capture_ts_ns: observation.capture_ts_ns,
                observation_age_ms: frame_age_ms,
            })
        };
        let predicted_offset_x = prediction.x.safe_offset;
        let predicted_offset_y = prediction.y.safe_offset;
        let mode = ControlMode::Continuous;
        let Some(control_result) = self.control_law.and_then(|law| {
            law.evaluate(AimControlInput {
                measured_error_px: AxisPair::new(error_x, error_y),
                predicted_offset_px: AxisPair::new(predicted_offset_x, predicted_offset_y),
            })
        }) else {
            self.release_trigger();
            return AimResult::blocked(BlockReason::GeometryInvalid);
        };
        let filtered_error_x = control_result.predicted_error_px.x;
        let filtered_error_y = control_result.predicted_error_px.y;
        let full_x = control_result.projected_error_counts.x;
        let full_y = control_result.projected_error_counts.y;
        let demand_x = control_result.demand_counts.x;
        let demand_y = control_result.demand_counts.y;

        let (dx, dy, block_reason) = if observation.trigger_active {
            let quantized = match self.quantizer.quantize(demand_x, demand_y) {
                Ok(value) => value,
                Err(_) => {
                    self.release_trigger();
                    return AimResult::blocked(BlockReason::DemandOutOfRange);
                }
            };
            let dx = quantized.dx;
            let dy = quantized.dy;
            let reason = if dx == 0 && dy == 0 {
                BlockReason::DeadZone
            } else {
                BlockReason::None
            };
            (dx, dy, reason)
        } else {
            self.release_trigger();
            (0, 0, BlockReason::TriggerInactive)
        };

        self.last_generation = Some(observation.generation);
        self.last_capture_ts_ns = Some(observation.capture_ts_ns);
        self.target_id = Some(observation.target_id);

        let (quantizer_residual_x, quantizer_residual_y) = self.quantizer.residuals();
        AimResult {
            sample_available: true,
            generation: observation.generation,
            target_id: observation.target_id,
            capture_ts_ns: observation.capture_ts_ns,
            control_now_ns: observation.control_now_ns,
            frame_age_ms,
            aim_x: observation.aim_x,
            aim_y: observation.aim_y,
            crosshair_x: observation.crosshair_x,
            crosshair_y: observation.crosshair_y,
            observation_width: self.config.observation_width,
            observation_height: self.config.observation_height,
            trigger_active: observation.trigger_active,
            dx,
            dy,
            emit_allowed: block_reason == BlockReason::None,
            block_reason,
            quantizer_residual_x,
            quantizer_residual_y,
            mode,
            velocity_x: prediction.x.velocity,
            velocity_y: prediction.y.velocity,
            motion_state: prediction.x.motion_state,
            motion_state_y: prediction.y.motion_state,
            trend_consistency: prediction.x.trend_consistency,
            trend_consistency_y: prediction.y.trend_consistency,
            acceleration_px_ms2: prediction.x.acceleration_px_ms2,
            acceleration_y_px_ms2: prediction.y.acceleration_px_ms2,
            predicted_offset_x,
            predicted_offset_y,
            motion_confidence: prediction.x.motion_confidence,
            history_position_count: prediction.history_position_count,
            velocity_samples: prediction.x.velocity_samples,
            mean_velocity: prediction.x.mean_velocity,
            medoid_velocity: prediction.x.medoid_velocity,
            velocity_spread: prediction.x.velocity_spread,
            measurement_dt_ms: prediction.x.measurement_dt_ms,
            reference_dt_ms: prediction.x.reference_dt_ms,
            prediction_actuation_delay_ms: prediction.actuation_delay_ms,
            prediction_lead_ms: prediction.lead_ms,
            prediction_horizon_ms: prediction.x.horizon_ms,
            prediction_raw_offset_x: prediction.x.raw_offset,
            prediction_weighted_offset_x: prediction.x.weighted_offset,
            prediction_allowed_cap_x: prediction.x.allowed_cap,
            prediction_allowed: prediction.x.allowed,
            motion_confidence_y: prediction.y.motion_confidence,
            velocity_samples_y: prediction.y.velocity_samples,
            mean_velocity_y: prediction.y.mean_velocity,
            medoid_velocity_y: prediction.y.medoid_velocity,
            velocity_spread_y: prediction.y.velocity_spread,
            measurement_dt_ms_y: prediction.y.measurement_dt_ms,
            reference_dt_ms_y: prediction.y.reference_dt_ms,
            prediction_raw_offset_y: prediction.y.raw_offset,
            prediction_weighted_offset_y: prediction.y.weighted_offset,
            prediction_allowed_cap_y: prediction.y.allowed_cap,
            prediction_allowed_y: prediction.y.allowed,
            observed_error_x: error_x,
            observed_error_y: error_y,
            filtered_error_x,
            filtered_error_y,
            full_error_counts_x: full_x,
            full_error_counts_y: full_y,
            float_demand_x: demand_x,
            float_demand_y: demand_y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AimAlgorithm, AimAlgorithmConfig, AimSample, BlockReason, ControlMode};
    use crate::prediction::PredictionMotionState;

    #[test]
    fn vector_medoid_prediction_respects_single_cap() {
        let config = AimAlgorithmConfig {
            prediction_enabled: true,
            prediction_lead_ms: 2.0,
            prediction_cap_px: 3.0,
            ..AimAlgorithmConfig::default()
        };
        let mut control = AimAlgorithm::new(config);
        let mut decision = None;
        for (index, (error_x, error_y)) in [(40.0, 20.0), (44.0, 22.0), (48.0, 24.0), (52.0, 26.0)]
            .into_iter()
            .enumerate()
        {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.step(AimSample {
                generation,
                target_id: 7,
                capture_ts_ns,
                control_now_ns: capture_ts_ns + 8_000_000,
                aim_x: 160.0 + error_x,
                aim_y: 160.0 + error_y,
                crosshair_x: 160.0,
                crosshair_y: 160.0,
                detection_confidence: 0.95,
                track_confidence: 0.90,
                target_valid: true,
                trigger_active: true,
            }));
        }
        let decision = decision.expect("last decision");
        assert!(decision.sample_available);
        assert_eq!(decision.history_position_count, 4);
        assert_eq!(decision.velocity_samples, [Some(0.4); 3]);
        assert!((decision.mean_velocity.expect("mean") - 0.4).abs() < 1e-12);
        assert!((decision.medoid_velocity.expect("medoid") - 0.4).abs() < 1e-12);
        assert!((decision.velocity_x - 0.4).abs() < 1e-12);
        assert_eq!(decision.motion_state, PredictionMotionState::Continuous);
        assert!(decision.trend_consistency > 0.99);
        assert_eq!(decision.acceleration_px_ms2, 0.0);
        assert!((decision.reference_dt_ms - 10.0).abs() < 1e-12);
        assert!((decision.prediction_horizon_ms - 14.0).abs() < 1e-12);
        assert!((decision.prediction_raw_offset_x - 5.6).abs() < 1e-12);
        assert!((decision.prediction_weighted_offset_x - 5.6).abs() < 1e-12);
        assert_eq!(
            decision.prediction_allowed_cap_x,
            decision.prediction_allowed_cap_y
        );
        assert!(decision.prediction_allowed_cap_x <= 3.0);
        assert!(decision.prediction_allowed);
        assert!(
            decision
                .predicted_offset_x
                .hypot(decision.predicted_offset_y)
                <= decision.prediction_allowed_cap_x + 1e-12
        );
        assert!((decision.predicted_offset_x / decision.predicted_offset_y - 2.0).abs() < 1e-12);
        assert!((decision.filtered_error_x - (52.0 + decision.predicted_offset_x)).abs() < 1e-12);
        assert!((decision.velocity_y - 0.2).abs() < 1e-12);
        assert!(decision.prediction_allowed_y);
        assert!(decision.predicted_offset_y > 0.0);
        assert!(decision.filtered_error_y > 26.0);
        assert!(decision.full_error_counts_x.is_finite());
        assert!(decision.float_demand_x.is_finite());
    }

    #[test]
    fn admitted_target_confidence_does_not_rescale_prediction() {
        let mut control = AimAlgorithm::new(AimAlgorithmConfig {
            prediction_enabled: true,
            ..AimAlgorithmConfig::default()
        });
        let mut decision = None;
        for (index, error_x) in [40.0, 44.0, 48.0, 52.0].into_iter().enumerate() {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.step(AimSample {
                generation,
                target_id: 7,
                capture_ts_ns,
                control_now_ns: capture_ts_ns + 8_000_000,
                aim_x: 160.0 + error_x,
                aim_y: 160.0,
                crosshair_x: 160.0,
                crosshair_y: 160.0,
                detection_confidence: 0.95,
                track_confidence: 0.0,
                target_valid: true,
                trigger_active: true,
            }));
        }
        let decision = decision.expect("last decision");
        assert!((decision.velocity_x - 0.4).abs() < 1e-12);
        assert_eq!(decision.motion_state, PredictionMotionState::Continuous);
        assert_eq!(decision.motion_confidence, 1.0);
        assert!((decision.predicted_offset_x - 4.8).abs() < 1e-12);
        assert!((decision.filtered_error_x - 56.8).abs() < 1e-12);
    }

    #[test]
    fn prediction_offsets_feed_the_continuous_controller() {
        let mut control = AimAlgorithm::new(AimAlgorithmConfig {
            prediction_enabled: true,
            prediction_lead_ms: 1.0,
            ..AimAlgorithmConfig::default()
        });
        let mut decision = None;
        for (index, error_x) in [5.0, 7.0, 9.0, 11.0].into_iter().enumerate() {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.step(AimSample {
                generation,
                target_id: 7,
                capture_ts_ns,
                control_now_ns: capture_ts_ns + 8_000_000,
                aim_x: 160.0 + error_x,
                aim_y: 160.0,
                crosshair_x: 160.0,
                crosshair_y: 160.0,
                detection_confidence: 1.0,
                track_confidence: 1.0,
                target_valid: true,
                trigger_active: true,
            }));
        }

        let decision = decision.expect("last decision");
        assert!(decision.observed_error_x < 12.0);
        assert!(decision.filtered_error_x > 12.0);
        assert_eq!(decision.mode, ControlMode::Continuous);
    }

    #[test]
    fn trigger_release_clears_output_residual_but_keeps_motion_history_hot() {
        let mut control = AimAlgorithm::new(AimAlgorithmConfig {
            prediction_enabled: true,
            ..AimAlgorithmConfig::default()
        });
        let mut decision = None;
        for (index, error_x) in [40.0, 44.0, 48.0, 52.0].into_iter().enumerate() {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.step(AimSample {
                generation,
                target_id: 7,
                capture_ts_ns,
                control_now_ns: capture_ts_ns + 8_000_000,
                aim_x: 160.0 + error_x,
                aim_y: 160.0,
                crosshair_x: 160.0,
                crosshair_y: 160.0,
                detection_confidence: 1.0,
                track_confidence: 1.0,
                target_valid: true,
                trigger_active: generation == 4,
            }));
        }
        let decision = decision.expect("triggered decision");
        assert!((decision.velocity_x - 0.4).abs() < 1e-12);
        assert!(decision.motion_confidence > 0.0);
        assert!(decision.predicted_offset_x > 0.0);
        assert!(decision.emit_allowed);
    }
}

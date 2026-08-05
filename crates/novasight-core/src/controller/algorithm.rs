//! Stateful aim algorithm for one selected target trajectory.
//!
//! This module owns sequencing, bounded temporal state, and output policy.
//! Target prediction is delegated to `prediction`, pure projection and
//! nonlinear response math to `control_law`, and integer conversion to
//! `limiter`.
//!
//! * `dx`/`dy` are integer mouse counts the device should emit.
//! * `emit_allowed` is `true` only when the algorithm produced an
//!   emit-eligible decision. Triggers and target validity gate the
//!   state machine; stale or non-monotonic observations are rejected
//!   with a typed `BlockReason`.
//! * `quantizer_residual` reports fractional demand retained by the limiter
//!   until it becomes one actionable device count. Once the full correction is within
//!   half a count, the current integer position is already the nearest point
//!   the device can represent and that axis settles instead of limit-cycling.

use serde::{Deserialize, Serialize};

use super::control_law::{AimControlInput, AimControlLaw, AimControlParameters, AxisPair};
use crate::limiter::{DeviceCountLimiter, DeviceCountLimits};
use crate::prediction::{
    FocusTargetObservation, PredictionMotionState, SingleTargetPredictionConfig,
    SingleTargetPredictor,
};

/// Integer mouse output cannot represent a correction smaller than one count.
/// At half a count or less the current integer position is the nearest
/// representable point, so retaining residual would only create a +/-1 cycle.
const HALF_DEVICE_COUNT: f64 = 0.5;
const ARRIVAL_CONFIRM_NS: u64 = 8_000_000;
const ARRIVAL_CONFIRM_SAMPLES: u8 = 2;

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
    /// The target is inside the FOV-projected arrival region.
    AimSettled,
    /// A successful device movement is not yet observable by this frame.
    ActuationFeedbackPending,
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
    pub max_counts_per_update: f64,
    /// Half-size of the per-axis arrival box in projected device counts.
    pub arrival_radius_counts: f64,
    pub velocity_history_reset_gap_ms: f64,
    pub velocity_spread_base_px_ms: f64,
    pub velocity_spread_relative: f64,
    pub prediction_enabled: bool,
    /// Unified command-to-visible-response delay shared with the runtime's
    /// actuation feedback gate.
    pub prediction_actuation_delay_ms: f64,
    pub prediction_lead_ms: f64,
    pub prediction_cap_px: f64,
    pub source_width: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub observation_width: u32,
    pub observation_height: u32,
    /// Cap on the fractional residual retained across emits.
    pub residual_cap: f64,
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
            max_counts_per_update: 127.0,
            arrival_radius_counts: 3.0,
            velocity_history_reset_gap_ms: 80.0,
            velocity_spread_base_px_ms: 0.12,
            velocity_spread_relative: 0.50,
            prediction_enabled: true,
            prediction_actuation_delay_ms: 4.0,
            prediction_lead_ms: 16.0,
            prediction_cap_px: 10.0,
            source_width: 640,
            roi_width: 640,
            roi_height: 640,
            observation_width: 640,
            observation_height: 640,
            residual_cap: 1.0,
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
            max_counts_per_update: self.max_counts_per_update,
        }
    }

    fn prediction_config(self) -> SingleTargetPredictionConfig {
        SingleTargetPredictionConfig {
            enabled: self.prediction_enabled,
            history_reset_gap_ms: self.velocity_history_reset_gap_ms,
            spread_base_px_ms: self.velocity_spread_base_px_ms,
            spread_relative: self.velocity_spread_relative,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AimFeedback {
    pub pending_x: bool,
    pub pending_y: bool,
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
    pub arrival_settled_x: bool,
    pub arrival_settled_y: bool,
    pub arrival_hold_x: bool,
    pub arrival_hold_y: bool,
    pub arrival_enter_counts: f64,
    pub arrival_exit_counts: f64,
    pub actuation_pending_x: bool,
    pub actuation_pending_y: bool,
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
            arrival_settled_x: false,
            arrival_settled_y: false,
            arrival_hold_x: false,
            arrival_hold_y: false,
            arrival_enter_counts: 0.0,
            arrival_exit_counts: 0.0,
            actuation_pending_x: false,
            actuation_pending_y: false,
        }
    }
}

impl Default for AimResult {
    fn default() -> Self {
        Self::blocked(BlockReason::TriggerInactive)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AxisArrivalState {
    settled: bool,
    arrival_since_ns: Option<u64>,
    arrival_samples: u8,
    departure_since_ns: Option<u64>,
    departure_samples: u8,
}

impl AxisArrivalState {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn hold(&mut self, full_error_counts: f64, capture_ts_ns: u64, enter_counts: f64) -> bool {
        let absolute = full_error_counts.abs();
        let exit_counts = (enter_counts * 1.5).max(enter_counts + HALF_DEVICE_COUNT);
        let hard_exit_counts = exit_counts * 2.0;

        if self.settled {
            self.arrival_since_ns = None;
            self.arrival_samples = 0;
            if absolute <= exit_counts {
                self.departure_since_ns = None;
                self.departure_samples = 0;
                return true;
            }
            if absolute >= hard_exit_counts {
                self.reset();
                return false;
            }
            let since = *self.departure_since_ns.get_or_insert(capture_ts_ns);
            self.departure_samples = self.departure_samples.saturating_add(1);
            if self.departure_samples >= ARRIVAL_CONFIRM_SAMPLES
                && capture_ts_ns.saturating_sub(since) >= ARRIVAL_CONFIRM_NS
            {
                self.reset();
                return false;
            }
            return true;
        }

        self.departure_since_ns = None;
        self.departure_samples = 0;
        if absolute <= enter_counts {
            let since = *self.arrival_since_ns.get_or_insert(capture_ts_ns);
            self.arrival_samples = self.arrival_samples.saturating_add(1);
            if self.arrival_samples >= ARRIVAL_CONFIRM_SAMPLES
                && capture_ts_ns.saturating_sub(since) >= ARRIVAL_CONFIRM_NS
            {
                self.settled = true;
            }
            return true;
        }
        self.arrival_since_ns = None;
        self.arrival_samples = 0;
        false
    }
}

/// Deep stateful interface for one target trajectory.
#[derive(Clone, Debug)]
pub struct AimAlgorithm {
    config: AimAlgorithmConfig,
    control_law: Option<AimControlLaw>,
    limiter: DeviceCountLimiter,
    last_generation: Option<u64>,
    last_capture_ts_ns: Option<u64>,
    target_id: Option<u64>,
    previous_error_x: f64,
    previous_error_y: f64,
    measured_error_history_valid: bool,
    prediction: SingleTargetPredictor,
    arrival_x: AxisArrivalState,
    arrival_y: AxisArrivalState,
}

impl AimAlgorithm {
    pub fn new(config: AimAlgorithmConfig) -> Self {
        Self {
            config,
            control_law: AimControlLaw::new(config.control_parameters()),
            limiter: DeviceCountLimiter::new(),
            last_generation: None,
            last_capture_ts_ns: None,
            target_id: None,
            previous_error_x: 0.0,
            previous_error_y: 0.0,
            measured_error_history_valid: false,
            prediction: SingleTargetPredictor::new(config.prediction_config()),
            arrival_x: AxisArrivalState::default(),
            arrival_y: AxisArrivalState::default(),
        }
    }

    pub fn set_config(&mut self, config: AimAlgorithmConfig) {
        self.config = config;
        self.control_law = AimControlLaw::new(config.control_parameters());
        self.prediction.set_config(config.prediction_config());
        self.limiter.reset();
        self.arrival_x.reset();
        self.arrival_y.reset();
    }

    pub fn reset(&mut self) {
        self.limiter.reset();
        self.last_generation = None;
        self.last_capture_ts_ns = None;
        self.target_id = None;
        self.previous_error_x = 0.0;
        self.previous_error_y = 0.0;
        self.measured_error_history_valid = false;
        self.prediction.reset(None);
        self.arrival_x.reset();
        self.arrival_y.reset();
    }

    pub fn release_trigger(&mut self) {
        self.limiter.reset();
        self.arrival_x.reset();
        self.arrival_y.reset();
    }

    /// Clear target-relative state while preserving observation sequence
    /// guards. Used when a tracker restores or rebuilds an identity.
    pub fn reset_target_state(&mut self) {
        self.release_trigger();
        self.prediction.reset(None);
        self.target_id = None;
        self.previous_error_x = 0.0;
        self.previous_error_y = 0.0;
        self.measured_error_history_valid = false;
    }

    pub fn step(&mut self, sample: AimSample) -> AimResult {
        self.step_with_feedback(sample, AimFeedback::default())
    }

    pub fn step_with_feedback(&mut self, sample: AimSample, feedback: AimFeedback) -> AimResult {
        self.step_internal(sample, feedback)
    }

    fn step_internal(&mut self, observation: AimSample, feedback: AimFeedback) -> AimResult {
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
            self.measured_error_history_valid = false;
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
            self.measured_error_history_valid = false;
            self.prediction.reset(None);
            self.release_trigger();
            return AimResult::blocked(BlockReason::TargetInvalid);
        }
        if !self.config.arrival_radius_counts.is_finite()
            || self.config.arrival_radius_counts < HALF_DEVICE_COUNT
            || !self.prediction.config_valid()
        {
            self.release_trigger();
            return AimResult::blocked(BlockReason::GeometryInvalid);
        }
        let aim_x = observation.aim_x;
        let aim_y = observation.aim_y;
        let error_x = aim_x - observation.crosshair_x;
        let error_y = aim_y - observation.crosshair_y;
        if self.measured_error_history_valid {
            if crossed_center(self.previous_error_x, error_x) {
                self.limiter.reset_x();
            }
            if crossed_center(self.previous_error_y, error_y) {
                self.limiter.reset_y();
            }
        }
        let prediction = if capture_timestamp_discontinuity {
            self.prediction.unavailable()
        } else {
            self.prediction.predict(FocusTargetObservation {
                track_id: observation.target_id,
                aim_x,
                aim_y,
                capture_ts_ns: observation.capture_ts_ns,
                observation_age_ms: frame_age_ms,
                detection_confidence: observation.detection_confidence,
                identity_confidence: observation.track_confidence,
            })
        };
        let predicted_offset_x = prediction.x.safe_offset;
        let predicted_offset_y = prediction.y.safe_offset;
        let motion_strength = prediction
            .x
            .motion_confidence
            .max(prediction.y.motion_confidence)
            .clamp(0.0, 1.0);
        let mode = ControlMode::Continuous;
        let Some(control_result) = self.control_law.and_then(|law| {
            law.evaluate(AimControlInput {
                measured_error_px: AxisPair::new(error_x, error_y),
                predicted_offset_px: AxisPair::new(predicted_offset_x, predicted_offset_y),
                motion_strength,
            })
        }) else {
            self.release_trigger();
            return AimResult::blocked(BlockReason::GeometryInvalid);
        };
        let filtered_error_x = control_result.predicted_error_px.x;
        let filtered_error_y = control_result.predicted_error_px.y;
        let full_x = control_result.projected_error_counts.x;
        let full_y = control_result.projected_error_counts.y;
        let mut demand_x = control_result.demand_counts.x;
        let mut demand_y = control_result.demand_counts.y;

        let arrival_enter_counts = self.config.arrival_radius_counts.max(HALF_DEVICE_COUNT);
        let arrival_exit_counts =
            (arrival_enter_counts * 1.5).max(arrival_enter_counts + HALF_DEVICE_COUNT);
        let feedback_hold_x = actuation_feedback_should_hold(
            feedback.pending_x,
            self.measured_error_history_valid,
            self.previous_error_x,
            error_x,
            full_x,
            arrival_enter_counts,
        );
        let feedback_hold_y = actuation_feedback_should_hold(
            feedback.pending_y,
            self.measured_error_history_valid,
            self.previous_error_y,
            error_y,
            full_y,
            arrival_enter_counts,
        );
        let arrival_hold_x = feedback_hold_x
            || self
                .arrival_x
                .hold(full_x, observation.capture_ts_ns, arrival_enter_counts);
        let arrival_hold_y = feedback_hold_y
            || self
                .arrival_y
                .hold(full_y, observation.capture_ts_ns, arrival_enter_counts);

        if arrival_hold_x || full_x.abs() <= HALF_DEVICE_COUNT {
            self.limiter.reset_x();
            demand_x = 0.0;
        }
        if arrival_hold_y || full_y.abs() <= HALF_DEVICE_COUNT {
            self.limiter.reset_y();
            demand_y = 0.0;
        }

        let limits = DeviceCountLimits {
            max_counts_per_axis: control_result.max_counts_per_update,
            residual_cap: self.config.residual_cap,
        };

        let (dx, dy, block_reason) = if observation.trigger_active {
            let limited = match self.limiter.limit(demand_x, demand_y, limits) {
                Ok(value) => value,
                Err(_) => {
                    self.release_trigger();
                    return AimResult::blocked(BlockReason::DemandOutOfRange);
                }
            };
            let dx = limited.dx;
            let dy = limited.dy;
            let reason = if dx == 0 && dy == 0 && (feedback_hold_x || feedback_hold_y) {
                BlockReason::ActuationFeedbackPending
            } else if dx == 0 && dy == 0 && (arrival_hold_x || arrival_hold_y) {
                BlockReason::AimSettled
            } else if dx == 0 && dy == 0 {
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
        self.previous_error_x = error_x;
        self.previous_error_y = error_y;
        self.measured_error_history_valid = true;

        let (quantizer_residual_x, quantizer_residual_y) = self.limiter.residuals();
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
            arrival_settled_x: self.arrival_x.settled,
            arrival_settled_y: self.arrival_y.settled,
            arrival_hold_x,
            arrival_hold_y,
            arrival_enter_counts,
            arrival_exit_counts,
            actuation_pending_x: feedback.pending_x,
            actuation_pending_y: feedback.pending_y,
        }
    }
}

fn actuation_feedback_should_hold(
    pending: bool,
    history_valid: bool,
    previous_error: f64,
    current_error: f64,
    full_counts: f64,
    arrival_enter_counts: f64,
) -> bool {
    if !pending {
        return false;
    }
    if !history_valid || !previous_error.is_finite() || !current_error.is_finite() {
        return true;
    }
    if full_counts.abs() <= arrival_enter_counts.max(HALF_DEVICE_COUNT) * 2.0 {
        return true;
    }
    if crossed_center(previous_error, current_error) {
        return true;
    }
    current_error.abs() <= previous_error.abs() + HALF_DEVICE_COUNT
}

fn crossed_center(previous: f64, current: f64) -> bool {
    previous.is_finite() && current.is_finite() && previous * current < 0.0
}

#[cfg(test)]
mod tests {
    use super::{
        AimAlgorithm, AimAlgorithmConfig, AimFeedback, AimSample, AxisArrivalState, BlockReason,
        ControlMode,
    };
    use crate::prediction::PredictionMotionState;

    #[test]
    fn arrival_state_uses_hysteresis_but_never_traps_a_real_departure() {
        let mut state = AxisArrivalState::default();
        let enter = 3.0;

        assert!(state.hold(2.9, 1_000_000_000, enter));
        assert!(state.hold(2.8, 1_009_000_000, enter));
        assert!(state.settled);

        // Between the 3-count enter threshold and 4.5-count exit threshold,
        // detector chatter remains settled.
        assert!(state.hold(4.4, 1_018_000_000, enter));
        // A modest departure is confirmed over time instead of reacting to a
        // single noisy sample.
        assert!(state.hold(5.0, 1_027_000_000, enter));
        assert!(!state.hold(5.0, 1_036_000_000, enter));

        // A large displacement bypasses confirmation immediately, so the
        // dead zone cannot retain a genuinely moved/new target.
        state.hold(2.0, 1_045_000_000, enter);
        state.hold(2.0, 1_054_000_000, enter);
        assert!(state.settled);
        assert!(!state.hold(9.1, 1_055_000_000, enter));
    }

    #[test]
    fn confidence_weighted_vector_prediction_respects_single_cap() {
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
        assert!((decision.prediction_weighted_offset_x - 2.8).abs() < 1e-12);
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
    fn track_confidence_zero_suppresses_prediction_without_hiding_velocity() {
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
        assert_eq!(decision.motion_confidence, 0.0);
        assert_eq!(decision.predicted_offset_x, 0.0);
        assert_eq!(decision.filtered_error_x, 52.0);
    }

    #[test]
    fn pending_feedback_holds_stationary_axis_to_avoid_duplicate_correction() {
        let mut control = AimAlgorithm::new(AimAlgorithmConfig {
            prediction_enabled: false,
            ..AimAlgorithmConfig::default()
        });
        let first = AimSample {
            generation: 1,
            target_id: 7,
            capture_ts_ns: 1_000_000_000,
            control_now_ns: 1_008_000_000,
            aim_x: 240.0,
            aim_y: 160.0,
            crosshair_x: 160.0,
            crosshair_y: 160.0,
            detection_confidence: 1.0,
            track_confidence: 1.0,
            target_valid: true,
            trigger_active: true,
        };
        assert!(control.step(first).emit_allowed);

        let second = AimSample {
            generation: 2,
            capture_ts_ns: 1_010_000_000,
            control_now_ns: 1_018_000_000,
            ..first
        };
        let decision = control.step_with_feedback(
            second,
            AimFeedback {
                pending_x: true,
                pending_y: false,
            },
        );

        assert!(!decision.emit_allowed);
        assert_eq!(decision.block_reason, BlockReason::ActuationFeedbackPending);
        assert_eq!(decision.dx, 0);
        assert!(decision.arrival_hold_x);
        assert!(decision.actuation_pending_x);
    }

    #[test]
    fn pending_feedback_does_not_freeze_axis_when_error_keeps_growing() {
        let mut control = AimAlgorithm::new(AimAlgorithmConfig {
            prediction_enabled: false,
            ..AimAlgorithmConfig::default()
        });
        let first = AimSample {
            generation: 1,
            target_id: 7,
            capture_ts_ns: 1_000_000_000,
            control_now_ns: 1_008_000_000,
            aim_x: 240.0,
            aim_y: 160.0,
            crosshair_x: 160.0,
            crosshair_y: 160.0,
            detection_confidence: 1.0,
            track_confidence: 1.0,
            target_valid: true,
            trigger_active: true,
        };
        assert!(control.step(first).emit_allowed);

        let second = AimSample {
            generation: 2,
            capture_ts_ns: 1_010_000_000,
            control_now_ns: 1_018_000_000,
            aim_x: 260.0,
            ..first
        };
        let decision = control.step_with_feedback(
            second,
            AimFeedback {
                pending_x: true,
                pending_y: false,
            },
        );

        assert!(decision.emit_allowed);
        assert_eq!(decision.block_reason, BlockReason::None);
        assert_ne!(decision.dx, 0);
        assert!(!decision.arrival_hold_x);
        assert!(decision.actuation_pending_x);
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

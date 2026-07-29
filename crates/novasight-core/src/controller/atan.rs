//! Two-phase Atan feedback controller for the selected target.
//!
//! The measured error is projected into device counts and compressed by one
//! continuously blended FAR/NEAR Atan response. Optional two-axis target
//! prediction is owned by the dedicated `prediction` module. Integer device
//! conversion is owned by the dedicated `limiter` module.
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

use super::response_curve::{BlendedAtanConfig, ContinuousDemand, far_weight};
use crate::limiter::{DeviceCountLimiter, DeviceCountLimits};
use crate::prediction::{
    FocusTargetObservation, SingleTargetPredictionConfig, SingleTargetPredictor,
};

/// Integer mouse output cannot represent a correction smaller than one count.
/// At half a count or less the current integer position is the nearest
/// representable point, so retaining residual would only create a +/-1 cycle.
const HALF_DEVICE_COUNT: f64 = 0.5;
const ARRIVAL_CONFIRM_NS: u64 = 8_000_000;
const ARRIVAL_CONFIRM_SAMPLES: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ControlMode {
    Far,
    Near,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BlockReason {
    /// Frame timestamp is in the past relative to `control_now_ns`.
    TimestampDomainInvalid,
    /// Frame age exceeded the freshness threshold.
    StaleObservation,
    /// Generation or frame id is not strictly increasing.
    NonMonotonicObservation,
    /// Capture timestamp went backwards.
    CaptureTimestampDiscontinuity,
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
pub struct DualPhaseConfig {
    pub freshness_threshold_ms: f64,
    pub near_threshold_px: f64,
    pub projection_fov_x_deg: f64,
    pub projection_counts_per_360: f64,
    pub atan_scale_counts: f64,
    pub far_kp: f64,
    pub far_max_counts_per_update: f64,
    pub near_kp: f64,
    pub near_max_counts_per_update: f64,
    /// Half-size of the per-axis arrival box in projected device counts.
    pub arrival_radius_counts: f64,
    pub velocity_smoothing_frames: f64,
    pub velocity_history_reset_gap_ms: f64,
    pub velocity_spread_base_px_ms: f64,
    pub velocity_spread_relative: f64,
    pub velocity_change_base_px_ms: f64,
    pub velocity_change_relative: f64,
    pub prediction_enabled: bool,
    pub prediction_lead_frames: f64,
    pub prediction_far_absolute_cap_px: f64,
    pub prediction_far_base_cap_px: f64,
    pub prediction_far_relative_cap: f64,
    pub prediction_near_absolute_cap_px: f64,
    pub prediction_near_base_cap_px: f64,
    pub prediction_near_relative_cap: f64,
    pub source_width: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub observation_width: u32,
    pub observation_height: u32,
    /// Cap on the fractional residual retained across emits.
    pub residual_cap: f64,
}

impl Default for DualPhaseConfig {
    fn default() -> Self {
        Self {
            freshness_threshold_ms: 55.0,
            near_threshold_px: 12.0,
            projection_fov_x_deg: 105.0,
            projection_counts_per_360: 9_980.0,
            atan_scale_counts: 256.0,
            far_kp: 0.22,
            far_max_counts_per_update: 127.0,
            near_kp: 0.20,
            near_max_counts_per_update: 72.0,
            arrival_radius_counts: 3.0,
            velocity_smoothing_frames: 3.0,
            velocity_history_reset_gap_ms: 80.0,
            velocity_spread_base_px_ms: 0.12,
            velocity_spread_relative: 0.50,
            velocity_change_base_px_ms: 0.20,
            velocity_change_relative: 0.75,
            prediction_enabled: false,
            prediction_lead_frames: 1.0,
            prediction_far_absolute_cap_px: 10.0,
            prediction_far_base_cap_px: 1.25,
            prediction_far_relative_cap: 0.30,
            prediction_near_absolute_cap_px: 3.0,
            prediction_near_base_cap_px: 0.75,
            prediction_near_relative_cap: 0.20,
            source_width: 640,
            roi_width: 640,
            roi_height: 640,
            observation_width: 640,
            observation_height: 640,
            residual_cap: 1.0,
        }
    }
}

impl DualPhaseConfig {
    fn prediction_config(self) -> SingleTargetPredictionConfig {
        SingleTargetPredictionConfig {
            enabled: self.prediction_enabled,
            smoothing_frames: self.velocity_smoothing_frames,
            history_reset_gap_ms: self.velocity_history_reset_gap_ms,
            spread_base_px_ms: self.velocity_spread_base_px_ms,
            spread_relative: self.velocity_spread_relative,
            change_base_px_ms: self.velocity_change_base_px_ms,
            change_relative: self.velocity_change_relative,
            lead_frames: self.prediction_lead_frames,
            far_absolute_cap_px: self.prediction_far_absolute_cap_px,
            far_base_cap_px: self.prediction_far_base_cap_px,
            far_relative_cap: self.prediction_far_relative_cap,
            near_absolute_cap_px: self.prediction_near_absolute_cap_px,
            near_base_cap_px: self.prediction_near_base_cap_px,
            near_relative_cap: self.prediction_near_relative_cap,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlObservation {
    pub generation: u64,
    pub frame_id: u64,
    pub target_id: u64,
    pub capture_ts_ns: u64,
    pub inference_end_ts_ns: u64,
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
pub struct ActuationFeedback {
    pub pending_x: bool,
    pub pending_y: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlDecision {
    pub sample_available: bool,
    pub generation: u64,
    pub frame_id: u64,
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
    pub predicted_offset_x: f64,
    pub predicted_offset_y: f64,
    pub motion_confidence: f64,
    pub history_position_count: usize,
    pub velocity_samples: [Option<f64>; 3],
    pub median_velocity: Option<f64>,
    pub velocity_spread: Option<f64>,
    pub measurement_dt_ms: Option<f64>,
    pub reference_dt_ms: f64,
    pub prediction_lead_frames: f64,
    pub prediction_horizon_ms: f64,
    pub prediction_raw_offset_x: f64,
    pub prediction_weighted_offset_x: f64,
    pub prediction_allowed_cap_x: f64,
    pub prediction_allowed: bool,
    pub motion_confidence_y: f64,
    pub velocity_samples_y: [Option<f64>; 3],
    pub median_velocity_y: Option<f64>,
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

impl ControlDecision {
    pub fn blocked(reason: BlockReason) -> Self {
        Self {
            sample_available: false,
            generation: 0,
            frame_id: 0,
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
            mode: ControlMode::Far,
            velocity_x: 0.0,
            velocity_y: 0.0,
            predicted_offset_x: 0.0,
            predicted_offset_y: 0.0,
            motion_confidence: 0.0,
            history_position_count: 0,
            velocity_samples: [None; 3],
            median_velocity: None,
            velocity_spread: None,
            measurement_dt_ms: None,
            reference_dt_ms: 0.0,
            prediction_lead_frames: 0.0,
            prediction_horizon_ms: 0.0,
            prediction_raw_offset_x: 0.0,
            prediction_weighted_offset_x: 0.0,
            prediction_allowed_cap_x: 0.0,
            prediction_allowed: false,
            motion_confidence_y: 0.0,
            velocity_samples_y: [None; 3],
            median_velocity_y: None,
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

impl Default for ControlDecision {
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

/// reach into a stale decision.
#[derive(Clone, Debug)]
pub struct DualPhaseControl {
    config: DualPhaseConfig,
    limiter: DeviceCountLimiter,
    last_generation: Option<u64>,
    last_frame_id: Option<u64>,
    last_capture_ts_ns: Option<u64>,
    target_id: Option<u64>,
    previous_error_x: f64,
    previous_error_y: f64,
    measured_error_history_valid: bool,
    prediction: SingleTargetPredictor,
    arrival_x: AxisArrivalState,
    arrival_y: AxisArrivalState,
}

impl DualPhaseControl {
    pub fn new(config: DualPhaseConfig) -> Self {
        Self {
            config,
            limiter: DeviceCountLimiter::new(),
            last_generation: None,
            last_frame_id: None,
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

    pub fn reset(&mut self) {
        self.limiter.reset();
        self.last_generation = None;
        self.last_frame_id = None;
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

    pub fn calculate(&mut self, observation: ControlObservation) -> ControlDecision {
        self.calculate_with_feedback(observation, ActuationFeedback::default())
    }

    pub fn calculate_with_feedback(
        &mut self,
        observation: ControlObservation,
        feedback: ActuationFeedback,
    ) -> ControlDecision {
        self.calculate_internal(observation, feedback)
    }

    fn calculate_internal(
        &mut self,
        observation: ControlObservation,
        feedback: ActuationFeedback,
    ) -> ControlDecision {
        let frame_age_ns = observation.control_now_ns as i128 - observation.capture_ts_ns as i128;
        let inference_end_ns = observation.inference_end_ts_ns as i128;
        if frame_age_ns < 0
            || inference_end_ns < observation.capture_ts_ns as i128
            || inference_end_ns > observation.control_now_ns as i128
        {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::TimestampDomainInvalid);
        }
        let frame_age_ms = frame_age_ns as f64 / 1_000_000.0;
        if frame_age_ms > self.config.freshness_threshold_ms {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::StaleObservation);
        }
        if let Some(prev_gen) = self.last_generation
            && observation.generation <= prev_gen
        {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::NonMonotonicObservation);
        }
        if let Some(prev_frame) = self.last_frame_id
            && observation.frame_id <= prev_frame
        {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::NonMonotonicObservation);
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
            self.last_frame_id = Some(observation.frame_id);
            self.last_capture_ts_ns = Some(observation.capture_ts_ns);
            self.target_id = None;
            self.measured_error_history_valid = false;
            self.prediction.reset(None);
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::TargetInvalid);
        }
        if !self.config.arrival_radius_counts.is_finite()
            || self.config.arrival_radius_counts < HALF_DEVICE_COUNT
            || !self.prediction.config_valid()
        {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::GeometryInvalid);
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
        let observed_distance = error_x.hypot(error_y);
        // Prediction caps are scheduled from the measured observation so the
        // predictor cannot enlarge its own envelope recursively.
        let prediction_far_weight = far_weight(observed_distance, self.config.near_threshold_px);
        let prediction = if capture_timestamp_discontinuity {
            self.prediction
                .unavailable(error_x, error_y, prediction_far_weight)
        } else {
            self.prediction.predict(FocusTargetObservation {
                track_id: observation.target_id,
                aim_x,
                aim_y,
                measured_error_x: error_x,
                measured_error_y: error_y,
                capture_ts_ns: observation.capture_ts_ns,
                observation_age_ms: frame_age_ms,
                detection_confidence: observation.detection_confidence,
                identity_confidence: observation.track_confidence,
                far_weight: prediction_far_weight,
            })
        };
        let predicted_offset_x = prediction.x.safe_offset;
        let predicted_offset_y = prediction.y.safe_offset;
        let filtered_error_x = error_x + predicted_offset_x;
        let filtered_error_y = error_y + predicted_offset_y;
        // The response region must follow the point we actually intend to
        // control, not the stale pre-prediction position.
        let control_distance = filtered_error_x.hypot(filtered_error_y);
        let mode = if control_distance <= self.config.near_threshold_px {
            ControlMode::Near
        } else {
            ControlMode::Far
        };
        let Some((mut response, full_x, full_y)) =
            self.project_demand(filtered_error_x, filtered_error_y, control_distance)
        else {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::GeometryInvalid);
        };

        let arrival_enter_counts = self.config.arrival_radius_counts.max(HALF_DEVICE_COUNT);
        let arrival_exit_counts =
            (arrival_enter_counts * 1.5).max(arrival_enter_counts + HALF_DEVICE_COUNT);
        let arrival_hold_x = feedback.pending_x
            || self
                .arrival_x
                .hold(full_x, observation.capture_ts_ns, arrival_enter_counts);
        let arrival_hold_y = feedback.pending_y
            || self
                .arrival_y
                .hold(full_y, observation.capture_ts_ns, arrival_enter_counts);

        if arrival_hold_x || full_x.abs() <= HALF_DEVICE_COUNT {
            self.limiter.reset_x();
            response.x = 0.0;
        }
        if arrival_hold_y || full_y.abs() <= HALF_DEVICE_COUNT {
            self.limiter.reset_y();
            response.y = 0.0;
        }

        let demand_x = response.x;
        let demand_y = response.y;
        let limits = DeviceCountLimits {
            max_counts_per_axis: response.limit_counts,
            residual_cap: self.config.residual_cap,
        };

        let (dx, dy, block_reason) = if observation.trigger_active {
            let limited = match self.limiter.limit(demand_x, demand_y, limits) {
                Ok(value) => value,
                Err(_) => {
                    self.release_trigger();
                    return ControlDecision::blocked(BlockReason::DemandOutOfRange);
                }
            };
            let dx = limited.dx;
            let dy = limited.dy;
            let reason = if dx == 0 && dy == 0 && (feedback.pending_x || feedback.pending_y) {
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
        self.last_frame_id = Some(observation.frame_id);
        self.last_capture_ts_ns = Some(observation.capture_ts_ns);
        self.target_id = Some(observation.target_id);
        self.previous_error_x = error_x;
        self.previous_error_y = error_y;
        self.measured_error_history_valid = true;

        let (quantizer_residual_x, quantizer_residual_y) = self.limiter.residuals();
        ControlDecision {
            sample_available: true,
            generation: observation.generation,
            frame_id: observation.frame_id,
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
            predicted_offset_x,
            predicted_offset_y,
            motion_confidence: prediction.x.motion_confidence,
            history_position_count: prediction.history_position_count,
            velocity_samples: prediction.x.velocity_samples,
            median_velocity: prediction.x.median_velocity,
            velocity_spread: prediction.x.velocity_spread,
            measurement_dt_ms: prediction.x.measurement_dt_ms,
            reference_dt_ms: prediction.x.reference_dt_ms,
            prediction_lead_frames: prediction.lead_frames,
            prediction_horizon_ms: prediction.x.horizon_ms,
            prediction_raw_offset_x: prediction.x.raw_offset,
            prediction_weighted_offset_x: prediction.x.weighted_offset,
            prediction_allowed_cap_x: prediction.x.allowed_cap,
            prediction_allowed: prediction.x.allowed,
            motion_confidence_y: prediction.y.motion_confidence,
            velocity_samples_y: prediction.y.velocity_samples,
            median_velocity_y: prediction.y.median_velocity,
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

    fn project_demand(
        &self,
        error_x: f64,
        error_y: f64,
        measured_distance_px: f64,
    ) -> Option<(ContinuousDemand, f64, f64)> {
        let config = self.config;
        if config.source_width == 0
            || config.roi_width == 0
            || config.roi_height == 0
            || config.observation_width == 0
            || config.observation_height == 0
            || !config.projection_fov_x_deg.is_finite()
            || !(0.0..180.0).contains(&config.projection_fov_x_deg)
            || !config.projection_counts_per_360.is_finite()
            || config.projection_counts_per_360 <= 0.0
            || !config.atan_scale_counts.is_finite()
            || config.atan_scale_counts <= 0.0
        {
            return None;
        }
        let source_error_x =
            error_x * f64::from(config.roi_width) / f64::from(config.observation_width);
        let source_error_y =
            error_y * f64::from(config.roi_height) / f64::from(config.observation_height);
        let fov_x_rad = config.projection_fov_x_deg.to_radians();
        let focal_px = (f64::from(config.source_width) * 0.5) / (fov_x_rad * 0.5).tan();
        if !focal_px.is_finite() || focal_px <= 0.0 {
            return None;
        }
        let counts_per_rad = config.projection_counts_per_360 / std::f64::consts::TAU;
        let full_x = (source_error_x / focal_px).atan() * counts_per_rad;
        let full_y = (source_error_y / focal_px).atan() * counts_per_rad;
        let response = BlendedAtanConfig {
            near_threshold_px: config.near_threshold_px,
            scale_counts: config.atan_scale_counts,
            far_kp: config.far_kp,
            far_limit_counts: config.far_max_counts_per_update,
            near_kp: config.near_kp,
            near_limit_counts: config.near_max_counts_per_update,
        }
        .evaluate(measured_distance_px, full_x, full_y)?;
        Some((response, full_x, full_y))
    }
}

fn crossed_center(previous: f64, current: f64) -> bool {
    previous.is_finite() && current.is_finite() && previous * current < 0.0
}

#[cfg(test)]
mod tests {
    use super::{
        AxisArrivalState, ControlMode, ControlObservation, DualPhaseConfig, DualPhaseControl,
    };

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
    fn confidence_weighted_prediction_respects_far_cap() {
        let config = DualPhaseConfig {
            prediction_enabled: true,
            prediction_lead_frames: 2.0,
            prediction_far_absolute_cap_px: 3.0,
            prediction_far_base_cap_px: 0.0,
            prediction_far_relative_cap: 0.05,
            ..DualPhaseConfig::default()
        };
        let mut control = DualPhaseControl::new(config);
        let mut decision = None;
        for (index, (error_x, error_y)) in [(40.0, 20.0), (44.0, 22.0), (48.0, 24.0), (52.0, 26.0)]
            .into_iter()
            .enumerate()
        {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.calculate(ControlObservation {
                generation,
                frame_id: generation,
                target_id: 7,
                capture_ts_ns,
                inference_end_ts_ns: capture_ts_ns + 4_000_000,
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
        assert_eq!(decision.median_velocity, Some(0.4));
        assert!((decision.velocity_x - 0.4).abs() < 1e-12);
        assert!((decision.reference_dt_ms - 10.0).abs() < 1e-12);
        assert!((decision.prediction_horizon_ms - 28.0).abs() < 1e-12);
        assert!((decision.prediction_raw_offset_x - 11.2).abs() < 1e-12);
        assert!((decision.prediction_allowed_cap_x - 2.6).abs() < 1e-12);
        assert!(decision.prediction_allowed);
        assert!((decision.predicted_offset_x - 2.6).abs() < 1e-12);
        assert!((decision.filtered_error_x - 54.6).abs() < 1e-12);
        assert!((decision.velocity_y - 0.2).abs() < 1e-12);
        assert!(decision.prediction_allowed_y);
        assert!(decision.predicted_offset_y > 0.0);
        assert!(decision.filtered_error_y > 26.0);
        assert!(decision.full_error_counts_x.is_finite());
        assert!(decision.float_demand_x.is_finite());
    }

    #[test]
    fn track_confidence_zero_suppresses_prediction_without_hiding_velocity() {
        let mut control = DualPhaseControl::new(DualPhaseConfig {
            prediction_enabled: true,
            ..DualPhaseConfig::default()
        });
        let mut decision = None;
        for (index, error_x) in [40.0, 44.0, 48.0, 52.0].into_iter().enumerate() {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.calculate(ControlObservation {
                generation,
                frame_id: generation,
                target_id: 7,
                capture_ts_ns,
                inference_end_ts_ns: capture_ts_ns + 4_000_000,
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
        assert_eq!(decision.motion_confidence, 0.0);
        assert_eq!(decision.predicted_offset_x, 0.0);
        assert_eq!(decision.filtered_error_x, 52.0);
    }

    #[test]
    fn response_region_follows_the_predicted_control_point() {
        let mut control = DualPhaseControl::new(DualPhaseConfig {
            prediction_enabled: true,
            prediction_lead_frames: 1.0,
            ..DualPhaseConfig::default()
        });
        let mut decision = None;
        for (index, error_x) in [5.0, 7.0, 9.0, 11.0].into_iter().enumerate() {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.calculate(ControlObservation {
                generation,
                frame_id: generation,
                target_id: 7,
                capture_ts_ns,
                inference_end_ts_ns: capture_ts_ns + 4_000_000,
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
        assert_eq!(decision.mode, ControlMode::Far);
    }

    #[test]
    fn trigger_release_clears_output_residual_but_keeps_motion_history_hot() {
        let mut control = DualPhaseControl::new(DualPhaseConfig {
            prediction_enabled: true,
            ..DualPhaseConfig::default()
        });
        let mut decision = None;
        for (index, error_x) in [40.0, 44.0, 48.0, 52.0].into_iter().enumerate() {
            let generation = index as u64 + 1;
            let capture_ts_ns = 1_000_000_000 + generation * 10_000_000;
            decision = Some(control.calculate(ControlObservation {
                generation,
                frame_id: generation,
                target_id: 7,
                capture_ts_ns,
                inference_end_ts_ns: capture_ts_ns + 4_000_000,
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

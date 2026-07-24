//! Rust-owned dual-phase atan feedback with robust X-axis prediction.
//!
//! This implements the production semantics of
//! ``novasight.control.algorithms.dual_phase_atan_robust_predictive_v2``:
//! four positions produce three segment velocities, the segment median is
//! smoothed by a capture-time-aware EMA, and a confidence-weighted prediction
//! is bounded independently for FAR and NEAR modes.
//!
//! * `dx`/`dy` are integer mouse counts the device should emit. We do
//!   not promise 1:1 floating-point parity with the Python telemetry;
//!   instead we pin the integer outcome and the contract edges that
//!   the runtime actually depends on.
//! * `emit_allowed` is `true` only when the algorithm produced an
//!   emit-eligible decision. Triggers and target validity gate the
//!   state machine; stale or non-monotonic observations are rejected
//!   with a typed `BlockReason`.
//! * `quantizer_residual` carries the fractional count past the
//!   integer boundary so the next call can absorb sub-count motion
//!   without losing precision.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::control::humanized_motion::{
    HumanizedMotionGenerator, HumanizedMotionInput, HumanizedMotionTelemetry, MotionProfile,
};
use crate::error::AppError;

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
    /// Filtered position fell outside the configured dead-zone.
    DeadZone,
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
    pub projection_invert_y: bool,
    pub atan_scale_counts: f64,
    pub far_kp: f64,
    pub far_max_counts_per_update: f64,
    pub near_kp: f64,
    pub near_max_counts_per_update: f64,
    pub velocity_smoothing_frames: f64,
    pub velocity_history_reset_gap_ms: f64,
    pub velocity_spread_base_px_ms: f64,
    pub velocity_spread_relative: f64,
    pub velocity_change_base_px_ms: f64,
    pub velocity_change_relative: f64,
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
            projection_invert_y: false,
            atan_scale_counts: 256.0,
            far_kp: 0.45,
            far_max_counts_per_update: 127.0,
            near_kp: 0.22,
            near_max_counts_per_update: 72.0,
            velocity_smoothing_frames: 3.0,
            velocity_history_reset_gap_ms: 80.0,
            velocity_spread_base_px_ms: 0.12,
            velocity_spread_relative: 0.50,
            velocity_change_base_px_ms: 0.20,
            velocity_change_relative: 0.75,
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
    pub prediction_raw_offset_x: f64,
    pub prediction_weighted_offset_x: f64,
    pub prediction_allowed_cap_x: f64,
    pub prediction_allowed: bool,
    pub observed_error_x: f64,
    pub observed_error_y: f64,
    pub filtered_error_x: f64,
    pub filtered_error_y: f64,
    pub full_error_counts_x: f64,
    pub full_error_counts_y: f64,
    pub float_demand_x: f64,
    pub float_demand_y: f64,
    pub humanized_motion: HumanizedMotionTelemetry,
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
            prediction_raw_offset_x: 0.0,
            prediction_weighted_offset_x: 0.0,
            prediction_allowed_cap_x: 0.0,
            prediction_allowed: false,
            observed_error_x: 0.0,
            observed_error_y: 0.0,
            filtered_error_x: 0.0,
            filtered_error_y: 0.0,
            full_error_counts_x: 0.0,
            full_error_counts_y: 0.0,
            float_demand_x: 0.0,
            float_demand_y: 0.0,
            humanized_motion: HumanizedMotionTelemetry::default(),
        }
    }
}

impl Default for ControlDecision {
    fn default() -> Self {
        Self::blocked(BlockReason::TriggerInactive)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PredictionResult {
    raw_offset_x: f64,
    weighted_offset_x: f64,
    allowed_cap_x: f64,
    safe_offset_x: f64,
    allowed: bool,
}

const VELOCITY_POSITION_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
struct PositionSample {
    aim_x: f64,
    capture_ts_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityEstimate {
    pub raw_velocities: [f64; 3],
    pub median_velocity: f64,
    pub filtered_velocity: f64,
    pub spread: f64,
    pub motion_confidence: f64,
    pub measurement_dt_ms: f64,
    pub reference_dt_ms: f64,
    pub history_quality: f64,
    pub spread_quality: f64,
    pub trend_quality: f64,
    pub detection_quality: f64,
    pub track_quality: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RobustVelocityConfig {
    smoothing_frames: f64,
    history_reset_gap_ms: f64,
    spread_base_px_ms: f64,
    spread_relative: f64,
    change_base_px_ms: f64,
    change_relative: f64,
}

impl From<DualPhaseConfig> for RobustVelocityConfig {
    fn from(config: DualPhaseConfig) -> Self {
        Self {
            smoothing_frames: config.velocity_smoothing_frames,
            history_reset_gap_ms: config.velocity_history_reset_gap_ms,
            spread_base_px_ms: config.velocity_spread_base_px_ms,
            spread_relative: config.velocity_spread_relative,
            change_base_px_ms: config.velocity_change_base_px_ms,
            change_relative: config.velocity_change_relative,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RobustVelocityEstimator {
    config: RobustVelocityConfig,
    target_id: Option<u64>,
    samples: VecDeque<PositionSample>,
    filtered_velocity: f64,
    initialized_velocity: bool,
    complete_window_updates: u64,
}

impl RobustVelocityEstimator {
    pub fn new(config: DualPhaseConfig) -> Self {
        Self {
            config: config.into(),
            target_id: None,
            samples: VecDeque::with_capacity(VELOCITY_POSITION_COUNT),
            filtered_velocity: 0.0,
            initialized_velocity: false,
            complete_window_updates: 0,
        }
    }

    pub fn history_position_count(&self) -> usize {
        self.samples.len()
    }

    pub fn reset(&mut self, target_id: Option<u64>) {
        self.target_id = target_id;
        self.samples.clear();
        self.filtered_velocity = 0.0;
        self.initialized_velocity = false;
        self.complete_window_updates = 0;
    }

    pub fn update(
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
        let reference_dt_ms = intervals_ms.iter().sum::<f64>() / 3.0;
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
        let detection_quality = detection_confidence.clamp(0.0, 1.0);
        let track_quality = track_confidence.clamp(0.0, 1.0);
        let motion_confidence =
            (history_quality * spread_quality * trend_quality * detection_quality * track_quality)
                .clamp(0.0, 1.0);
        Some(VelocityEstimate {
            raw_velocities: velocities,
            median_velocity,
            filtered_velocity: self.filtered_velocity,
            spread,
            motion_confidence,
            measurement_dt_ms: latest_dt_ms,
            reference_dt_ms,
            history_quality,
            spread_quality,
            trend_quality,
            detection_quality,
            track_quality,
        })
    }
}

fn median_three(mut values: [f64; 3]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[1]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Quantizer {
    accumulator: f64,
}

impl Quantizer {
    fn new() -> Self {
        Self { accumulator: 0.0 }
    }

    fn reset(&mut self) {
        self.accumulator = 0.0;
    }

    fn quantize(
        &mut self,
        demand: f64,
        max_counts_per_axis: i32,
        residual_cap: f64,
    ) -> Result<i32, AppError> {
        if !demand.is_finite() {
            self.reset();
            return Err(AppError::DeviceCountOutOfRange);
        }
        if self.accumulator != 0.0 && demand != 0.0 && self.accumulator * demand < 0.0 {
            self.reset();
        }
        self.accumulator += demand;
        let counts = self.accumulator.trunc().clamp(
            -f64::from(max_counts_per_axis),
            f64::from(max_counts_per_axis),
        );
        if !counts.is_finite() || counts < f64::from(i32::MIN) || counts > f64::from(i32::MAX) {
            self.reset();
            return Err(AppError::DeviceCountOutOfRange);
        }
        let counts_int = counts as i32;
        self.accumulator -= f64::from(counts_int);
        self.accumulator = self.accumulator.clamp(-residual_cap, residual_cap);
        Ok(counts_int)
    }
}

/// reach into a stale decision.
#[derive(Clone, Debug)]
pub struct DualPhaseControl {
    config: DualPhaseConfig,
    quantizer_x: Quantizer,
    quantizer_y: Quantizer,
    last_generation: Option<u64>,
    last_frame_id: Option<u64>,
    last_capture_ts_ns: Option<u64>,
    target_id: Option<u64>,
    previous_error_x: f64,
    previous_error_y: f64,
    measured_error_history_valid: bool,
    velocity_x: RobustVelocityEstimator,
    humanized_motion: HumanizedMotionGenerator,
}

impl DualPhaseControl {
    pub fn new(config: DualPhaseConfig) -> Self {
        Self {
            config,
            quantizer_x: Quantizer::new(),
            quantizer_y: Quantizer::new(),
            last_generation: None,
            last_frame_id: None,
            last_capture_ts_ns: None,
            target_id: None,
            previous_error_x: 0.0,
            previous_error_y: 0.0,
            measured_error_history_valid: false,
            velocity_x: RobustVelocityEstimator::new(config),
            humanized_motion: HumanizedMotionGenerator::default(),
        }
    }

    pub fn reset(&mut self) {
        self.quantizer_x.reset();
        self.quantizer_y.reset();
        self.last_generation = None;
        self.last_frame_id = None;
        self.last_capture_ts_ns = None;
        self.target_id = None;
        self.previous_error_x = 0.0;
        self.previous_error_y = 0.0;
        self.measured_error_history_valid = false;
        self.velocity_x.reset(None);
        self.humanized_motion.reset();
    }

    pub fn release_trigger(&mut self) {
        self.quantizer_x.reset();
        self.quantizer_y.reset();
    }

    /// Clear target-relative state while preserving observation sequence
    /// guards. Used when a tracker restores or rebuilds an identity.
    pub fn reset_target_state(&mut self) {
        self.release_trigger();
        self.velocity_x.reset(None);
        self.humanized_motion.reset();
        self.target_id = None;
        self.previous_error_x = 0.0;
        self.previous_error_y = 0.0;
        self.measured_error_history_valid = false;
    }

    pub fn calculate(&mut self, observation: ControlObservation) -> ControlDecision {
        self.calculate_with_profile(observation, None, 1.0)
    }

    /// Calculate against the currently active immutable motion profile.
    ///
    /// The profile shapes floating-point demand before quantization. The
    /// existing per-axis mode limit is then applied again, so a trained curve
    /// cannot bypass the controller's safety envelope.
    pub fn calculate_with_profile(
        &mut self,
        observation: ControlObservation,
        profile: Option<&MotionProfile>,
        target_width_px: f64,
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
            self.velocity_x.reset(self.target_id);
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
            self.velocity_x.reset(None);
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::TargetInvalid);
        }
        if !self.prediction_config_valid() {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::GeometryInvalid);
        }
        let aim_x = observation.aim_x;
        let aim_y = observation.aim_y;
        let error_x = aim_x - observation.crosshair_x;
        let error_y = aim_y - observation.crosshair_y;
        if self.measured_error_history_valid {
            if crossed_center(self.previous_error_x, error_x) {
                self.quantizer_x.reset();
            }
            if crossed_center(self.previous_error_y, error_y) {
                self.quantizer_y.reset();
            }
        }
        let distance = error_x.hypot(error_y);
        let mode = if distance <= self.config.near_threshold_px {
            ControlMode::Near
        } else {
            ControlMode::Far
        };
        let estimate = if capture_timestamp_discontinuity {
            None
        } else {
            self.velocity_x.update(
                observation.target_id,
                aim_x,
                observation.capture_ts_ns,
                observation.detection_confidence,
                observation.track_confidence,
            )
        };
        let velocity_x = estimate.map_or(0.0, |value| value.filtered_velocity);
        let motion_confidence = estimate.map_or(0.0, |value| value.motion_confidence);
        let reference_dt_ms = estimate.map_or(0.0, |value| value.reference_dt_ms);
        let prediction = self.predict_offset(
            mode,
            error_x,
            velocity_x,
            motion_confidence,
            reference_dt_ms,
            estimate.is_some(),
        );
        let predicted_offset_x = prediction.safe_offset_x;
        // The authoritative Python robust predictor intentionally predicts X only.
        let predicted_offset_y = 0.0;
        let filtered_error_x = error_x + predicted_offset_x;
        let filtered_error_y = error_y + predicted_offset_y;
        let Some((base_x, base_y, full_x, full_y, max_counts_per_axis)) =
            self.project_demand(filtered_error_x, filtered_error_y, mode)
        else {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::GeometryInvalid);
        };

        let shaped = self.humanized_motion.apply(
            profile,
            HumanizedMotionInput {
                base_x,
                base_y,
                full_x,
                full_y,
                error_x_px: filtered_error_x,
                error_y_px: filtered_error_y,
                target_width_px,
                target_id: observation.target_id,
                trigger_active: observation.trigger_active,
                control_time_ms: observation.control_now_ns as f64 / 1_000_000.0,
            },
        );
        let maximum = f64::from(max_counts_per_axis);
        let demand_x = shaped.x.clamp(-maximum, maximum);
        let demand_y = shaped.y.clamp(-maximum, maximum);

        let (dx, dy, block_reason) = if observation.trigger_active {
            let dx = match self.quantizer_x.quantize(
                demand_x,
                max_counts_per_axis,
                self.config.residual_cap,
            ) {
                Ok(value) => value,
                Err(_) => {
                    self.release_trigger();
                    return ControlDecision::blocked(BlockReason::DemandOutOfRange);
                }
            };
            let dy = match self.quantizer_y.quantize(
                demand_y,
                max_counts_per_axis,
                self.config.residual_cap,
            ) {
                Ok(value) => value,
                Err(_) => {
                    self.release_trigger();
                    return ControlDecision::blocked(BlockReason::DemandOutOfRange);
                }
            };
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
        self.last_frame_id = Some(observation.frame_id);
        self.last_capture_ts_ns = Some(observation.capture_ts_ns);
        self.target_id = Some(observation.target_id);
        self.previous_error_x = error_x;
        self.previous_error_y = error_y;
        self.measured_error_history_valid = true;

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
            quantizer_residual_x: self.quantizer_x.accumulator,
            quantizer_residual_y: self.quantizer_y.accumulator,
            mode,
            velocity_x,
            velocity_y: 0.0,
            predicted_offset_x,
            predicted_offset_y,
            motion_confidence,
            history_position_count: self.velocity_x.history_position_count(),
            velocity_samples: estimate.map_or([None; 3], |value| value.raw_velocities.map(Some)),
            median_velocity: estimate.map(|value| value.median_velocity),
            velocity_spread: estimate.map(|value| value.spread),
            measurement_dt_ms: estimate.map(|value| value.measurement_dt_ms),
            reference_dt_ms,
            prediction_lead_frames: self.config.prediction_lead_frames,
            prediction_raw_offset_x: prediction.raw_offset_x,
            prediction_weighted_offset_x: prediction.weighted_offset_x,
            prediction_allowed_cap_x: prediction.allowed_cap_x,
            prediction_allowed: prediction.allowed,
            observed_error_x: error_x,
            observed_error_y: error_y,
            filtered_error_x,
            filtered_error_y,
            full_error_counts_x: full_x,
            full_error_counts_y: full_y,
            float_demand_x: demand_x,
            float_demand_y: demand_y,
            humanized_motion: shaped.telemetry,
        }
    }

    fn predict_offset(
        &self,
        mode: ControlMode,
        measured_error_x: f64,
        filtered_velocity_x: f64,
        motion_confidence: f64,
        reference_dt_ms: f64,
        estimate_available: bool,
    ) -> PredictionResult {
        let allowed = estimate_available
            && reference_dt_ms.is_finite()
            && reference_dt_ms > 0.0
            && self.config.prediction_lead_frames > 0.0;
        let confidence = if allowed {
            motion_confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let raw_offset =
            filtered_velocity_x * reference_dt_ms.max(0.0) * self.config.prediction_lead_frames;
        let (absolute_cap, base_cap, relative_cap) = match mode {
            ControlMode::Far => (
                self.config.prediction_far_absolute_cap_px,
                self.config.prediction_far_base_cap_px,
                self.config.prediction_far_relative_cap,
            ),
            ControlMode::Near => (
                self.config.prediction_near_absolute_cap_px,
                self.config.prediction_near_base_cap_px,
                self.config.prediction_near_relative_cap,
            ),
        };
        let allowed_cap = absolute_cap.min(base_cap + relative_cap * measured_error_x.abs());
        let weighted_offset = raw_offset * confidence;
        PredictionResult {
            raw_offset_x: raw_offset,
            weighted_offset_x: weighted_offset,
            allowed_cap_x: allowed_cap,
            safe_offset_x: weighted_offset.clamp(-allowed_cap, allowed_cap),
            allowed,
        }
    }

    fn prediction_config_valid(&self) -> bool {
        self.config.velocity_smoothing_frames.is_finite()
            && self.config.velocity_smoothing_frames > 0.0
            && self.config.velocity_history_reset_gap_ms.is_finite()
            && self.config.velocity_history_reset_gap_ms > 0.0
            && self.config.velocity_spread_base_px_ms.is_finite()
            && self.config.velocity_spread_base_px_ms > 0.0
            && self.config.velocity_spread_relative.is_finite()
            && self.config.velocity_spread_relative >= 0.0
            && self.config.velocity_change_base_px_ms.is_finite()
            && self.config.velocity_change_base_px_ms > 0.0
            && self.config.velocity_change_relative.is_finite()
            && self.config.velocity_change_relative >= 0.0
            && self.config.prediction_lead_frames.is_finite()
            && (0.0..=10.0).contains(&self.config.prediction_lead_frames)
            && [
                self.config.prediction_far_absolute_cap_px,
                self.config.prediction_far_base_cap_px,
                self.config.prediction_far_relative_cap,
                self.config.prediction_near_absolute_cap_px,
                self.config.prediction_near_base_cap_px,
                self.config.prediction_near_relative_cap,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
    }

    fn project_demand(
        &self,
        error_x: f64,
        error_y: f64,
        mode: ControlMode,
    ) -> Option<(f64, f64, f64, f64, i32)> {
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
        let mut full_y = (source_error_y / focal_px).atan() * counts_per_rad;
        if config.projection_invert_y {
            full_y = -full_y;
        }
        let (kp, maximum) = match mode {
            ControlMode::Far => (config.far_kp, config.far_max_counts_per_update),
            ControlMode::Near => (config.near_kp, config.near_max_counts_per_update),
        };
        if !kp.is_finite()
            || kp < 0.0
            || !maximum.is_finite()
            || maximum < 1.0
            || maximum > f64::from(i16::MAX)
        {
            return None;
        }
        let demand = |full: f64| {
            (kp * config.atan_scale_counts * (full / config.atan_scale_counts).atan())
                .clamp(-maximum, maximum)
        };
        Some((
            demand(full_x),
            demand(full_y),
            full_x,
            full_y,
            maximum.ceil() as i32,
        ))
    }
}

fn crossed_center(previous: f64, current: f64) -> bool {
    (previous < 0.0 && current >= 0.0) || (previous > 0.0 && current <= 0.0)
}

#[cfg(test)]
mod tests {
    use super::{
        ControlObservation, DualPhaseConfig, DualPhaseControl, Quantizer, RobustVelocityEstimator,
    };

    #[test]
    fn quantizer_truncates_whole_counts_and_carries_only_fractional_residual() {
        let mut quantizer = Quantizer::new();
        assert_eq!(quantizer.quantize(80.6, 100, 1.0).unwrap(), 80);
        assert!((quantizer.accumulator - 0.6).abs() < 1e-9);
        assert_eq!(quantizer.quantize(0.6, 100, 1.0).unwrap(), 1);
        assert!((quantizer.accumulator - 0.2).abs() < 1e-9);
        assert_eq!(quantizer.quantize(-0.6, 100, 1.0).unwrap(), 0);
        assert!((quantizer.accumulator + 0.6).abs() < 1e-9);
    }

    #[test]
    fn robust_velocity_matches_python_three_segment_median_and_spread() {
        let mut estimator = RobustVelocityEstimator::new(DualPhaseConfig::default());
        let mut estimate = None;
        for (index, position) in [100.0, 102.0, 160.0, 106.0].into_iter().enumerate() {
            estimate = estimator.update(
                1,
                position,
                1_000_000_000 + index as u64 * 10_000_000,
                1.0,
                1.0,
            );
        }
        let estimate = estimate.expect("four positions complete the robust window");
        for (actual, expected) in estimate.raw_velocities.into_iter().zip([0.2, 5.8, -5.4]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((estimate.median_velocity - 0.2).abs() < 1e-12);
        assert!((estimate.filtered_velocity - 0.2).abs() < 1e-12);
        assert!((estimate.spread - 5.6).abs() < 1e-12);
        assert!(estimate.motion_confidence < 0.05);
    }

    #[test]
    fn robust_velocity_uses_real_capture_intervals_and_ema_decay() {
        let mut estimator = RobustVelocityEstimator::new(DualPhaseConfig::default());
        let mut estimate = None;
        for elapsed_ms in [0_u64, 8, 20, 29] {
            estimate = estimator.update(
                1,
                100.0 + 0.5 * elapsed_ms as f64,
                1_000_000_000 + elapsed_ms * 1_000_000,
                1.0,
                1.0,
            );
        }
        let estimate = estimate.expect("variable interval window");
        assert_eq!(estimate.raw_velocities, [0.5, 0.5, 0.5]);
        assert!((estimate.reference_dt_ms - 29.0 / 3.0).abs() < 1e-12);

        let mut estimator = RobustVelocityEstimator::new(DualPhaseConfig::default());
        let mut estimates = Vec::new();
        for (index, position) in [100.0, 105.0, 110.0, 115.0, 115.0, 115.0, 115.0]
            .into_iter()
            .enumerate()
        {
            if let Some(estimate) = estimator.update(
                1,
                position,
                1_000_000_000 + index as u64 * 10_000_000,
                1.0,
                1.0,
            ) {
                estimates.push(estimate);
            }
        }
        let last = estimates.last().expect("rolling estimate");
        let expected = 0.5 * (-2.0_f64 / 3.0).exp();
        assert_eq!(last.median_velocity, 0.0);
        assert!((last.filtered_velocity - expected).abs() < 1e-12);
    }

    #[test]
    fn confidence_weighted_prediction_matches_python_far_cap() {
        let config = DualPhaseConfig {
            prediction_lead_frames: 2.0,
            prediction_far_absolute_cap_px: 3.0,
            prediction_far_base_cap_px: 0.0,
            prediction_far_relative_cap: 0.05,
            ..DualPhaseConfig::default()
        };
        let mut control = DualPhaseControl::new(config);
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
        assert!((decision.prediction_raw_offset_x - 8.0).abs() < 1e-12);
        assert!((decision.prediction_allowed_cap_x - 2.6).abs() < 1e-12);
        assert!(decision.prediction_allowed);
        assert!((decision.predicted_offset_x - 2.6).abs() < 1e-12);
        assert!((decision.filtered_error_x - 54.6).abs() < 1e-12);
        assert!(decision.full_error_counts_x.is_finite());
        assert!(decision.float_demand_x.is_finite());
    }

    #[test]
    fn track_confidence_zero_suppresses_prediction_without_hiding_velocity() {
        let mut control = DualPhaseControl::new(DualPhaseConfig::default());
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
    fn trigger_release_clears_output_residual_but_keeps_motion_history_hot() {
        let mut control = DualPhaseControl::new(DualPhaseConfig::default());
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

//! Phase 2 dual-phase control algorithm.
//!
//! Minimal Rust port of
//! ``novasight.control.algorithms.dual_phase_atan_robust_predictive_v2``
//! sufficient to drive the dual-phase-control.jsonl fixtures. Live
//! Python implementation includes a Kalman filter, a robust velocity
//! estimator with median+spread filtering, a multi-mode predictor,
//! and a humanized motion profile generator. We reproduce the same
//! *interface* and the *typed invariants* the runtime relies on:
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

use serde::{Deserialize, Serialize};

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
            atan_scale_counts: 1_024.0,
            far_kp: 0.90,
            far_max_counts_per_update: 600.0,
            near_kp: 0.30,
            near_max_counts_per_update: 120.0,
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
    pub target_valid: bool,
    pub trigger_active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlDecision {
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
    pub filtered_error_x: f64,
    pub filtered_error_y: f64,
}

impl ControlDecision {
    pub const fn blocked(reason: BlockReason) -> Self {
        Self {
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
            filtered_error_x: 0.0,
            filtered_error_y: 0.0,
        }
    }
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
    last_aim: Option<(f64, f64)>,
    last_capture_for_velocity: Option<u64>,
    velocity_x: f64,
    velocity_y: f64,
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
            last_aim: None,
            last_capture_for_velocity: None,
            velocity_x: 0.0,
            velocity_y: 0.0,
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
        self.last_aim = None;
        self.last_capture_for_velocity = None;
        self.velocity_x = 0.0;
        self.velocity_y = 0.0;
    }

    pub fn release_trigger(&mut self) {
        self.quantizer_x.reset();
        self.quantizer_y.reset();
    }

    pub fn calculate(&mut self, observation: ControlObservation) -> ControlDecision {
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
        if self
            .last_capture_ts_ns
            .is_some_and(|prev| observation.capture_ts_ns <= prev)
        {
            self.velocity_x = 0.0;
            self.velocity_y = 0.0;
            self.last_aim = None;
            self.last_capture_for_velocity = None;
        }

        if self
            .target_id
            .is_some_and(|target| target != observation.target_id)
        {
            self.release_trigger();
            self.last_aim = None;
            self.last_capture_for_velocity = None;
            self.velocity_x = 0.0;
            self.velocity_y = 0.0;
            self.measured_error_history_valid = false;
        }

        if !observation.target_valid {
            self.last_generation = Some(observation.generation);
            self.last_frame_id = Some(observation.frame_id);
            self.last_capture_ts_ns = Some(observation.capture_ts_ns);
            self.target_id = None;
            self.measured_error_history_valid = false;
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::TargetInvalid);
        }
        if !observation.trigger_active {
            self.last_generation = Some(observation.generation);
            self.last_frame_id = Some(observation.frame_id);
            self.last_capture_ts_ns = Some(observation.capture_ts_ns);
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::TriggerInactive);
        }

        let aim_x = observation.aim_x;
        let aim_y = observation.aim_y;
        let (velocity_x, velocity_y, predicted_offset_x, predicted_offset_y) =
            self.update_velocity_and_predict(aim_x, aim_y, observation.capture_ts_ns);

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
        let filtered_error_x = error_x + predicted_offset_x;
        let filtered_error_y = error_y + predicted_offset_y;
        let Some((demand_x, demand_y, max_counts_per_axis)) =
            self.project_demand(filtered_error_x, filtered_error_y, mode)
        else {
            self.release_trigger();
            return ControlDecision::blocked(BlockReason::GeometryInvalid);
        };

        let dx =
            match self
                .quantizer_x
                .quantize(demand_x, max_counts_per_axis, self.config.residual_cap)
            {
                Ok(value) => value,
                Err(_) => {
                    self.release_trigger();
                    return ControlDecision::blocked(BlockReason::DemandOutOfRange);
                }
            };
        let dy =
            match self
                .quantizer_y
                .quantize(demand_y, max_counts_per_axis, self.config.residual_cap)
            {
                Ok(value) => value,
                Err(_) => {
                    self.release_trigger();
                    return ControlDecision::blocked(BlockReason::DemandOutOfRange);
                }
            };

        self.last_generation = Some(observation.generation);
        self.last_frame_id = Some(observation.frame_id);
        self.last_capture_ts_ns = Some(observation.capture_ts_ns);
        self.target_id = Some(observation.target_id);
        self.last_aim = Some((aim_x, aim_y));
        self.last_capture_for_velocity = Some(observation.capture_ts_ns);
        self.previous_error_x = error_x;
        self.previous_error_y = error_y;
        self.measured_error_history_valid = true;

        ControlDecision {
            dx,
            dy,
            emit_allowed: dx != 0 || dy != 0,
            block_reason: if dx == 0 && dy == 0 {
                BlockReason::DeadZone
            } else {
                BlockReason::None
            },
            quantizer_residual_x: self.quantizer_x.accumulator,
            quantizer_residual_y: self.quantizer_y.accumulator,
            mode,
            velocity_x,
            velocity_y,
            predicted_offset_x,
            predicted_offset_y,
            filtered_error_x,
            filtered_error_y,
        }
    }

    fn update_velocity_and_predict(
        &mut self,
        aim_x: f64,
        aim_y: f64,
        capture_ts_ns: u64,
    ) -> (f64, f64, f64, f64) {
        let mut velocity_x = 0.0;
        let mut velocity_y = 0.0;
        if let (Some((prev_aim_x, prev_aim_y)), Some(prev_capture)) =
            (self.last_aim, self.last_capture_for_velocity)
        {
            let dt_ns = capture_ts_ns.saturating_sub(prev_capture);
            if dt_ns > 0 {
                let dt_ms = dt_ns as f64 / 1_000_000.0;
                velocity_x = (aim_x - prev_aim_x) / dt_ms;
                velocity_y = (aim_y - prev_aim_y) / dt_ms;
            }
        }
        // Keep prediction in shadow until the Rust port owns the Python
        // confidence/spread/cap contract. Applying one raw frame delta here
        // would amplify jitter and is less safe than pure feedback.
        let predicted_offset_x = 0.0;
        let predicted_offset_y = 0.0;
        self.velocity_x = velocity_x;
        self.velocity_y = velocity_y;
        (
            velocity_x,
            velocity_y,
            predicted_offset_x,
            predicted_offset_y,
        )
    }

    fn project_demand(
        &self,
        error_x: f64,
        error_y: f64,
        mode: ControlMode,
    ) -> Option<(f64, f64, i32)> {
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
        Some((demand(full_x), demand(full_y), maximum.ceil() as i32))
    }
}

fn crossed_center(previous: f64, current: f64) -> bool {
    (previous < 0.0 && current >= 0.0) || (previous > 0.0 && current <= 0.0)
}

#[cfg(test)]
mod tests {
    use super::Quantizer;

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
}

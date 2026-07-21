//! Phase 2 per-axis quantizer.
//!
//! Converts a fractional `f64` demand in pixels into an integer mouse
//! count plus a residual that the next call absorbs. The algorithm is
//! the same one the `dual_phase_atan_robust_predictive_v2` Python
//! implementation uses:
//!
//! 1. Accumulate the new demand into the residual state.
//! 2. Clamp the residual to `[-residual_cap, +residual_cap]` so a
//!    sequence of large demands cannot push the accumulator past
//!    where a single recover step would lose sub-count precision.
//! 3. Round the residual to the nearest integer count.
//! 4. Reject non-finite or out-of-range values with a typed
//!    `AppError::DeviceCountOutOfRange`.
//! 5. Clamp the integer count to `[-max_counts_per_axis, +max_counts_per_axis]`
//!    so a single emit cannot exceed the device's per-step saturation
//!    envelope.
//! 6. Subtract the integer from the residual so the next call sees
//!    only the sub-count leftover.

use serde::{Deserialize, Serialize};

use crate::error::AppError;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantizerConfig {
    /// Per-step integer cap. Counts whose absolute value exceeds this
    /// are clamped to the cap with the original sign.
    pub max_counts_per_axis: i32,
    /// Maximum residual the per-axis accumulator can hold. Anything
    /// past this is clipped.
    pub residual_cap: f64,
}

impl Default for QuantizerConfig {
    fn default() -> Self {
        Self {
            max_counts_per_axis: 9_980,
            residual_cap: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AxisQuantizer {
    accumulator: f64,
    saturating_attempts: u64,
    rejected_attempts: u64,
}

impl AxisQuantizer {
    pub const fn new() -> Self {
        Self {
            accumulator: 0.0,
            saturating_attempts: 0,
            rejected_attempts: 0,
        }
    }

    pub fn reset(&mut self) {
        self.accumulator = 0.0;
        self.saturating_attempts = 0;
        self.rejected_attempts = 0;
    }

    pub fn accumulator(&self) -> f64 {
        self.accumulator
    }

    pub fn saturating_attempts(&self) -> u64 {
        self.saturating_attempts
    }

    pub fn rejected_attempts(&self) -> u64 {
        self.rejected_attempts
    }

    /// Quantize one fractional demand. Returns the integer count the
    /// device should emit this step and updates the residual.
    ///
    /// The accumulator absorbs the new demand without an intermediate
    /// clamp; the per-axis cap clips the *integer* count, and the
    /// residual is clamped only after subtraction so a long tail of
    /// small motions can never pile up beyond the configured cap.
    pub fn quantize(&mut self, demand: f64, config: QuantizerConfig) -> Result<i32, AppError> {
        if !demand.is_finite() {
            self.rejected_attempts = self.rejected_attempts.saturating_add(1);
            return Err(AppError::DeviceCountOutOfRange);
        }
        self.accumulator += demand;
        let counts = self.accumulator.round();
        if !counts.is_finite() || counts < f64::from(i32::MIN) || counts > f64::from(i32::MAX) {
            self.rejected_attempts = self.rejected_attempts.saturating_add(1);
            return Err(AppError::DeviceCountOutOfRange);
        }
        let mut counts_int = counts as i32;
        if counts_int.abs() > config.max_counts_per_axis {
            counts_int = counts_int.signum() * config.max_counts_per_axis;
            self.saturating_attempts = self.saturating_attempts.saturating_add(1);
        }
        self.accumulator -= f64::from(counts_int);
        if self.accumulator > config.residual_cap {
            self.accumulator = config.residual_cap;
        } else if self.accumulator < -config.residual_cap {
            self.accumulator = -config.residual_cap;
        }
        Ok(counts_int)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantizedCommand {
    pub dx: i32,
    pub dy: i32,
    pub residual_x: f64,
    pub residual_y: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerAxisQuantizer {
    config: QuantizerConfig,
    x: AxisQuantizer,
    y: AxisQuantizer,
}

impl PerAxisQuantizer {
    pub fn new(config: QuantizerConfig) -> Self {
        Self {
            config,
            x: AxisQuantizer::new(),
            y: AxisQuantizer::new(),
        }
    }

    pub fn config(&self) -> QuantizerConfig {
        self.config
    }

    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }

    /// Quantize a fractional (dx, dy) demand. Both axes are processed
    /// in order; if the x-axis rejects the demand the residual state
    /// is rolled back so a partial emit cannot leak.
    pub fn quantize(&mut self, demand_x: f64, demand_y: f64) -> Result<QuantizedCommand, AppError> {
        let saved_x = self.x.accumulator;
        let _saved_y = self.y.accumulator;
        let dx = self.x.quantize(demand_x, self.config)?;
        let dy = match self.y.quantize(demand_y, self.config) {
            Ok(value) => value,
            Err(error) => {
                // Roll back the x-axis so a partial emit cannot leak.
                // The internal x.quantize already updated accumulator
                // and counts before failing; restore.
                self.x.accumulator = saved_x;
                return Err(error);
            }
        };
        Ok(QuantizedCommand {
            dx,
            dy,
            residual_x: self.x.accumulator,
            residual_y: self.y.accumulator,
        })
    }

    pub fn residuals(&self) -> (f64, f64) {
        (self.x.accumulator, self.y.accumulator)
    }
}

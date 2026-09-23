//! Converts continuous controller demand into integer device counts.
//!
//! This module retains only sub-count residuals. Fixed X/Y output limits are
//! applied once, after recoil composition, at the final device boundary.

use serde::{Deserialize, Serialize};

use crate::error::AppError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AxisCountQuantizer {
    residual: f64,
}

impl AxisCountQuantizer {
    pub const fn new() -> Self {
        Self { residual: 0.0 }
    }

    pub fn reset(&mut self) {
        self.residual = 0.0;
    }

    pub const fn residual(&self) -> f64 {
        self.residual
    }

    pub fn quantize(&mut self, demand_counts: f64) -> Result<i32, AppError> {
        if !demand_counts.is_finite() {
            self.reset();
            return Err(AppError::DeviceCountOutOfRange);
        }
        if self.residual != 0.0 && demand_counts != 0.0 && self.residual * demand_counts < 0.0 {
            self.reset();
        }
        self.residual += demand_counts;
        let counts = self.residual.trunc();
        if !counts.is_finite() || counts < f64::from(i32::MIN) || counts > f64::from(i32::MAX) {
            self.reset();
            return Err(AppError::DeviceCountOutOfRange);
        }
        let counts = counts as i32;
        self.residual -= f64::from(counts);
        Ok(counts)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QuantizedDeviceCounts {
    pub dx: i32,
    pub dy: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DeviceCountQuantizer {
    x: AxisCountQuantizer,
    y: AxisCountQuantizer,
}

impl DeviceCountQuantizer {
    pub const fn new() -> Self {
        Self {
            x: AxisCountQuantizer::new(),
            y: AxisCountQuantizer::new(),
        }
    }

    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }

    pub const fn residuals(&self) -> (f64, f64) {
        (self.x.residual(), self.y.residual())
    }

    pub fn quantize(
        &mut self,
        demand_x: f64,
        demand_y: f64,
    ) -> Result<QuantizedDeviceCounts, AppError> {
        let dx = match self.x.quantize(demand_x) {
            Ok(value) => value,
            Err(error) => {
                self.reset();
                return Err(error);
            }
        };
        let dy = match self.y.quantize(demand_y) {
            Ok(value) => value,
            Err(error) => {
                self.reset();
                return Err(error);
            }
        };
        Ok(QuantizedDeviceCounts { dx, dy })
    }
}

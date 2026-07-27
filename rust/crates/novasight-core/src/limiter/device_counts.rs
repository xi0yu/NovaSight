use serde::{Deserialize, Serialize};

use crate::error::AppError;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceCountLimits {
    pub max_counts_per_axis: f64,
    pub residual_cap: f64,
}

impl DeviceCountLimits {
    fn valid(self) -> bool {
        self.max_counts_per_axis.is_finite()
            && (1.0..=f64::from(i16::MAX)).contains(&self.max_counts_per_axis)
            && self.residual_cap.is_finite()
            && (0.0..=1.0).contains(&self.residual_cap)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AxisCountLimiter {
    residual: f64,
}

impl AxisCountLimiter {
    pub const fn new() -> Self {
        Self { residual: 0.0 }
    }

    pub fn reset(&mut self) {
        self.residual = 0.0;
    }

    pub const fn residual(&self) -> f64 {
        self.residual
    }

    pub fn limit(
        &mut self,
        demand_counts: f64,
        limits: DeviceCountLimits,
    ) -> Result<i32, AppError> {
        if !demand_counts.is_finite() || !limits.valid() {
            self.reset();
            return Err(AppError::DeviceCountOutOfRange);
        }
        let demand_counts =
            demand_counts.clamp(-limits.max_counts_per_axis, limits.max_counts_per_axis);
        if self.residual != 0.0 && demand_counts != 0.0 && self.residual * demand_counts < 0.0 {
            self.reset();
        }
        self.residual += demand_counts;
        let integer_limit = limits.max_counts_per_axis.ceil();
        let counts = self.residual.trunc().clamp(-integer_limit, integer_limit);
        if !counts.is_finite() || counts < f64::from(i32::MIN) || counts > f64::from(i32::MAX) {
            self.reset();
            return Err(AppError::DeviceCountOutOfRange);
        }
        let counts = counts as i32;
        self.residual -= f64::from(counts);
        self.residual = self
            .residual
            .clamp(-limits.residual_cap, limits.residual_cap);
        Ok(counts)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LimitedDeviceCounts {
    pub dx: i32,
    pub dy: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DeviceCountLimiter {
    x: AxisCountLimiter,
    y: AxisCountLimiter,
}

impl DeviceCountLimiter {
    pub const fn new() -> Self {
        Self {
            x: AxisCountLimiter::new(),
            y: AxisCountLimiter::new(),
        }
    }

    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }

    pub fn reset_x(&mut self) {
        self.x.reset();
    }

    pub fn reset_y(&mut self) {
        self.y.reset();
    }

    pub const fn residuals(&self) -> (f64, f64) {
        (self.x.residual(), self.y.residual())
    }

    pub fn limit(
        &mut self,
        demand_x: f64,
        demand_y: f64,
        limits: DeviceCountLimits,
    ) -> Result<LimitedDeviceCounts, AppError> {
        let dx = match self.x.limit(demand_x, limits) {
            Ok(value) => value,
            Err(error) => {
                self.reset();
                return Err(error);
            }
        };
        let dy = match self.y.limit(demand_y, limits) {
            Ok(value) => value,
            Err(error) => {
                self.reset();
                return Err(error);
            }
        };
        Ok(LimitedDeviceCounts { dx, dy })
    }
}

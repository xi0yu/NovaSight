//! Converts continuous controller demand into device-executable counts.
//!
//! The controller owns target response. This module owns per-update limits,
//! integer quantization and fractional residual state. Hardware delivery and
//! stale-command replacement remain in `output`.

mod device_counts;

pub use device_counts::{AxisCountLimiter, DeviceCountLimiter, LimitedDeviceCounts};

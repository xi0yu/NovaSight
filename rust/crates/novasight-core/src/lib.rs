#![forbid(unsafe_code)]

pub mod control;
pub mod error;
pub mod output;
pub mod perception;
pub mod ports;
pub mod targeting;

pub use control::{ControlDecision, ProportionalReplayControl};
pub use error::AppError;
pub use output::{DeviceCommand, DeviceReceipt, RecordingPointerDevice};
pub use perception::{
    Detection, DetectionBatch, FrameStamp, Generation, MAX_DETECTIONS, MonotonicNanos,
    ReplayPerceptionSource, RuntimeEpoch,
};
pub use ports::{Clock, PerceptionSource, PointerDevice};
pub use targeting::{NearestCenterTargeting, SelectedTarget};

#![forbid(unsafe_code)]

pub mod control;
pub mod error;
pub mod freshness;
pub mod geometry;
pub mod output;
pub mod perception;
pub mod ports;
pub mod runtime;
pub mod targeting;
pub mod telemetry;
pub mod tracking;
pub mod units;

pub use control::{ControlDecision, ProportionalReplayControl};
pub use error::AppError;
pub use output::{DeviceCommand, DeviceReceipt, RecordingPointerDevice};
pub use perception::{
    Detection, DetectionBatch, FrameStamp, Generation, MAX_DETECTIONS, MonotonicNanos,
    ReplayPerceptionSource, RuntimeEpoch,
};
pub use ports::{Clock, PerceptionSource, PointerDevice};
pub use runtime::{
    OperationalSnapshot, RunIntent, RuntimeAlgorithm, RuntimeCommandReceipt, RuntimeDependencies,
    RuntimeHandle, RuntimeManager, RuntimePhase,
};
pub use targeting::{NearestCenterTargeting, SelectedTarget};
pub use telemetry::ErrorSnapshot;

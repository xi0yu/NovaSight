#![forbid(unsafe_code)]

pub mod capture;
pub mod control;
pub mod error;
pub mod freshness;
pub mod geometry;
pub mod output;
pub mod perception;
pub mod ports;
pub mod targeting;
pub mod tracking;
pub mod units;

pub use capture::{
    CaptureCapabilities, CaptureCapability, CaptureCapabilityProbe, CaptureProbeError,
    CaptureSelectionError, CaptureSelectionPreference, SelectedCaptureProfile,
    select_capture_profile,
};
pub use control::{ControlDecision, ProportionalReplayControl};
pub use error::AppError;
pub use output::{
    DeviceCommand, DeviceReceipt, RecordingPointerDevice, UncommissionedPointerDevice,
};
pub use perception::{
    Detection, DetectionBatch, FrameStamp, Generation, MAX_DETECTIONS, MonotonicNanos,
    ReplayPerceptionSource, RuntimeEpoch,
};
pub use ports::{Clock, PerceptionSource, PointerButtons, PointerDevice, PointerDeviceMode};
pub use targeting::{NearestCenterTargeting, SelectedTarget};

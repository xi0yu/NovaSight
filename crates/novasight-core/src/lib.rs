#![forbid(unsafe_code)]

#[cfg(feature = "diagnostics")]
pub mod assessment;
pub mod capture;
pub mod controller;
pub mod error;
pub mod freshness;
pub mod geometry;
pub mod output;
pub mod perception;
pub mod ports;
pub mod prediction;
pub mod quantizer;
#[cfg(feature = "replay-tools")]
pub mod targeting;
pub mod tracking;
pub mod units;

#[cfg(feature = "diagnostics")]
pub use assessment::{
    AlgorithmScore, AlgorithmScoreConfig, AlgorithmTraceSample, CountResponseLagEstimate,
    CountResponseLagScore, CountResponseModel, PredictionTruthConfig, PredictionTruthHorizonScore,
    PredictionTruthMotionClass, PredictionTruthMotionClassScore, PredictionTruthProjection,
    PredictionTruthReport, PredictionTruthSample, estimate_count_response_lag,
    score_algorithm_trace, score_prediction_truth,
};
pub use capture::{
    CaptureCapabilities, CaptureCapability, CaptureCapabilityProbe, CaptureFrameRate,
    CaptureProbeError, CaptureSelectionError, CaptureSelectionPreference, SelectedCaptureProfile,
    select_capture_profile, select_capture_profile_for_formats,
};
#[cfg(feature = "replay-tools")]
pub use controller::replay::{ControlDecision, ProportionalReplayControl};
pub use error::AppError;
pub use output::{
    DeviceCommand, DeviceReceipt, RecordingPointerDevice, UncommissionedPointerDevice,
};
pub use perception::{
    Detection, DetectionBatch, FrameStamp, Generation, MAX_DETECTIONS, MonotonicNanos,
    ReplayPerceptionSource, RuntimeEpoch,
};
pub use ports::{Clock, PerceptionSource, PointerButtons, PointerDevice, PointerDeviceMode};
#[cfg(feature = "replay-tools")]
pub use targeting::{NearestCenterTargeting, SelectedTarget};

#![forbid(unsafe_code)]

pub mod assessment;
pub mod capture;
pub mod controller;
pub mod error;
pub mod freshness;
pub mod geometry;
pub mod limiter;
pub mod output;
pub mod perception;
pub mod ports;
pub mod prediction;
pub mod targeting;
pub mod tracking;
pub mod units;

pub use assessment::{
    AlgorithmScore, AlgorithmScoreConfig, AlgorithmTraceSample, CountResponseLagEstimate,
    CountResponseLagScore, CountResponseModel, PredictionTruthConfig, PredictionTruthHorizonScore,
    PredictionTruthMotionClass, PredictionTruthMotionClassScore, PredictionTruthProjection,
    PredictionTruthReport, PredictionTruthSample, estimate_count_response_lag,
    score_algorithm_trace, score_prediction_truth,
};
pub use capture::{
    CaptureCapabilities, CaptureCapability, CaptureCapabilityProbe, CaptureProbeError,
    CaptureSelectionError, CaptureSelectionPreference, SelectedCaptureProfile,
    select_capture_profile, select_capture_profile_for_formats,
};
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
pub use targeting::{NearestCenterTargeting, SelectedTarget};

#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod frame;
mod pipeline;
#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod session;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use frame::{FrameLease, FrameLeaseError, LatestFrameExchange};
pub use pipeline::{
    CaptureFormat, CaptureProfile, DeepStreamPipelineSpec, ModelInput, PipelineSpecError, Roi,
};
#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use session::{
    DeepStreamAdapter, DeepStreamSession, DeepStreamSessionConfig, SessionError, SessionEvent,
    SessionMetrics,
};

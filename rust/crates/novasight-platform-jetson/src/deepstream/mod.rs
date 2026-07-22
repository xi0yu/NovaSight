#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod frame;
mod pipeline;
#[cfg(all(feature = "cuda-preprocess", target_os = "linux"))]
mod preprocess;
#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod session;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use frame::{FrameLease, FrameLeaseError, LatestFrameExchange};
#[cfg(all(feature = "cuda-preprocess", target_os = "linux"))]
pub use novasight_jetson_preprocess::{DeviceTensor, PreprocessError, TensorContract, TensorDtype};
pub use pipeline::{
    CaptureFormat, CaptureProfile, DeepStreamPipelineSpec, ModelInput, PipelineSpecError, Roi,
};
#[cfg(all(feature = "cuda-preprocess", target_os = "linux"))]
pub use preprocess::CudaFramePreprocessor;
#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use session::{
    DeepStreamAdapter, DeepStreamSession, DeepStreamSessionConfig, SessionError, SessionEvent,
    SessionMetrics,
};

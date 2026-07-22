#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod frame;
#[cfg(all(feature = "tensorrt", target_os = "linux"))]
mod inference;
mod pipeline;
#[cfg(all(feature = "cuda-preprocess", target_os = "linux"))]
mod preprocess;
#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod session;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use frame::{FrameLease, FrameLeaseError, LatestFrameExchange};
#[cfg(all(feature = "tensorrt", target_os = "linux"))]
pub use inference::{CudaTensorRtOwner, CudaTensorRtOwnerError};
#[cfg(all(feature = "cuda-preprocess", target_os = "linux"))]
pub use novasight_jetson_preprocess::{DeviceTensor, PreprocessError, TensorContract, TensorDtype};
#[cfg(all(feature = "tensorrt", target_os = "linux"))]
pub use novasight_tensorrt::{EngineContract, ExecutionOutputs, HostTensor, TensorRtError};
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

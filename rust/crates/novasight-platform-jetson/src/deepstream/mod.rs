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
pub use inference::{CudaTensorRtConfig, CudaTensorRtOwner, CudaTensorRtOwnerError};
#[cfg(all(feature = "cuda-preprocess", target_os = "linux"))]
pub use novasight_jetson_preprocess::{DeviceTensor, PreprocessError, TensorContract, TensorDtype};
#[cfg(all(feature = "tensorrt", target_os = "linux"))]
pub use novasight_tensorrt::{
    DecodeContract, DecodeError, DetectionDecoder, EngineContract, TensorRtError,
};
pub use pipeline::{
    CaptureFormat, CaptureProfile, DeepStreamPipelineSpec, InferenceStage, ModelInput,
    PipelineSpecError, PreviewPipelineConfig, Roi,
};
#[cfg(all(feature = "cuda-preprocess", target_os = "linux"))]
pub use preprocess::CudaFramePreprocessor;
#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use session::{
    DeepStreamAdapter, DeepStreamSession, DeepStreamSessionConfig, SessionError, SessionEvent,
    SessionMetrics, preflight_deepstream_runtime,
};

#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod frame;
#[cfg(feature = "deepstream")]
mod model_contract;
mod pipeline;
#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod session;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use frame::{FrameLease, FrameLeaseError, LatestFrameExchange};
#[cfg(feature = "deepstream")]
pub use model_contract::{
    DeepStreamModelContractError, DeepStreamParserContract, deepstream_parser_contract,
    render_deepstream_nvinfer_config,
};
pub use pipeline::{
    CaptureFormat, CaptureProfile, CrosshairPipelineConfig, DeepStreamPipelineSpec, InferenceStage,
    ModelInput, PipelineSpecError, PreviewPipelineConfig, Roi,
};
#[cfg(all(feature = "deepstream", target_os = "linux"))]
pub use session::{
    DeepStreamAdapter, DeepStreamSession, DeepStreamSessionConfig, SessionError, SessionEvent,
    SessionMetrics, preflight_deepstream_runtime,
};

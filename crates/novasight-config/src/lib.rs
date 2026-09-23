#![deny(unsafe_code)]

mod endpoints;
mod model;
mod repository;

pub use endpoints::{NetworkEndpoint, StudioEndpointContract, studio_endpoint_contract};
pub use model::{
    AppConfig, CURRENT_SCHEMA_VERSION, CaptureConfig, CaptureMemory, CapturePreference,
    ComputeDevice, ConfigValidationError, ConsumerConfig, CrosshairConfig, DeepStreamBackend,
    DeviceBackend, DeviceConfig, InferenceBackend, InferenceConfig, InferenceInputSource,
    LimitsConfig, PathConfig, PipelineRuntimeConfig, ProductionAdapterConfig, QueueLeaky,
    RecoilConfig, ReplayConfig, ServerConfig, TriggerMode, VisionAdapterConfig,
    parse_target_class_aim_y_ratios, parse_target_class_filter, parse_target_class_priority,
};
pub use repository::{ConfigError, ConfigRepository, YamlConfigRepository};

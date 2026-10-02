#![deny(unsafe_code)]

mod class_presets;
mod endpoints;
mod model;
mod repository;

pub use class_presets::{ClassPreset, builtin_class_presets};
pub use endpoints::{NetworkEndpoint, StudioEndpointContract, studio_endpoint_contract};
pub use model::{
    AppConfig, CURRENT_SCHEMA_VERSION, CaptureConfig, CaptureMemory, CapturePreference,
    ComputeDevice, ConfigValidationError, ConsumerConfig, DeepStreamBackend, DeviceBackend,
    DeviceConfig, InferenceBackend, InferenceConfig, InferenceInputSource, LimitsConfig,
    PathConfig, PipelineRuntimeConfig, ProductionAdapterConfig, QueueLeaky, ReplayConfig,
    ServerConfig, TriggerMode, VisionAdapterConfig, parse_class_values,
    parse_target_class_aim_y_ratios, parse_target_class_filter,
};
pub use repository::{ConfigError, ConfigRepository, YamlConfigRepository};

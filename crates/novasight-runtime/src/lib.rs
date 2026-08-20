//! `novasight-runtime` — daemon lifecycle, RuntimeSupervisor, configuration
//! transactions, model activation, PipelineRuntime orchestration, and immutable
//! runtime snapshots.
//!
//! Runtime commands are serialized through one supervisor actor. HTTP and other
//! control surfaces use `RuntimeHandle` and must not become parallel runtime
//! authorities.

#![forbid(unsafe_code)]

mod application;
mod command;
mod config_service;
mod error;
mod model_activation;
mod model_ingress;
#[cfg(feature = "tensorrt-model-ingress")]
mod model_ingress_native;
mod protocol;
mod runtime_config;
pub mod snapshot;
mod state;
pub mod supervisor;

pub use application::{Application, ApplicationError, LoadedApplication};
pub use config_service::{
    ConfigApplyMode, ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate,
};
pub use error::{RuntimeError, RuntimeErrorKind};
pub use model_activation::{ModelActivationError, ModelActivationRequest, ModelActivationResult};
pub use model_ingress::{
    ModelIngressError, ModelIngressRequest, ModelIngressResult, ModelJobRunner,
    ModelProbeInputMode, ModelProfileConfigureRequest,
};
#[cfg(feature = "tensorrt-model-ingress")]
pub use model_ingress_native::NativeModelJobRunner;
pub use novasight_core::RuntimeEpoch;
pub use novasight_pipeline::{
    CrosshairSnapshot, CrosshairTemplateSummary, DetectionTelemetryItem, OutputDeliveryState,
    PreviewFrame, PreviewHub, PreviewSnapshot, PreviewSubscription,
};
pub use novasight_store::config::AppConfig;
pub use protocol::{RuntimeErrorSummary, SubsystemState};
pub use runtime_config::compose_pipeline_config;
pub use snapshot::{
    DaemonSnapshot, DeviceMetrics, ModelSnapshot, PipelineSnapshot, RuntimeSnapshot,
    RuntimeTelemetrySnapshot, SubsystemSnapshot, SubsystemSnapshots,
};
pub use state::{DaemonState, PipelineState};
pub use supervisor::{
    PointerDeviceInstallation, RuntimeDependencies, RuntimeHandle, RuntimeSupervisor,
};

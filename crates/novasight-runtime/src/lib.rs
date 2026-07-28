//! `novasight-runtime` — daemon lifecycle, RuntimeSupervisor,
//! PipelineRuntime orchestration, and API server wiring.
//!
//! Commit 2 introduces the RuntimeSupervisor, RuntimeHandle,
//! internal RuntimeCommand actor protocol and public RuntimeSnapshot type
//! families. The
//! PipelineRuntime, ConfigService, and HTTP transport land in
//! later commits; `ReplaceConfig` is intentionally absent from
//! the command enum until the ConfigService surface exists.

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
pub use config_service::{ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate};
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
    CrosshairSnapshot, CrosshairTemplateSummary, DetectionTelemetryItem, PreviewFrame, PreviewHub,
    PreviewSnapshot, PreviewSubscription,
};
pub use novasight_store::config::AppConfig;
pub use protocol::{RuntimeErrorSummary, SubsystemState};
pub use runtime_config::compose_pipeline_config;
pub use snapshot::{
    DaemonSnapshot, DeviceMetrics, ModelSnapshot, PipelineSnapshot, RuntimeSnapshot,
    RuntimeTelemetrySnapshot, SubsystemSnapshot, SubsystemSnapshots,
};
pub use state::{DaemonState, PipelineState};
pub use supervisor::{RuntimeDependencies, RuntimeHandle, RuntimeSupervisor};

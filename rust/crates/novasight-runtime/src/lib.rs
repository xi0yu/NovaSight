//! `novasight-runtime` — daemon lifecycle, RuntimeSupervisor,
//! PipelineRuntime orchestration, and API server wiring.
//!
//! Commit 2 introduces the RuntimeSupervisor, RuntimeHandle,
//! RuntimeCommand, and RuntimeSnapshot type families. The
//! PipelineRuntime, ConfigService, and HTTP transport land in
//! later commits; `ReplaceConfig` is intentionally absent from
//! the command enum until the ConfigService surface exists.

#![forbid(unsafe_code)]

mod application;
pub mod command;
mod config_service;
mod error;
mod protocol;
pub mod snapshot;
mod state;
pub mod supervisor;

pub use application::{Application, ApplicationError, LoadedApplication};
pub use command::RuntimeCommand;
pub use config_service::{ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate};
pub use error::{RuntimeError, RuntimeErrorKind};
pub use novasight_core::RuntimeEpoch;
pub use novasight_store::config::AppConfig;
pub use protocol::{RuntimeErrorSummary, SubsystemState};
pub use snapshot::{
    DaemonSnapshot, DeviceMetrics, PipelineSnapshot, RuntimeSnapshot, SubsystemSnapshot,
    SubsystemSnapshots,
};
pub use state::{DaemonState, PipelineState};
pub use supervisor::{RuntimeDependencies, RuntimeHandle, RuntimeSupervisor};

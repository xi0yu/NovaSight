//! Runtime command enum. Each variant carries a typed
//! `oneshot::Sender<...>` reply channel; supervisors must respond
//! to every command, even on failure, so the caller never blocks
//! waiting for a reply that will never come.
//!
//! Commands are added only with a concrete owner and reply contract. Device
//! diagnostics are serialized here because they must never compete with the
//! live DeviceLane. Hot output configuration is also serialized here so a
//! persisted revision and its physical gate transition have one owner.

use tokio::sync::oneshot;

use novasight_core::DeviceReceipt;
use novasight_store::config::AppConfig;

use crate::config_service::{ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate};
use crate::error::RuntimeError;
use crate::model_activation::{
    ModelActivationError, ModelActivationRequest, ModelActivationResult,
};
use crate::model_ingress::{ModelIngressError, ModelIngressRequest, ModelIngressResult};
use crate::snapshot::RuntimeSnapshot;
use crate::supervisor::UrgentStopToken;
use novasight_pipeline::{PreviewSnapshot, TriggerMode};

#[derive(Debug)]
pub(crate) enum RuntimeCommand {
    Start {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    Stop {
        urgent: UrgentStopToken,
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    Restart {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    InstallStoppedConfig {
        config: Box<AppConfig>,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    PreflightPerception {
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    ActivateModel {
        request: ModelActivationRequest,
        reply: oneshot::Sender<Result<ModelActivationResult, ModelActivationError>>,
    },
    ModelIngress {
        request: ModelIngressRequest,
        reply: oneshot::Sender<Result<ModelIngressResult, ModelIngressError>>,
    },
    EmergencyStop {
        urgent: UrgentStopToken,
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    SetTriggerActive {
        active: bool,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    SetTriggerMode {
        mode: TriggerMode,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    UpdateOutputConfig {
        service: ConfigService,
        update: ConfigFieldUpdate,
        reply: oneshot::Sender<Result<ConfigUpdate, ConfigServiceError>>,
    },
    UpdateTriggerModeConfig {
        service: ConfigService,
        update: ConfigFieldUpdate,
        reply: oneshot::Sender<Result<ConfigUpdate, ConfigServiceError>>,
    },
    UpdateRecoilConfig {
        service: ConfigService,
        update: ConfigFieldUpdate,
        reply: oneshot::Sender<Result<ConfigUpdate, ConfigServiceError>>,
    },
    SetPreviewActive {
        active: bool,
        reply: oneshot::Sender<Result<PreviewSnapshot, RuntimeError>>,
    },
    DiagnoseDeviceMove {
        delta_x_counts: i32,
        delta_y_counts: i32,
        reply: oneshot::Sender<Result<DeviceReceipt, RuntimeError>>,
    },
    ConnectDevice {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    DisconnectDevice {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    ShutdownDaemon {
        urgent: UrgentStopToken,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
}

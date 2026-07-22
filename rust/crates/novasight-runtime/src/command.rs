//! Runtime command enum. Each variant carries a typed
//! `oneshot::Sender<...>` reply channel; supervisors must respond
//! to every command, even on failure, so the caller never blocks
//! waiting for a reply that will never come.
//!
//! Commands are added only with a concrete owner and reply contract. Device
//! diagnostics are serialized here because they must never compete with the
//! live DeviceLane; config persistence remains owned by ConfigService.

use tokio::sync::oneshot;

use novasight_core::DeviceReceipt;

use crate::error::RuntimeError;
use crate::snapshot::RuntimeSnapshot;

#[derive(Debug)]
pub enum RuntimeCommand {
    Start {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    Stop {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    Restart {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    EmergencyStop {
        reply: oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>,
    },
    SetTriggerActive {
        active: bool,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    DiagnoseDeviceMove {
        delta_x_counts: i32,
        delta_y_counts: i32,
        reply: oneshot::Sender<Result<DeviceReceipt, RuntimeError>>,
    },
    ShutdownDaemon {
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
}

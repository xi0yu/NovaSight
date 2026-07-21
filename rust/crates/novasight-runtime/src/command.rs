//! Runtime command enum. Each variant carries a typed
//! `oneshot::Sender<...>` reply channel; supervisors must respond
//! to every command, even on failure, so the caller never blocks
//! waiting for a reply that will never come.
//!
//! `ReplaceConfig`, `ReloadConfig`, `ValidateConfig`, and
//! `Diagnose` are intentionally absent in Commit 2. They will land
//! in Commit 7 or Commit 8 once the ConfigService surface exists;
//! pre-allocating them now would create fake interfaces.

use tokio::sync::oneshot;

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
    ShutdownDaemon {
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
}

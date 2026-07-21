//! Runtime command actor and its cloneable handle.
//!
//! Commit 2 deliberately owns only an in-memory lifecycle state
//! machine. A later commit can put a pipeline behind the same typed
//! command boundary without exposing workers to control-plane callers.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use novasight_core::RuntimeEpoch;
use tokio::sync::{mpsc, oneshot, watch};

use crate::command::RuntimeCommand;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::protocol::SubsystemState;
use crate::snapshot::{DaemonSnapshot, PipelineSnapshot, RuntimeSnapshot, SubsystemSnapshots};
use crate::state::{DaemonState, PipelineState};

const COMMAND_CAPACITY: usize = 32;

#[derive(Clone, Debug)]
struct SupervisorState {
    daemon: DaemonState,
    pipeline: PipelineState,
    pipeline_epoch: Option<RuntimeEpoch>,
    next_epoch: u64,
    pipeline_started_at_ms: Option<u64>,
    subsystems: SubsystemSnapshots,
    started_at_unix_ms: u64,
}

impl Default for SupervisorState {
    fn default() -> Self {
        Self {
            daemon: DaemonState::Starting,
            pipeline: PipelineState::Stopped,
            pipeline_epoch: None,
            next_epoch: 0,
            pipeline_started_at_ms: None,
            subsystems: SubsystemSnapshots::default(),
            started_at_unix_ms: now_ms(),
        }
    }
}

impl SupervisorState {
    fn snapshot(&self, updated_at_ms: u64) -> RuntimeSnapshot {
        RuntimeSnapshot {
            daemon: DaemonSnapshot {
                state: self.daemon,
                version: env!("CARGO_PKG_VERSION").to_owned(),
                uptime_ms: updated_at_ms.saturating_sub(self.started_at_unix_ms),
            },
            pipeline: PipelineSnapshot {
                state: self.pipeline,
                epoch: self.pipeline_epoch,
                started_at_ms: self.pipeline_started_at_ms,
                last_error: None,
            },
            subsystems: self.subsystems.clone(),
            updated_at_ms,
        }
    }

    fn begin_start(&mut self) -> Result<bool, RuntimeError> {
        match self.pipeline {
            PipelineState::Running | PipelineState::Starting | PipelineState::Standby => {
                return Ok(false);
            }
            PipelineState::Stopping => {
                return Err(RuntimeError::invalid_pipeline_state(
                    "start requested while the pipeline is stopping",
                ));
            }
            PipelineState::Stopped | PipelineState::Faulted => {}
        }

        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or_else(RuntimeError::runtime_epoch_exhausted)?;
        self.pipeline_epoch = Some(RuntimeEpoch(self.next_epoch));
        self.pipeline = PipelineState::Starting;
        self.pipeline_started_at_ms = None;
        self.subsystems.capture.state = SubsystemState::Starting;
        self.subsystems.inference.state = SubsystemState::Starting;
        self.subsystems.control.state = SubsystemState::Starting;
        self.subsystems.device.state = SubsystemState::Starting;
        Ok(true)
    }

    fn finish_start(&mut self, started_at_ms: u64) {
        self.pipeline = PipelineState::Running;
        self.pipeline_started_at_ms = Some(started_at_ms);
        self.subsystems.capture.state = SubsystemState::Running;
        self.subsystems.inference.state = SubsystemState::Running;
        self.subsystems.control.state = SubsystemState::Running;
        self.subsystems.device.state = SubsystemState::Ready;
    }

    fn begin_stop(&mut self) -> bool {
        if self.pipeline == PipelineState::Stopped {
            return false;
        }

        self.pipeline = PipelineState::Stopping;
        self.subsystems.capture.state = SubsystemState::Stopping;
        self.subsystems.inference.state = SubsystemState::Stopping;
        self.subsystems.control.state = SubsystemState::Stopping;
        self.subsystems.device.state = SubsystemState::Stopping;
        true
    }

    fn finish_stop(&mut self) {
        self.pipeline = PipelineState::Stopped;
        self.pipeline_started_at_ms = None;
        self.subsystems.capture.state = SubsystemState::Stopped;
        self.subsystems.inference.state = SubsystemState::Stopped;
        self.subsystems.control.state = SubsystemState::Stopped;
        self.subsystems.device.state = SubsystemState::Stopped;
    }

    fn emergency_stop(&mut self, stopped_at_ms: u64) {
        self.finish_stop();
        self.subsystems.device.state = SubsystemState::Unavailable;
        self.subsystems.control.last_error = Some(crate::protocol::RuntimeErrorSummary::new(
            "emergency_stop",
            format!("emergency stop activated at {stopped_at_ms}"),
        ));
        self.subsystems.control.restart_count =
            self.subsystems.control.restart_count.saturating_add(1);
    }
}

/// Owns the supervisor task. Runtime commands are issued through the
/// paired [`RuntimeHandle`].
pub struct RuntimeSupervisor {
    join: Option<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for RuntimeSupervisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeSupervisor")
            .field("running", &self.join.is_some())
            .finish()
    }
}

impl RuntimeSupervisor {
    /// Spawn the command actor on the current Tokio runtime.
    pub fn spawn() -> (Self, RuntimeHandle) {
        let state = SupervisorState::default();
        let initial_snapshot = Arc::new(state.snapshot(now_ms()));
        let (snapshot_tx, snapshot_rx) = watch::channel(initial_snapshot);
        let (command_tx, command_rx) = mpsc::channel(COMMAND_CAPACITY);
        let join = tokio::spawn(supervisor_loop(command_rx, snapshot_tx, state));

        (
            Self { join: Some(join) },
            RuntimeHandle {
                command_tx,
                snapshot_rx,
            },
        )
    }

    /// Await actor termination after `ShutdownDaemon` or after the
    /// final [`RuntimeHandle`] has been dropped.
    pub async fn join(mut self) -> Result<(), RuntimeError> {
        let join = self
            .join
            .take()
            .ok_or_else(RuntimeError::supervisor_unavailable)?;
        join.await.map_err(|error| {
            RuntimeError::new(
                RuntimeErrorKind::SupervisorUnavailable,
                format!("runtime supervisor task failed: {error}"),
            )
        })
    }
}

impl Drop for RuntimeSupervisor {
    fn drop(&mut self) {
        if let Some(join) = &self.join {
            join.abort();
        }
    }
}

/// Cheap clone of the command sender and latest immutable snapshot.
#[derive(Clone)]
pub struct RuntimeHandle {
    command_tx: mpsc::Sender<RuntimeCommand>,
    snapshot_rx: watch::Receiver<Arc<RuntimeSnapshot>>,
}

impl std::fmt::Debug for RuntimeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeHandle")
            .field("command_tx", &"<mpsc::Sender>")
            .field("snapshot_rx", &"<watch::Receiver>")
            .finish()
    }
}

impl RuntimeHandle {
    /// Return the most recently published immutable snapshot.
    pub fn snapshot(&self) -> RuntimeSnapshot {
        self.snapshot_rx.borrow().as_ref().clone()
    }

    /// Subscribe to latest-only runtime snapshots.
    pub fn subscribe(&self) -> watch::Receiver<Arc<RuntimeSnapshot>> {
        self.snapshot_rx.clone()
    }

    pub async fn start(&self) -> Result<RuntimeSnapshot, RuntimeError> {
        self.send_command(|reply| RuntimeCommand::Start { reply })
            .await
    }

    pub async fn stop(&self) -> Result<RuntimeSnapshot, RuntimeError> {
        self.send_command(|reply| RuntimeCommand::Stop { reply })
            .await
    }

    pub async fn restart(&self) -> Result<RuntimeSnapshot, RuntimeError> {
        self.send_command(|reply| RuntimeCommand::Restart { reply })
            .await
    }

    pub async fn emergency_stop(&self) -> Result<RuntimeSnapshot, RuntimeError> {
        self.send_command(|reply| RuntimeCommand::EmergencyStop { reply })
            .await
    }

    pub async fn shutdown_daemon(&self) -> Result<(), RuntimeError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::ShutdownDaemon { reply: reply_tx })
            .await
            .map_err(|_| RuntimeError::supervisor_closed())?;
        reply_rx
            .await
            .map_err(|_| RuntimeError::supervisor_reply_lost())?
    }

    async fn send_command<F>(&self, build: F) -> Result<RuntimeSnapshot, RuntimeError>
    where
        F: FnOnce(oneshot::Sender<Result<RuntimeSnapshot, RuntimeError>>) -> RuntimeCommand,
    {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(build(reply_tx))
            .await
            .map_err(|_| RuntimeError::supervisor_closed())?;
        reply_rx
            .await
            .map_err(|_| RuntimeError::supervisor_reply_lost())?
    }
}

async fn supervisor_loop(
    mut command_rx: mpsc::Receiver<RuntimeCommand>,
    snapshot_tx: watch::Sender<Arc<RuntimeSnapshot>>,
    mut state: SupervisorState,
) {
    while let Some(command) = command_rx.recv().await {
        match command {
            RuntimeCommand::Start { reply } => {
                let result = start_state(&snapshot_tx, &mut state);
                let _ = reply.send(result);
            }
            RuntimeCommand::Stop { reply } => {
                let snapshot = stop_state(&snapshot_tx, &mut state);
                let _ = reply.send(Ok(snapshot));
            }
            RuntimeCommand::Restart { reply } => {
                stop_state(&snapshot_tx, &mut state);
                let result = start_state(&snapshot_tx, &mut state);
                let _ = reply.send(result);
            }
            RuntimeCommand::EmergencyStop { reply } => {
                let timestamp = now_ms();
                state.emergency_stop(timestamp);
                let snapshot = publish(&snapshot_tx, &state, timestamp);
                let _ = reply.send(Ok(snapshot));
            }
            RuntimeCommand::ShutdownDaemon { reply } => {
                state.daemon = DaemonState::ShuttingDown;
                publish(&snapshot_tx, &state, now_ms());
                let _ = reply.send(Ok(()));
                break;
            }
        }
    }
}

fn start_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    state: &mut SupervisorState,
) -> Result<RuntimeSnapshot, RuntimeError> {
    if state.begin_start()? {
        publish(snapshot_tx, state, now_ms());
        state.finish_start(now_ms());
    }
    Ok(publish(snapshot_tx, state, now_ms()))
}

fn stop_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    state: &mut SupervisorState,
) -> RuntimeSnapshot {
    if state.begin_stop() {
        publish(snapshot_tx, state, now_ms());
        state.finish_stop();
    }
    publish(snapshot_tx, state, now_ms())
}

fn publish(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    state: &SupervisorState,
    timestamp: u64,
) -> RuntimeSnapshot {
    let snapshot = state.snapshot(timestamp);
    snapshot_tx.send_replace(Arc::new(snapshot.clone()));
    snapshot
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

//! Sole owner of the live [`PipelineRuntime`] lifecycle.
//!
//! Control-plane callers send typed commands through [`RuntimeHandle`].
//! Detection producers submit caller-owned batches through the same
//! handle; worker objects and platform adapters never escape.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use novasight_core::{
    Clock, DetectionBatch, MonotonicNanos, PointerDevice, RecordingPointerDevice, RuntimeEpoch,
};
use novasight_pipeline::{
    PerceptionAdapter, PerceptionEvent, PerceptionMetrics, PerceptionSession, PipelineConfig,
    PipelineEvent, PipelineIngress, PipelineRuntime, PipelineStatus,
};
use tokio::sync::{mpsc, oneshot, watch};

use crate::command::RuntimeCommand;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::protocol::{RuntimeErrorSummary, SubsystemState};
use crate::snapshot::{DaemonSnapshot, PipelineSnapshot, RuntimeSnapshot, SubsystemSnapshots};
use crate::state::{DaemonState, PipelineState};

const COMMAND_CAPACITY: usize = 32;
const PIPELINE_EVENT_CAPACITY: usize = 8;
const PERCEPTION_EVENT_CAPACITY: usize = 4;

/// Concrete resources used to create each runtime epoch.
#[derive(Clone)]
pub struct RuntimeDependencies {
    clock: Arc<dyn Clock>,
    device: Arc<dyn PointerDevice>,
    pipeline: PipelineConfig,
    perception: Option<Arc<dyn PerceptionAdapter>>,
}

impl std::fmt::Debug for RuntimeDependencies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeDependencies")
            .field("pipeline", &self.pipeline)
            .field("clock", &"<dyn Clock>")
            .field("device", &"<dyn PointerDevice>")
            .field(
                "perception",
                &self.perception.as_ref().map(|_| "<dyn PerceptionAdapter>"),
            )
            .finish()
    }
}

impl RuntimeDependencies {
    pub fn new(
        clock: Arc<dyn Clock>,
        device: Arc<dyn PointerDevice>,
        pipeline: PipelineConfig,
    ) -> Self {
        Self {
            clock,
            device,
            pipeline,
            perception: None,
        }
    }

    pub fn with_perception(mut self, perception: Arc<dyn PerceptionAdapter>) -> Self {
        self.perception = Some(perception);
        self
    }

    fn pipeline_config(&self, epoch: RuntimeEpoch) -> PipelineConfig {
        PipelineConfig {
            epoch,
            ..self.pipeline.clone()
        }
    }

    /// Explicit recording adapter for tests, diagnostics, and dry-run
    /// startup. Production composition should call [`Self::new`].
    pub fn recording() -> Self {
        let clock: Arc<dyn Clock> = Arc::new(ProcessMonotonicClock::default());
        let device: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
        Self::new(clock, device, PipelineConfig::default())
    }
}

#[derive(Debug)]
struct ProcessMonotonicClock {
    origin: Instant,
}

impl Default for ProcessMonotonicClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for ProcessMonotonicClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }
}

#[derive(Clone, Debug)]
struct SupervisorState {
    daemon: DaemonState,
    pipeline: PipelineState,
    pipeline_epoch: Option<RuntimeEpoch>,
    next_epoch: u64,
    pipeline_started_at_ms: Option<u64>,
    pipeline_error: Option<RuntimeErrorSummary>,
    subsystems: SubsystemSnapshots,
    perception_metrics: PerceptionMetrics,
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
            pipeline_error: None,
            subsystems: SubsystemSnapshots::default(),
            perception_metrics: PerceptionMetrics::default(),
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
                last_error: self.pipeline_error.clone(),
            },
            subsystems: self.subsystems.clone(),
            perception_metrics: self.perception_metrics,
            updated_at_ms,
        }
    }

    fn begin_start(&mut self) -> Result<Option<RuntimeEpoch>, RuntimeError> {
        match self.pipeline {
            PipelineState::Running | PipelineState::Starting | PipelineState::Standby => {
                return Ok(None);
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
        let epoch = RuntimeEpoch(self.next_epoch);
        self.pipeline_epoch = Some(epoch);
        self.pipeline = PipelineState::Starting;
        self.pipeline_started_at_ms = None;
        self.pipeline_error = None;
        self.perception_metrics = PerceptionMetrics::default();
        self.subsystems.control.state = SubsystemState::Starting;
        self.subsystems.device.state = SubsystemState::Starting;
        Ok(Some(epoch))
    }

    fn finish_start(&mut self, started_at_ms: u64, perception_running: bool) {
        self.pipeline = PipelineState::Running;
        self.pipeline_started_at_ms = Some(started_at_ms);
        if perception_running {
            self.subsystems.capture.state = SubsystemState::Running;
            self.subsystems.inference.state = SubsystemState::Running;
        }
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

    fn finish_fault(&mut self, message: impl Into<String>) {
        let error = RuntimeErrorSummary::new("pipeline_faulted", message);
        self.pipeline = PipelineState::Faulted;
        self.pipeline_started_at_ms = None;
        self.pipeline_error = Some(error.clone());
        if self.subsystems.capture.state != SubsystemState::Stopped {
            self.subsystems.capture.state = SubsystemState::Failed;
        }
        if self.subsystems.inference.state != SubsystemState::Stopped {
            self.subsystems.inference.state = SubsystemState::Failed;
        }
        self.subsystems.control.state = SubsystemState::Failed;
        self.subsystems.device.state = SubsystemState::Unavailable;
        self.subsystems.control.last_error = Some(error);
    }

    fn emergency_stop(&mut self, stopped_at_ms: u64) {
        self.finish_stop();
        self.subsystems.device.state = SubsystemState::Unavailable;
        self.subsystems.control.last_error = Some(RuntimeErrorSummary::new(
            "emergency_stop",
            format!("emergency stop activated at {stopped_at_ms}"),
        ));
        self.subsystems.control.restart_count =
            self.subsystems.control.restart_count.saturating_add(1);
    }
}

struct ActivePipeline {
    epoch: RuntimeEpoch,
    runtime: PipelineRuntime,
    ingress: PipelineIngress,
    perception: Option<Box<dyn PerceptionSession>>,
    event_bridge: tokio::task::JoinHandle<()>,
    perception_event_bridge: Option<tokio::task::JoinHandle<()>>,
    perception_event_cancel: Option<Arc<AtomicBool>>,
}

#[derive(Debug)]
struct PipelineNotice {
    epoch: RuntimeEpoch,
    event: PipelineEvent,
}

/// Owns the supervisor task. Runtime commands are issued through the
/// paired [`RuntimeHandle`].
pub struct RuntimeSupervisor {
    join: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: watch::Sender<bool>,
}

impl std::fmt::Debug for RuntimeSupervisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeSupervisor")
            .field("running", &self.join.is_some())
            .finish()
    }
}

impl RuntimeSupervisor {
    /// Spawn the sole lifecycle actor with production-selected adapters.
    pub fn spawn(dependencies: RuntimeDependencies) -> (Self, RuntimeHandle) {
        let state = SupervisorState {
            daemon: DaemonState::Ready,
            ..SupervisorState::default()
        };
        let initial_snapshot = Arc::new(state.snapshot(now_ms()));
        let (snapshot_tx, snapshot_rx) = watch::channel(initial_snapshot);
        let (ingress_tx, ingress_rx) = watch::channel(None::<PipelineIngress>);
        let (command_tx, command_rx) = mpsc::channel(COMMAND_CAPACITY);
        let (notice_tx, notice_rx) = mpsc::channel(PIPELINE_EVENT_CAPACITY);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let join = tokio::spawn(supervisor_loop(
            command_rx,
            notice_rx,
            notice_tx,
            shutdown_rx,
            snapshot_tx,
            ingress_tx,
            state,
            dependencies,
        ));

        (
            Self {
                join: Some(join),
                shutdown_tx,
            },
            RuntimeHandle {
                command_tx,
                snapshot_rx,
                ingress_rx,
            },
        )
    }

    pub fn spawn_recording() -> (Self, RuntimeHandle) {
        Self::spawn(RuntimeDependencies::recording())
    }

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
        if self.join.is_some() {
            self.shutdown_tx.send_replace(true);
        }
    }
}

/// Cheap clone of the command sender and latest immutable state.
#[derive(Clone)]
pub struct RuntimeHandle {
    command_tx: mpsc::Sender<RuntimeCommand>,
    snapshot_rx: watch::Receiver<Arc<RuntimeSnapshot>>,
    ingress_rx: watch::Receiver<Option<PipelineIngress>>,
}

impl std::fmt::Debug for RuntimeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeHandle")
            .field("command_tx", &"<mpsc::Sender>")
            .field("snapshot_rx", &"<watch::Receiver>")
            .field("ingress_rx", &"<watch::Receiver>")
            .finish()
    }
}

impl RuntimeHandle {
    pub fn snapshot(&self) -> RuntimeSnapshot {
        self.snapshot_rx.borrow().as_ref().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<Arc<RuntimeSnapshot>> {
        self.snapshot_rx.clone()
    }

    pub fn submit_detection_batch(&self, batch: DetectionBatch) -> Result<(), RuntimeError> {
        let ingress = self
            .ingress_rx
            .borrow()
            .clone()
            .ok_or_else(RuntimeError::pipeline_unavailable)?;
        ingress
            .submit(batch)
            .map_err(|error| RuntimeError::pipeline_rejected(error.to_string()))
    }

    pub async fn set_trigger_active(&self, active: bool) -> Result<(), RuntimeError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::SetTriggerActive {
                active,
                reply: reply_tx,
            })
            .await
            .map_err(|_| RuntimeError::supervisor_closed())?;
        reply_rx
            .await
            .map_err(|_| RuntimeError::supervisor_reply_lost())?
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

#[allow(clippy::too_many_arguments)]
async fn supervisor_loop(
    mut command_rx: mpsc::Receiver<RuntimeCommand>,
    mut notice_rx: mpsc::Receiver<PipelineNotice>,
    notice_tx: mpsc::Sender<PipelineNotice>,
    mut shutdown_rx: watch::Receiver<bool>,
    snapshot_tx: watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: watch::Sender<Option<PipelineIngress>>,
    mut state: SupervisorState,
    dependencies: RuntimeDependencies,
) {
    let mut active: Option<ActivePipeline> = None;
    let mut metrics_tick = tokio::time::interval(Duration::from_millis(200));
    metrics_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    shutdown_for_exit(
                        &snapshot_tx,
                        &ingress_tx,
                        &mut state,
                        &mut active,
                    ).await;
                    break;
                }
            }
            Some(notice) = notice_rx.recv() => {
                handle_pipeline_notice(
                    notice,
                    &snapshot_tx,
                    &ingress_tx,
                    &mut state,
                    &mut active,
                ).await;
            }
            _ = metrics_tick.tick(), if active.is_some() => {
                let metrics = active
                    .as_ref()
                    .and_then(|pipeline| pipeline.perception.as_ref())
                    .map(|perception| perception.metrics())
                    .unwrap_or_default();
                if metrics != state.perception_metrics {
                    state.perception_metrics = metrics;
                    publish(&snapshot_tx, &state, now_ms());
                }
            }
            command = command_rx.recv() => {
                let Some(command) = command else { break };
                if handle_command(
                    command,
                    &snapshot_tx,
                    &ingress_tx,
                    &notice_tx,
                    &mut state,
                    &mut active,
                    &dependencies,
                ).await {
                    break;
                }
            }
        }
    }
    if active.is_some() {
        shutdown_for_exit(&snapshot_tx, &ingress_tx, &mut state, &mut active).await;
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_command(
    command: RuntimeCommand,
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
) -> bool {
    match command {
        RuntimeCommand::Start { reply } => {
            let result = start_state(
                snapshot_tx,
                ingress_tx,
                notice_tx,
                state,
                active,
                dependencies,
            )
            .await;
            let _ = reply.send(result);
        }
        RuntimeCommand::Stop { reply } => {
            let result = stop_state(snapshot_tx, ingress_tx, state, active).await;
            let _ = reply.send(result);
        }
        RuntimeCommand::Restart { reply } => {
            let result = async {
                stop_state(snapshot_tx, ingress_tx, state, active).await?;
                start_state(
                    snapshot_tx,
                    ingress_tx,
                    notice_tx,
                    state,
                    active,
                    dependencies,
                )
                .await
            }
            .await;
            let _ = reply.send(result);
        }
        RuntimeCommand::EmergencyStop { reply } => {
            ingress_tx.send_replace(None);
            if state.begin_stop() {
                publish(snapshot_tx, state, now_ms());
            }
            let result = shutdown_active(active).await;
            match result {
                Ok(()) => {
                    let timestamp = now_ms();
                    state.emergency_stop(timestamp);
                    let snapshot = publish(snapshot_tx, state, timestamp);
                    let _ = reply.send(Ok(snapshot));
                }
                Err(error) => {
                    state.finish_fault(error.to_string());
                    publish(snapshot_tx, state, now_ms());
                    let _ = reply.send(Err(error));
                }
            }
        }
        RuntimeCommand::SetTriggerActive {
            active: requested,
            reply,
        } => {
            let result = match live_ingress(active) {
                Ok(ingress) => set_trigger_state(ingress, requested).await,
                Err(error) => Err(error),
            };
            let _ = reply.send(result);
        }
        RuntimeCommand::ShutdownDaemon { reply } => {
            state.daemon = DaemonState::ShuttingDown;
            ingress_tx.send_replace(None);
            if state.begin_stop() {
                publish(snapshot_tx, state, now_ms());
            }
            let result = shutdown_active(active).await;
            if result.is_ok() {
                state.finish_stop();
            } else if let Err(error) = &result {
                state.finish_fault(error.to_string());
            }
            publish(snapshot_tx, state, now_ms());
            let _ = reply.send(result);
            return true;
        }
    }
    false
}

async fn start_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
) -> Result<RuntimeSnapshot, RuntimeError> {
    let Some(epoch) = state.begin_start()? else {
        return Ok(publish(snapshot_tx, state, now_ms()));
    };
    let has_perception = dependencies.perception.is_some();
    if has_perception {
        state.subsystems.capture.state = SubsystemState::Starting;
        state.subsystems.inference.state = SubsystemState::Starting;
    }
    publish(snapshot_tx, state, now_ms());

    let config = dependencies.pipeline_config(epoch);
    let clock = Arc::clone(&dependencies.clock);
    let perception_clock = Arc::clone(&clock);
    let device = Arc::clone(&dependencies.device);
    let perception_adapter = dependencies.perception.clone();
    let started =
        tokio::task::spawn_blocking(move || PipelineRuntime::start(config, clock, device)).await;
    let (mut pipeline, ingress) = match started {
        Ok(Ok(started)) => started,
        Ok(Err(error)) => {
            let error = RuntimeError::pipeline_rejected(error.to_string());
            state.finish_fault(error.to_string());
            publish(snapshot_tx, state, now_ms());
            return Err(error);
        }
        Err(error) => {
            let error =
                RuntimeError::pipeline_rejected(format!("pipeline startup task failed: {error}"));
            state.finish_fault(error.to_string());
            publish(snapshot_tx, state, now_ms());
            return Err(error);
        }
    };
    let Some(events) = pipeline.take_event_receiver() else {
        let error = RuntimeError::pipeline_rejected("pipeline event receiver unavailable");
        let cleanup = tokio::task::spawn_blocking(move || pipeline.shutdown()).await;
        let shutdown_error = match cleanup {
            Ok(Ok(_)) => None,
            Ok(Err(cleanup_error)) => Some(format!("; cleanup failed: {cleanup_error}")),
            Err(join_error) => Some(format!("; cleanup task failed: {join_error}")),
        };
        let message = format!(
            "{}{cleanup}",
            error,
            cleanup = shutdown_error.unwrap_or_default()
        );
        state.finish_fault(message);
        publish(snapshot_tx, state, now_ms());
        return Err(error);
    };
    let (perception, perception_event_bridge, perception_event_cancel) =
        if let Some(adapter) = perception_adapter {
            let (perception_event_tx, perception_event_rx) =
                std::sync::mpsc::sync_channel(PERCEPTION_EVENT_CAPACITY);
            let perception_ingress = ingress.clone();
            let started = tokio::task::spawn_blocking(move || {
                adapter.start(
                    epoch,
                    perception_ingress,
                    perception_clock,
                    perception_event_tx,
                )
            })
            .await;
            let perception = match started {
                Ok(Ok(perception)) => perception,
                Ok(Err(error)) => {
                    let cleanup = tokio::task::spawn_blocking(move || pipeline.shutdown()).await;
                    let cleanup = cleanup_error("pipeline", cleanup);
                    let error = RuntimeError::pipeline_rejected(format!(
                        "perception startup failed: {error}{cleanup}"
                    ));
                    state.finish_fault(error.to_string());
                    publish(snapshot_tx, state, now_ms());
                    return Err(error);
                }
                Err(error) => {
                    let cleanup = tokio::task::spawn_blocking(move || pipeline.shutdown()).await;
                    let cleanup = cleanup_error("pipeline", cleanup);
                    let error = RuntimeError::pipeline_rejected(format!(
                        "perception startup task failed: {error}{cleanup}"
                    ));
                    state.finish_fault(error.to_string());
                    publish(snapshot_tx, state, now_ms());
                    return Err(error);
                }
            };
            let event_tx = notice_tx.clone();
            let cancel = Arc::new(AtomicBool::new(false));
            let bridge_cancel = Arc::clone(&cancel);
            let bridge = tokio::task::spawn_blocking(move || {
                loop {
                    match perception_event_rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(event) => {
                            let event = match event {
                                PerceptionEvent::Faulted { message } => {
                                    PipelineEvent::Faulted { message }
                                }
                                PerceptionEvent::Stopped => PipelineEvent::Stopped,
                            };
                            let _ = event_tx.blocking_send(PipelineNotice { epoch, event });
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                            if bridge_cancel.load(Ordering::Acquire) =>
                        {
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            });
            (Some(perception), Some(bridge), Some(cancel))
        } else {
            (None, None, None)
        };
    // Start the blocking receiver only after every epoch-scoped producer has
    // reached its readiness gate. If perception startup fails, shutting down
    // the pipeline closes this receiver without leaving an unjoinable blocking
    // task behind.
    let event_tx = notice_tx.clone();
    let event_bridge = tokio::task::spawn_blocking(move || {
        if let Ok(event) = events.recv() {
            let _ = event_tx.blocking_send(PipelineNotice { epoch, event });
        }
    });
    ingress_tx.send_replace(Some(ingress.clone()));
    *active = Some(ActivePipeline {
        epoch,
        runtime: pipeline,
        ingress,
        perception,
        event_bridge,
        perception_event_bridge,
        perception_event_cancel,
    });
    state.finish_start(now_ms(), has_perception);
    Ok(publish(snapshot_tx, state, now_ms()))
}

async fn stop_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
) -> Result<RuntimeSnapshot, RuntimeError> {
    if state.begin_stop() {
        publish(snapshot_tx, state, now_ms());
        ingress_tx.send_replace(None);
        if let Err(error) = shutdown_active(active).await {
            state.finish_fault(error.to_string());
            publish(snapshot_tx, state, now_ms());
            return Err(error);
        }
        state.finish_stop();
    }
    Ok(publish(snapshot_tx, state, now_ms()))
}

async fn handle_pipeline_notice(
    notice: PipelineNotice,
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
) {
    let is_live_state = matches!(
        state.pipeline,
        PipelineState::Starting | PipelineState::Running | PipelineState::Standby
    );
    let is_active_epoch = active
        .as_ref()
        .map(|pipeline| pipeline.epoch == notice.epoch)
        .unwrap_or(false);
    if !is_live_state || !is_active_epoch {
        return;
    }
    match notice.event {
        PipelineEvent::Faulted { message } => {
            ingress_tx.send_replace(None);
            let cleanup = shutdown_active(active).await.err();
            let message = match cleanup {
                Some(error) => format!("{message}; pipeline cleanup failed: {error}"),
                None => message,
            };
            state.finish_fault(message);
            publish(snapshot_tx, state, now_ms());
        }
        PipelineEvent::Stopped => {}
    }
}

async fn shutdown_active(active: &mut Option<ActivePipeline>) -> Result<(), RuntimeError> {
    let Some(active) = active.take() else {
        return Ok(());
    };
    let ActivePipeline {
        epoch: _,
        mut runtime,
        ingress: _,
        mut perception,
        event_bridge,
        perception_event_bridge,
        perception_event_cancel,
    } = active;
    let shutdown = tokio::task::spawn_blocking(move || {
        let mut failures = Vec::new();
        runtime.close_output_gate();
        if let Some(perception) = &mut perception
            && let Err(error) = perception.shutdown()
        {
            failures.push(format!("perception shutdown failed: {error}"));
        }
        if let Some(cancel) = &perception_event_cancel {
            cancel.store(true, Ordering::Release);
        }
        if let Err(error) = runtime.shutdown() {
            failures.push(format!("pipeline shutdown failed: {error}"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    })
    .await;
    let bridge = event_bridge.await;
    let perception_bridge = match perception_event_bridge {
        Some(bridge) => bridge.await.err(),
        None => None,
    };
    let shutdown_error = match shutdown {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error),
        Err(error) => Some(format!("pipeline shutdown task failed: {error}")),
    };
    let mut failures = Vec::new();
    if let Some(error) = shutdown_error {
        failures.push(error);
    }
    if let Err(error) = bridge {
        failures.push(format!("pipeline event bridge failed: {error}"));
    }
    if let Some(error) = perception_bridge {
        failures.push(format!("perception event bridge failed: {error}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(RuntimeError::pipeline_rejected(failures.join("; ")))
    }
}

fn cleanup_error<T, E>(label: &str, cleanup: Result<Result<T, E>, tokio::task::JoinError>) -> String
where
    E: std::fmt::Display,
{
    match cleanup {
        Ok(Ok(_)) => String::new(),
        Ok(Err(error)) => format!("; {label} cleanup failed: {error}"),
        Err(error) => format!("; {label} cleanup task failed: {error}"),
    }
}

fn live_ingress(active: &Option<ActivePipeline>) -> Result<PipelineIngress, RuntimeError> {
    active
        .as_ref()
        .filter(|pipeline| {
            matches!(
                pipeline.runtime.status(),
                PipelineStatus::Running | PipelineStatus::Standby
            )
        })
        .map(|pipeline| pipeline.ingress.clone())
        .ok_or_else(RuntimeError::pipeline_unavailable)
}

async fn set_trigger_state(ingress: PipelineIngress, requested: bool) -> Result<(), RuntimeError> {
    tokio::task::spawn_blocking(move || ingress.set_trigger_active(requested))
        .await
        .map_err(|error| {
            RuntimeError::pipeline_rejected(format!("trigger transition task failed: {error}"))
        })
}

async fn shutdown_for_exit(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
) {
    state.daemon = DaemonState::ShuttingDown;
    ingress_tx.send_replace(None);
    if state.begin_stop() {
        publish(snapshot_tx, state, now_ms());
    }
    match shutdown_active(active).await {
        Ok(()) => state.finish_stop(),
        Err(error) => state.finish_fault(error.to_string()),
    }
    publish(snapshot_tx, state, now_ms());
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

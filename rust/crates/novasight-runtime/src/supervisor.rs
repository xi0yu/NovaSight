//! Sole owner of the live [`PipelineRuntime`] lifecycle.
//!
//! Control-plane callers send typed commands through [`RuntimeHandle`].
//! Detection producers submit caller-owned batches through the same
//! handle; worker objects and platform adapters never escape.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use novasight_core::control::humanized_motion::MotionProfile;
use novasight_core::{
    Clock, DetectionBatch, DeviceCommand, DeviceReceipt, Generation, MonotonicNanos, PointerDevice,
    RecordingPointerDevice, RuntimeEpoch,
};
use novasight_pipeline::{
    CrosshairHub, CrosshairSnapshot, CrosshairTemplateSummary, ModelCandidate, MotionProfileHub,
    MotionProfileStatus, PerceptionAdapter, PerceptionEvent, PerceptionMetrics, PerceptionSession,
    PipelineConfig, PipelineEvent, PipelineIngress, PipelineMetrics, PipelineRuntime,
    PipelineStatus, PreviewHub, PreviewSnapshot, PreviewSubscription,
};
use novasight_store::model_catalog::{DeploymentChange, ModelCatalogError, SqliteModelCatalog};
use novasight_store::motion_profile::{
    MotionProfileRepository, MotionSampleInput, MotionSampleResult, MotionSessionSummary,
};
use tokio::sync::{mpsc, oneshot, watch};

use crate::command::RuntimeCommand;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::model_activation::{
    ModelActivationError, ModelActivationRequest, ModelActivationResult,
};
use crate::model_ingress::{
    ModelIngressError, ModelIngressRequest, ModelIngressResult, ModelIngressStage,
    ModelManifestTransaction, OfflineModelJobRunner, load_profile, validate_worker_output,
};
use crate::protocol::{RuntimeErrorSummary, SubsystemState};
use crate::snapshot::{
    DaemonSnapshot, DeviceMetrics, ModelSnapshot, PipelineSnapshot, RuntimeSnapshot,
    SubsystemSnapshots,
};
use crate::state::{DaemonState, PipelineState};

const COMMAND_CAPACITY: usize = 32;
const PIPELINE_EVENT_CAPACITY: usize = 8;
const PERCEPTION_EVENT_CAPACITY: usize = 4;

#[derive(Debug)]
struct UrgentStopSignal {
    pending: Arc<AtomicUsize>,
}

impl UrgentStopSignal {
    fn new() -> Self {
        Self {
            pending: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn register(self: &Arc<Self>) -> UrgentStopToken {
        self.pending.fetch_add(1, Ordering::AcqRel);
        UrgentStopToken {
            signal: Arc::clone(self),
        }
    }

    fn is_pending(&self) -> bool {
        self.pending.load(Ordering::Acquire) != 0
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct UrgentStopToken {
    signal: Arc<UrgentStopSignal>,
}

impl Drop for UrgentStopToken {
    fn drop(&mut self) {
        self.signal.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Concrete resources used to create each runtime epoch.
#[derive(Clone)]
pub struct RuntimeDependencies {
    clock: Arc<dyn Clock>,
    device: Arc<dyn PointerDevice>,
    pipeline: PipelineConfig,
    perception: Option<Arc<dyn PerceptionAdapter>>,
    model_catalog: Option<SqliteModelCatalog>,
    model_jobs: Option<OfflineModelJobRunner>,
    preview: Option<PreviewHub>,
    crosshair: Option<CrosshairHub>,
    motion_profiles: Option<MotionProfileHub>,
    motion_repository: Option<MotionProfileRepository>,
    urgent_stop: Arc<UrgentStopSignal>,
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
            model_catalog: None,
            model_jobs: None,
            preview: None,
            crosshair: None,
            motion_profiles: None,
            motion_repository: None,
            urgent_stop: Arc::new(UrgentStopSignal::new()),
        }
    }

    pub fn with_perception(mut self, perception: Arc<dyn PerceptionAdapter>) -> Self {
        self.perception = Some(perception);
        self
    }

    pub fn with_model_catalog(mut self, model_catalog: SqliteModelCatalog) -> Self {
        self.model_catalog = Some(model_catalog);
        self
    }

    pub fn with_model_jobs(mut self, model_jobs: OfflineModelJobRunner) -> Self {
        self.model_jobs = Some(model_jobs);
        self
    }

    pub fn with_preview(mut self, preview: PreviewHub) -> Self {
        self.preview = Some(preview);
        self
    }

    pub fn with_crosshair(mut self, crosshair: CrosshairHub) -> Self {
        self.pipeline.crosshair = Some(crosshair.clone());
        self.crosshair = Some(crosshair);
        self
    }

    pub fn with_motion_profiles(
        mut self,
        hub: MotionProfileHub,
        repository: MotionProfileRepository,
    ) -> Self {
        self.pipeline.motion_profiles = Some(hub.clone());
        self.motion_profiles = Some(hub);
        self.motion_repository = Some(repository);
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
    next_diagnostic_generation: u64,
    pipeline_started_at_ms: Option<u64>,
    pipeline_error: Option<RuntimeErrorSummary>,
    subsystems: SubsystemSnapshots,
    perception_metrics: PerceptionMetrics,
    pipeline_metrics: PipelineMetrics,
    device_metrics: DeviceMetrics,
    model: ModelSnapshot,
    started_at_unix_ms: u64,
}

impl Default for SupervisorState {
    fn default() -> Self {
        Self {
            daemon: DaemonState::Starting,
            pipeline: PipelineState::Stopped,
            pipeline_epoch: None,
            next_epoch: 0,
            next_diagnostic_generation: 0,
            pipeline_started_at_ms: None,
            pipeline_error: None,
            subsystems: SubsystemSnapshots::default(),
            perception_metrics: PerceptionMetrics::default(),
            pipeline_metrics: PipelineMetrics::default(),
            device_metrics: DeviceMetrics::default(),
            model: ModelSnapshot::default(),
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
            pipeline_metrics: self.pipeline_metrics.clone(),
            device_metrics: self.device_metrics,
            model: self.model.clone(),
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
        self.pipeline_metrics = PipelineMetrics {
            status: PipelineStatus::Starting,
            ..PipelineMetrics::default()
        };
        self.subsystems.capture.last_error = None;
        self.subsystems.inference.last_error = None;
        self.subsystems.control.last_error = None;
        self.subsystems.device.last_error = None;
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
        self.pipeline_metrics.status = PipelineStatus::Stopping;
        self.subsystems.capture.state = SubsystemState::Stopping;
        self.subsystems.inference.state = SubsystemState::Stopping;
        self.subsystems.control.state = SubsystemState::Stopping;
        self.subsystems.device.state = SubsystemState::Stopping;
        true
    }

    fn finish_stop(&mut self) {
        self.pipeline = PipelineState::Stopped;
        self.pipeline_metrics.status = PipelineStatus::Stopped;
        self.pipeline_started_at_ms = None;
        self.subsystems.capture.state = SubsystemState::Stopped;
        self.subsystems.inference.state = SubsystemState::Stopped;
        self.subsystems.control.state = SubsystemState::Stopped;
        self.subsystems.device.state = SubsystemState::Stopped;
    }

    fn finish_fault(&mut self, message: impl Into<String>) {
        let error = RuntimeErrorSummary::new("pipeline_faulted", message);
        self.pipeline = PipelineState::Faulted;
        self.pipeline_metrics.status = PipelineStatus::Faulted;
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

fn initial_model_snapshot(catalog: Option<&SqliteModelCatalog>) -> ModelSnapshot {
    let Some(catalog) = catalog else {
        return ModelSnapshot::default();
    };
    match catalog.active_model() {
        Ok(active) => ModelSnapshot {
            active,
            catalog_error: None,
        },
        Err(error) => ModelSnapshot {
            active: None,
            catalog_error: Some(error.to_string()),
        },
    }
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
        let model = initial_model_snapshot(dependencies.model_catalog.as_ref());
        let state = SupervisorState {
            daemon: DaemonState::Ready,
            model,
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
            dependencies.clone(),
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
                urgent_stop: Arc::clone(&dependencies.urgent_stop),
                preview: dependencies.preview.clone(),
                crosshair: dependencies.crosshair.clone(),
                motion_profiles: dependencies.motion_profiles.clone(),
                motion_repository: dependencies.motion_repository.clone(),
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
    urgent_stop: Arc<UrgentStopSignal>,
    preview: Option<PreviewHub>,
    crosshair: Option<CrosshairHub>,
    motion_profiles: Option<MotionProfileHub>,
    motion_repository: Option<MotionProfileRepository>,
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
    #[doc(hidden)]
    pub fn pending_urgent_stop_count(&self) -> usize {
        self.urgent_stop.pending.load(Ordering::Acquire)
    }

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

    pub fn preview_snapshot(&self) -> Option<PreviewSnapshot> {
        self.preview.as_ref().map(PreviewHub::snapshot)
    }

    pub fn subscribe_preview(&self) -> Result<PreviewSubscription, RuntimeError> {
        self.preview
            .as_ref()
            .ok_or_else(RuntimeError::pipeline_unavailable)?
            .subscribe()
            .map_err(|error| RuntimeError::invalid_pipeline_state(error.to_string()))
    }

    pub async fn set_preview_active(&self, active: bool) -> Result<PreviewSnapshot, RuntimeError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::SetPreviewActive {
                active,
                reply: reply_tx,
            })
            .await
            .map_err(|_| RuntimeError::supervisor_closed())?;
        reply_rx
            .await
            .map_err(|_| RuntimeError::supervisor_reply_lost())?
    }

    pub fn crosshair_snapshot(&self) -> Option<CrosshairSnapshot> {
        self.crosshair.as_ref().map(CrosshairHub::snapshot)
    }

    pub fn learn_crosshair(&self) -> Result<CrosshairTemplateSummary, RuntimeError> {
        self.crosshair
            .as_ref()
            .ok_or_else(RuntimeError::pipeline_unavailable)?
            .learn()
            .map_err(|error| RuntimeError::invalid_pipeline_state(error.to_string()))
    }

    pub fn clear_crosshair(&self) -> Result<CrosshairSnapshot, RuntimeError> {
        self.crosshair
            .as_ref()
            .ok_or_else(RuntimeError::pipeline_unavailable)?
            .clear_template()
            .map_err(|error| RuntimeError::invalid_pipeline_state(error.to_string()))
    }

    pub fn crosshair_template_preview(&self) -> Result<Option<Vec<u8>>, RuntimeError> {
        self.crosshair
            .as_ref()
            .ok_or_else(RuntimeError::pipeline_unavailable)?
            .template_preview_png()
            .map_err(|error| RuntimeError::invalid_pipeline_state(error.to_string()))
    }

    pub fn motion_sessions(&self) -> Result<Vec<MotionSessionSummary>, RuntimeError> {
        self.motion_repository()?
            .list_sessions()
            .map_err(motion_error)
    }

    pub fn create_motion_session(&self, name: &str) -> Result<MotionSessionSummary, RuntimeError> {
        self.motion_repository()?
            .create_session(name)
            .map_err(motion_error)
    }

    pub fn add_motion_sample(
        &self,
        session_id: &str,
        sample: MotionSampleInput,
    ) -> Result<MotionSampleResult, RuntimeError> {
        self.motion_repository()?
            .add_sample(session_id, sample)
            .map_err(motion_error)
    }

    pub fn train_motion_profile(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<MotionProfile, RuntimeError> {
        self.motion_repository()?
            .train_profile(session_id, name)
            .map_err(motion_error)
    }

    pub fn motion_profiles(&self) -> Result<Vec<MotionProfile>, RuntimeError> {
        self.motion_repository()?
            .list_profiles()
            .map_err(motion_error)
    }

    pub fn activate_motion_profile(
        &self,
        profile_id: &str,
    ) -> Result<(MotionProfile, MotionProfileStatus), RuntimeError> {
        let profile = self
            .motion_repository()?
            .profile(profile_id)
            .map_err(motion_error)?;
        let status = self
            .motion_hub()?
            .activate(profile.clone())
            .map_err(RuntimeError::invalid_pipeline_state)?;
        Ok((profile, status))
    }

    pub fn activate_builtin_motion(&self) -> Result<MotionProfileStatus, RuntimeError> {
        Ok(self.motion_hub()?.activate_builtin())
    }

    pub fn disable_motion_profile(&self) -> Result<MotionProfileStatus, RuntimeError> {
        Ok(self.motion_hub()?.disable())
    }

    pub fn motion_profile_status(&self) -> Result<MotionProfileStatus, RuntimeError> {
        Ok(self.motion_hub()?.status())
    }

    fn motion_repository(&self) -> Result<&MotionProfileRepository, RuntimeError> {
        self.motion_repository
            .as_ref()
            .ok_or_else(RuntimeError::pipeline_unavailable)
    }

    fn motion_hub(&self) -> Result<&MotionProfileHub, RuntimeError> {
        self.motion_profiles
            .as_ref()
            .ok_or_else(RuntimeError::pipeline_unavailable)
    }

    pub async fn diagnose_device_move(
        &self,
        delta_x_counts: i32,
        delta_y_counts: i32,
    ) -> Result<DeviceReceipt, RuntimeError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::DiagnoseDeviceMove {
                delta_x_counts,
                delta_y_counts,
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
        let urgent = self.urgent_stop.register();
        if let Some(ingress) = self.ingress_rx.borrow().clone() {
            ingress.request_output_stop();
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::Stop {
                urgent,
                reply: reply_tx,
            })
            .await
            .map_err(|_| RuntimeError::supervisor_closed())?;
        reply_rx
            .await
            .map_err(|_| RuntimeError::supervisor_reply_lost())?
    }

    pub async fn restart(&self) -> Result<RuntimeSnapshot, RuntimeError> {
        self.send_command(|reply| RuntimeCommand::Restart { reply })
            .await
    }

    pub async fn preflight_perception(&self) -> Result<(), RuntimeError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::PreflightPerception { reply: reply_tx })
            .await
            .map_err(|_| RuntimeError::supervisor_closed())?;
        reply_rx
            .await
            .map_err(|_| RuntimeError::supervisor_reply_lost())?
    }

    pub async fn activate_model(
        &self,
        request: ModelActivationRequest,
    ) -> Result<ModelActivationResult, ModelActivationError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::ActivateModel {
                request,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ModelActivationError::Runtime(RuntimeError::supervisor_closed()))?;
        reply_rx
            .await
            .map_err(|_| ModelActivationError::Runtime(RuntimeError::supervisor_reply_lost()))?
    }

    pub async fn model_ingress(
        &self,
        request: ModelIngressRequest,
    ) -> Result<ModelIngressResult, ModelIngressError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::ModelIngress {
                request,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ModelIngressError::Failed("runtime supervisor is closed".to_owned()))?;
        reply_rx.await.map_err(|_| {
            ModelIngressError::Failed("runtime supervisor reply was lost".to_owned())
        })?
    }

    pub async fn emergency_stop(&self) -> Result<RuntimeSnapshot, RuntimeError> {
        let urgent = self.urgent_stop.register();
        if let Some(ingress) = self.ingress_rx.borrow().clone() {
            ingress.request_output_stop();
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::EmergencyStop {
                urgent,
                reply: reply_tx,
            })
            .await
            .map_err(|_| RuntimeError::supervisor_closed())?;
        reply_rx
            .await
            .map_err(|_| RuntimeError::supervisor_reply_lost())?
    }

    pub async fn shutdown_daemon(&self) -> Result<(), RuntimeError> {
        let urgent = self.urgent_stop.register();
        if let Some(ingress) = self.ingress_rx.borrow().clone() {
            ingress.request_output_stop();
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand::ShutdownDaemon {
                urgent,
                reply: reply_tx,
            })
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

fn motion_error(error: novasight_store::motion_profile::MotionProfileError) -> RuntimeError {
    RuntimeError::invalid_pipeline_state(error.to_string())
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
                let perception_metrics = active
                    .as_ref()
                    .and_then(|pipeline| pipeline.perception.as_ref())
                    .map(|perception| perception.metrics())
                    .unwrap_or_default();
                let pipeline_metrics = active
                    .as_ref()
                    .map(|pipeline| pipeline.runtime.metrics())
                    .unwrap_or_default();
                if perception_metrics != state.perception_metrics
                    || pipeline_metrics != state.pipeline_metrics
                {
                    state.perception_metrics = perception_metrics;
                    state.pipeline_metrics = pipeline_metrics;
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
            let result = if dependencies.urgent_stop.is_pending() {
                Err(RuntimeError::invalid_pipeline_state(
                    "start cancelled by a pending stop request",
                ))
            } else {
                start_state(
                    snapshot_tx,
                    ingress_tx,
                    notice_tx,
                    state,
                    active,
                    dependencies,
                )
                .await
            };
            let _ = reply.send(result);
        }
        RuntimeCommand::Stop { urgent, reply } => {
            let result = stop_state(snapshot_tx, ingress_tx, state, active).await;
            let _ = reply.send(result);
            drop(urgent);
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
        RuntimeCommand::PreflightPerception { reply } => {
            let result = preflight_perception(state, dependencies).await;
            let _ = reply.send(result);
        }
        RuntimeCommand::ActivateModel { request, reply } => {
            let result = activate_model_state(
                request,
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
        RuntimeCommand::ModelIngress { request, reply } => {
            let result = model_ingress_state(
                request,
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
        RuntimeCommand::EmergencyStop { urgent, reply } => {
            ingress_tx.send_replace(None);
            refresh_pipeline_metrics(state, active);
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
            drop(urgent);
        }
        RuntimeCommand::SetTriggerActive {
            active: requested,
            reply,
        } => {
            let result = match live_ingress(active) {
                Ok(ingress) => set_trigger_state(ingress, requested).await,
                Err(error) => Err(error),
            };
            if result.is_ok() {
                refresh_pipeline_metrics(state, active);
                publish(snapshot_tx, state, now_ms());
            }
            let _ = reply.send(result);
        }
        RuntimeCommand::SetPreviewActive {
            active: requested,
            reply,
        } => {
            let result = dependencies
                .preview
                .as_ref()
                .ok_or_else(RuntimeError::pipeline_unavailable)
                .and_then(|preview| {
                    preview
                        .set_active(requested)
                        .map_err(|error| RuntimeError::invalid_pipeline_state(error.to_string()))
                });
            let _ = reply.send(result);
        }
        RuntimeCommand::DiagnoseDeviceMove {
            delta_x_counts,
            delta_y_counts,
            reply,
        } => {
            let result = diagnose_device_move(
                snapshot_tx,
                state,
                dependencies,
                delta_x_counts,
                delta_y_counts,
            )
            .await;
            let _ = reply.send(result);
        }
        RuntimeCommand::ShutdownDaemon { urgent, reply } => {
            state.daemon = DaemonState::ShuttingDown;
            ingress_tx.send_replace(None);
            refresh_pipeline_metrics(state, active);
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
            drop(urgent);
            return true;
        }
    }
    false
}

async fn preflight_perception(
    state: &SupervisorState,
    dependencies: &RuntimeDependencies,
) -> Result<(), RuntimeError> {
    if !matches!(
        state.pipeline,
        PipelineState::Stopped | PipelineState::Faulted
    ) {
        return Err(RuntimeError::invalid_pipeline_state(
            "perception preflight requires a stopped pipeline",
        ));
    }
    let Some(adapter) = dependencies.perception.clone() else {
        return Ok(());
    };
    tokio::task::spawn_blocking(move || adapter.preflight())
        .await
        .map_err(|error| {
            RuntimeError::pipeline_rejected(format!("perception preflight task failed: {error}"))
        })?
        .map_err(|error| RuntimeError::pipeline_rejected(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
async fn activate_model_state(
    request: ModelActivationRequest,
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active_pipeline: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
) -> Result<ModelActivationResult, ModelActivationError> {
    let catalog = dependencies
        .model_catalog
        .clone()
        .ok_or(ModelActivationError::Unavailable)?;
    let action = request.label();
    let project_id = request.project_id();
    let plan_catalog = catalog.clone();
    let plan_request = request.clone();
    let (candidate, parser_preset, no_op) =
        tokio::task::spawn_blocking(move || -> Result<_, ModelCatalogError> {
            match plan_request {
                ModelActivationRequest::Publish {
                    artifact_id,
                    parser_preset,
                    ..
                } => Ok((
                    plan_catalog.runtime_artifact(project_id, artifact_id)?,
                    parser_preset,
                    None,
                )),
                ModelActivationRequest::Rollback { .. } => {
                    let current = plan_catalog.deployment(project_id)?;
                    let artifact_id = current.previous_artifact_id.unwrap_or(current.artifact_id);
                    let candidate = plan_catalog.runtime_artifact(project_id, artifact_id)?;
                    let no_op = (artifact_id == current.artifact_id).then_some(current);
                    Ok((candidate, "auto".to_owned(), no_op))
                }
            }
        })
        .await
        .map_err(|error| ModelActivationError::Failed {
            action,
            message: format!("catalog planning task failed: {error}"),
        })??;

    if let Some(deployment) = no_op {
        let active = novasight_store::model_catalog::ActiveModelDeployment {
            deployment: deployment.clone(),
            project: candidate.project.clone(),
            version: candidate.version.clone(),
            artifact: candidate.artifact.clone(),
            artifact_path: candidate.artifact_path.clone(),
        };
        state.model = ModelSnapshot {
            active: Some(active.clone()),
            catalog_error: None,
        };
        let runtime = publish(snapshot_tx, state, now_ms());
        return Ok(ModelActivationResult {
            action,
            deployment,
            active,
            candidate,
            contract: None,
            runtime,
            restarted: false,
            changed: false,
        });
    }

    let receipt_catalog = catalog.clone();
    let receipt_artifact_id = candidate.artifact.id;
    tokio::task::spawn_blocking(move || {
        receipt_catalog.validate_ingress_receipt(receipt_artifact_id)
    })
    .await
    .map_err(|error| ModelActivationError::Failed {
        action,
        message: format!("model validation receipt task failed: {error}"),
    })?
    .map_err(ModelActivationError::Catalog)?;

    let adapter = dependencies
        .perception
        .clone()
        .ok_or(ModelActivationError::PerceptionUnavailable)?;
    let model = ModelCandidate {
        project_id,
        artifact_id: candidate.artifact.id,
        parser_preset,
    };
    let contract = tokio::task::spawn_blocking(move || adapter.preflight_model(&model))
        .await
        .map_err(|error| ModelActivationError::Failed {
            action,
            message: format!("candidate validation task failed: {error}"),
        })?
        .map_err(|error| ModelActivationError::Failed {
            action,
            message: format!("candidate validation failed: {error}"),
        })?;

    if dependencies.urgent_stop.is_pending() {
        return Err(ModelActivationError::Failed {
            action,
            message: "candidate activation cancelled by a pending stop request".to_owned(),
        });
    }

    let was_running = matches!(
        state.pipeline,
        PipelineState::Running | PipelineState::Standby
    );
    if was_running {
        stop_state(snapshot_tx, ingress_tx, state, active_pipeline).await?;
    }
    if dependencies.urgent_stop.is_pending() {
        return Err(ModelActivationError::Failed {
            action,
            message: "candidate activation cancelled before catalog mutation".to_owned(),
        });
    }

    let change_catalog = catalog.clone();
    let change = tokio::task::spawn_blocking(move || match request {
        ModelActivationRequest::Publish { artifact_id, .. } => {
            change_catalog.publish_change(project_id, artifact_id)
        }
        ModelActivationRequest::Rollback { .. } => change_catalog.rollback_change(project_id),
    })
    .await
    .map_err(|error| ModelActivationError::Failed {
        action,
        message: format!("catalog mutation task failed: {error}"),
    });
    let change = match change {
        Ok(Ok(change)) => change,
        Ok(Err(error)) => {
            if was_running
                && let Err(recovery) = start_state(
                    snapshot_tx,
                    ingress_tx,
                    notice_tx,
                    state,
                    active_pipeline,
                    dependencies,
                )
                .await
            {
                return Err(ModelActivationError::Failed {
                    action,
                    message: format!(
                        "catalog mutation failed: {error}; previous runtime recovery failed: {recovery}"
                    ),
                });
            }
            return Err(ModelActivationError::Catalog(error));
        }
        Err(error) => {
            if was_running {
                let recovery = start_state(
                    snapshot_tx,
                    ingress_tx,
                    notice_tx,
                    state,
                    active_pipeline,
                    dependencies,
                )
                .await;
                if let Err(recovery) = recovery {
                    return Err(ModelActivationError::Failed {
                        action,
                        message: format!("{error}; previous runtime recovery failed: {recovery}"),
                    });
                }
            }
            return Err(error);
        }
    };

    let resolved_catalog = catalog.clone();
    let resolved = tokio::task::spawn_blocking(move || resolved_catalog.active_model())
        .await
        .map_err(|error| ModelActivationError::Failed {
            action,
            message: format!("active model read task failed: {error}"),
        })?;
    let active_model = match resolved {
        Ok(Some(active))
            if active.deployment == change.after && active.artifact.id == candidate.artifact.id =>
        {
            active
        }
        Ok(Some(active)) => {
            return Err(compensate_model_activation(
                action,
                format!(
                    "catalog resolved artifact {} instead of candidate {}",
                    active.artifact.id, candidate.artifact.id
                ),
                change,
                was_running,
                &catalog,
                snapshot_tx,
                ingress_tx,
                notice_tx,
                state,
                active_pipeline,
                dependencies,
            )
            .await);
        }
        Ok(None) => {
            return Err(compensate_model_activation(
                action,
                "catalog committed a deployment but resolved no active model".to_owned(),
                change,
                was_running,
                &catalog,
                snapshot_tx,
                ingress_tx,
                notice_tx,
                state,
                active_pipeline,
                dependencies,
            )
            .await);
        }
        Err(error) => {
            return Err(compensate_model_activation(
                action,
                format!("active model read failed: {error}"),
                change,
                was_running,
                &catalog,
                snapshot_tx,
                ingress_tx,
                notice_tx,
                state,
                active_pipeline,
                dependencies,
            )
            .await);
        }
    };

    if dependencies.urgent_stop.is_pending() {
        return Err(compensate_model_activation(
            action,
            "candidate activation cancelled after catalog mutation".to_owned(),
            change,
            was_running,
            &catalog,
            snapshot_tx,
            ingress_tx,
            notice_tx,
            state,
            active_pipeline,
            dependencies,
        )
        .await);
    }

    if was_running {
        match start_state(
            snapshot_tx,
            ingress_tx,
            notice_tx,
            state,
            active_pipeline,
            dependencies,
        )
        .await
        {
            Ok(_) => {}
            Err(error) => {
                return Err(compensate_model_activation(
                    action,
                    error.to_string(),
                    change,
                    was_running,
                    &catalog,
                    snapshot_tx,
                    ingress_tx,
                    notice_tx,
                    state,
                    active_pipeline,
                    dependencies,
                )
                .await);
            }
        }
    }

    state.model = ModelSnapshot {
        active: Some(active_model.clone()),
        catalog_error: None,
    };
    let runtime = publish(snapshot_tx, state, now_ms());

    Ok(ModelActivationResult {
        action,
        deployment: change.after,
        active: active_model,
        candidate,
        contract,
        runtime,
        restarted: was_running,
        changed: true,
    })
}

#[allow(clippy::too_many_arguments)]
async fn model_ingress_state(
    request: ModelIngressRequest,
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active_pipeline: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
) -> Result<ModelIngressResult, ModelIngressError> {
    let catalog = dependencies
        .model_catalog
        .clone()
        .ok_or(ModelIngressError::Unavailable)?;
    let artifact_id = request.artifact_id();
    let artifact_catalog = catalog.clone();
    let artifact =
        tokio::task::spawn_blocking(move || artifact_catalog.runtime_artifact_by_id(artifact_id))
            .await
            .map_err(|error| {
                ModelIngressError::Failed(format!("artifact lookup task failed: {error}"))
            })??;
    if artifact.artifact.kind != "engine" {
        return Err(ModelIngressError::Catalog(
            ModelCatalogError::IngressRequiresEngine(artifact_id),
        ));
    }
    let display_root = catalog
        .path()
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""))
        .to_owned();

    if matches!(request, ModelIngressRequest::GetProfile { .. }) {
        return tokio::task::spawn_blocking(move || {
            load_profile(&artifact, &display_root, 1024 * 1024)
        })
        .await
        .map_err(|error| {
            ModelIngressError::Failed(format!("profile read task failed: {error}"))
        })?;
    }

    let deployed_catalog = catalog.clone();
    let deployed =
        tokio::task::spawn_blocking(move || deployed_catalog.artifact_is_deployed(artifact_id))
            .await
            .map_err(|error| {
                ModelIngressError::Failed(format!("deployment lookup task failed: {error}"))
            })??;
    if deployed {
        return Err(ModelIngressError::Catalog(
            ModelCatalogError::ArtifactCurrentlyDeployed(artifact_id),
        ));
    }
    let jobs = dependencies
        .model_jobs
        .clone()
        .ok_or(ModelIngressError::Unavailable)?;
    let cancellation = Arc::clone(&dependencies.urgent_stop.pending);
    let was_running = matches!(
        state.pipeline,
        PipelineState::Running | PipelineState::Standby
    );
    let stage = match &request {
        ModelIngressRequest::Inspect { .. } => ModelIngressStage::Inspect,
        ModelIngressRequest::Configure { .. } => ModelIngressStage::Configure,
        ModelIngressRequest::Probe { .. } => ModelIngressStage::Probe,
        ModelIngressRequest::GetProfile { .. } => unreachable!("handled above"),
    };
    let manifest_engine = artifact.artifact_path.clone();
    let manifest_transaction = tokio::task::spawn_blocking(move || {
        ModelManifestTransaction::begin(&manifest_engine, 1024 * 1024)
    })
    .await
    .map_err(|error| {
        ModelIngressError::Failed(format!("manifest snapshot task failed: {error}"))
    })??;
    if stage == ModelIngressStage::Probe && was_running {
        stop_state(snapshot_tx, ingress_tx, state, active_pipeline).await?;
    }
    let worker = match &request {
        ModelIngressRequest::Inspect { .. } => {
            jobs.inspect(
                &artifact.artifact_path,
                &artifact.project.name,
                cancellation,
            )
            .await
        }
        ModelIngressRequest::Configure { profile, .. } => {
            jobs.configure(&artifact.artifact_path, profile, cancellation)
                .await
        }
        ModelIngressRequest::Probe { input_mode, .. } => {
            jobs.probe(&artifact.artifact_path, *input_mode, cancellation)
                .await
        }
        ModelIngressRequest::GetProfile { .. } => unreachable!("handled above"),
    };
    let result = match worker {
        Ok(worker) => {
            let artifact_for_validation = artifact.clone();
            let validation_root = display_root.clone();
            let catalog_for_commit = catalog.clone();
            tokio::task::spawn_blocking(move || {
                let operation = (|| {
                    let (result, update) = validate_worker_output(
                        &artifact_for_validation,
                        &validation_root,
                        stage,
                        worker,
                    )?;
                    catalog_for_commit.commit_model_ingress(update)?;
                    Ok::<_, ModelIngressError>(result)
                })();
                match operation {
                    Ok(result) => {
                        manifest_transaction.commit();
                        Ok(result)
                    }
                    Err(error) => {
                        manifest_transaction.rollback().map_err(|rollback| {
                            ModelIngressError::Failed(format!(
                                "{error}; manifest rollback also failed: {rollback}"
                            ))
                        })?;
                        Err(error)
                    }
                }
            })
            .await
            .map_err(|error| {
                ModelIngressError::Failed(format!("model-ingress commit task failed: {error}"))
            })?
        }
        Err(error) => {
            tokio::task::spawn_blocking(move || manifest_transaction.rollback())
                .await
                .map_err(|join| {
                    ModelIngressError::Failed(format!("manifest rollback task failed: {join}"))
                })?
                .map_err(|rollback| {
                    ModelIngressError::Failed(format!(
                        "{error}; manifest rollback also failed: {rollback}"
                    ))
                })?;
            Err(error)
        }
    };

    if stage == ModelIngressStage::Probe
        && was_running
        && !dependencies.urgent_stop.is_pending()
        && let Err(recovery) = start_state(
            snapshot_tx,
            ingress_tx,
            notice_tx,
            state,
            active_pipeline,
            dependencies,
        )
        .await
    {
        return Err(ModelIngressError::Failed(format!(
            "model probe completed but previous runtime recovery failed: {recovery}"
        )));
    }
    result
}

#[allow(clippy::too_many_arguments)]
async fn compensate_model_activation(
    action: &'static str,
    failure: String,
    change: DeploymentChange,
    was_running: bool,
    catalog: &SqliteModelCatalog,
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active_pipeline: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
) -> ModelActivationError {
    let compensation_catalog = catalog.clone();
    let compensation =
        tokio::task::spawn_blocking(move || compensation_catalog.compensate(change)).await;
    let (compensated, compensation_message) = match compensation {
        Ok(Ok(_)) => (true, "deployment was restored".to_owned()),
        Ok(Err(error)) => (false, format!("deployment compensation failed: {error}")),
        Err(error) => (
            false,
            format!("deployment compensation task failed: {error}"),
        ),
    };
    let recovery = if compensated && was_running && !dependencies.urgent_stop.is_pending() {
        start_state(
            snapshot_tx,
            ingress_tx,
            notice_tx,
            state,
            active_pipeline,
            dependencies,
        )
        .await
        .err()
    } else {
        None
    };
    let recovery_message = recovery
        .map(|error| format!("; previous runtime recovery failed: {error}"))
        .unwrap_or_default();
    ModelActivationError::Failed {
        action,
        message: format!(
            "candidate activation failed: {failure}; {compensation_message}{recovery_message}"
        ),
    }
}

async fn diagnose_device_move(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    state: &mut SupervisorState,
    dependencies: &RuntimeDependencies,
    delta_x_counts: i32,
    delta_y_counts: i32,
) -> Result<DeviceReceipt, RuntimeError> {
    if state.pipeline != PipelineState::Stopped {
        return Err(RuntimeError::invalid_pipeline_state(
            "device diagnostics require a stopped pipeline",
        ));
    }
    if state.subsystems.device.state == SubsystemState::Unavailable {
        return Err(RuntimeError::device_unavailable(
            "device diagnostics remain disabled after emergency stop or device fault",
        ));
    }
    if delta_x_counts == 0 && delta_y_counts == 0 {
        return Err(RuntimeError::invalid_device_command(
            "device diagnostic move must change at least one axis",
        ));
    }
    if i16::try_from(delta_x_counts).is_err() || i16::try_from(delta_y_counts).is_err() {
        return Err(RuntimeError::invalid_device_command(
            "device diagnostic counts must fit the kmNet signed 16-bit contract",
        ));
    }
    state.next_diagnostic_generation = state
        .next_diagnostic_generation
        .checked_add(1)
        .ok_or_else(|| RuntimeError::device_unavailable("diagnostic generation is exhausted"))?;
    let command = DeviceCommand {
        epoch: RuntimeEpoch(0),
        generation: Generation(state.next_diagnostic_generation),
        issued_at: dependencies.clock.now(),
        target_object_id: 0,
        delta_x_counts,
        delta_y_counts,
    };
    state.subsystems.device.state = SubsystemState::Starting;
    publish(snapshot_tx, state, now_ms());
    let device = Arc::clone(&dependencies.device);
    let result = tokio::task::spawn_blocking(move || {
        device.connect()?;
        let send = device.send(command);
        let disconnect = device.disconnect();
        match send {
            Ok(receipt) => disconnect.map(|()| receipt),
            Err(error) => Err(error),
        }
    })
    .await;
    match result {
        Ok(Ok(receipt)) => {
            state.device_metrics.diagnostic_move_count =
                state.device_metrics.diagnostic_move_count.saturating_add(1);
            state.device_metrics.last_diagnostic_dx = Some(receipt.delta_x_counts);
            state.device_metrics.last_diagnostic_dy = Some(receipt.delta_y_counts);
            state.subsystems.device.state = SubsystemState::Ready;
            state.subsystems.device.last_error = None;
            publish(snapshot_tx, state, now_ms());
            Ok(receipt)
        }
        Ok(Err(error)) => {
            let error = RuntimeError::device_unavailable(error.to_string());
            state.subsystems.device.state = SubsystemState::Unavailable;
            state.subsystems.device.last_error = Some(error.summary());
            publish(snapshot_tx, state, now_ms());
            Err(error)
        }
        Err(error) => {
            let error =
                RuntimeError::device_unavailable(format!("device diagnostic task failed: {error}"));
            state.subsystems.device.state = SubsystemState::Unavailable;
            state.subsystems.device.last_error = Some(error.summary());
            publish(snapshot_tx, state, now_ms());
            Err(error)
        }
    }
}

async fn start_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
) -> Result<RuntimeSnapshot, RuntimeError> {
    if dependencies.urgent_stop.is_pending() {
        return Err(RuntimeError::invalid_pipeline_state(
            "runtime start cancelled by a pending stop request",
        ));
    }
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
    let urgent_stop = Arc::clone(&dependencies.urgent_stop.pending);
    let perception_adapter = dependencies.perception.clone();
    let started = tokio::task::spawn_blocking(move || {
        PipelineRuntime::start_suspended_with_cancel(config, clock, device, urgent_stop)
    })
    .await;
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
    if dependencies.urgent_stop.is_pending() {
        let cleanup = tokio::task::spawn_blocking(move || {
            let mut failures = Vec::new();
            if let Some(mut perception) = perception
                && let Err(error) = perception.shutdown()
            {
                failures.push(format!("perception shutdown failed: {error}"));
            }
            if let Err(error) = pipeline.shutdown() {
                failures.push(format!("pipeline shutdown failed: {error}"));
            }
            failures
        })
        .await;
        if let Some(cancel) = &perception_event_cancel {
            cancel.store(true, Ordering::Release);
        }
        if let Some(bridge) = perception_event_bridge {
            let _ = bridge.await;
        }
        state.finish_stop();
        publish(snapshot_tx, state, now_ms());
        let cleanup = cleanup
            .map_err(|error| format!("cleanup task failed: {error}"))
            .and_then(|failures| {
                if failures.is_empty() {
                    Ok(())
                } else {
                    Err(failures.join("; "))
                }
            })
            .err()
            .map(|message| format!("; {message}"))
            .unwrap_or_default();
        return Err(RuntimeError::invalid_pipeline_state(format!(
            "runtime start cancelled by a pending stop request{cleanup}"
        )));
    }
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
    pipeline.open_output_gate();
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
    state.pipeline_metrics = active
        .as_ref()
        .map(|active| active.runtime.metrics())
        .unwrap_or_default();
    state.finish_start(now_ms(), has_perception);
    Ok(publish(snapshot_tx, state, now_ms()))
}

async fn stop_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
) -> Result<RuntimeSnapshot, RuntimeError> {
    refresh_pipeline_metrics(state, active);
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
            refresh_pipeline_metrics(state, active);
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

fn refresh_pipeline_metrics(state: &mut SupervisorState, active: &Option<ActivePipeline>) {
    if let Some(active) = active {
        state.pipeline_metrics = active.runtime.metrics();
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
    refresh_pipeline_metrics(state, active);
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

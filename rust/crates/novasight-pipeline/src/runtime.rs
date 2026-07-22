use std::sync::{
    Arc, Mutex, TryLockError,
    atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    mpsc::{Receiver, SyncSender, sync_channel},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use novasight_core::control::dual_phase_v2::{
    ControlObservation, DualPhaseConfig, DualPhaseControl,
};
use novasight_core::tracking::{TargetingConfig, TargetingCore};
use novasight_core::{
    Clock, DetectionBatch, DeviceCommand, Generation, PointerDevice, RuntimeEpoch,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{LatestSlot, TryPublishError};

const STATUS_STARTING: u8 = 0;
const STATUS_RUNNING: u8 = 1;
const STATUS_STOPPING: u8 = 2;
const STATUS_STOPPED: u8 = 3;
const STATUS_FAULTED: u8 = 4;
const STATUS_STANDBY: u8 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PipelineStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Faulted,
    Standby,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineEvent {
    Faulted { message: String },
    Stopped,
}

impl PipelineStatus {
    fn from_atomic(value: u8) -> Self {
        match value {
            STATUS_STARTING => Self::Starting,
            STATUS_RUNNING => Self::Running,
            STATUS_STOPPING => Self::Stopping,
            STATUS_STOPPED => Self::Stopped,
            STATUS_FAULTED => Self::Faulted,
            STATUS_STANDBY => Self::Standby,
            _ => Self::Faulted,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub epoch: RuntimeEpoch,
    pub targeting: TargetingConfig,
    pub control: DualPhaseConfig,
    /// Commands older than this many monotonic nanoseconds are dropped
    /// immediately before the device call.
    pub max_command_age_ns: u64,
    /// Independent output scheduler cadence. Production configuration is
    /// constrained to the Python-compatible 1-10 ms range.
    pub output_interval_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            epoch: RuntimeEpoch(1),
            targeting: TargetingConfig::default(),
            control: DualPhaseConfig::default(),
            max_command_age_ns: 55_000_000,
            output_interval_ms: 4,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub status: PipelineStatus,
    pub received_batches: u64,
    pub input_overwrites: u64,
    pub targeting_batches: u64,
    pub control_decisions: u64,
    pub blocked_decisions: u64,
    pub command_overwrites: u64,
    pub superseded_commands: u64,
    pub stale_commands: u64,
    pub device_receipts: u64,
    pub live_workers: u64,
    pub last_generation: Option<Generation>,
    pub last_fault: Option<String>,
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("pipeline is not running")]
    NotRunning,
    #[error("pipeline expected epoch {expected}, received {actual}")]
    EpochMismatch { expected: u64, actual: u64 },
    #[error("pipeline generation must increase: previous {previous}, received {actual}")]
    NonMonotonicGeneration { previous: u64, actual: u64 },
    #[error("pipeline ingress is busy; realtime producer must drop this batch")]
    IngressBusy,
    #[error("output scheduler interval must be within 1..=10 ms, got {actual_ms}")]
    InvalidOutputInterval { actual_ms: u64 },
    #[error("failed to spawn {worker} worker: {source}")]
    Spawn {
        worker: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("one or more pipeline workers panicked during shutdown")]
    WorkerPanicked,
}

#[derive(Debug, Default)]
struct AtomicMetrics {
    received_batches: AtomicU64,
    targeting_batches: AtomicU64,
    control_decisions: AtomicU64,
    blocked_decisions: AtomicU64,
    superseded_commands: AtomicU64,
    stale_commands: AtomicU64,
    device_receipts: AtomicU64,
    live_workers: AtomicU64,
    last_generation: Mutex<Option<Generation>>,
    last_fault: Mutex<Option<String>>,
}

#[derive(Debug)]
struct SharedState {
    status: AtomicU8,
    output_gate: AtomicBool,
    trigger_active: AtomicBool,
    device_lane: Mutex<()>,
    event_tx: SyncSender<PipelineEvent>,
    metrics: AtomicMetrics,
}

impl SharedState {
    fn new(event_tx: SyncSender<PipelineEvent>) -> Self {
        Self {
            status: AtomicU8::new(STATUS_STARTING),
            output_gate: AtomicBool::new(false),
            trigger_active: AtomicBool::new(false),
            device_lane: Mutex::new(()),
            event_tx,
            metrics: AtomicMetrics::default(),
        }
    }

    fn status(&self) -> PipelineStatus {
        PipelineStatus::from_atomic(self.status.load(Ordering::Acquire))
    }

    fn fault(&self, message: impl Into<String>) {
        let message = message.into();
        self.output_gate.store(false, Ordering::Release);
        self.trigger_active.store(false, Ordering::Release);
        *self
            .metrics
            .last_fault
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(message.clone());
        let previous = self.status.swap(STATUS_FAULTED, Ordering::AcqRel);
        if previous != STATUS_FAULTED {
            let _ = self.event_tx.try_send(PipelineEvent::Faulted { message });
        }
    }
}

struct WorkerGuard(Arc<SharedState>);

impl WorkerGuard {
    fn new(shared: Arc<SharedState>) -> Self {
        shared.metrics.live_workers.fetch_add(1, Ordering::AcqRel);
        Self(shared)
    }
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        self.0.metrics.live_workers.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone)]
pub struct PipelineIngress {
    epoch: RuntimeEpoch,
    batches: LatestSlot<DetectionBatch>,
    shared: Arc<SharedState>,
}

impl std::fmt::Debug for PipelineIngress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineIngress")
            .field("epoch", &self.epoch)
            .field("status", &self.shared.status())
            .finish()
    }
}

impl PipelineIngress {
    /// Update the live trigger cache. Output is disabled by default and
    /// the device lane rechecks this value immediately before sending.
    pub fn set_trigger_active(&self, active: bool) {
        self.shared.trigger_active.store(active, Ordering::Release);
        if !active {
            // Returning from trigger release is a synchronization point:
            // no device send from the previous active interval remains.
            drop(
                self.shared
                    .device_lane
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
        }
    }

    pub fn trigger_active(&self) -> bool {
        self.shared.trigger_active.load(Ordering::Acquire)
    }

    /// Admit one already-validated object-metadata batch. DeepStream
    /// and replay adapters use the same ingress; if the targeting lane
    /// is behind, this replaces its unread batch.
    pub fn submit(&self, batch: DetectionBatch) -> Result<(), PipelineError> {
        if self.shared.status() != PipelineStatus::Running {
            return Err(PipelineError::NotRunning);
        }
        let stamp = batch.stamp();
        if stamp.epoch != self.epoch {
            return Err(PipelineError::EpochMismatch {
                expected: self.epoch.0,
                actual: stamp.epoch.0,
            });
        }
        let mut last = self
            .shared
            .metrics
            .last_generation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(previous) = *last
            && stamp.generation <= previous
        {
            return Err(PipelineError::NonMonotonicGeneration {
                previous: previous.0,
                actual: stamp.generation.0,
            });
        }
        self.batches
            .publish(batch)
            .map_err(|_| PipelineError::NotRunning)?;
        *last = Some(stamp.generation);
        drop(last);
        self.shared
            .metrics
            .received_batches
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Submit from a realtime producer without waiting for another producer or
    /// the targeting lane. Busy means the caller must drop this frame.
    pub fn try_submit(&self, batch: DetectionBatch) -> Result<(), PipelineError> {
        if self.shared.status() != PipelineStatus::Running {
            return Err(PipelineError::NotRunning);
        }
        let stamp = batch.stamp();
        if stamp.epoch != self.epoch {
            return Err(PipelineError::EpochMismatch {
                expected: self.epoch.0,
                actual: stamp.epoch.0,
            });
        }
        let mut last = match self.shared.metrics.last_generation.try_lock() {
            Ok(last) => last,
            Err(TryLockError::WouldBlock) => return Err(PipelineError::IngressBusy),
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        if let Some(previous) = *last
            && stamp.generation <= previous
        {
            return Err(PipelineError::NonMonotonicGeneration {
                previous: previous.0,
                actual: stamp.generation.0,
            });
        }
        match self.batches.try_publish(batch) {
            Ok(_) => {}
            Err(TryPublishError::Busy) => return Err(PipelineError::IngressBusy),
            Err(TryPublishError::Closed) => return Err(PipelineError::NotRunning),
        }
        *last = Some(stamp.generation);
        drop(last);
        self.shared
            .metrics
            .received_batches
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct TargetedObservation {
    stamp: novasight_core::FrameStamp,
    target_id: Option<u64>,
    aim_x: f64,
    aim_y: f64,
    crosshair_x: f64,
    crosshair_y: f64,
    inference_end_ns: u64,
    control_now_ns: u64,
}

/// Owns the post-inference real-time lanes. Capture/DeepStream keeps
/// ownership of vendor objects and submits caller-owned batches through
/// [`PipelineIngress`]; targeting, control, and device calls each run on
/// a dedicated OS thread.
pub struct PipelineRuntime {
    config: PipelineConfig,
    shared: Arc<SharedState>,
    batch_slot: LatestSlot<DetectionBatch>,
    target_slot: LatestSlot<TargetedObservation>,
    command_slot: LatestSlot<DeviceCommand>,
    event_rx: Option<Receiver<PipelineEvent>>,
    workers: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for PipelineRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineRuntime")
            .field("epoch", &self.config.epoch)
            .field("status", &self.shared.status())
            .field("workers", &self.workers.len())
            .finish()
    }
}

impl PipelineRuntime {
    pub fn start(
        config: PipelineConfig,
        clock: Arc<dyn Clock>,
        device: Arc<dyn PointerDevice>,
    ) -> Result<(Self, PipelineIngress), PipelineError> {
        if !(1..=10).contains(&config.output_interval_ms) {
            return Err(PipelineError::InvalidOutputInterval {
                actual_ms: config.output_interval_ms,
            });
        }
        let (event_tx, event_rx) = sync_channel(4);
        let shared = Arc::new(SharedState::new(event_tx));
        let batch_slot = LatestSlot::new();
        let target_slot = LatestSlot::new();
        let command_slot = LatestSlot::new();
        let mut workers = Vec::with_capacity(3);

        let targeting_handle = spawn_targeting_worker(
            batch_slot.clone(),
            target_slot.clone(),
            Arc::clone(&shared),
            Arc::clone(&clock),
            config.targeting.clone(),
        )?;
        workers.push(targeting_handle);

        let control_handle = match spawn_control_worker(
            target_slot.clone(),
            command_slot.clone(),
            Arc::clone(&shared),
            config.epoch,
            config.control,
        ) {
            Ok(handle) => handle,
            Err(error) => {
                close_slots(&batch_slot, &target_slot, &command_slot);
                join_workers(&mut workers);
                return Err(error);
            }
        };
        workers.push(control_handle);

        let device_handle = match spawn_device_worker(
            command_slot.clone(),
            Arc::clone(&shared),
            clock,
            device,
            config.epoch,
            config.max_command_age_ns,
            config.output_interval_ms,
        ) {
            Ok(handle) => handle,
            Err(error) => {
                close_slots(&batch_slot, &target_slot, &command_slot);
                join_workers(&mut workers);
                return Err(error);
            }
        };
        workers.push(device_handle);

        shared.output_gate.store(true, Ordering::Release);
        shared.status.store(STATUS_RUNNING, Ordering::Release);
        let ingress = PipelineIngress {
            epoch: config.epoch,
            batches: batch_slot.clone(),
            shared: Arc::clone(&shared),
        };
        Ok((
            Self {
                config,
                shared,
                batch_slot,
                target_slot,
                command_slot,
                event_rx: Some(event_rx),
                workers,
            },
            ingress,
        ))
    }

    pub fn status(&self) -> PipelineStatus {
        self.shared.status()
    }

    pub fn metrics(&self) -> PipelineMetrics {
        snapshot_metrics(&self.shared, &self.batch_slot, &self.command_slot)
    }

    /// Prevent every future device side effect without tearing down ingress.
    /// The supervisor calls this before stopping an upstream perception owner,
    /// so reverse-order cleanup cannot leak one last command.
    pub fn close_output_gate(&self) {
        let _lane = self
            .shared
            .device_lane
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.shared.output_gate.store(false, Ordering::Release);
        self.shared.trigger_active.store(false, Ordering::Release);
    }

    /// Transfer the single lifecycle event stream to the supervisor.
    pub fn take_event_receiver(&mut self) -> Option<Receiver<PipelineEvent>> {
        self.event_rx.take()
    }

    /// Close the output gate first, wake every blocked lane, then join
    /// all worker threads before reporting `Stopped`.
    pub fn shutdown(&mut self) -> Result<PipelineMetrics, PipelineError> {
        self.close_output_gate();
        self.shared.status.store(STATUS_STOPPING, Ordering::Release);
        close_slots(&self.batch_slot, &self.target_slot, &self.command_slot);
        let panicked = join_workers(&mut self.workers);
        if panicked {
            self.shared
                .fault("pipeline worker panicked during shutdown");
            return Err(PipelineError::WorkerPanicked);
        }
        self.shared.status.store(STATUS_STOPPED, Ordering::Release);
        let _ = self.shared.event_tx.try_send(PipelineEvent::Stopped);
        Ok(self.metrics())
    }
}

impl Drop for PipelineRuntime {
    fn drop(&mut self) {
        if !self.workers.is_empty() {
            let _ = self.shutdown();
        }
    }
}

fn guard_worker(shared: &SharedState, name: &'static str, run: impl FnOnce()) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err() {
        shared.fault(format!("{name} worker panicked"));
    }
}

fn spawn_targeting_worker(
    input: LatestSlot<DetectionBatch>,
    output: LatestSlot<TargetedObservation>,
    shared: Arc<SharedState>,
    clock: Arc<dyn Clock>,
    config: TargetingConfig,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-targeting".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "targeting", || {
                let mut targeting = TargetingCore::new(config);
                while let Some(batch) = input.wait_take() {
                    if shared.status() != PipelineStatus::Running {
                        break;
                    }
                    shared
                        .metrics
                        .targeting_batches
                        .fetch_add(1, Ordering::Relaxed);
                    let selection = targeting.select(batch.detections());
                    let (target_id, aim_x, aim_y) = match (
                        selection.target_object_id,
                        selection.target_track_id,
                        selection.target_class_id,
                    ) {
                        (Some(object_id), Some(track_id), Some(class_id)) => {
                            match batch.detections().iter().find(|item| {
                                item.object_id() == object_id && item.class_id() == class_id
                            }) {
                                Some(target) => {
                                    (Some(track_id.0), target.center_x(), target.center_y())
                                }
                                None => {
                                    shared.fault(
                                        "target selection did not belong to its detection batch",
                                    );
                                    output.close();
                                    break;
                                }
                            }
                        }
                        _ => {
                            let (center_x, center_y) = batch.center();
                            (None, center_x, center_y)
                        }
                    };
                    let (crosshair_x, crosshair_y) = batch.center();
                    let now = clock.now().0;
                    let observation = TargetedObservation {
                        stamp: batch.stamp(),
                        target_id,
                        aim_x,
                        aim_y,
                        crosshair_x,
                        crosshair_y,
                        inference_end_ns: now,
                        control_now_ns: now,
                    };
                    if output.publish(observation).is_err() {
                        break;
                    }
                }
            });
            output.close();
        })
        .map_err(|source| PipelineError::Spawn {
            worker: "targeting",
            source,
        })
}

fn spawn_control_worker(
    input: LatestSlot<TargetedObservation>,
    output: LatestSlot<DeviceCommand>,
    shared: Arc<SharedState>,
    epoch: RuntimeEpoch,
    config: DualPhaseConfig,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-control".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "control", || {
                let mut control = DualPhaseControl::new(config);
                while let Some(target) = input.wait_take() {
                    if shared.status() != PipelineStatus::Running {
                        break;
                    }
                    let target_id = target.target_id.unwrap_or(0);
                    let decision = control.calculate(ControlObservation {
                        generation: target.stamp.generation.0,
                        frame_id: target.stamp.generation.0,
                        target_id,
                        capture_ts_ns: target.stamp.captured_at.0,
                        inference_end_ts_ns: target.inference_end_ns,
                        control_now_ns: target.control_now_ns,
                        aim_x: target.aim_x,
                        aim_y: target.aim_y,
                        crosshair_x: target.crosshair_x,
                        crosshair_y: target.crosshair_y,
                        target_valid: target.target_id.is_some(),
                        trigger_active: shared.trigger_active.load(Ordering::Acquire),
                    });
                    shared
                        .metrics
                        .control_decisions
                        .fetch_add(1, Ordering::Relaxed);
                    if !decision.emit_allowed {
                        shared
                            .metrics
                            .blocked_decisions
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    let command = DeviceCommand {
                        epoch,
                        generation: target.stamp.generation,
                        issued_at: novasight_core::MonotonicNanos(target.control_now_ns),
                        target_object_id: target_id,
                        delta_x_counts: decision.dx,
                        delta_y_counts: decision.dy,
                    };
                    if output.publish(command).is_err() {
                        break;
                    }
                }
            });
            output.close();
        })
        .map_err(|source| PipelineError::Spawn {
            worker: "control",
            source,
        })
}

fn spawn_device_worker(
    input: LatestSlot<DeviceCommand>,
    shared: Arc<SharedState>,
    clock: Arc<dyn Clock>,
    device: Arc<dyn PointerDevice>,
    epoch: RuntimeEpoch,
    max_command_age_ns: u64,
    output_interval_ms: u64,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-device".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "device", || {
                let interval = Duration::from_millis(output_interval_ms);
                while input.wait_interval(interval) {
                    let Some(command) = input.try_take() else {
                        continue;
                    };
                    let _lane = shared
                        .device_lane
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if !shared.output_gate.load(Ordering::Acquire)
                        || shared.status() != PipelineStatus::Running
                    {
                        break;
                    }
                    if !shared.trigger_active.load(Ordering::Acquire) {
                        continue;
                    }
                    if command.epoch != epoch {
                        shared.fault(format!(
                            "device lane rejected epoch {}, expected {}",
                            command.epoch.0, epoch.0
                        ));
                        break;
                    }
                    let age = clock.now().0.saturating_sub(command.issued_at.0);
                    if age > max_command_age_ns {
                        shared
                            .metrics
                            .stale_commands
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    let latest_generation = *shared
                        .metrics
                        .last_generation
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if latest_generation != Some(command.generation) {
                        shared
                            .metrics
                            .superseded_commands
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    if !shared.trigger_active.load(Ordering::Acquire) {
                        continue;
                    }
                    match device.send(*command) {
                        Ok(_) => {
                            shared
                                .metrics
                                .device_receipts
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(error) => {
                            shared.fault(format!("pointer device failed: {error}"));
                            break;
                        }
                    }
                }
            });
        })
        .map_err(|source| PipelineError::Spawn {
            worker: "device",
            source,
        })
}

fn close_slots(
    batches: &LatestSlot<DetectionBatch>,
    targets: &LatestSlot<TargetedObservation>,
    commands: &LatestSlot<DeviceCommand>,
) {
    batches.close();
    targets.close();
    commands.close();
}

fn join_workers(workers: &mut Vec<JoinHandle<()>>) -> bool {
    let mut panicked = false;
    while let Some(worker) = workers.pop() {
        if worker.join().is_err() {
            panicked = true;
        }
    }
    panicked
}

fn snapshot_metrics(
    shared: &SharedState,
    batches: &LatestSlot<DetectionBatch>,
    commands: &LatestSlot<DeviceCommand>,
) -> PipelineMetrics {
    PipelineMetrics {
        status: shared.status(),
        received_batches: shared.metrics.received_batches.load(Ordering::Relaxed),
        input_overwrites: batches.metrics().overwritten,
        targeting_batches: shared.metrics.targeting_batches.load(Ordering::Relaxed),
        control_decisions: shared.metrics.control_decisions.load(Ordering::Relaxed),
        blocked_decisions: shared.metrics.blocked_decisions.load(Ordering::Relaxed),
        command_overwrites: commands.metrics().overwritten,
        superseded_commands: shared.metrics.superseded_commands.load(Ordering::Relaxed),
        stale_commands: shared.metrics.stale_commands.load(Ordering::Relaxed),
        device_receipts: shared.metrics.device_receipts.load(Ordering::Relaxed),
        live_workers: shared.metrics.live_workers.load(Ordering::Acquire),
        last_generation: *shared
            .metrics
            .last_generation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        last_fault: shared
            .metrics
            .last_fault
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone(),
    }
}

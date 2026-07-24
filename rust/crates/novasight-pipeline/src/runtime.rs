use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex, TryLockError,
    atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
    mpsc::{Receiver, SyncSender, sync_channel},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use novasight_core::control::dual_phase_v2::{
    ControlDecision as DualPhaseDecision, ControlObservation, DualPhaseConfig, DualPhaseControl,
};
use novasight_core::control::humanized_motion::HumanizedMotionTelemetry;
use novasight_core::control::recoil::{
    RecoilConfig, RecoilDecision, RecoilInput, TargetRelativeRecoilController,
    mix_tracking_and_recoil,
};
use novasight_core::tracking::{TargetSelection, TargetingConfig, TargetingCore};
use novasight_core::{
    Clock, DetectionBatch, DeviceCommand, Generation, PointerDevice, RuntimeEpoch,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{CrosshairHub, MotionProfileHub};
use crate::{LatestSlot, TryPublishError};

const STATUS_STARTING: u8 = 0;
const STATUS_RUNNING: u8 = 1;
const STATUS_STOPPING: u8 = 2;
const STATUS_STOPPED: u8 = 3;
const STATUS_FAULTED: u8 = 4;
const STATUS_STANDBY: u8 = 5;
const MAX_TELEMETRY_DETECTIONS: usize = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum PipelineStatus {
    Starting,
    Running,
    Stopping,
    #[default]
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
    /// positive; deployment policy may impose a tighter latency target.
    pub output_interval_ms: u64,
    /// Hardware trigger polling cadence. `None` leaves trigger ownership with
    /// the control plane (recording/replay); production devices set this.
    pub trigger_poll_interval_ms: Option<u64>,
    /// Optional vision-verified control origin. The hub owns its template and
    /// observation state; targeting only performs a cheap resolved-point read.
    pub crosshair: Option<CrosshairHub>,
    /// Hot-swappable immutable trajectory profile shared with the daemon.
    pub motion_profiles: Option<MotionProfileHub>,
    /// Independent target-relative recoil branch, evaluated at the output
    /// scheduler cadence rather than the detector cadence.
    pub recoil: RecoilConfig,
}

#[derive(Clone, Copy, Debug)]
struct DeviceWorkerConfig {
    epoch: RuntimeEpoch,
    max_command_age_ns: u64,
    output_interval_ms: u64,
    recoil: RecoilConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            epoch: RuntimeEpoch(1),
            targeting: TargetingConfig::default(),
            control: DualPhaseConfig::default(),
            max_command_age_ns: 55_000_000,
            output_interval_ms: 4,
            trigger_poll_interval_ms: None,
            crosshair: None,
            motion_profiles: None,
            recoil: RecoilConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    #[serde(default)]
    pub device_connected: bool,
    #[serde(default)]
    pub device_error_count: u64,
    #[serde(default)]
    pub device_recovery_count: u64,
    #[serde(default)]
    pub last_device_error: Option<String>,
    pub buttons_available: bool,
    pub button_left: bool,
    pub button_right: bool,
    pub output_gate_open: bool,
    pub live_workers: u64,
    pub last_generation: Option<Generation>,
    pub last_fault: Option<String>,
    pub detections: DetectionTelemetry,
    pub target_selection: TargetSelection,
    pub dual_phase: DualPhaseDecision,
    pub humanized_motion: HumanizedMotionTelemetry,
    pub recoil: RecoilDecision,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DetectionTelemetry {
    pub generation: Option<Generation>,
    pub coordinate_width: u32,
    pub coordinate_height: u32,
    pub items: Vec<DetectionTelemetryItem>,
    pub truncated: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectionTelemetryItem {
    pub object_id: u64,
    pub class_id: u32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
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
    #[error("output scheduler interval must be positive, got {actual_ms}")]
    InvalidOutputInterval { actual_ms: u64 },
    #[error("trigger polling interval must be within 1..=50 ms, got {actual_ms}")]
    InvalidTriggerPollInterval { actual_ms: u64 },
    #[error("invalid recoil configuration: {message}")]
    InvalidRecoilConfig { message: &'static str },
    #[error("pointer device connection failed: {0}")]
    DeviceConnect(String),
    #[error("pointer device disconnect failed: {0}")]
    DeviceDisconnect(String),
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
    device_error_count: AtomicU64,
    device_recovery_count: AtomicU64,
    live_workers: AtomicU64,
    last_generation: Mutex<Option<Generation>>,
    last_fault: Mutex<Option<String>>,
    last_device_error: Mutex<Option<String>>,
    detections: Mutex<Arc<DetectionTelemetry>>,
    target_selection: Mutex<TargetSelection>,
    dual_phase: Mutex<DualPhaseDecision>,
    humanized_motion: Mutex<HumanizedMotionTelemetry>,
    recoil: Mutex<RecoilDecision>,
}

#[derive(Debug)]
struct SharedState {
    status: AtomicU8,
    output_gate: AtomicBool,
    output_gate_min_generation: AtomicU64,
    trigger_active: AtomicBool,
    buttons_available: AtomicBool,
    button_left: AtomicBool,
    button_right: AtomicBool,
    device_connected: AtomicBool,
    external_stop: Arc<AtomicUsize>,
    device_lane: Mutex<()>,
    event_tx: SyncSender<PipelineEvent>,
    metrics: AtomicMetrics,
    latest_recoil_observation: Mutex<Option<RecoilObservation>>,
}

impl SharedState {
    fn new(event_tx: SyncSender<PipelineEvent>, external_stop: Arc<AtomicUsize>) -> Self {
        Self {
            status: AtomicU8::new(STATUS_STARTING),
            output_gate: AtomicBool::new(false),
            output_gate_min_generation: AtomicU64::new(0),
            trigger_active: AtomicBool::new(false),
            buttons_available: AtomicBool::new(false),
            button_left: AtomicBool::new(false),
            button_right: AtomicBool::new(false),
            device_connected: AtomicBool::new(true),
            external_stop,
            device_lane: Mutex::new(()),
            event_tx,
            metrics: AtomicMetrics::default(),
            latest_recoil_observation: Mutex::new(None),
        }
    }

    fn status(&self) -> PipelineStatus {
        PipelineStatus::from_atomic(self.status.load(Ordering::Acquire))
    }

    /// Telemetry must never stall the realtime control lane. A concurrent
    /// status snapshot may keep the previous complete sample for one poll.
    fn record_humanized_motion(&self, value: HumanizedMotionTelemetry) {
        match self.metrics.humanized_motion.try_lock() {
            Ok(mut telemetry) => *telemetry = value,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(poisoned)) => *poisoned.into_inner() = value,
        }
    }

    fn record_dual_phase(&self, value: DualPhaseDecision) {
        match self.metrics.dual_phase.try_lock() {
            Ok(mut telemetry) => *telemetry = value,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(poisoned)) => *poisoned.into_inner() = value,
        }
    }

    fn record_target_selection(&self, value: TargetSelection) {
        match self.metrics.target_selection.try_lock() {
            Ok(mut telemetry) => *telemetry = value,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(poisoned)) => *poisoned.into_inner() = value,
        }
    }

    fn record_detections(&self, batch: &DetectionBatch) {
        // Build the bounded immutable snapshot without holding the publication
        // lock. Readers only clone an Arc while holding it, so publishing a
        // real frame cannot be lost to concurrent status polling and the
        // targeting lane never waits on JSON/vector cloning.
        let telemetry = DetectionTelemetry {
            generation: Some(batch.stamp().generation),
            coordinate_width: batch.coordinate_width(),
            coordinate_height: batch.coordinate_height(),
            items: batch
                .detections()
                .iter()
                .take(MAX_TELEMETRY_DETECTIONS)
                .map(|detection| DetectionTelemetryItem {
                    object_id: detection.object_id(),
                    class_id: detection.class_id(),
                    x: detection.x(),
                    y: detection.y(),
                    width: detection.width(),
                    height: detection.height(),
                    confidence: detection.confidence(),
                })
                .collect(),
            truncated: batch
                .detections()
                .len()
                .saturating_sub(MAX_TELEMETRY_DETECTIONS),
        };
        *self
            .metrics
            .detections
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Arc::new(telemetry);
    }

    fn record_recoil(&self, value: RecoilDecision) {
        match self.metrics.recoil.try_lock() {
            Ok(mut telemetry) => *telemetry = value,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(poisoned)) => *poisoned.into_inner() = value,
        }
    }

    fn clear_control_telemetry(&self) {
        self.record_dual_phase(DualPhaseDecision::default());
        self.record_humanized_motion(HumanizedMotionTelemetry::default());
        self.record_recoil(RecoilDecision::default());
    }

    fn fault(&self, message: impl Into<String>) {
        let message = message.into();
        self.output_gate.store(false, Ordering::Release);
        self.trigger_active.store(false, Ordering::Release);
        self.buttons_available.store(false, Ordering::Release);
        self.button_left.store(false, Ordering::Release);
        self.button_right.store(false, Ordering::Release);
        self.device_connected.store(false, Ordering::Release);
        self.clear_control_telemetry();
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

    fn close_output_gate(&self) {
        let _lane = self
            .device_lane
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.output_gate.store(false, Ordering::Release);
        self.trigger_active.store(false, Ordering::Release);
        self.buttons_available.store(false, Ordering::Release);
        self.button_left.store(false, Ordering::Release);
        self.button_right.store(false, Ordering::Release);
        self.device_connected.store(false, Ordering::Release);
        self.clear_control_telemetry();
    }

    fn record_device_success(&self) {
        if !self.device_connected.swap(true, Ordering::AcqRel) {
            self.metrics
                .device_recovery_count
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_device_error(&self, error: &novasight_core::AppError) {
        self.device_connected.store(false, Ordering::Release);
        if matches!(
            error,
            novasight_core::AppError::PointerDevice {
                code: "reconnect_cooldown",
                ..
            }
        ) {
            return;
        }
        self.metrics
            .device_error_count
            .fetch_add(1, Ordering::Relaxed);
        *self
            .metrics
            .last_device_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error.to_string());
    }

    fn record_device_error_message(&self, message: impl Into<String>) {
        self.device_connected.store(false, Ordering::Release);
        self.metrics
            .device_error_count
            .fetch_add(1, Ordering::Relaxed);
        *self
            .metrics
            .last_device_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(message.into());
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
    /// Non-blocking stop request used by RuntimeHandle. The shared external
    /// cancellation flag retires future sends; this clears the local gate and
    /// trigger cache without waiting for an already-running vendor call.
    pub fn request_output_stop(&self) {
        self.shared.output_gate.store(false, Ordering::Release);
        self.shared.trigger_active.store(false, Ordering::Release);
        self.shared
            .buttons_available
            .store(false, Ordering::Release);
        self.shared.button_left.store(false, Ordering::Release);
        self.shared.button_right.store(false, Ordering::Release);
        self.shared.clear_control_telemetry();
    }

    /// Synchronously retire every future device side effect for this epoch.
    /// RuntimeHandle uses this fast path before enqueueing stop commands so a
    /// busy supervisor cannot delay the safety boundary.
    pub fn close_output_gate(&self) {
        self.shared.close_output_gate();
    }

    /// Update the live trigger cache. Output is disabled by default and
    /// the device lane rechecks this value immediately before sending.
    pub fn set_trigger_active(&self, active: bool) {
        self.shared.trigger_active.store(active, Ordering::Release);
        self.shared.buttons_available.store(true, Ordering::Release);
        self.shared.button_left.store(active, Ordering::Release);
        self.shared.button_right.store(false, Ordering::Release);
        if !active {
            // Returning from trigger release is a synchronization point:
            // no device send from the previous active interval remains.
            drop(
                self.shared
                    .device_lane
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            self.shared.clear_control_telemetry();
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
    detection_confidence: f64,
    track_confidence: f64,
    target_width_px: f64,
    inference_end_ns: u64,
}

#[derive(Clone, Copy, Debug)]
struct RecoilObservation {
    generation: Generation,
    target_id: u64,
    capture_ts_ns: u64,
    error_y_norm: f64,
}

/// Owns the post-inference real-time lanes. Capture/DeepStream keeps
/// ownership of vendor objects and submits caller-owned batches through
/// [`PipelineIngress`]; targeting, control, and device calls each run on
/// a dedicated OS thread.
pub struct PipelineRuntime {
    config: PipelineConfig,
    device: Arc<dyn PointerDevice>,
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
            .field("device", &"<dyn PointerDevice>")
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
        Self::start_with_output_gate(config, clock, device, true)
    }

    /// Start every worker while keeping device output closed. The runtime
    /// supervisor uses this during epoch preparation and opens the gate only
    /// after perception readiness and cancellation checks succeed.
    pub fn start_suspended(
        config: PipelineConfig,
        clock: Arc<dyn Clock>,
        device: Arc<dyn PointerDevice>,
    ) -> Result<(Self, PipelineIngress), PipelineError> {
        Self::start_with_output_gate(config, clock, device, false)
    }

    pub fn start_suspended_with_cancel(
        config: PipelineConfig,
        clock: Arc<dyn Clock>,
        device: Arc<dyn PointerDevice>,
        external_stop: Arc<AtomicUsize>,
    ) -> Result<(Self, PipelineIngress), PipelineError> {
        Self::start_with_output_gate_and_cancel(config, clock, device, false, external_stop)
    }

    fn start_with_output_gate(
        config: PipelineConfig,
        clock: Arc<dyn Clock>,
        device: Arc<dyn PointerDevice>,
        output_gate_open: bool,
    ) -> Result<(Self, PipelineIngress), PipelineError> {
        Self::start_with_output_gate_and_cancel(
            config,
            clock,
            device,
            output_gate_open,
            Arc::new(AtomicUsize::new(0)),
        )
    }

    fn start_with_output_gate_and_cancel(
        config: PipelineConfig,
        clock: Arc<dyn Clock>,
        device: Arc<dyn PointerDevice>,
        output_gate_open: bool,
        external_stop: Arc<AtomicUsize>,
    ) -> Result<(Self, PipelineIngress), PipelineError> {
        if config.output_interval_ms == 0 {
            return Err(PipelineError::InvalidOutputInterval {
                actual_ms: config.output_interval_ms,
            });
        }
        if let Some(actual_ms) = config.trigger_poll_interval_ms
            && !(1..=50).contains(&actual_ms)
        {
            return Err(PipelineError::InvalidTriggerPollInterval { actual_ms });
        }
        config
            .recoil
            .validate()
            .map_err(|message| PipelineError::InvalidRecoilConfig { message })?;
        device
            .connect()
            .map_err(|error| PipelineError::DeviceConnect(error.to_string()))?;
        let mut device_guard = DeviceConnectionGuard::new(Arc::clone(&device));
        let (event_tx, event_rx) = sync_channel(4);
        let shared = Arc::new(SharedState::new(event_tx, external_stop));
        let batch_slot = LatestSlot::new();
        let target_slot = LatestSlot::new();
        let command_slot = LatestSlot::new();
        let mut workers = Vec::with_capacity(4);

        let targeting_handle = spawn_targeting_worker(
            batch_slot.clone(),
            target_slot.clone(),
            Arc::clone(&shared),
            Arc::clone(&clock),
            config.targeting.clone(),
            config.crosshair.clone(),
        )?;
        workers.push(targeting_handle);

        let control_handle = match spawn_control_worker(
            target_slot.clone(),
            command_slot.clone(),
            Arc::clone(&shared),
            Arc::clone(&clock),
            config.epoch,
            config.control,
            config.motion_profiles.clone(),
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
            Arc::clone(&device),
            DeviceWorkerConfig {
                epoch: config.epoch,
                max_command_age_ns: config.max_command_age_ns,
                output_interval_ms: config.output_interval_ms,
                recoil: config.recoil,
            },
        ) {
            Ok(handle) => handle,
            Err(error) => {
                close_slots(&batch_slot, &target_slot, &command_slot);
                join_workers(&mut workers);
                return Err(error);
            }
        };
        workers.push(device_handle);

        if let Some(interval_ms) = config.trigger_poll_interval_ms {
            let trigger_handle =
                match spawn_trigger_worker(Arc::clone(&shared), Arc::clone(&device), interval_ms) {
                    Ok(handle) => handle,
                    Err(error) => {
                        shared.status.store(STATUS_STOPPING, Ordering::Release);
                        close_slots(&batch_slot, &target_slot, &command_slot);
                        join_workers(&mut workers);
                        return Err(error);
                    }
                };
            workers.push(trigger_handle);
        }

        shared
            .output_gate
            .store(output_gate_open, Ordering::Release);
        shared.status.store(STATUS_RUNNING, Ordering::Release);
        let ingress = PipelineIngress {
            epoch: config.epoch,
            batches: batch_slot.clone(),
            shared: Arc::clone(&shared),
        };
        let started = (
            Self {
                config,
                device: Arc::clone(&device),
                shared,
                batch_slot,
                target_slot,
                command_slot,
                event_rx: Some(event_rx),
                workers,
            },
            ingress,
        );
        device_guard.disarm();
        Ok(started)
    }

    pub fn status(&self) -> PipelineStatus {
        self.shared.status()
    }

    pub fn metrics(&self) -> PipelineMetrics {
        snapshot_metrics(&self.shared, &self.batch_slot, &self.command_slot)
    }

    /// Publish a fully prepared epoch to the device lane. Opening is separate
    /// from worker startup so cancelled model switches never expose candidate
    /// output.
    pub fn open_output_gate(&self) {
        if self.shared.status() == PipelineStatus::Running
            && self.shared.external_stop.load(Ordering::Acquire) == 0
        {
            let next_generation = self
                .shared
                .metrics
                .last_generation
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .map_or(0, |generation| generation.0.saturating_add(1));
            self.shared
                .output_gate_min_generation
                .store(next_generation, Ordering::Release);
            let _ = self.command_slot.try_take();
            self.shared.output_gate.store(true, Ordering::Release);
        }
    }

    /// Prevent every future device side effect without tearing down ingress.
    /// The supervisor calls this before stopping an upstream perception owner,
    /// so reverse-order cleanup cannot leak one last command.
    pub fn close_output_gate(&self) {
        self.shared.close_output_gate();
        let _ = self.command_slot.try_take();
    }

    /// Pause device delivery while keeping trigger observation and all
    /// upstream calculation lanes alive. Reopening requires a newer source
    /// generation, so a command calculated during the pause cannot leak.
    pub fn pause_output_gate(&self) {
        let _lane = self
            .shared
            .device_lane
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.shared.output_gate.store(false, Ordering::Release);
        let _ = self.command_slot.try_take();
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
        let disconnect = self.device.disconnect();
        if panicked {
            self.shared
                .fault("pipeline worker panicked during shutdown");
            return Err(PipelineError::WorkerPanicked);
        }
        disconnect.map_err(|error| PipelineError::DeviceDisconnect(error.to_string()))?;
        self.shared.status.store(STATUS_STOPPED, Ordering::Release);
        let _ = self.shared.event_tx.try_send(PipelineEvent::Stopped);
        Ok(self.metrics())
    }
}

struct DeviceConnectionGuard {
    device: Option<Arc<dyn PointerDevice>>,
}

impl DeviceConnectionGuard {
    fn new(device: Arc<dyn PointerDevice>) -> Self {
        Self {
            device: Some(device),
        }
    }

    fn disarm(&mut self) {
        self.device = None;
    }
}

impl Drop for DeviceConnectionGuard {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            let _ = device.disconnect();
        }
    }
}

fn spawn_trigger_worker(
    shared: Arc<SharedState>,
    device: Arc<dyn PointerDevice>,
    interval_ms: u64,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-trigger".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "trigger", || {
                let interval = Duration::from_millis(interval_ms);
                while shared.status() == PipelineStatus::Starting {
                    thread::yield_now();
                }
                while shared.status() == PipelineStatus::Running {
                    match device.buttons() {
                        Ok(Some(buttons)) => {
                            shared.record_device_success();
                            let active = buttons.trigger_active();
                            shared.trigger_active.store(active, Ordering::Release);
                            shared.buttons_available.store(true, Ordering::Release);
                            shared.button_left.store(buttons.left, Ordering::Release);
                            shared.button_right.store(buttons.right, Ordering::Release);
                            if !active {
                                drop(
                                    shared
                                        .device_lane
                                        .lock()
                                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                                );
                                shared.clear_control_telemetry();
                            }
                        }
                        Ok(None) => {
                            shared.buttons_available.store(false, Ordering::Release);
                            let message = "pointer device does not expose a hardware trigger";
                            shared.record_device_error_message(message);
                            shared.fault(message);
                            break;
                        }
                        Err(error) => {
                            shared.record_device_error(&error);
                            shared.trigger_active.store(false, Ordering::Release);
                            shared.clear_control_telemetry();
                            shared.buttons_available.store(false, Ordering::Release);
                            shared.button_left.store(false, Ordering::Release);
                            shared.button_right.store(false, Ordering::Release);
                            if !is_recoverable_pointer_error(&error) {
                                shared.fault(format!("pointer trigger failed: {error}"));
                                break;
                            }
                        }
                    }
                    thread::sleep(interval);
                }
            });
        })
        .map_err(|source| PipelineError::Spawn {
            worker: "trigger",
            source,
        })
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
    crosshair: Option<CrosshairHub>,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-targeting".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "targeting", || {
                let mut targeting = TargetingCore::new(config);
                let mut recoil_target_id = None;
                let mut recoil_heights = VecDeque::with_capacity(3);
                while let Some(batch) = input.wait_take() {
                    if shared.status() != PipelineStatus::Running {
                        break;
                    }
                    shared
                        .metrics
                        .targeting_batches
                        .fetch_add(1, Ordering::Relaxed);
                    shared.record_detections(&batch);
                    let geometric_center = batch.center();
                    let reference = crosshair.as_ref().map(|hub| {
                        hub.resolve(
                            geometric_center.0,
                            geometric_center.1,
                            batch.coordinate_width(),
                            batch.coordinate_height(),
                        )
                    });
                    let targeting_center = reference
                        .as_ref()
                        .map_or(geometric_center, |item| (item.x, item.y));
                    let selection = targeting.select_at(
                        batch.detections(),
                        targeting_center,
                        batch.stamp().captured_at.0,
                    );
                    shared.record_target_selection(selection.clone());
                    let track_confidence = selection.target_identity_confidence.unwrap_or(0.0);
                    let (
                        target_id,
                        aim_x,
                        aim_y,
                        detection_confidence,
                        target_width_px,
                        target_height_px,
                    ) = match (
                        selection.target_object_id,
                        selection.target_track_id,
                        selection.target_class_id,
                        selection.target_aim_x,
                        selection.target_aim_y,
                    ) {
                        (
                            Some(object_id),
                            Some(track_id),
                            Some(class_id),
                            Some(aim_x),
                            Some(aim_y),
                        ) => {
                            match batch.detections().iter().find(|item| {
                                item.object_id() == object_id && item.class_id() == class_id
                            }) {
                                Some(target) => (
                                    Some(track_id.0),
                                    aim_x,
                                    aim_y,
                                    f64::from(target.confidence()),
                                    f64::from(target.width()),
                                    f64::from(target.height()),
                                ),
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
                            let (center_x, center_y) = targeting_center;
                            (None, center_x, center_y, 0.0, 1.0, 1.0)
                        }
                    };
                    let (crosshair_x, crosshair_y) = targeting_center;
                    let now = clock.now().0;
                    let recoil_observation = target_id.map(|target_id| {
                        if recoil_target_id != Some(target_id) {
                            recoil_heights.clear();
                            recoil_target_id = Some(target_id);
                        }
                        if recoil_heights.len() == 3 {
                            recoil_heights.pop_front();
                        }
                        recoil_heights.push_back(target_height_px.max(1.0));
                        let stable_height = median_height(&recoil_heights);
                        RecoilObservation {
                            generation: batch.stamp().generation,
                            target_id,
                            capture_ts_ns: batch.stamp().captured_at.0,
                            error_y_norm: ((aim_y - crosshair_y) / stable_height).clamp(-1.0, 1.0),
                        }
                    });
                    *shared
                        .latest_recoil_observation
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = recoil_observation;
                    let observation = TargetedObservation {
                        stamp: batch.stamp(),
                        target_id,
                        aim_x,
                        aim_y,
                        crosshair_x,
                        crosshair_y,
                        detection_confidence,
                        track_confidence,
                        target_width_px,
                        inference_end_ns: now,
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

fn median_height(values: &VecDeque<f64>) -> f64 {
    let mut ordered = values.iter().copied().collect::<Vec<_>>();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    if ordered.len() % 2 == 0 {
        (ordered[middle - 1] + ordered[middle]) * 0.5
    } else {
        ordered[middle]
    }
}

fn spawn_control_worker(
    input: LatestSlot<TargetedObservation>,
    output: LatestSlot<DeviceCommand>,
    shared: Arc<SharedState>,
    clock: Arc<dyn Clock>,
    epoch: RuntimeEpoch,
    config: DualPhaseConfig,
    motion_profiles: Option<MotionProfileHub>,
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
                    let control_now_ns = clock.now().0;
                    let target_id = target.target_id.unwrap_or(0);
                    let observation = ControlObservation {
                        generation: target.stamp.generation.0,
                        frame_id: target.stamp.generation.0,
                        target_id,
                        capture_ts_ns: target.stamp.captured_at.0,
                        inference_end_ts_ns: target.inference_end_ns,
                        control_now_ns,
                        aim_x: target.aim_x,
                        aim_y: target.aim_y,
                        crosshair_x: target.crosshair_x,
                        crosshair_y: target.crosshair_y,
                        detection_confidence: target.detection_confidence,
                        track_confidence: target.track_confidence,
                        target_valid: target.target_id.is_some(),
                        trigger_active: shared.trigger_active.load(Ordering::Acquire),
                    };
                    let active_profile =
                        motion_profiles.as_ref().and_then(MotionProfileHub::active);
                    let decision = control.calculate_with_profile(
                        observation,
                        active_profile.as_deref(),
                        target.target_width_px,
                    );
                    shared.record_dual_phase(decision);
                    shared.record_humanized_motion(decision.humanized_motion);
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
                        issued_at: novasight_core::MonotonicNanos(control_now_ns),
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
    config: DeviceWorkerConfig,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-device".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "device", || {
                let interval = Duration::from_millis(config.output_interval_ms);
                let interval_s = config.output_interval_ms as f64 / 1_000.0;
                let mut last_tick_ns = None;
                let mut recoil = TargetRelativeRecoilController::new(config.recoil)
                    .expect("pipeline validates recoil config before worker startup");
                while input.wait_interval(interval) {
                    let now_ns = clock.now().0;
                    let dt_s = last_tick_ns.map_or(interval_s, |last_tick_ns| {
                        (now_ns.saturating_sub(last_tick_ns) as f64 / 1_000_000_000.0)
                            .clamp(0.0, interval_s * 2.0)
                    });
                    last_tick_ns = Some(now_ns);
                    let observation = *shared
                        .latest_recoil_observation
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let observation_age_ms = observation.map_or(0.0, |observation| {
                        now_ns.saturating_sub(observation.capture_ts_ns) as f64 / 1_000_000.0
                    });
                    let recoil_decision = recoil.calculate(RecoilInput {
                        firing: shared.buttons_available.load(Ordering::Acquire)
                            && shared.button_left.load(Ordering::Acquire)
                            && shared.output_gate.load(Ordering::Acquire)
                            && shared.external_stop.load(Ordering::Acquire) == 0
                            && shared.status() == PipelineStatus::Running,
                        now_ns,
                        dt_s,
                        target_valid: observation.is_some(),
                        target_id: observation.map(|value| value.target_id),
                        source_generation: observation.map(|value| value.generation.0),
                        observation_age_ms,
                        error_y_norm: observation.map_or(0.0, |value| value.error_y_norm),
                    });
                    shared.record_recoil(recoil_decision);

                    let tracking = input.try_take().map(|command| *command);
                    let _lane = shared
                        .device_lane
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if shared.external_stop.load(Ordering::Acquire) != 0
                        || shared.status() != PipelineStatus::Running
                    {
                        break;
                    }
                    if !shared.output_gate.load(Ordering::Acquire) {
                        continue;
                    }
                    if !shared.trigger_active.load(Ordering::Acquire) {
                        continue;
                    }
                    let command = match tracking {
                        Some(mut command) => {
                            if recoil_decision.source_generation == Some(command.generation.0) {
                                command.delta_y_counts = mix_tracking_and_recoil(
                                    command.delta_y_counts,
                                    recoil_decision,
                                );
                            }
                            command
                        }
                        None => {
                            let Some(observation) = observation else {
                                continue;
                            };
                            if !recoil_decision.engaged() || recoil_decision.emitted_counts_y == 0 {
                                continue;
                            }
                            DeviceCommand {
                                epoch: config.epoch,
                                generation: observation.generation,
                                issued_at: novasight_core::MonotonicNanos(now_ns),
                                target_object_id: observation.target_id,
                                delta_x_counts: 0,
                                delta_y_counts: recoil_decision.emitted_counts_y,
                            }
                        }
                    };
                    if command.generation.0
                        < shared.output_gate_min_generation.load(Ordering::Acquire)
                    {
                        continue;
                    }
                    if command.epoch != config.epoch {
                        shared.fault(format!(
                            "device lane rejected epoch {}, expected {}",
                            command.epoch.0, config.epoch.0
                        ));
                        break;
                    }
                    let age = now_ns.saturating_sub(command.issued_at.0);
                    if age > config.max_command_age_ns {
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
                    match device.send(command) {
                        Ok(_) => {
                            shared.record_device_success();
                            shared
                                .metrics
                                .device_receipts
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(error) => {
                            shared.record_device_error(&error);
                            shared.trigger_active.store(false, Ordering::Release);
                            shared.clear_control_telemetry();
                            if !is_recoverable_pointer_error(&error) {
                                shared.fault(format!("pointer device failed: {error}"));
                                break;
                            }
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

fn is_recoverable_pointer_error(error: &novasight_core::AppError) -> bool {
    matches!(
        error,
        novasight_core::AppError::PointerDevice {
            code: "driver_timeout"
                | "helper_exited"
                | "helper_spawn_failed"
                | "reconnect_cooldown"
                | "driver_send_failed"
                | "driver_protocol_failed"
                | "monitor_stale",
            ..
        }
    )
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
    let detections = Arc::clone(
        &shared
            .metrics
            .detections
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    );
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
        device_connected: shared.device_connected.load(Ordering::Acquire),
        device_error_count: shared.metrics.device_error_count.load(Ordering::Relaxed),
        device_recovery_count: shared.metrics.device_recovery_count.load(Ordering::Relaxed),
        last_device_error: shared
            .metrics
            .last_device_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone(),
        buttons_available: shared.buttons_available.load(Ordering::Acquire),
        button_left: shared.button_left.load(Ordering::Acquire),
        button_right: shared.button_right.load(Ordering::Acquire),
        output_gate_open: shared.output_gate.load(Ordering::Acquire),
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
        detections: (*detections).clone(),
        target_selection: shared
            .metrics
            .target_selection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone(),
        dual_phase: *shared
            .metrics
            .dual_phase
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        humanized_motion: *shared
            .metrics
            .humanized_motion
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        recoil: *shared
            .metrics
            .recoil
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    }
}

use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex, TryLockError,
    atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU64, AtomicUsize, Ordering, fence},
    mpsc::{Receiver, SyncSender, sync_channel},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use novasight_core::controller::recoil::{
    IntervalRecoilController, RecoilConfig, RecoilDecision, RecoilInput, mix_tracking_and_recoil,
};
use novasight_core::controller::{
    ActuationFeedback, ContinuousControl, ContinuousControlConfig, ControlDecision,
    ControlObservation,
};
use novasight_core::tracking::{TargetSelection, TargetingConfig, TargetingCore, TrackState};
use novasight_core::{
    Clock, DetectionBatch, DeviceCommand, DeviceReceipt, Generation, PointerDevice,
    PredictionTruthConfig, PredictionTruthReport, PredictionTruthSample, RuntimeEpoch,
    score_prediction_truth,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::CrosshairHub;
use crate::LatestSlot;
use crate::slot::{MonotonicPublishError, TryMonotonicPublishError};

const STATUS_STARTING: u8 = 0;
const STATUS_RUNNING: u8 = 1;
const STATUS_STOPPING: u8 = 2;
const STATUS_STOPPED: u8 = 3;
const STATUS_FAULTED: u8 = 4;
const STATUS_STANDBY: u8 = 5;
const MAX_TELEMETRY_DETECTIONS: usize = 64;
const DETECTION_TELEMETRY_MIN_INTERVAL_NS: u64 = 200_000_000;
const CONTROL_TELEMETRY_MIN_INTERVAL_NS: u64 = 50_000_000;
const PREDICTION_TRUTH_SAMPLE_CAPACITY: usize = 240;

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TriggerMode {
    #[default]
    Always,
    Hardware,
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub epoch: RuntimeEpoch,
    pub targeting: TargetingConfig,
    pub control: ContinuousControlConfig,
    /// Hardware trigger polling cadence. `None` leaves trigger ownership with
    /// the control plane (recording/replay); production devices set this.
    pub trigger_poll_interval_ms: Option<u64>,
    /// Runtime activation policy. `Always` still respects the
    /// output gate, device connection, freshness, and target-validity guards.
    pub trigger_mode: TriggerMode,
    /// Minimum device-to-capture visibility delay after a successful move.
    pub actuation_feedback_delay_ns: u64,
    /// Optional vision-verified control origin. The hub owns its template and
    /// observation state; targeting only performs a cheap resolved-point read.
    pub crosshair: Option<CrosshairHub>,
    /// Positive-Y recoil contribution mixed into the newest safe output plan.
    /// A plan may carry zero tracking demand when the aim is already settled
    /// or when target gating is disabled and no target is present.
    pub recoil: RecoilConfig,
}

#[derive(Clone, Debug)]
pub struct PipelineLiveConfig {
    pub targeting: TargetingConfig,
    pub control: ContinuousControlConfig,
    pub actuation_feedback_delay_ns: u64,
}

impl From<&PipelineConfig> for PipelineLiveConfig {
    fn from(config: &PipelineConfig) -> Self {
        Self {
            targeting: config.targeting.clone(),
            control: config.control,
            actuation_feedback_delay_ns: config.actuation_feedback_delay_ns,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DeviceWorkerConfig {
    epoch: RuntimeEpoch,
    recoil: RecoilConfig,
}

#[derive(Clone, Copy, Debug)]
struct ControlWorkerConfig {
    epoch: RuntimeEpoch,
    control: ContinuousControlConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            epoch: RuntimeEpoch(1),
            targeting: TargetingConfig::default(),
            control: ContinuousControlConfig::default(),
            trigger_poll_interval_ms: None,
            trigger_mode: TriggerMode::Always,
            actuation_feedback_delay_ns: 4_000_000,
            crosshair: None,
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
    pub device_receipts: u64,
    #[serde(default)]
    pub last_device_receipt: Option<DeviceReceipt>,
    #[serde(default)]
    pub device_connected: bool,
    #[serde(default)]
    pub device_connection_enabled: bool,
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
    pub control: ControlDecision,
    #[serde(default)]
    pub prediction_truth: PredictionTruthReport,
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
    #[error("trigger polling interval must be within 1..=50 ms, got {actual_ms}")]
    InvalidTriggerPollInterval { actual_ms: u64 },
    #[error("actuation feedback delay must be within 0..=100 ms, got {actual_ns} ns")]
    InvalidActuationFeedbackDelay { actual_ns: u64 },
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
    device_error_count: AtomicU64,
    device_recovery_count: AtomicU64,
    live_workers: AtomicU64,
    last_generation: AtomicU64,
    last_fault: Mutex<Option<String>>,
    last_device_error: Mutex<Option<String>>,
    vision: Mutex<VisionTelemetry>,
    last_device_receipt: AtomicDeviceReceipt,
    control: Mutex<ControlDecision>,
    prediction_truth: Mutex<VecDeque<PredictionTruthSample>>,
    recoil: Mutex<RecoilDecision>,
}

impl AtomicMetrics {
    fn last_generation(&self) -> Option<Generation> {
        (self.received_batches.load(Ordering::Acquire) != 0)
            .then(|| Generation(self.last_generation.load(Ordering::Acquire)))
    }

    fn record_received_generation(&self, generation: Generation) {
        self.last_generation
            .fetch_max(generation.0, Ordering::AcqRel);
        self.received_batches.fetch_add(1, Ordering::Release);
    }
}

#[derive(Debug, Default)]
struct VisionTelemetry {
    detections: DetectionTelemetry,
    target_selection: TargetSelection,
}

#[derive(Debug, Default)]
struct AtomicDeviceReceipt {
    sequence: AtomicU64,
    accepted_count: AtomicU64,
    attempt: AtomicU64,
    epoch: AtomicU64,
    generation: AtomicU64,
    issued_at: AtomicU64,
    target_object_id: AtomicU64,
    delta_x_counts: AtomicI32,
    delta_y_counts: AtomicI32,
}

impl AtomicDeviceReceipt {
    fn store(&self, receipt: DeviceReceipt) {
        self.sequence.fetch_add(1, Ordering::AcqRel);
        let accepted_count = self
            .accepted_count
            .load(Ordering::Relaxed)
            .saturating_add(1);
        self.accepted_count.store(accepted_count, Ordering::Relaxed);
        self.attempt.store(receipt.attempt, Ordering::Relaxed);
        self.epoch.store(receipt.epoch.0, Ordering::Relaxed);
        self.generation
            .store(receipt.generation.0, Ordering::Relaxed);
        self.issued_at.store(receipt.issued_at.0, Ordering::Relaxed);
        self.target_object_id
            .store(receipt.target_object_id, Ordering::Relaxed);
        self.delta_x_counts
            .store(receipt.delta_x_counts, Ordering::Relaxed);
        self.delta_y_counts
            .store(receipt.delta_y_counts, Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
    }

    fn snapshot(&self) -> Option<(u64, DeviceReceipt)> {
        let mut attempts = 0_u32;
        loop {
            let before = self.sequence.load(Ordering::Acquire);
            if before == 0 {
                return None;
            }
            if before & 1 != 0 {
                attempts = attempts.saturating_add(1);
                if attempts.is_multiple_of(16) {
                    thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }
            let accepted_count = self.accepted_count.load(Ordering::Relaxed);
            let receipt = DeviceReceipt {
                attempt: self.attempt.load(Ordering::Relaxed),
                epoch: RuntimeEpoch(self.epoch.load(Ordering::Relaxed)),
                generation: Generation(self.generation.load(Ordering::Relaxed)),
                issued_at: novasight_core::MonotonicNanos(self.issued_at.load(Ordering::Relaxed)),
                target_object_id: self.target_object_id.load(Ordering::Relaxed),
                delta_x_counts: self.delta_x_counts.load(Ordering::Relaxed),
                delta_y_counts: self.delta_y_counts.load(Ordering::Relaxed),
            };
            fence(Ordering::Acquire);
            if self.sequence.load(Ordering::Acquire) == before {
                return Some((accepted_count, receipt));
            }
            attempts = attempts.saturating_add(1);
            if attempts.is_multiple_of(16) {
                thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }
}

#[derive(Debug)]
struct SharedState {
    status: AtomicU8,
    output_gate: AtomicBool,
    output_gate_min_generation: AtomicU64,
    hardware_trigger_required: AtomicBool,
    trigger_active: AtomicBool,
    buttons_available: AtomicBool,
    button_left: AtomicBool,
    button_left_epoch: AtomicU64,
    button_right: AtomicBool,
    device_connected: AtomicBool,
    device_connection_enabled: AtomicBool,
    external_stop: Arc<AtomicUsize>,
    last_vision_telemetry_at_ns: AtomicU64,
    latest_successful_send_x_ts_ns: AtomicU64,
    latest_successful_send_y_ts_ns: AtomicU64,
    targeting_config_version: AtomicU64,
    control_config_version: AtomicU64,
    actuation_feedback_delay_ns: AtomicU64,
    recoil_config_version: AtomicU64,
    recoil_enabled: AtomicBool,
    recoil_require_target: AtomicBool,
    latest_seen_generation: AtomicU64,
    device_lane: Mutex<()>,
    event_tx: SyncSender<PipelineEvent>,
    metrics: AtomicMetrics,
    targeting_config: Mutex<TargetingConfig>,
    control_config: Mutex<ContinuousControlConfig>,
    recoil_config: Mutex<RecoilConfig>,
}

impl SharedState {
    fn new(
        event_tx: SyncSender<PipelineEvent>,
        external_stop: Arc<AtomicUsize>,
        device_connected: bool,
        trigger_mode: TriggerMode,
        live_config: PipelineLiveConfig,
        recoil_config: RecoilConfig,
    ) -> Self {
        let metrics = AtomicMetrics {
            vision: Mutex::new(VisionTelemetry {
                detections: DetectionTelemetry {
                    items: Vec::with_capacity(MAX_TELEMETRY_DETECTIONS),
                    ..DetectionTelemetry::default()
                },
                target_selection: TargetSelection {
                    rejected_class_ids: Vec::with_capacity(
                        novasight_core::tracking::MAX_TRACK_CANDIDATES,
                    ),
                    ..TargetSelection::default()
                },
            }),
            prediction_truth: Mutex::new(VecDeque::with_capacity(PREDICTION_TRUTH_SAMPLE_CAPACITY)),
            ..AtomicMetrics::default()
        };
        Self {
            status: AtomicU8::new(STATUS_STARTING),
            output_gate: AtomicBool::new(false),
            output_gate_min_generation: AtomicU64::new(0),
            hardware_trigger_required: AtomicBool::new(trigger_mode == TriggerMode::Hardware),
            trigger_active: AtomicBool::new(false),
            buttons_available: AtomicBool::new(false),
            button_left: AtomicBool::new(false),
            button_left_epoch: AtomicU64::new(1),
            button_right: AtomicBool::new(false),
            device_connected: AtomicBool::new(device_connected),
            device_connection_enabled: AtomicBool::new(true),
            external_stop,
            last_vision_telemetry_at_ns: AtomicU64::new(0),
            latest_successful_send_x_ts_ns: AtomicU64::new(0),
            latest_successful_send_y_ts_ns: AtomicU64::new(0),
            targeting_config_version: AtomicU64::new(1),
            control_config_version: AtomicU64::new(1),
            actuation_feedback_delay_ns: AtomicU64::new(live_config.actuation_feedback_delay_ns),
            recoil_config_version: AtomicU64::new(1),
            recoil_enabled: AtomicBool::new(recoil_config.enabled),
            recoil_require_target: AtomicBool::new(recoil_config.require_target),
            latest_seen_generation: AtomicU64::new(0),
            device_lane: Mutex::new(()),
            event_tx,
            metrics,
            targeting_config: Mutex::new(live_config.targeting),
            control_config: Mutex::new(live_config.control),
            recoil_config: Mutex::new(recoil_config),
        }
    }

    fn status(&self) -> PipelineStatus {
        PipelineStatus::from_atomic(self.status.load(Ordering::Acquire))
    }

    fn latest_seen_generation(&self) -> Option<Generation> {
        (self.metrics.received_batches.load(Ordering::Acquire) != 0)
            .then(|| Generation(self.latest_seen_generation.load(Ordering::Acquire)))
    }

    fn set_button_left(&self, pressed: bool) {
        if self.button_left.swap(pressed, Ordering::AcqRel) != pressed {
            self.button_left_epoch.fetch_add(1, Ordering::Release);
            if !pressed {
                // Release is an immediate business-state transition even when
                // settled tracking produces no subsequent device plan.
                self.record_recoil(RecoilDecision::default());
            }
        }
    }

    /// Telemetry must never stall the realtime control lane. A concurrent
    /// status snapshot may keep the previous complete sample for one poll.
    fn record_control_decision(&self, value: ControlDecision) {
        match self.metrics.control.try_lock() {
            Ok(mut telemetry) => *telemetry = value,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(poisoned)) => *poisoned.into_inner() = value,
        }
    }

    fn record_prediction_truth(&self, value: ControlDecision) {
        let sample = PredictionTruthSample::from_control_decision(&value);
        match self.metrics.prediction_truth.try_lock() {
            Ok(mut samples) => {
                if samples.len() == PREDICTION_TRUTH_SAMPLE_CAPACITY {
                    samples.pop_front();
                }
                samples.push_back(sample);
            }
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut samples = poisoned.into_inner();
                samples.clear();
                samples.push_back(sample);
            }
        }
    }

    fn record_vision(&self, batch: &DetectionBatch, selection: &TargetSelection) {
        // UI telemetry is best-effort: status polling may retain the previous
        // complete sample, but it must never stall or allocate in targeting.
        let mut telemetry = match self.metrics.vision.try_lock() {
            Ok(telemetry) => telemetry,
            Err(TryLockError::WouldBlock) => return,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        telemetry.detections.generation = Some(batch.stamp().generation);
        telemetry.detections.coordinate_width = batch.coordinate_width();
        telemetry.detections.coordinate_height = batch.coordinate_height();
        telemetry.detections.items.clear();
        telemetry.detections.items.extend(
            batch
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
                }),
        );
        telemetry.detections.truncated = batch
            .detections()
            .len()
            .saturating_sub(MAX_TELEMETRY_DETECTIONS);
        telemetry.target_selection.clone_from(selection);
    }

    fn vision_telemetry_due(&self, captured_at_ns: u64) -> bool {
        let encoded_now = captured_at_ns.saturating_add(1);
        let previous = self.last_vision_telemetry_at_ns.load(Ordering::Relaxed);
        if previous != 0
            && encoded_now.saturating_sub(previous) < DETECTION_TELEMETRY_MIN_INTERVAL_NS
        {
            return false;
        }
        self.last_vision_telemetry_at_ns
            .store(encoded_now, Ordering::Relaxed);
        true
    }

    fn record_device_receipt(
        &self,
        receipt: DeviceReceipt,
        accepted_at_ns: u64,
        tracking_x_counts: i32,
        tracking_y_counts: i32,
    ) {
        // The feedback gate prevents a visual controller from issuing the
        // same tracking correction against a frame that predates that move.
        // Recoil is an independent feed-forward disturbance compensation: a
        // recoil-only tick must not keep visual Y tracking permanently pending.
        if tracking_x_counts != 0 {
            self.latest_successful_send_x_ts_ns
                .store(accepted_at_ns, Ordering::Release);
        }
        if tracking_y_counts != 0 {
            self.latest_successful_send_y_ts_ns
                .store(accepted_at_ns, Ordering::Release);
        }
        self.metrics.last_device_receipt.store(receipt);
    }

    fn record_recoil(&self, value: RecoilDecision) {
        match self.metrics.recoil.try_lock() {
            Ok(mut telemetry) => *telemetry = value,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(poisoned)) => *poisoned.into_inner() = value,
        }
    }

    fn clear_control_telemetry(&self) {
        self.record_control_decision(ControlDecision::default());
        self.record_recoil(RecoilDecision::default());
        self.metrics
            .prediction_truth
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    fn fault(&self, message: impl Into<String>) {
        let message = message.into();
        self.output_gate.store(false, Ordering::Release);
        self.trigger_active.store(false, Ordering::Release);
        self.buttons_available.store(false, Ordering::Release);
        self.set_button_left(false);
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
        self.set_button_left(false);
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
        if matches!(
            error,
            novasight_core::AppError::PointerDevice {
                code: "reconnect_cooldown",
                ..
            }
        ) {
            self.device_connected.store(false, Ordering::Release);
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
        // Publish the externally visible disconnect only after its diagnostic
        // fields are complete. Snapshot readers that observe `false` through
        // the acquire load can then always explain the degraded state.
        self.device_connected.store(false, Ordering::Release);
    }

    fn record_device_error_message(&self, message: impl Into<String>) {
        self.metrics
            .device_error_count
            .fetch_add(1, Ordering::Relaxed);
        *self
            .metrics
            .last_device_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(message.into());
        self.device_connected.store(false, Ordering::Release);
    }

    fn record_manual_device_disconnect(&self) {
        self.device_connected.store(false, Ordering::Release);
        self.trigger_active.store(false, Ordering::Release);
        self.buttons_available.store(false, Ordering::Release);
        self.set_button_left(false);
        self.button_right.store(false, Ordering::Release);
        self.clear_control_telemetry();
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
    command_slot: LatestSlot<OutputPlan>,
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
    /// Publish the newest generation while sharing the same safety boundary as
    /// the final device send. Once this returns, no older command can enter the
    /// vendor call.
    fn observe_generation(&self, generation: Generation) -> Result<(), PipelineError> {
        let _lane = self
            .shared
            .device_lane
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.observe_generation_locked(generation)
    }

    /// Realtime observation never waits for a device call. A busy lane drops
    /// this frame; the next capture can publish the newer generation.
    fn try_observe_generation(&self, generation: Generation) -> Result<(), PipelineError> {
        let _lane = match self.shared.device_lane.try_lock() {
            Ok(lane) => lane,
            Err(TryLockError::WouldBlock) => return Err(PipelineError::IngressBusy),
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        self.observe_generation_locked(generation)
    }

    /// Equality remains retryable until the latest slot accepts the batch;
    /// strictly older attempts fail immediately.
    fn observe_generation_locked(&self, generation: Generation) -> Result<(), PipelineError> {
        let mut previous = self.shared.latest_seen_generation.load(Ordering::Acquire);
        loop {
            if generation.0 < previous {
                return Err(PipelineError::NonMonotonicGeneration {
                    previous,
                    actual: generation.0,
                });
            }
            if generation.0 == previous {
                return Ok(());
            }
            match self.shared.latest_seen_generation.compare_exchange_weak(
                previous,
                generation.0,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => previous = actual,
            }
        }
    }

    /// Non-blocking stop request used by RuntimeHandle. The shared external
    /// cancellation flag retires future sends; this clears the local gate and
    /// trigger cache without waiting for an already-running vendor call.
    pub fn request_output_stop(&self) {
        self.shared.output_gate.store(false, Ordering::Release);
        self.shared.trigger_active.store(false, Ordering::Release);
        self.shared
            .buttons_available
            .store(false, Ordering::Release);
        self.shared.set_button_left(false);
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
        self.shared.set_button_left(active);
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

    /// Change the current epoch's activation policy without rebuilding the
    /// capture or inference pipeline. Entering hardware mode retires any
    /// in-flight unconditional command before returning.
    pub fn set_trigger_mode(&self, mode: TriggerMode) {
        let hardware_required = mode == TriggerMode::Hardware;
        self.shared
            .hardware_trigger_required
            .store(hardware_required, Ordering::Release);
        if hardware_required {
            drop(
                self.shared
                    .device_lane
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            self.shared.clear_control_telemetry();
        }
    }

    pub fn trigger_mode(&self) -> TriggerMode {
        if self
            .shared
            .hardware_trigger_required
            .load(Ordering::Acquire)
        {
            TriggerMode::Hardware
        } else {
            TriggerMode::Always
        }
    }

    /// Replace recoil behavior for the active epoch without rebuilding capture
    /// or inference. The device worker reads the mutex only after this version
    /// changes; ordinary output ticks pay one atomic load and no config lock.
    pub fn set_recoil_config(&self, config: RecoilConfig) -> Result<(), PipelineError> {
        config
            .validate()
            .map_err(|message| PipelineError::InvalidRecoilConfig { message })?;
        let _lane = self
            .shared
            .device_lane
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *self
            .shared
            .recoil_config
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = config;
        self.shared
            .recoil_enabled
            .store(config.enabled, Ordering::Release);
        self.shared
            .recoil_require_target
            .store(config.require_target, Ordering::Release);
        self.shared
            .recoil_config_version
            .fetch_add(1, Ordering::Release);
        self.shared.clear_control_telemetry();
        Ok(())
    }

    /// Replace targeting and mouse-control tuning for the active epoch without
    /// rebuilding capture, inference, or the pointer device lane.
    pub fn set_live_config(&self, config: PipelineLiveConfig) -> Result<(), PipelineError> {
        if config.actuation_feedback_delay_ns > 100_000_000 {
            return Err(PipelineError::InvalidActuationFeedbackDelay {
                actual_ns: config.actuation_feedback_delay_ns,
            });
        }
        let _lane = self
            .shared
            .device_lane
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *self
            .shared
            .targeting_config
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = config.targeting;
        *self
            .shared
            .control_config
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = config.control;
        self.shared
            .actuation_feedback_delay_ns
            .store(config.actuation_feedback_delay_ns, Ordering::Release);
        self.shared
            .targeting_config_version
            .fetch_add(1, Ordering::Release);
        self.shared
            .control_config_version
            .fetch_add(1, Ordering::Release);
        let next_generation = self
            .shared
            .latest_seen_generation()
            .map_or(0, |generation| generation.0.saturating_add(1));
        self.shared
            .output_gate_min_generation
            .store(next_generation, Ordering::Release);
        let _ = self.command_slot.try_take();
        self.shared.clear_control_telemetry();
        Ok(())
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
        self.observe_generation(stamp.generation)?;
        match self.batches.publish_monotonic(stamp.generation.0, batch) {
            Ok(_) => {}
            Err(MonotonicPublishError::Closed) => return Err(PipelineError::NotRunning),
            Err(MonotonicPublishError::NonMonotonic { previous }) => {
                return Err(PipelineError::NonMonotonicGeneration {
                    previous,
                    actual: stamp.generation.0,
                });
            }
        }
        self.shared
            .metrics
            .record_received_generation(stamp.generation);
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
        self.try_observe_generation(stamp.generation)?;
        match self
            .batches
            .try_publish_monotonic(stamp.generation.0, batch)
        {
            Ok(_) => {}
            Err(TryMonotonicPublishError::Busy) => return Err(PipelineError::IngressBusy),
            Err(TryMonotonicPublishError::Closed) => return Err(PipelineError::NotRunning),
            Err(TryMonotonicPublishError::NonMonotonic { previous }) => {
                return Err(PipelineError::NonMonotonicGeneration {
                    previous,
                    actual: stamp.generation.0,
                });
            }
        }
        self.shared
            .metrics
            .record_received_generation(stamp.generation);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct TargetedObservation {
    stamp: novasight_core::FrameStamp,
    target_id: Option<u64>,
    recoil_target_valid: bool,
    aim_x: f64,
    aim_y: f64,
    crosshair_x: f64,
    crosshair_y: f64,
    detection_confidence: f64,
    track_confidence: f64,
    track_rebuilt: bool,
}

/// One latest-only opportunity to compose physical output from a fresh
/// perception observation. Tracking demand may be zero; recoil is evaluated
/// later against the freshest button/config state while the device lane lock
/// is held. This preserves one combined send without making recoil depend on
/// the controller having produced a non-zero tracking command.
#[derive(Clone, Copy, Debug)]
struct OutputPlan {
    command: DeviceCommand,
    recoil_target_valid: bool,
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
    command_slot: LatestSlot<OutputPlan>,
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
        if let Some(actual_ms) = config.trigger_poll_interval_ms
            && !(1..=50).contains(&actual_ms)
        {
            return Err(PipelineError::InvalidTriggerPollInterval { actual_ms });
        }
        if config.actuation_feedback_delay_ns > 100_000_000 {
            return Err(PipelineError::InvalidActuationFeedbackDelay {
                actual_ns: config.actuation_feedback_delay_ns,
            });
        }
        config
            .recoil
            .validate()
            .map_err(|message| PipelineError::InvalidRecoilConfig { message })?;
        let initial_device_error = match device.connect() {
            Ok(()) => None,
            Err(error) if is_recoverable_pointer_error(&error) => Some(error),
            Err(error) => return Err(PipelineError::DeviceConnect(error.to_string())),
        };
        let mut device_guard = initial_device_error
            .is_none()
            .then(|| DeviceConnectionGuard::new(Arc::clone(&device)));
        let (event_tx, event_rx) = sync_channel(4);
        let shared = Arc::new(SharedState::new(
            event_tx,
            external_stop,
            initial_device_error.is_none(),
            config.trigger_mode,
            PipelineLiveConfig::from(&config),
            config.recoil,
        ));
        if let Some(error) = &initial_device_error {
            // Capture and inference remain useful while a commissioned output
            // device is temporarily offline. Trigger polling owns reconnect;
            // the closed output gate and cleared trigger state prevent any
            // stale command from escaping in the meantime.
            shared.record_device_error(error);
        }
        let batch_slot = LatestSlot::new();
        let target_slot = LatestSlot::new();
        let command_slot = LatestSlot::new();
        let mut workers = Vec::with_capacity(4);

        let targeting_handle = spawn_targeting_worker(
            batch_slot.clone(),
            target_slot.clone(),
            Arc::clone(&shared),
            config.targeting.clone(),
            config.crosshair.clone(),
        )?;
        workers.push(targeting_handle);

        let control_handle = match spawn_control_worker(
            target_slot.clone(),
            command_slot.clone(),
            Arc::clone(&shared),
            Arc::clone(&clock),
            ControlWorkerConfig {
                epoch: config.epoch,
                control: config.control,
            },
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
            command_slot: command_slot.clone(),
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
        if let Some(device_guard) = &mut device_guard {
            device_guard.disarm();
        }
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
                .latest_seen_generation()
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

    /// Enable the low-rate device health lane and attempt an immediate
    /// connection. Recoverable failures remain eligible for background retry.
    pub fn connect_device(&self) -> Result<(), PipelineError> {
        self.shared
            .device_connection_enabled
            .store(true, Ordering::Release);
        match self.device.connect() {
            Ok(()) => {
                self.shared.record_device_success();
                Ok(())
            }
            Err(error) => {
                self.shared.record_device_error(&error);
                Err(PipelineError::DeviceConnect(error.to_string()))
            }
        }
    }

    /// Stop physical output and suppress automatic reconnect without stopping
    /// capture, inference, targeting, or preview workers.
    pub fn disconnect_device(&self) -> Result<(), PipelineError> {
        self.pause_output_gate();
        self.shared
            .device_connection_enabled
            .store(false, Ordering::Release);
        self.shared.record_manual_device_disconnect();
        self.device
            .disconnect()
            .map_err(|error| PipelineError::DeviceDisconnect(error.to_string()))
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
                    if !shared.device_connection_enabled.load(Ordering::Acquire) {
                        thread::sleep(interval);
                        continue;
                    }
                    match device.buttons() {
                        Ok(Some(buttons)) => {
                            shared.record_device_success();
                            let active = buttons.trigger_active();
                            shared.trigger_active.store(active, Ordering::Release);
                            shared.buttons_available.store(true, Ordering::Release);
                            shared.set_button_left(buttons.left);
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
                            shared.set_button_left(false);
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
    config: TargetingConfig,
    crosshair: Option<CrosshairHub>,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-targeting".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "targeting", || {
                let mut targeting = TargetingCore::new(config);
                let mut config_version = shared.targeting_config_version.load(Ordering::Acquire);
                while let Some(batch) = input.wait_take() {
                    if shared.status() != PipelineStatus::Running {
                        break;
                    }
                    let latest_config_version =
                        shared.targeting_config_version.load(Ordering::Acquire);
                    if latest_config_version != config_version {
                        let config = shared
                            .targeting_config
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .clone();
                        targeting.set_config(config);
                        config_version = latest_config_version;
                    }
                    shared
                        .metrics
                        .targeting_batches
                        .fetch_add(1, Ordering::Relaxed);
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
                    if shared.vision_telemetry_due(batch.stamp().captured_at.0) {
                        shared.record_vision(&batch, &selection);
                    }
                    let track_confidence = if selection.target_state_valid {
                        selection.target_identity_confidence.unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let (target_id, aim_x, aim_y, detection_confidence) = match (
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
                            (None, center_x, center_y, 0.0)
                        }
                    };
                    let (crosshair_x, crosshair_y) = targeting_center;
                    let recoil_target_valid = target_id.is_some()
                        || targeting
                            .locked()
                            .is_some_and(|track| track.state == TrackState::Lost);
                    let observation = TargetedObservation {
                        stamp: batch.stamp(),
                        target_id,
                        recoil_target_valid,
                        aim_x,
                        aim_y,
                        crosshair_x,
                        crosshair_y,
                        detection_confidence,
                        track_confidence,
                        track_rebuilt: selection.target_rebuilt,
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
    output: LatestSlot<OutputPlan>,
    shared: Arc<SharedState>,
    clock: Arc<dyn Clock>,
    config: ControlWorkerConfig,
) -> Result<JoinHandle<()>, PipelineError> {
    thread::Builder::new()
        .name("novasight-control".to_owned())
        .spawn(move || {
            let _guard = WorkerGuard::new(Arc::clone(&shared));
            guard_worker(&shared, "control", || {
                let mut control = ContinuousControl::new(config.control);
                let mut config_version = shared.control_config_version.load(Ordering::Acquire);
                let mut previous_capture_ts_ns = None;
                let mut next_telemetry_at_ns = 0;
                while let Some(target) = input.wait_take() {
                    if shared.status() != PipelineStatus::Running {
                        break;
                    }
                    let latest_config_version =
                        shared.control_config_version.load(Ordering::Acquire);
                    if latest_config_version != config_version {
                        let config = *shared
                            .control_config
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        control.set_config(config);
                        config_version = latest_config_version;
                    }
                    let control_now_ns = clock.now().0;
                    let target_id = target.target_id.unwrap_or(0);
                    let trigger_active = !shared.hardware_trigger_required.load(Ordering::Acquire)
                        || shared.trigger_active.load(Ordering::Acquire);
                    let observation = ControlObservation {
                        generation: target.stamp.generation.0,
                        target_id,
                        capture_ts_ns: target.stamp.captured_at.0,
                        control_now_ns,
                        aim_x: target.aim_x,
                        aim_y: target.aim_y,
                        crosshair_x: target.crosshair_x,
                        crosshair_y: target.crosshair_y,
                        detection_confidence: target.detection_confidence,
                        track_confidence: target.track_confidence,
                        target_valid: target.target_id.is_some(),
                        trigger_active,
                    };
                    if target.track_rebuilt {
                        control.reset_target_state();
                        previous_capture_ts_ns = None;
                    }
                    let measurement_guard_ns = previous_capture_ts_ns
                        .map(|previous: u64| {
                            target
                                .stamp
                                .captured_at
                                .0
                                .saturating_sub(previous)
                                .min(50_000_000)
                        })
                        .unwrap_or(0);
                    previous_capture_ts_ns = Some(target.stamp.captured_at.0);
                    let visible_after_delay_ns = shared
                        .actuation_feedback_delay_ns
                        .load(Ordering::Acquire)
                        .saturating_add(measurement_guard_ns);
                    let pending_for_axis = |accepted_at_ns: u64| {
                        accepted_at_ns != 0
                            && target.stamp.captured_at.0
                                <= accepted_at_ns.saturating_add(visible_after_delay_ns)
                    };
                    let feedback = ActuationFeedback {
                        pending_x: pending_for_axis(
                            shared
                                .latest_successful_send_x_ts_ns
                                .load(Ordering::Acquire),
                        ),
                        pending_y: pending_for_axis(
                            shared
                                .latest_successful_send_y_ts_ns
                                .load(Ordering::Acquire),
                        ),
                    };
                    let decision = control.calculate_with_feedback(observation, feedback);
                    shared.record_prediction_truth(decision);
                    if control_now_ns >= next_telemetry_at_ns {
                        shared.record_control_decision(decision);
                        next_telemetry_at_ns =
                            control_now_ns.saturating_add(CONTROL_TELEMETRY_MIN_INTERVAL_NS);
                    }
                    shared
                        .metrics
                        .control_decisions
                        .fetch_add(1, Ordering::Relaxed);
                    if !decision.emit_allowed {
                        shared
                            .metrics
                            .blocked_decisions
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    if !decision.emit_allowed
                        && !recoil_plan_allowed(
                            &shared,
                            target.recoil_target_valid,
                            decision.block_reason,
                        )
                    {
                        continue;
                    }
                    let command = DeviceCommand {
                        epoch: config.epoch,
                        generation: target.stamp.generation,
                        source_captured_at: target.stamp.captured_at,
                        issued_at: novasight_core::MonotonicNanos(control_now_ns),
                        target_object_id: target_id,
                        delta_x_counts: if decision.emit_allowed {
                            decision.dx
                        } else {
                            0
                        },
                        delta_y_counts: if decision.emit_allowed {
                            decision.dy
                        } else {
                            0
                        },
                    };
                    if output
                        .publish(OutputPlan {
                            command,
                            recoil_target_valid: target.recoil_target_valid,
                        })
                        .is_err()
                    {
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

/// Only observations that already passed the controller's timestamp,
/// freshness and sequence guards may authorize a recoil-only plan. Geometry
/// and numeric failures remain hard output stops. Target absence is allowed
/// here because `RecoilConfig::require_target` is evaluated at the device
/// boundary using the current live configuration.
fn recoil_plan_allowed(
    shared: &SharedState,
    target_valid: bool,
    reason: novasight_core::controller::BlockReason,
) -> bool {
    use novasight_core::controller::BlockReason;

    shared.recoil_enabled.load(Ordering::Acquire)
        && shared.buttons_available.load(Ordering::Acquire)
        && shared.button_left.load(Ordering::Acquire)
        && (target_valid || !shared.recoil_require_target.load(Ordering::Acquire))
        && matches!(
            reason,
            BlockReason::TargetInvalid
                | BlockReason::DeadZone
                | BlockReason::AimSettled
                | BlockReason::ActuationFeedbackPending
        )
}

fn spawn_device_worker(
    input: LatestSlot<OutputPlan>,
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
                let mut recoil = IntervalRecoilController::new(config.recoil)
                    .expect("pipeline validates recoil config before worker startup");
                let mut recoil_config_version =
                    shared.recoil_config_version.load(Ordering::Acquire);
                let mut button_left_epoch = shared.button_left_epoch.load(Ordering::Acquire);
                while let Some(plan) = input.wait_take() {
                    let latest_recoil_config_version =
                        shared.recoil_config_version.load(Ordering::Acquire);
                    if latest_recoil_config_version != recoil_config_version {
                        let config = *shared
                            .recoil_config
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        recoil = IntervalRecoilController::new(config)
                            .expect("live recoil config is validated before publication");
                        recoil_config_version = latest_recoil_config_version;
                    }
                    let _lane = shared
                        .device_lane
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if shared.recoil_config_version.load(Ordering::Acquire) != recoil_config_version
                    {
                        continue;
                    }
                    if shared.external_stop.load(Ordering::Acquire) != 0
                        || shared.status() != PipelineStatus::Running
                    {
                        break;
                    }
                    if !shared.output_gate.load(Ordering::Acquire) {
                        continue;
                    }
                    if !shared.device_connection_enabled.load(Ordering::Acquire) {
                        continue;
                    }
                    if shared.hardware_trigger_required.load(Ordering::Acquire)
                        && !shared.trigger_active.load(Ordering::Acquire)
                    {
                        continue;
                    }
                    let mut command = plan.command;
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
                    if shared.hardware_trigger_required.load(Ordering::Acquire)
                        && !shared.trigger_active.load(Ordering::Acquire)
                    {
                        continue;
                    }

                    let latest_button_left_epoch = shared.button_left_epoch.load(Ordering::Acquire);
                    if latest_button_left_epoch != button_left_epoch {
                        recoil.reset();
                        button_left_epoch = latest_button_left_epoch;
                    }
                    let now_ns = clock.now().0;
                    let mut recoil_decision = recoil.calculate(RecoilInput {
                        firing: shared.buttons_available.load(Ordering::Acquire)
                            && shared.button_left.load(Ordering::Acquire),
                        now_ns,
                        target_valid: plan.recoil_target_valid,
                        source_generation: Some(command.generation.0),
                    });
                    let tracking_x_counts = command.delta_x_counts;
                    let tracking_y_counts = command.delta_y_counts;
                    let tracking_requested = tracking_x_counts != 0 || tracking_y_counts != 0;
                    let recoil_mix = mix_tracking_and_recoil(tracking_y_counts, recoil_decision);
                    command.delta_y_counts = recoil_mix.command_y;
                    if !tracking_requested && !recoil_decision.should_add() {
                        shared.record_recoil(recoil_decision);
                        continue;
                    }
                    let latest_generation = shared.latest_seen_generation();
                    if latest_generation != Some(command.generation) {
                        shared
                            .metrics
                            .superseded_commands
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    match device.send(command) {
                        Ok(receipt) => {
                            let accepted_at_ns = clock.now().0;
                            if recoil_decision.mark_output_result(recoil_mix.applied_counts_y) {
                                recoil.mark_output_sent(accepted_at_ns);
                            }
                            shared.record_recoil(recoil_decision);
                            shared.record_device_success();
                            shared.record_device_receipt(
                                receipt,
                                accepted_at_ns,
                                tracking_x_counts,
                                recoil_mix.surviving_tracking_counts_y,
                            );
                        }
                        Err(error) => {
                            shared.record_recoil(recoil_decision);
                            shared.record_device_error(&error);
                            shared.trigger_active.store(false, Ordering::Release);
                            shared.set_button_left(false);
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
                | "helper_protocol_failed"
                | "driver_rejected"
                | "driver_unavailable"
                | "reconnect_cooldown"
                | "driver_send_failed"
                | "driver_protocol_failed"
                | "monitor_stale"
                | "monitor_failed"
                | "socket_setup_failed"
                | "not_connected",
            ..
        }
    )
}

fn close_slots(
    batches: &LatestSlot<DetectionBatch>,
    targets: &LatestSlot<TargetedObservation>,
    commands: &LatestSlot<OutputPlan>,
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
    commands: &LatestSlot<OutputPlan>,
) -> PipelineMetrics {
    let (detections, target_selection) = {
        let vision = shared
            .metrics
            .vision
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (vision.detections.clone(), vision.target_selection.clone())
    };
    let last_device_receipt = shared.metrics.last_device_receipt.snapshot();
    let prediction_truth = {
        let samples = shared
            .metrics
            .prediction_truth
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let samples = samples.iter().copied().collect::<Vec<_>>();
        score_prediction_truth(&samples, PredictionTruthConfig::default())
    };
    PipelineMetrics {
        status: shared.status(),
        received_batches: shared.metrics.received_batches.load(Ordering::Relaxed),
        input_overwrites: batches.metrics().overwritten,
        targeting_batches: shared.metrics.targeting_batches.load(Ordering::Relaxed),
        control_decisions: shared.metrics.control_decisions.load(Ordering::Relaxed),
        blocked_decisions: shared.metrics.blocked_decisions.load(Ordering::Relaxed),
        command_overwrites: commands.metrics().overwritten,
        superseded_commands: shared.metrics.superseded_commands.load(Ordering::Relaxed),
        device_receipts: last_device_receipt.map_or(0, |(count, _)| count),
        last_device_receipt: last_device_receipt.map(|(_, receipt)| receipt),
        device_connected: shared.device_connected.load(Ordering::Acquire),
        device_connection_enabled: shared.device_connection_enabled.load(Ordering::Acquire),
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
        last_generation: shared.metrics.last_generation(),
        last_fault: shared
            .metrics
            .last_fault
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone(),
        detections,
        target_selection,
        control: *shared
            .metrics
            .control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        prediction_truth,
        recoil: *shared
            .metrics
            .recoil
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    }
}

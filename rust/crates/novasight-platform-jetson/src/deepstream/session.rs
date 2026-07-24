use std::collections::VecDeque;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_queue::ArrayQueue;
use gst::glib::prelude::*;
use gst::prelude::*;
use gstreamer as gst;
use novasight_core::{Clock, Generation, RuntimeEpoch};
#[cfg(feature = "tensorrt")]
use novasight_deepstream_bridge::admit_capture_snapshot;
use novasight_deepstream_bridge::{
    AdmissionContext, FRAME_BUFFER_PTS_VALID, FRAME_META_PTS_VALID, PipelineClockSample,
    admit_snapshot, extract_frame_into, validate_loaded_abi,
};
use novasight_pipeline::{
    CrosshairEpoch, CrosshairHub, PerceptionAdapter, PerceptionError, PerceptionEvent,
    PerceptionMetrics, PerceptionSession, PipelineError, PipelineIngress, PreviewHub,
};
use thiserror::Error;

#[cfg(feature = "tensorrt")]
use super::{CudaTensorRtConfig, CudaTensorRtOwner};
use super::{DeepStreamPipelineSpec, FrameLease, InferenceStage, LatestFrameExchange};

const EVENT_CAPACITY: usize = 8;
const BUS_POLL_INTERVAL: Duration = Duration::from_millis(50);
const SNAPSHOT_SLOT_COUNT: usize = 3;
const INFERENCE_INPUT_TIMELINE_CAPACITY: usize = 256;

#[derive(Clone, Debug)]
pub struct DeepStreamSessionConfig {
    pub pipeline: DeepStreamPipelineSpec,
    pub probe_pad: String,
    pub source_id: u32,
    pub inference_component_id: i32,
    pub max_batch_age_ns: Option<u64>,
    pub startup_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub preview: Option<PreviewHub>,
    pub crosshair: Option<CrosshairHub>,
    #[cfg(feature = "tensorrt")]
    pub rust_tensorrt: Option<CudaTensorRtConfig>,
}

impl DeepStreamSessionConfig {
    fn validate(&self) -> Result<(), SessionError> {
        self.pipeline.build()?;
        if self.probe_pad.trim().is_empty() {
            return Err(SessionError::BlankProbePad);
        }
        if self.inference_component_id < 0 {
            return Err(SessionError::NegativeInferenceComponent(
                self.inference_component_id,
            ));
        }
        if self.startup_timeout.is_zero() {
            return Err(SessionError::ZeroStartupTimeout);
        }
        if self.shutdown_timeout.is_zero() {
            return Err(SessionError::ZeroShutdownTimeout);
        }
        if self.max_batch_age_ns == Some(0) {
            return Err(SessionError::ZeroBatchAge);
        }
        #[cfg(feature = "tensorrt")]
        if matches!(
            self.pipeline.inference,
            InferenceStage::DeepStreamNvinfer { .. }
        ) && self.rust_tensorrt.is_some()
        {
            return Err(SessionError::InferenceStageMismatch);
        }
        if matches!(self.pipeline.inference, InferenceStage::RustTensorRt) {
            #[cfg(not(feature = "tensorrt"))]
            return Err(SessionError::RustTensorRtNotCompiled);
            #[cfg(feature = "tensorrt")]
            if self.rust_tensorrt.is_none() {
                return Err(SessionError::InferenceStageMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionEvent {
    Faulted { message: String },
    Stopped,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SessionMetrics {
    pub input_buffers: u64,
    pub probed_buffers: u64,
    pub published_batches: u64,
    pub latest_published_capture_at_ns: Option<u64>,
    pub latest_inference_duration_ns: Option<u64>,
    pub inference_duration_samples: u64,
    pub busy_dropped_batches: u64,
    pub overwritten_snapshots: u64,
    pub unavailable_snapshot_slots: u64,
    pub extraction_rejections: u64,
    pub admission_rejections: u64,
    pub truncated_detections: u64,
    pub ingress_rejections: u64,
    pub timestamp_buffer_pts_matches: u64,
    pub timestamp_frame_meta_pts_matches: u64,
    pub timestamp_correlation_misses: u64,
}

#[derive(Debug, Default)]
struct AtomicSessionMetrics {
    input_buffers: AtomicU64,
    probed_buffers: AtomicU64,
    published_batches: AtomicU64,
    latest_published_capture_at_ns: AtomicU64,
    latest_inference_duration_ns: AtomicU64,
    inference_duration_samples: AtomicU64,
    busy_dropped_batches: AtomicU64,
    overwritten_snapshots: AtomicU64,
    unavailable_snapshot_slots: AtomicU64,
    extraction_rejections: AtomicU64,
    admission_rejections: AtomicU64,
    truncated_detections: AtomicU64,
    ingress_rejections: AtomicU64,
    timestamp_buffer_pts_matches: AtomicU64,
    timestamp_frame_meta_pts_matches: AtomicU64,
    timestamp_correlation_misses: AtomicU64,
}

impl AtomicSessionMetrics {
    fn snapshot(&self) -> SessionMetrics {
        SessionMetrics {
            input_buffers: self.input_buffers.load(Ordering::Relaxed),
            probed_buffers: self.probed_buffers.load(Ordering::Relaxed),
            published_batches: self.published_batches.load(Ordering::Relaxed),
            latest_published_capture_at_ns: self
                .latest_published_capture_at_ns
                .load(Ordering::Relaxed)
                .checked_sub(1),
            latest_inference_duration_ns: self
                .latest_inference_duration_ns
                .load(Ordering::Relaxed)
                .checked_sub(1),
            inference_duration_samples: self.inference_duration_samples.load(Ordering::Relaxed),
            busy_dropped_batches: self.busy_dropped_batches.load(Ordering::Relaxed),
            overwritten_snapshots: self.overwritten_snapshots.load(Ordering::Relaxed),
            unavailable_snapshot_slots: self.unavailable_snapshot_slots.load(Ordering::Relaxed),
            extraction_rejections: self.extraction_rejections.load(Ordering::Relaxed),
            admission_rejections: self.admission_rejections.load(Ordering::Relaxed),
            truncated_detections: self.truncated_detections.load(Ordering::Relaxed),
            ingress_rejections: self.ingress_rejections.load(Ordering::Relaxed),
            timestamp_buffer_pts_matches: self.timestamp_buffer_pts_matches.load(Ordering::Relaxed),
            timestamp_frame_meta_pts_matches: self
                .timestamp_frame_meta_pts_matches
                .load(Ordering::Relaxed),
            timestamp_correlation_misses: self.timestamp_correlation_misses.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct SnapshotSlot {
    snapshot: MaybeUninit<novasight_deepstream_bridge::FrameSnapshot>,
    pipeline_running_now_ns: u64,
    monotonic_now: novasight_core::MonotonicNanos,
    inference_duration_ns: Option<u64>,
    observed_at_inference_input: bool,
    frame: Option<gst::Buffer>,
}

impl SnapshotSlot {
    fn empty() -> Self {
        Self {
            snapshot: MaybeUninit::uninit(),
            pipeline_running_now_ns: 0,
            monotonic_now: novasight_core::MonotonicNanos(0),
            inference_duration_ns: None,
            observed_at_inference_input: false,
            frame: None,
        }
    }
}

#[derive(Debug, Default)]
struct InferenceInputTimeline {
    samples: Mutex<VecDeque<(u64, novasight_core::MonotonicNanos)>>,
}

impl InferenceInputTimeline {
    fn observe(&self, pts_ns: u64, observed_at: novasight_core::MonotonicNanos) {
        let mut samples = self
            .samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        samples.push_back((pts_ns, observed_at));
        while samples.len() > INFERENCE_INPUT_TIMELINE_CAPACITY {
            samples.pop_front();
        }
    }

    fn take(&self, pts_ns: u64) -> Option<novasight_core::MonotonicNanos> {
        let mut samples = self
            .samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let position = samples
            .iter()
            .position(|(candidate, _)| *candidate == pts_ns)?;
        samples.remove(position).map(|(_, observed_at)| observed_at)
    }
}

#[derive(Debug)]
struct SnapshotExchange {
    free: ArrayQueue<Box<SnapshotSlot>>,
    ready: ArrayQueue<Box<SnapshotSlot>>,
    latest_frames: LatestFrameExchange,
    closed: AtomicBool,
    wake_sequence: AtomicU64,
    wake_lock: Mutex<()>,
    wake: Condvar,
}

impl SnapshotExchange {
    fn new(latest_frames: LatestFrameExchange) -> Arc<Self> {
        let exchange = Arc::new(Self {
            free: ArrayQueue::new(SNAPSHOT_SLOT_COUNT),
            ready: ArrayQueue::new(1),
            latest_frames,
            closed: AtomicBool::new(false),
            wake_sequence: AtomicU64::new(0),
            wake_lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        for _ in 0..SNAPSHOT_SLOT_COUNT {
            exchange
                .free
                .push(Box::new(SnapshotSlot::empty()))
                .expect("preallocated snapshot pool has exact capacity");
        }
        exchange
    }

    fn claim(&self) -> Option<Box<SnapshotSlot>> {
        self.free.pop()
    }

    fn publish(&self, mut slot: Box<SnapshotSlot>) -> Result<bool, Box<SnapshotSlot>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(slot);
        }
        let mut overwritten = false;
        loop {
            match self.ready.push(slot) {
                Ok(()) => {
                    self.wake_sequence.fetch_add(1, Ordering::Release);
                    self.wake.notify_one();
                    return Ok(overwritten);
                }
                Err(returned) => {
                    slot = returned;
                    if let Some(mut stale) = self.ready.pop() {
                        overwritten = true;
                        // Releasing a strong GstBuffer reference is normal on a
                        // GStreamer streaming thread. Mapping, CUDA work, model
                        // execution, and arbitrary user cleanup remain forbidden
                        // here; only the replaced pending lease is retired.
                        drop(stale.frame.take());
                        self.recycle(stale);
                    }
                }
            }
            if self.closed.load(Ordering::Acquire) {
                return Err(slot);
            }
        }
    }

    fn wait_take(&self) -> Option<Box<SnapshotSlot>> {
        loop {
            if let Some(slot) = self.ready.pop() {
                return Some(slot);
            }
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            let observed = self.wake_sequence.load(Ordering::Acquire);
            let guard = self
                .wake_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if self.ready.is_empty()
                && !self.closed.load(Ordering::Acquire)
                && self.wake_sequence.load(Ordering::Acquire) == observed
            {
                let _ = self
                    .wake
                    .wait_timeout(guard, BUS_POLL_INTERVAL)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    fn recycle(&self, slot: Box<SnapshotSlot>) {
        if let Err(_unexpected) = self.free.push(slot) {
            debug_assert!(false, "snapshot pool accounting overflow");
        }
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.wake_sequence.fetch_add(1, Ordering::Release);
        self.wake.notify_all();
    }
}

#[derive(Debug)]
struct ProbeState {
    closed: AtomicBool,
    generation: AtomicU64,
    first_published: AtomicBool,
    metrics: AtomicSessionMetrics,
    last_rejection: Mutex<Option<String>>,
    events: SyncSender<SessionEvent>,
    perception_events: Option<SyncSender<PerceptionEvent>>,
}

impl ProbeState {
    fn record_rejection(&self, stage: &'static str, detail: impl std::fmt::Display) {
        let mut last = self
            .last_rejection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *last = Some(format!("{stage}: {detail}"));
    }

    fn last_rejection(&self) -> String {
        self.last_rejection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .unwrap_or_else(|| "none recorded".to_owned())
    }

    fn fault(&self, message: impl Into<String>) {
        self.closed.store(true, Ordering::Release);
        let message = message.into();
        let _ = self.events.try_send(SessionEvent::Faulted {
            message: message.clone(),
        });
        if let Some(events) = &self.perception_events {
            let _ = events.try_send(PerceptionEvent::Faulted { message });
        }
    }
}

#[derive(Debug)]
enum SessionCommand {
    Stop,
}

pub struct DeepStreamSession {
    commands: SyncSender<SessionCommand>,
    events: Receiver<SessionEvent>,
    state: Arc<ProbeState>,
    join: Option<JoinHandle<Result<(), SessionError>>>,
}

struct StartedPipeline {
    pipeline: gst::Pipeline,
    inference_input_pad: gst::Pad,
    inference_input_probe_id: gst::PadProbeId,
    probe_pad: gst::Pad,
    probe_id: gst::PadProbeId,
    preview_probe: Option<(gst::Pad, gst::PadProbeId)>,
    crosshair_probe: Option<(gst::Pad, gst::PadProbeId)>,
    crosshair_epoch: Option<CrosshairEpoch>,
    bus: gst::Bus,
    exchange: Arc<SnapshotExchange>,
    perception_worker: JoinHandle<()>,
}

struct PreparedPipeline {
    pipeline: gst::Pipeline,
    inference_input_pad: gst::Pad,
    probe_pad: gst::Pad,
    bus: gst::Bus,
}

struct PreviewEpochGuard {
    hub: PreviewHub,
    epoch: RuntimeEpoch,
}

impl Drop for PreviewEpochGuard {
    fn drop(&mut self) {
        self.hub.end_epoch(self.epoch);
    }
}

#[derive(Clone, Debug)]
pub struct DeepStreamAdapter {
    config: DeepStreamSessionConfig,
    latest_frames: LatestFrameExchange,
}

impl DeepStreamAdapter {
    pub fn new(config: DeepStreamSessionConfig) -> Self {
        Self {
            config,
            latest_frames: LatestFrameExchange::new(),
        }
    }

    pub fn with_latest_frames(
        config: DeepStreamSessionConfig,
        latest_frames: LatestFrameExchange,
    ) -> Self {
        Self {
            config,
            latest_frames,
        }
    }

    pub fn config(&self) -> &DeepStreamSessionConfig {
        &self.config
    }

    pub fn latest_frames(&self) -> &LatestFrameExchange {
        &self.latest_frames
    }
}

impl PerceptionAdapter for DeepStreamAdapter {
    fn start(
        &self,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        clock: Arc<dyn Clock>,
        events: SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        DeepStreamSession::start_internal(
            self.config.clone(),
            epoch,
            ingress,
            clock,
            Some(events),
            self.latest_frames.clone(),
        )
        .map(|session| Box::new(session) as Box<dyn PerceptionSession>)
        .map_err(|error| PerceptionError::new(error.to_string()))
    }
}

impl std::fmt::Debug for DeepStreamSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepStreamSession")
            .field("metrics", &self.metrics())
            .field("running", &self.join.is_some())
            .finish()
    }
}

impl DeepStreamSession {
    pub fn start(
        config: DeepStreamSessionConfig,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, SessionError> {
        Self::start_internal(
            config,
            epoch,
            ingress,
            clock,
            None,
            LatestFrameExchange::new(),
        )
    }

    fn start_internal(
        config: DeepStreamSessionConfig,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        clock: Arc<dyn Clock>,
        perception_events: Option<SyncSender<PerceptionEvent>>,
        latest_frames: LatestFrameExchange,
    ) -> Result<Self, SessionError> {
        config.validate()?;
        latest_frames.clear();
        let (command_tx, command_rx) = sync_channel(1);
        let (event_tx, event_rx) = sync_channel(EVENT_CAPACITY);
        let (startup_tx, startup_rx) = sync_channel(1);
        let state = Arc::new(ProbeState {
            closed: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            first_published: AtomicBool::new(false),
            metrics: AtomicSessionMetrics::default(),
            last_rejection: Mutex::new(None),
            events: event_tx,
            perception_events,
        });
        let worker_state = Arc::clone(&state);
        let join = thread::Builder::new()
            .name("novasight-deepstream".to_owned())
            .spawn(move || {
                run_session(
                    config,
                    epoch,
                    ingress,
                    clock,
                    command_rx,
                    startup_tx,
                    worker_state,
                    latest_frames,
                )
            })
            .map_err(SessionError::Spawn)?;
        match startup_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                commands: command_tx,
                events: event_rx,
                state,
                join: Some(join),
            }),
            Ok(Err(message)) => match join.join() {
                Ok(Err(error)) => Err(error),
                Ok(Ok(())) => Err(SessionError::StartupFailed(message)),
                Err(_) => Err(SessionError::WorkerPanicked),
            },
            Err(_) => match join.join() {
                Ok(Err(error)) => Err(error),
                Ok(Ok(())) => Err(SessionError::StartupChannelClosed),
                Err(_) => Err(SessionError::WorkerPanicked),
            },
        }
    }

    pub fn metrics(&self) -> SessionMetrics {
        self.state.metrics.snapshot()
    }

    pub fn try_event(&self) -> Option<SessionEvent> {
        self.events.try_recv().ok()
    }

    pub fn shutdown(&mut self) -> Result<SessionMetrics, SessionError> {
        let Some(join) = self.join.take() else {
            return Ok(self.metrics());
        };
        self.state.closed.store(true, Ordering::Release);
        let _ = self.commands.try_send(SessionCommand::Stop);
        match join.join() {
            Ok(result) => result?,
            Err(_) => return Err(SessionError::WorkerPanicked),
        }
        Ok(self.metrics())
    }
}

/// Validate the linked DeepStream/GStreamer runtime and construct the configured
/// pipeline without changing its state. This checks the same ABI, plugin,
/// element-name, and probe-pad contract used by a real session, but does not
/// open capture or start inference.
pub fn preflight_deepstream_runtime(config: &DeepStreamSessionConfig) -> Result<(), SessionError> {
    config.validate()?;
    let _prepared = prepare_pipeline(config)?;
    #[cfg(feature = "tensorrt")]
    if let Some(tensorrt) = config.rust_tensorrt.clone() {
        let _owner = CudaTensorRtOwner::from_config(tensorrt)
            .map_err(|error| SessionError::InferenceInitialize(error.to_string()))?;
    }
    Ok(())
}

impl PerceptionSession for DeepStreamSession {
    fn metrics(&self) -> PerceptionMetrics {
        let metrics = self.metrics();
        PerceptionMetrics {
            input_buffers: metrics.input_buffers,
            probed_buffers: metrics.probed_buffers,
            published_batches: metrics.published_batches,
            latest_published_capture_at_ns: metrics.latest_published_capture_at_ns,
            latest_inference_duration_ns: metrics.latest_inference_duration_ns,
            inference_duration_samples: metrics.inference_duration_samples,
            busy_dropped_batches: metrics.busy_dropped_batches,
            overwritten_snapshots: metrics.overwritten_snapshots,
            unavailable_snapshot_slots: metrics.unavailable_snapshot_slots,
            extraction_rejections: metrics.extraction_rejections,
            admission_rejections: metrics.admission_rejections,
            truncated_detections: metrics.truncated_detections,
            ingress_rejections: metrics.ingress_rejections,
            timestamp_buffer_pts_matches: metrics.timestamp_buffer_pts_matches,
            timestamp_frame_meta_pts_matches: metrics.timestamp_frame_meta_pts_matches,
            timestamp_correlation_misses: metrics.timestamp_correlation_misses,
        }
    }

    fn poll_health(&mut self) -> Result<(), PerceptionError> {
        let finished = self.join.as_ref().is_some_and(JoinHandle::is_finished);
        if !finished {
            return Ok(());
        }
        let join = self
            .join
            .take()
            .expect("finished DeepStream owner was checked above");
        match join.join() {
            Ok(Ok(())) => Err(PerceptionError::new(
                "DeepStream owner exited before supervisor shutdown",
            )),
            Ok(Err(error)) => Err(PerceptionError::new(error.to_string())),
            Err(_) => Err(PerceptionError::new(
                SessionError::WorkerPanicked.to_string(),
            )),
        }
    }

    fn shutdown(&mut self) -> Result<(), PerceptionError> {
        DeepStreamSession::shutdown(self)
            .map(|_| ())
            .map_err(|error| PerceptionError::new(error.to_string()))
    }
}

impl Drop for DeepStreamSession {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_session(
    config: DeepStreamSessionConfig,
    epoch: RuntimeEpoch,
    ingress: PipelineIngress,
    clock: Arc<dyn Clock>,
    commands: Receiver<SessionCommand>,
    startup: SyncSender<Result<(), String>>,
    state: Arc<ProbeState>,
    latest_frames: LatestFrameExchange,
) -> Result<(), SessionError> {
    let startup_result = start_pipeline(
        &config,
        epoch,
        ingress,
        clock,
        Arc::clone(&state),
        latest_frames,
    );
    let StartedPipeline {
        pipeline,
        inference_input_pad,
        inference_input_probe_id,
        probe_pad,
        probe_id,
        preview_probe,
        crosshair_probe,
        mut crosshair_epoch,
        bus,
        exchange,
        perception_worker,
    } = match startup_result {
        Ok(started) => started,
        Err(error) => {
            let _ = startup.send(Err(error.to_string()));
            return Err(error);
        }
    };
    let _preview_epoch = config.preview.as_ref().map(|hub| PreviewEpochGuard {
        hub: hub.clone(),
        epoch,
    });
    let mut perception_worker = Some(perception_worker);

    let ready = wait_until_ready(&pipeline, &bus, &commands, &state, config.startup_timeout);
    if let Err(error) = ready {
        let error = combine_with_cleanup(
            error,
            cleanup_pipeline(
                &pipeline,
                &inference_input_pad,
                inference_input_probe_id,
                &probe_pad,
                probe_id,
                preview_probe,
                crosshair_probe,
                &mut crosshair_epoch,
                config.shutdown_timeout,
                &state,
                &exchange,
                &mut perception_worker,
            ),
        );
        let _ = startup.send(Err(error.to_string()));
        return Err(error);
    }
    if startup.send(Ok(())).is_err() {
        return Err(combine_with_cleanup(
            SessionError::StartupChannelClosed,
            cleanup_pipeline(
                &pipeline,
                &inference_input_pad,
                inference_input_probe_id,
                &probe_pad,
                probe_id,
                preview_probe,
                crosshair_probe,
                &mut crosshair_epoch,
                config.shutdown_timeout,
                &state,
                &exchange,
                &mut perception_worker,
            ),
        ));
    }

    let outcome = monitor_pipeline(
        &pipeline,
        &bus,
        &commands,
        &state,
        config.preview.as_ref(),
        epoch,
    );
    let cleanup = cleanup_pipeline(
        &pipeline,
        &inference_input_pad,
        inference_input_probe_id,
        &probe_pad,
        probe_id,
        preview_probe,
        crosshair_probe,
        &mut crosshair_epoch,
        config.shutdown_timeout,
        &state,
        &exchange,
        &mut perception_worker,
    );
    match (outcome, cleanup) {
        (Ok(()), Ok(())) => {
            let _ = state.events.try_send(SessionEvent::Stopped);
            if let Some(events) = &state.perception_events {
                let _ = events.try_send(PerceptionEvent::Stopped);
            }
            Ok(())
        }
        (Ok(()), Err(error)) | (Err(error), Ok(())) => {
            state.fault(error.to_string());
            Err(error)
        }
        (Err(error), Err(cleanup)) => {
            let error = combine_with_cleanup(error, Err(cleanup));
            state.fault(error.to_string());
            Err(error)
        }
    }
}

fn start_pipeline(
    config: &DeepStreamSessionConfig,
    epoch: RuntimeEpoch,
    ingress: PipelineIngress,
    monotonic_clock: Arc<dyn Clock>,
    state: Arc<ProbeState>,
    latest_frames: LatestFrameExchange,
) -> Result<StartedPipeline, SessionError> {
    let PreparedPipeline {
        pipeline,
        inference_input_pad,
        probe_pad,
        bus,
    } = prepare_pipeline(config)?;
    let weak_pipeline = pipeline.downgrade();
    let source_id = config.source_id;
    let exchange = SnapshotExchange::new(latest_frames);
    let inference_input_timeline = Arc::new(InferenceInputTimeline::default());
    let inference_input_probe_timeline = Arc::clone(&inference_input_timeline);
    let inference_input_probe_clock = Arc::clone(&monotonic_clock);
    let inference_input_probe_state = Arc::clone(&state);
    let inference_input_probe_id = inference_input_pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            let Some(buffer) = info.buffer() else {
                return gst::PadProbeReturn::Ok;
            };
            inference_input_probe_state
                .metrics
                .input_buffers
                .fetch_add(1, Ordering::Relaxed);
            if let Some(pts) = buffer.pts() {
                inference_input_probe_timeline
                    .observe(pts.nseconds(), inference_input_probe_clock.now());
            }
            gst::PadProbeReturn::Ok
        })
        .ok_or(SessionError::InferenceInputProbeInstallFailed)?;
    let probe_exchange = Arc::clone(&exchange);
    let probe_state = Arc::clone(&state);
    let probe_clock = Arc::clone(&monotonic_clock);
    let probe_inference_input_timeline = Arc::clone(&inference_input_timeline);
    let probe_id = probe_pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            if probe_state.closed.load(Ordering::Acquire) {
                return gst::PadProbeReturn::Ok;
            }
            let Some(buffer) = info.buffer() else {
                return gst::PadProbeReturn::Ok;
            };
            // Timestamp pad arrival before bridge extraction so the metric
            // matches Python's nvinfer sink-to-src scope and does not charge
            // NovaSight metadata copying to TensorRT/parser execution.
            let inference_output_observed_at = probe_clock.now();
            probe_state
                .metrics
                .probed_buffers
                .fetch_add(1, Ordering::Relaxed);
            let Some(mut slot) = probe_exchange.claim() else {
                probe_state
                    .metrics
                    .unavailable_snapshot_slots
                    .fetch_add(1, Ordering::Relaxed);
                return gst::PadProbeReturn::Ok;
            };
            let Some(buffer_ptr) = NonNull::new(buffer.as_ptr().cast_mut().cast::<c_void>()) else {
                probe_state
                    .metrics
                    .extraction_rejections
                    .fetch_add(1, Ordering::Relaxed);
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            let (frame_pts_ns, buffer_pts_ns, snapshot_flags) = match unsafe {
                extract_frame_into(buffer_ptr, source_id, &mut slot.snapshot)
            } {
                Ok(snapshot) => (
                    snapshot.frame_pts_ns,
                    snapshot.buffer_pts_ns,
                    snapshot.flags,
                ),
                Err(error) => {
                    probe_state
                        .metrics
                        .extraction_rejections
                        .fetch_add(1, Ordering::Relaxed);
                    probe_state.record_rejection(
                        "bridge extraction",
                        format_args!("status={:?}, raw_status={}", error.status, error.raw_status),
                    );
                    probe_exchange.recycle(slot);
                    return gst::PadProbeReturn::Ok;
                }
            };
            let buffer_pts_observed_at = (snapshot_flags & FRAME_BUFFER_PTS_VALID != 0)
                .then(|| probe_inference_input_timeline.take(buffer_pts_ns))
                .flatten();
            let (inference_input_observed_at, matched_buffer_pts) =
                if let Some(observed_at) = buffer_pts_observed_at {
                    (Some(observed_at), true)
                } else {
                    (
                        (snapshot_flags & FRAME_META_PTS_VALID != 0)
                            .then(|| probe_inference_input_timeline.take(frame_pts_ns))
                            .flatten(),
                        false,
                    )
                };
            if let Some(input_observed_at) = inference_input_observed_at {
                if matched_buffer_pts {
                    probe_state
                        .metrics
                        .timestamp_buffer_pts_matches
                        .fetch_add(1, Ordering::Relaxed);
                } else {
                    probe_state
                        .metrics
                        .timestamp_frame_meta_pts_matches
                        .fetch_add(1, Ordering::Relaxed);
                }
                // Correlate the nvinfer input and output by buffer PTS, matching
                // the Python DeepStream backend. The configured inference input
                // deadline measures inference work, not V4L2/decoder clock-domain
                // offsets that occur before the inference element.
                slot.pipeline_running_now_ns = frame_pts_ns;
                slot.monotonic_now = input_observed_at;
                slot.inference_duration_ns = inference_output_observed_at
                    .0
                    .checked_sub(input_observed_at.0);
                slot.observed_at_inference_input = true;
                slot.frame = Some(buffer.to_owned());
                match probe_exchange.publish(slot) {
                    Ok(true) => {
                        probe_state
                            .metrics
                            .overwritten_snapshots
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(false) => {}
                    Err(slot) => probe_exchange.recycle(slot),
                }
                return gst::PadProbeReturn::Ok;
            }
            probe_state
                .metrics
                .timestamp_correlation_misses
                .fetch_add(1, Ordering::Relaxed);
            let Some(pipeline) = weak_pipeline.upgrade() else {
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            let Some(gst_clock) = pipeline.clock() else {
                probe_state
                    .metrics
                    .admission_rejections
                    .fetch_add(1, Ordering::Relaxed);
                probe_state.record_rejection("pipeline clock", "clock unavailable");
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            let Some(base_time) = pipeline.base_time() else {
                probe_state
                    .metrics
                    .admission_rejections
                    .fetch_add(1, Ordering::Relaxed);
                probe_state.record_rejection("pipeline clock", "base time unavailable");
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            let Some(running_now_ns) = gst_clock
                .time()
                .nseconds()
                .checked_sub(base_time.nseconds())
            else {
                probe_state
                    .metrics
                    .admission_rejections
                    .fetch_add(1, Ordering::Relaxed);
                probe_state
                    .record_rejection("pipeline clock", "clock time precedes pipeline base time");
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            slot.pipeline_running_now_ns = running_now_ns;
            slot.monotonic_now = probe_clock.now();
            slot.inference_duration_ns = None;
            slot.observed_at_inference_input = false;
            slot.frame = Some(buffer.to_owned());
            match probe_exchange.publish(slot) {
                Ok(true) => {
                    probe_state
                        .metrics
                        .overwritten_snapshots
                        .fetch_add(1, Ordering::Relaxed);
                }
                Ok(false) => {}
                Err(slot) => probe_exchange.recycle(slot),
            }
            gst::PadProbeReturn::Ok
        })
        .ok_or(SessionError::ProbeInstallFailed)?;

    finish_started_pipeline(
        config,
        epoch,
        ingress,
        monotonic_clock,
        state,
        pipeline,
        inference_input_pad,
        inference_input_probe_id,
        probe_pad,
        probe_id,
        bus,
        exchange,
    )
}

fn prepare_pipeline(config: &DeepStreamSessionConfig) -> Result<PreparedPipeline, SessionError> {
    validate_loaded_abi().map_err(SessionError::BridgeAbi)?;
    gst::init().map_err(|error| SessionError::GstreamerInit(error.to_string()))?;
    let description = config.pipeline.build()?;
    let element = gst::parse::launch(&description)
        .map_err(|error| SessionError::PipelineParse(error.to_string()))?;
    let pipeline = element
        .downcast::<gst::Pipeline>()
        .map_err(|_| SessionError::ParsedElementNotPipeline)?;
    let bus = pipeline.bus().ok_or(SessionError::MissingBus)?;
    let inference = pipeline
        .by_name(&config.pipeline.inference_element)
        .ok_or_else(|| SessionError::MissingInferenceElement {
            name: config.pipeline.inference_element.clone(),
        })?;
    let inference_input_pad =
        inference
            .static_pad("sink")
            .ok_or_else(|| SessionError::MissingInferenceInputPad {
                element: config.pipeline.inference_element.clone(),
            })?;
    let probe_pad =
        inference
            .static_pad(&config.probe_pad)
            .ok_or_else(|| SessionError::MissingProbePad {
                element: config.pipeline.inference_element.clone(),
                pad: config.probe_pad.clone(),
            })?;
    if config.pipeline.preview.is_some() {
        pipeline
            .by_name("preview-valve")
            .ok_or(SessionError::MissingPreviewElement("preview-valve"))?;
        let preview_sink = pipeline
            .by_name("preview-sink")
            .ok_or(SessionError::MissingPreviewElement("preview-sink"))?;
        preview_sink
            .static_pad("sink")
            .ok_or(SessionError::MissingPreviewPad)?;
    }
    if config.pipeline.crosshair.is_some() {
        pipeline
            .by_name("crosshair-sink")
            .ok_or(SessionError::MissingCrosshairElement)?
            .static_pad("sink")
            .ok_or(SessionError::MissingCrosshairPad)?;
    }
    Ok(PreparedPipeline {
        pipeline,
        inference_input_pad,
        probe_pad,
        bus,
    })
}

#[allow(clippy::too_many_arguments)]
fn finish_started_pipeline(
    config: &DeepStreamSessionConfig,
    epoch: RuntimeEpoch,
    ingress: PipelineIngress,
    monotonic_clock: Arc<dyn Clock>,
    state: Arc<ProbeState>,
    pipeline: gst::Pipeline,
    inference_input_pad: gst::Pad,
    inference_input_probe_id: gst::PadProbeId,
    probe_pad: gst::Pad,
    probe_id: gst::PadProbeId,
    bus: gst::Bus,
    exchange: Arc<SnapshotExchange>,
) -> Result<StartedPipeline, SessionError> {
    let preview_probe = match install_preview_probe(&pipeline, config.preview.as_ref(), epoch) {
        Ok(probe) => probe,
        Err(error) => {
            inference_input_pad.remove_probe(inference_input_probe_id);
            probe_pad.remove_probe(probe_id);
            exchange.close();
            return Err(error);
        }
    };
    let mut crosshair_epoch = match config.crosshair.as_ref() {
        Some(hub) => {
            match hub.begin_epoch(epoch, config.pipeline.roi.width, config.pipeline.roi.height) {
                Ok(owner) => Some(owner),
                Err(error) => {
                    inference_input_pad.remove_probe(inference_input_probe_id);
                    probe_pad.remove_probe(probe_id);
                    if let Some((pad, id)) = preview_probe {
                        pad.remove_probe(id);
                    }
                    exchange.close();
                    return Err(SessionError::Crosshair(error.to_string()));
                }
            }
        }
        None => None,
    };
    let crosshair_probe = match install_crosshair_probe(
        &pipeline,
        crosshair_epoch.as_ref(),
        Arc::clone(&monotonic_clock),
    ) {
        Ok(probe) => probe,
        Err(error) => {
            inference_input_pad.remove_probe(inference_input_probe_id);
            probe_pad.remove_probe(probe_id);
            if let Some((pad, id)) = preview_probe {
                pad.remove_probe(id);
            }
            exchange.close();
            return Err(error);
        }
    };

    let worker_context = SnapshotWorkerContext {
        epoch,
        source_id: config.source_id,
        inference_component_id: config.inference_component_id,
        max_batch_age_ns: config.max_batch_age_ns,
        monotonic_clock,
        ingress,
        #[cfg(feature = "tensorrt")]
        rust_tensorrt: config.rust_tensorrt.clone(),
    };
    let mut perception_worker =
        match spawn_snapshot_worker(Arc::clone(&exchange), Arc::clone(&state), worker_context) {
            Ok(worker) => Some(worker),
            Err(error) => {
                inference_input_pad.remove_probe(inference_input_probe_id);
                probe_pad.remove_probe(probe_id);
                if let Some((pad, id)) = preview_probe {
                    pad.remove_probe(id);
                }
                if let Some((pad, id)) = crosshair_probe {
                    pad.remove_probe(id);
                }
                drop(crosshair_epoch.take());
                exchange.close();
                return Err(error);
            }
        };

    if let Err(error) = pipeline.set_state(gst::State::Playing) {
        return Err(combine_with_cleanup(
            SessionError::StateChange(error.to_string()),
            cleanup_pipeline(
                &pipeline,
                &inference_input_pad,
                inference_input_probe_id,
                &probe_pad,
                probe_id,
                preview_probe,
                crosshair_probe,
                &mut crosshair_epoch,
                config.shutdown_timeout,
                &state,
                &exchange,
                &mut perception_worker,
            ),
        ));
    }
    let (state_result, current, _pending) =
        pipeline.state(Some(duration_to_clock_time(config.startup_timeout)));
    if let Err(error) = state_result {
        return Err(combine_with_cleanup(
            SessionError::StateChange(error.to_string()),
            cleanup_pipeline(
                &pipeline,
                &inference_input_pad,
                inference_input_probe_id,
                &probe_pad,
                probe_id,
                preview_probe,
                crosshair_probe,
                &mut crosshair_epoch,
                config.shutdown_timeout,
                &state,
                &exchange,
                &mut perception_worker,
            ),
        ));
    }
    if current != gst::State::Playing {
        return Err(combine_with_cleanup(
            SessionError::DidNotReachPlaying { current },
            cleanup_pipeline(
                &pipeline,
                &inference_input_pad,
                inference_input_probe_id,
                &probe_pad,
                probe_id,
                preview_probe,
                crosshair_probe,
                &mut crosshair_epoch,
                config.shutdown_timeout,
                &state,
                &exchange,
                &mut perception_worker,
            ),
        ));
    }
    if let Some(preview) = config.preview.as_ref() {
        preview.begin_epoch(epoch);
    }
    Ok(StartedPipeline {
        pipeline,
        inference_input_pad,
        inference_input_probe_id,
        probe_pad,
        probe_id,
        preview_probe,
        crosshair_probe,
        crosshair_epoch,
        bus,
        exchange,
        perception_worker: perception_worker
            .expect("perception worker remains owned after successful startup"),
    })
}

fn install_preview_probe(
    pipeline: &gst::Pipeline,
    preview: Option<&PreviewHub>,
    epoch: RuntimeEpoch,
) -> Result<Option<(gst::Pad, gst::PadProbeId)>, SessionError> {
    let Some(preview) = preview else {
        return Ok(None);
    };
    let sink = pipeline
        .by_name("preview-sink")
        .ok_or(SessionError::MissingPreviewElement("preview-sink"))?;
    let pad = sink
        .static_pad("sink")
        .ok_or(SessionError::MissingPreviewPad)?;
    let publisher = preview.clone();
    let id = pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            let Some(buffer) = info.buffer() else {
                return gst::PadProbeReturn::Ok;
            };
            if let Ok(map) = buffer.map_readable() {
                let _ = publisher.publish_jpeg(epoch, map.as_slice().to_vec());
            }
            gst::PadProbeReturn::Ok
        })
        .ok_or(SessionError::PreviewProbeInstallFailed)?;
    Ok(Some((pad, id)))
}

fn install_crosshair_probe(
    pipeline: &gst::Pipeline,
    crosshair: Option<&CrosshairEpoch>,
    clock: Arc<dyn Clock>,
) -> Result<Option<(gst::Pad, gst::PadProbeId)>, SessionError> {
    let Some(crosshair) = crosshair else {
        return Ok(None);
    };
    let sink = pipeline
        .by_name("crosshair-sink")
        .ok_or(SessionError::MissingCrosshairElement)?;
    let pad = sink
        .static_pad("sink")
        .ok_or(SessionError::MissingCrosshairPad)?;
    let publisher = crosshair.publisher();
    let id = pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            let Some(buffer) = info.buffer() else {
                return gst::PadProbeReturn::Ok;
            };
            if let Ok(map) = buffer.map_readable() {
                let _ = publisher.publish_jpeg(clock.now().0, map.as_slice().to_vec());
            }
            gst::PadProbeReturn::Ok
        })
        .ok_or(SessionError::CrosshairProbeInstallFailed)?;
    Ok(Some((pad, id)))
}

struct SnapshotWorkerContext {
    epoch: RuntimeEpoch,
    source_id: u32,
    inference_component_id: i32,
    max_batch_age_ns: Option<u64>,
    monotonic_clock: Arc<dyn Clock>,
    ingress: PipelineIngress,
    #[cfg(feature = "tensorrt")]
    rust_tensorrt: Option<CudaTensorRtConfig>,
}

enum SnapshotProcessor {
    DeepStreamMetadata,
    #[cfg(feature = "tensorrt")]
    RustTensorRt(Box<CudaTensorRtOwner>),
}

impl SnapshotProcessor {
    #[cfg(feature = "tensorrt")]
    fn initialize(config: Option<CudaTensorRtConfig>) -> Result<Self, SessionError> {
        match config {
            Some(config) => CudaTensorRtOwner::from_config(config)
                .map(Box::new)
                .map(Self::RustTensorRt)
                .map_err(|error| SessionError::InferenceInitialize(error.to_string())),
            None => Ok(Self::DeepStreamMetadata),
        }
    }

    #[cfg(not(feature = "tensorrt"))]
    fn initialize() -> Self {
        Self::DeepStreamMetadata
    }
}

fn spawn_snapshot_worker(
    exchange: Arc<SnapshotExchange>,
    state: Arc<ProbeState>,
    context: SnapshotWorkerContext,
) -> Result<JoinHandle<()>, SessionError> {
    let (initialized_tx, initialized_rx) = sync_channel(1);
    let worker = thread::Builder::new()
        .name("novasight-perception".to_owned())
        .spawn(move || {
            let worker_state = Arc::clone(&state);
            #[cfg(feature = "tensorrt")]
            let processor = SnapshotProcessor::initialize(context.rust_tensorrt.clone());
            #[cfg(not(feature = "tensorrt"))]
            let processor: Result<SnapshotProcessor, SessionError> =
                Ok(SnapshotProcessor::initialize());
            let mut processor = match processor {
                Ok(processor) => {
                    let _ = initialized_tx.send(Ok(()));
                    processor
                }
                Err(error) => {
                    let message = error.to_string();
                    let _ = initialized_tx.send(Err(message.clone()));
                    worker_state.fault(message);
                    return;
                }
            };
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_snapshot_worker(&exchange, &state, &context, &mut processor);
            }));
            if outcome.is_err() {
                worker_state.fault("DeepStream perception worker panicked");
            }
        })
        .map_err(SessionError::SpawnPerceptionWorker)?;
    match initialized_rx.recv() {
        Ok(Ok(())) => Ok(worker),
        Ok(Err(message)) => {
            let _ = worker.join();
            Err(SessionError::InferenceInitialize(message))
        }
        Err(_) => {
            let _ = worker.join();
            Err(SessionError::InferenceInitializeChannelClosed)
        }
    }
}

fn run_snapshot_worker(
    exchange: &SnapshotExchange,
    state: &ProbeState,
    context: &SnapshotWorkerContext,
    processor: &mut SnapshotProcessor,
) {
    while let Some(mut slot) = exchange.wait_take() {
        if state.closed.load(Ordering::Acquire) {
            recycle_from_worker(exchange, slot);
            break;
        }
        if let Some(inference_duration_ns) = slot.inference_duration_ns {
            state
                .metrics
                .latest_inference_duration_ns
                .store(inference_duration_ns.saturating_add(1), Ordering::Relaxed);
            state
                .metrics
                .inference_duration_samples
                .fetch_add(1, Ordering::Relaxed);
        }
        let generation =
            match state
                .generation
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current.checked_add(1)
                }) {
                Ok(previous) => Generation(previous + 1),
                Err(_) => {
                    state.fault("DeepStream runtime generation exhausted");
                    recycle_from_worker(exchange, slot);
                    break;
                }
            };
        // The producer publishes a slot only after the C bridge reports a
        // successful full initialization of FrameSnapshot. Ownership of this
        // Box stays with the worker until it is recycled into the fixed pool.
        let snapshot = unsafe { slot.snapshot.assume_init_ref() };
        let admission_context = AdmissionContext {
            epoch: context.epoch,
            generation,
            clock: PipelineClockSample::new(slot.pipeline_running_now_ns, slot.monotonic_now),
            source_id: context.source_id,
            inference_component_id: context.inference_component_id,
        };
        let admitted = match processor {
            SnapshotProcessor::DeepStreamMetadata => admit_snapshot(snapshot, admission_context)
                .map(|admitted| {
                    state.metrics.truncated_detections.fetch_add(
                        u64::from(admitted.truncated_detections()),
                        Ordering::Relaxed,
                    );
                    let batch = admitted.into_batch();
                    (
                        batch.stamp(),
                        batch.coordinate_width(),
                        batch.coordinate_height(),
                        Some(batch),
                    )
                }),
            #[cfg(feature = "tensorrt")]
            SnapshotProcessor::RustTensorRt(_) => {
                admit_capture_snapshot(snapshot, admission_context).map(|admitted| {
                    let (width, height) = admitted.dimensions();
                    (admitted.stamp(), width, height, None)
                })
            }
        };
        let admitted = match admitted {
            Ok(admitted) => admitted,
            Err(error) => {
                state
                    .metrics
                    .admission_rejections
                    .fetch_add(1, Ordering::Relaxed);
                state.record_rejection(
                    "snapshot admission",
                    format_args!("{}: {error}", error.code()),
                );
                recycle_from_worker(exchange, slot);
                continue;
            }
        };
        let now = context.monotonic_clock.now();
        let (stamp, width, height, deepstream_batch) = admitted;
        let Some(batch_age_ns) = now.0.checked_sub(stamp.captured_at.0) else {
            state
                .metrics
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            state.record_rejection(
                "batch freshness",
                format_args!(
                    "capture time {} is ahead of monotonic time {}",
                    stamp.captured_at.0, now.0
                ),
            );
            recycle_from_worker(exchange, slot);
            continue;
        };
        if context
            .max_batch_age_ns
            .is_some_and(|maximum| batch_age_ns > maximum)
        {
            state
                .metrics
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            state.record_rejection(
                "batch freshness",
                format_args!(
                    "age {batch_age_ns} ns exceeds configured maximum {:?} ns; timestamp_source={}",
                    context.max_batch_age_ns,
                    if slot.observed_at_inference_input {
                        "nvinfer_input_probe"
                    } else {
                        "pipeline_clock_pts"
                    }
                ),
            );
            recycle_from_worker(exchange, slot);
            continue;
        }
        if state.closed.load(Ordering::Acquire) {
            recycle_from_worker(exchange, slot);
            break;
        }
        let frame = slot
            .frame
            .take()
            .expect("published DeepStream snapshot owns its GstBuffer lease");
        let frame = FrameLease::new(
            frame,
            stamp.epoch,
            stamp.generation,
            stamp.captured_at,
            width,
            height,
        );
        let batch = match processor {
            SnapshotProcessor::DeepStreamMetadata => {
                deepstream_batch.expect("DeepStream admission produces its detection batch")
            }
            #[cfg(feature = "tensorrt")]
            SnapshotProcessor::RustTensorRt(owner) => match owner.infer(&frame) {
                Ok(batch) => batch,
                Err(error) => {
                    state.fault(format!("Rust TensorRT inference failed: {error}"));
                    recycle_from_worker(exchange, slot);
                    break;
                }
            },
        };
        let _ = exchange.latest_frames.publish(frame);
        match context.ingress.try_submit(batch) {
            Ok(()) => {
                state
                    .metrics
                    .latest_published_capture_at_ns
                    .store(stamp.captured_at.0.saturating_add(1), Ordering::Relaxed);
                state
                    .metrics
                    .published_batches
                    .fetch_add(1, Ordering::Relaxed);
                state.first_published.store(true, Ordering::Release);
            }
            Err(PipelineError::IngressBusy) => {
                state
                    .metrics
                    .busy_dropped_batches
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                state
                    .metrics
                    .ingress_rejections
                    .fetch_add(1, Ordering::Relaxed);
                state.record_rejection("pipeline ingress", &error);
                state.fault(format!("DeepStream ingress rejected a batch: {error}"));
                recycle_from_worker(exchange, slot);
                break;
            }
        }
        recycle_from_worker(exchange, slot);
    }
}

fn recycle_from_worker(exchange: &SnapshotExchange, mut slot: Box<SnapshotSlot>) {
    drop(slot.frame.take());
    exchange.recycle(slot);
}

fn wait_until_ready(
    pipeline: &gst::Pipeline,
    bus: &gst::Bus,
    commands: &Receiver<SessionCommand>,
    state: &ProbeState,
    timeout: Duration,
) -> Result<(), SessionError> {
    let deadline = Instant::now() + timeout;
    loop {
        if state.first_published.load(Ordering::Acquire) {
            return Ok(());
        }
        if state.closed.load(Ordering::Acquire) {
            return Err(SessionError::StoppedBeforeReady);
        }
        match commands.try_recv() {
            Ok(SessionCommand::Stop) | Err(TryRecvError::Disconnected) => {
                return Err(SessionError::StoppedBeforeReady);
            }
            Err(TryRecvError::Empty) => {}
        }
        if let Some(error) = poll_terminal_bus(bus, BUS_POLL_INTERVAL)? {
            return Err(error);
        }
        if Instant::now() >= deadline {
            let metrics = state.metrics.snapshot();
            return Err(SessionError::FirstBatchTimeout {
                timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
                current: pipeline.current_state(),
                metrics,
                last_rejection: state.last_rejection(),
            });
        }
    }
}

fn monitor_pipeline(
    pipeline: &gst::Pipeline,
    bus: &gst::Bus,
    commands: &Receiver<SessionCommand>,
    state: &ProbeState,
    preview: Option<&PreviewHub>,
    epoch: RuntimeEpoch,
) -> Result<(), SessionError> {
    let mut encoder_active = false;
    loop {
        if let Some(preview) = preview {
            let requested = preview.encoder_requested(epoch);
            if requested != encoder_active {
                let valve = pipeline
                    .by_name("preview-valve")
                    .ok_or(SessionError::MissingPreviewElement("preview-valve"))?;
                valve.set_property("drop", !requested);
                encoder_active = requested;
                preview.set_encoder_active(epoch, requested);
            }
        }
        match commands.try_recv() {
            Ok(SessionCommand::Stop) | Err(TryRecvError::Disconnected) => return Ok(()),
            Err(TryRecvError::Empty) => {}
        }
        if state.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        if let Some(error) = poll_terminal_bus(bus, BUS_POLL_INTERVAL)? {
            return Err(error);
        }
    }
}

fn poll_terminal_bus(
    bus: &gst::Bus,
    timeout: Duration,
) -> Result<Option<SessionError>, SessionError> {
    let Some(message) = bus.timed_pop_filtered(
        Some(duration_to_clock_time(timeout)),
        &[gst::MessageType::Error, gst::MessageType::Eos],
    ) else {
        return Ok(None);
    };
    match message.view() {
        gst::MessageView::Error(error) => Ok(Some(SessionError::BusError {
            element: error
                .src()
                .map(|source| source.path_string().to_string())
                .unwrap_or_else(|| "unknown".to_owned()),
            message: error.error().to_string(),
            debug: error.debug().map(|debug| debug.to_string()),
        })),
        gst::MessageView::Eos(_) => Ok(Some(SessionError::UnexpectedEos)),
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn cleanup_pipeline(
    pipeline: &gst::Pipeline,
    inference_input_pad: &gst::Pad,
    inference_input_probe_id: gst::PadProbeId,
    probe_pad: &gst::Pad,
    probe_id: gst::PadProbeId,
    preview_probe: Option<(gst::Pad, gst::PadProbeId)>,
    crosshair_probe: Option<(gst::Pad, gst::PadProbeId)>,
    crosshair_epoch: &mut Option<CrosshairEpoch>,
    timeout: Duration,
    state: &ProbeState,
    exchange: &SnapshotExchange,
    perception_worker: &mut Option<JoinHandle<()>>,
) -> Result<(), SessionError> {
    state.closed.store(true, Ordering::Release);
    inference_input_pad.remove_probe(inference_input_probe_id);
    probe_pad.remove_probe(probe_id);
    if let Some((pad, id)) = preview_probe {
        pad.remove_probe(id);
    }
    if let Some((pad, id)) = crosshair_probe {
        pad.remove_probe(id);
    }
    drop(crosshair_epoch.take());
    exchange.close();
    let mut failures = Vec::new();
    match pipeline.set_state(gst::State::Null) {
        Ok(_) => {
            let (result, current, pending) = pipeline.state(Some(duration_to_clock_time(timeout)));
            if let Err(error) = result {
                failures.push(format!("state wait failed: {error}"));
            } else if current != gst::State::Null {
                failures.push(format!(
                    "did not reach NULL; current={current:?}, pending={pending:?}"
                ));
            }
        }
        Err(error) => failures.push(format!("set_state(NULL) failed: {error}")),
    }
    if let Some(worker) = perception_worker.take()
        && worker.join().is_err()
    {
        failures.push("perception worker panicked".to_owned());
    }
    // The worker is the only publisher. Clearing after join prevents a
    // stop-vs-publish race from leaking an old epoch or NVMM pool lease.
    exchange.latest_frames.clear();
    if !failures.is_empty() {
        return Err(SessionError::ShutdownCleanup(failures.join("; ")));
    }
    Ok(())
}

fn combine_with_cleanup(primary: SessionError, cleanup: Result<(), SessionError>) -> SessionError {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup) => SessionError::CleanupAfterFailure {
            primary: primary.to_string(),
            cleanup: cleanup.to_string(),
        },
    }
}

fn duration_to_clock_time(duration: Duration) -> gst::ClockTime {
    gst::ClockTime::from_nseconds(duration.as_nanos().min((u64::MAX - 1) as u128) as u64)
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error(transparent)]
    PipelineSpec(#[from] super::PipelineSpecError),
    #[error("DeepStream probe pad must not be blank")]
    BlankProbePad,
    #[error("DeepStream inference component ID must be non-negative, got {0}")]
    NegativeInferenceComponent(i32),
    #[error("DeepStream startup timeout must be positive")]
    ZeroStartupTimeout,
    #[error("DeepStream shutdown timeout must be positive")]
    ZeroShutdownTimeout,
    #[error("DeepStream batch age limit must be positive when enabled")]
    ZeroBatchAge,
    #[error("DeepStream bridge ABI validation failed: {0}")]
    BridgeAbi(&'static str),
    #[error("GStreamer initialization failed: {0}")]
    GstreamerInit(String),
    #[error("GStreamer pipeline parse failed: {0}")]
    PipelineParse(String),
    #[error("GStreamer parse result is not a Pipeline")]
    ParsedElementNotPipeline,
    #[error("DeepStream pipeline has no bus")]
    MissingBus,
    #[error("DeepStream inference element {element} is missing its sink pad")]
    MissingInferenceInputPad { element: String },
    #[error("failed to install DeepStream inference-input timestamp probe")]
    InferenceInputProbeInstallFailed,
    #[error("DeepStream pipeline is missing inference element {name}")]
    MissingInferenceElement { name: String },
    #[error("DeepStream inference element {element} is missing probe pad {pad}")]
    MissingProbePad { element: String, pad: String },
    #[error("failed to install DeepStream metadata probe")]
    ProbeInstallFailed,
    #[error("DeepStream pipeline is missing preview element {0}")]
    MissingPreviewElement(&'static str),
    #[error("DeepStream preview sink is missing its sink pad")]
    MissingPreviewPad,
    #[error("failed to install DeepStream JPEG preview probe")]
    PreviewProbeInstallFailed,
    #[error("DeepStream pipeline is missing the crosshair observer sink")]
    MissingCrosshairElement,
    #[error("DeepStream crosshair observer sink is missing its sink pad")]
    MissingCrosshairPad,
    #[error("failed to install DeepStream crosshair JPEG probe")]
    CrosshairProbeInstallFailed,
    #[error("crosshair observer failed: {0}")]
    Crosshair(String),
    #[error("DeepStream pipeline state change failed: {0}")]
    StateChange(String),
    #[error("DeepStream shutdown cleanup failed: {0}")]
    ShutdownCleanup(String),
    #[error("{primary}; DeepStream cleanup also failed: {cleanup}")]
    CleanupAfterFailure { primary: String, cleanup: String },
    #[error("DeepStream pipeline did not reach PLAYING; current state is {current:?}")]
    DidNotReachPlaying { current: gst::State },
    #[error(
        "DeepStream pipeline produced no admitted batch within {timeout_ms} ms; current state is {current:?}; metrics={{probed_buffers:{}, published_batches:{}, busy_dropped_batches:{}, extraction_rejections:{}, admission_rejections:{}, ingress_rejections:{}, unavailable_snapshot_slots:{}, overwritten_snapshots:{}}}; last rejection: {last_rejection}",
        metrics.probed_buffers,
        metrics.published_batches,
        metrics.busy_dropped_batches,
        metrics.extraction_rejections,
        metrics.admission_rejections,
        metrics.ingress_rejections,
        metrics.unavailable_snapshot_slots,
        metrics.overwritten_snapshots
    )]
    FirstBatchTimeout {
        timeout_ms: u64,
        current: gst::State,
        metrics: SessionMetrics,
        last_rejection: String,
    },
    #[error("DeepStream pipeline was stopped before its first admitted batch")]
    StoppedBeforeReady,
    #[error("DeepStream pipeline bus error from {element}: {message}; debug={debug:?}")]
    BusError {
        element: String,
        message: String,
        debug: Option<String>,
    },
    #[error("DeepStream pipeline reached EOS unexpectedly")]
    UnexpectedEos,
    #[error("failed to spawn DeepStream owner thread: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("failed to spawn DeepStream perception worker: {0}")]
    SpawnPerceptionWorker(#[source] std::io::Error),
    #[error("Rust TensorRT stage selected but this binary was not compiled with TensorRT support")]
    RustTensorRtNotCompiled,
    #[error("pipeline inference stage does not match its worker configuration")]
    InferenceStageMismatch,
    #[error("failed to initialize inference worker: {0}")]
    InferenceInitialize(String),
    #[error("inference worker closed its initialization channel")]
    InferenceInitializeChannelClosed,
    #[error("DeepStream startup channel closed before readiness")]
    StartupChannelClosed,
    #[error("DeepStream startup failed: {0}")]
    StartupFailed(String),
    #[error("DeepStream owner thread panicked")]
    WorkerPanicked,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_exchange_keeps_the_newest_pending_frame_under_sustained_overload() {
        let exchange = SnapshotExchange::new(LatestFrameExchange::new());
        // Model the single perception worker holding the current frame while
        // the streaming thread repeatedly replaces the one pending frame.
        let worker_owned = exchange.claim().expect("worker slot");

        for generation in 0..128 {
            let mut pending = exchange.claim().expect("replacement slot remains reusable");
            assert!(pending.frame.is_none());
            pending.pipeline_running_now_ns = generation;
            pending.frame = Some(gst::Buffer::new());
            assert_eq!(exchange.publish(pending).unwrap(), generation > 0);
        }

        let mut newest = exchange.wait_take().expect("newest pending frame");
        assert_eq!(newest.pipeline_running_now_ns, 127);
        assert!(newest.frame.take().is_some());
        exchange.recycle(newest);
        exchange.recycle(worker_owned);
        assert_eq!(exchange.free.len(), SNAPSHOT_SLOT_COUNT);
    }
}

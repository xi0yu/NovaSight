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
use novasight_deepstream_bridge::{
    AdmissionContext, PipelineClockSample, admit_snapshot, extract_frame_into, validate_loaded_abi,
};
use novasight_pipeline::{
    PerceptionAdapter, PerceptionError, PerceptionEvent, PerceptionMetrics, PerceptionSession,
    PipelineError, PipelineIngress,
};
use thiserror::Error;

use super::DeepStreamPipelineSpec;

const EVENT_CAPACITY: usize = 8;
const BUS_POLL_INTERVAL: Duration = Duration::from_millis(50);
const SNAPSHOT_SLOT_COUNT: usize = 3;

#[derive(Clone, Debug)]
pub struct DeepStreamSessionConfig {
    pub pipeline: DeepStreamPipelineSpec,
    pub probe_pad: String,
    pub source_id: u32,
    pub inference_component_id: i32,
    pub max_batch_age_ns: Option<u64>,
    pub startup_timeout: Duration,
    pub shutdown_timeout: Duration,
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
    pub probed_buffers: u64,
    pub published_batches: u64,
    pub busy_dropped_batches: u64,
    pub overwritten_snapshots: u64,
    pub unavailable_snapshot_slots: u64,
    pub extraction_rejections: u64,
    pub admission_rejections: u64,
    pub ingress_rejections: u64,
}

#[derive(Debug, Default)]
struct AtomicSessionMetrics {
    probed_buffers: AtomicU64,
    published_batches: AtomicU64,
    busy_dropped_batches: AtomicU64,
    overwritten_snapshots: AtomicU64,
    unavailable_snapshot_slots: AtomicU64,
    extraction_rejections: AtomicU64,
    admission_rejections: AtomicU64,
    ingress_rejections: AtomicU64,
}

impl AtomicSessionMetrics {
    fn snapshot(&self) -> SessionMetrics {
        SessionMetrics {
            probed_buffers: self.probed_buffers.load(Ordering::Relaxed),
            published_batches: self.published_batches.load(Ordering::Relaxed),
            busy_dropped_batches: self.busy_dropped_batches.load(Ordering::Relaxed),
            overwritten_snapshots: self.overwritten_snapshots.load(Ordering::Relaxed),
            unavailable_snapshot_slots: self.unavailable_snapshot_slots.load(Ordering::Relaxed),
            extraction_rejections: self.extraction_rejections.load(Ordering::Relaxed),
            admission_rejections: self.admission_rejections.load(Ordering::Relaxed),
            ingress_rejections: self.ingress_rejections.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct SnapshotSlot {
    snapshot: MaybeUninit<novasight_deepstream_bridge::FrameSnapshot>,
    pipeline_running_now_ns: u64,
    monotonic_now: novasight_core::MonotonicNanos,
}

impl SnapshotSlot {
    fn empty() -> Self {
        Self {
            snapshot: MaybeUninit::uninit(),
            pipeline_running_now_ns: 0,
            monotonic_now: novasight_core::MonotonicNanos(0),
        }
    }
}

#[derive(Debug)]
struct SnapshotExchange {
    free: ArrayQueue<Box<SnapshotSlot>>,
    ready: ArrayQueue<Box<SnapshotSlot>>,
    closed: AtomicBool,
    wake_sequence: AtomicU64,
    wake_lock: Mutex<()>,
    wake: Condvar,
}

impl SnapshotExchange {
    fn new() -> Arc<Self> {
        let exchange = Arc::new(Self {
            free: ArrayQueue::new(SNAPSHOT_SLOT_COUNT),
            ready: ArrayQueue::new(1),
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

    fn claim(&self) -> Option<(Box<SnapshotSlot>, bool)> {
        self.free
            .pop()
            .map(|slot| (slot, false))
            .or_else(|| self.ready.pop().map(|slot| (slot, true)))
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
                    if let Some(stale) = self.ready.pop() {
                        overwritten = true;
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
    events: SyncSender<SessionEvent>,
    perception_events: Option<SyncSender<PerceptionEvent>>,
}

impl ProbeState {
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
    probe_pad: gst::Pad,
    probe_id: gst::PadProbeId,
    bus: gst::Bus,
    exchange: Arc<SnapshotExchange>,
    perception_worker: JoinHandle<()>,
}

#[derive(Clone, Debug)]
pub struct DeepStreamAdapter {
    config: DeepStreamSessionConfig,
}

impl DeepStreamAdapter {
    pub fn new(config: DeepStreamSessionConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DeepStreamSessionConfig {
        &self.config
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
        DeepStreamSession::start_internal(self.config.clone(), epoch, ingress, clock, Some(events))
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
        Self::start_internal(config, epoch, ingress, clock, None)
    }

    fn start_internal(
        config: DeepStreamSessionConfig,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        clock: Arc<dyn Clock>,
        perception_events: Option<SyncSender<PerceptionEvent>>,
    ) -> Result<Self, SessionError> {
        config.validate()?;
        let (command_tx, command_rx) = sync_channel(1);
        let (event_tx, event_rx) = sync_channel(EVENT_CAPACITY);
        let (startup_tx, startup_rx) = sync_channel(1);
        let state = Arc::new(ProbeState {
            closed: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            first_published: AtomicBool::new(false),
            metrics: AtomicSessionMetrics::default(),
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

impl PerceptionSession for DeepStreamSession {
    fn metrics(&self) -> PerceptionMetrics {
        let metrics = self.metrics();
        PerceptionMetrics {
            probed_buffers: metrics.probed_buffers,
            published_batches: metrics.published_batches,
            busy_dropped_batches: metrics.busy_dropped_batches,
            overwritten_snapshots: metrics.overwritten_snapshots,
            unavailable_snapshot_slots: metrics.unavailable_snapshot_slots,
            extraction_rejections: metrics.extraction_rejections,
            admission_rejections: metrics.admission_rejections,
            ingress_rejections: metrics.ingress_rejections,
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
) -> Result<(), SessionError> {
    let startup_result = start_pipeline(&config, epoch, ingress, clock, Arc::clone(&state));
    let StartedPipeline {
        pipeline,
        probe_pad,
        probe_id,
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
    let mut perception_worker = Some(perception_worker);

    let ready = wait_until_ready(&pipeline, &bus, &commands, &state, config.startup_timeout);
    if let Err(error) = ready {
        let error = combine_with_cleanup(
            error,
            cleanup_pipeline(
                &pipeline,
                &probe_pad,
                probe_id,
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
                &probe_pad,
                probe_id,
                config.shutdown_timeout,
                &state,
                &exchange,
                &mut perception_worker,
            ),
        ));
    }

    let outcome = monitor_pipeline(&bus, &commands, &state);
    let cleanup = cleanup_pipeline(
        &pipeline,
        &probe_pad,
        probe_id,
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
) -> Result<StartedPipeline, SessionError> {
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
    let probe_pad =
        inference
            .static_pad(&config.probe_pad)
            .ok_or_else(|| SessionError::MissingProbePad {
                element: config.pipeline.inference_element.clone(),
                pad: config.probe_pad.clone(),
            })?;
    let weak_pipeline = pipeline.downgrade();
    let source_id = config.source_id;
    let exchange = SnapshotExchange::new();
    let probe_exchange = Arc::clone(&exchange);
    let probe_state = Arc::clone(&state);
    let probe_clock = Arc::clone(&monotonic_clock);
    let probe_id = probe_pad
        .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            if probe_state.closed.load(Ordering::Acquire) {
                return gst::PadProbeReturn::Ok;
            }
            let Some(buffer) = info.buffer() else {
                return gst::PadProbeReturn::Ok;
            };
            probe_state
                .metrics
                .probed_buffers
                .fetch_add(1, Ordering::Relaxed);
            let Some((mut slot, claimed_ready)) = probe_exchange.claim() else {
                probe_state
                    .metrics
                    .unavailable_snapshot_slots
                    .fetch_add(1, Ordering::Relaxed);
                return gst::PadProbeReturn::Ok;
            };
            if claimed_ready {
                probe_state
                    .metrics
                    .overwritten_snapshots
                    .fetch_add(1, Ordering::Relaxed);
            }
            let Some(buffer) = NonNull::new(buffer.as_ptr().cast_mut().cast::<c_void>()) else {
                probe_state
                    .metrics
                    .extraction_rejections
                    .fetch_add(1, Ordering::Relaxed);
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            if unsafe { extract_frame_into(buffer, source_id, &mut slot.snapshot) }.is_err() {
                probe_state
                    .metrics
                    .extraction_rejections
                    .fetch_add(1, Ordering::Relaxed);
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            }
            let Some(pipeline) = weak_pipeline.upgrade() else {
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            let Some(gst_clock) = pipeline.clock() else {
                probe_state
                    .metrics
                    .admission_rejections
                    .fetch_add(1, Ordering::Relaxed);
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            let Some(base_time) = pipeline.base_time() else {
                probe_state
                    .metrics
                    .admission_rejections
                    .fetch_add(1, Ordering::Relaxed);
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
                probe_exchange.recycle(slot);
                return gst::PadProbeReturn::Ok;
            };
            slot.pipeline_running_now_ns = running_now_ns;
            slot.monotonic_now = probe_clock.now();
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

    let worker_context = SnapshotWorkerContext {
        epoch,
        source_id,
        inference_component_id: config.inference_component_id,
        max_batch_age_ns: config.max_batch_age_ns,
        monotonic_clock,
        ingress,
    };
    let mut perception_worker =
        match spawn_snapshot_worker(Arc::clone(&exchange), Arc::clone(&state), worker_context) {
            Ok(worker) => Some(worker),
            Err(error) => {
                probe_pad.remove_probe(probe_id);
                exchange.close();
                return Err(error);
            }
        };

    if let Err(error) = pipeline.set_state(gst::State::Playing) {
        return Err(combine_with_cleanup(
            SessionError::StateChange(error.to_string()),
            cleanup_pipeline(
                &pipeline,
                &probe_pad,
                probe_id,
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
                &probe_pad,
                probe_id,
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
                &probe_pad,
                probe_id,
                config.shutdown_timeout,
                &state,
                &exchange,
                &mut perception_worker,
            ),
        ));
    }
    Ok(StartedPipeline {
        pipeline,
        probe_pad,
        probe_id,
        bus,
        exchange,
        perception_worker: perception_worker
            .expect("perception worker remains owned after successful startup"),
    })
}

struct SnapshotWorkerContext {
    epoch: RuntimeEpoch,
    source_id: u32,
    inference_component_id: i32,
    max_batch_age_ns: Option<u64>,
    monotonic_clock: Arc<dyn Clock>,
    ingress: PipelineIngress,
}

fn spawn_snapshot_worker(
    exchange: Arc<SnapshotExchange>,
    state: Arc<ProbeState>,
    context: SnapshotWorkerContext,
) -> Result<JoinHandle<()>, SessionError> {
    thread::Builder::new()
        .name("novasight-perception".to_owned())
        .spawn(move || {
            let worker_state = Arc::clone(&state);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_snapshot_worker(&exchange, &state, &context);
            }));
            if outcome.is_err() {
                worker_state.fault("DeepStream perception worker panicked");
            }
        })
        .map_err(SessionError::SpawnPerceptionWorker)
}

fn run_snapshot_worker(
    exchange: &SnapshotExchange,
    state: &ProbeState,
    context: &SnapshotWorkerContext,
) {
    while let Some(slot) = exchange.wait_take() {
        if state.closed.load(Ordering::Acquire) {
            exchange.recycle(slot);
            break;
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
                    exchange.recycle(slot);
                    break;
                }
            };
        // The producer publishes a slot only after the C bridge reports a
        // successful full initialization of FrameSnapshot. Ownership of this
        // Box stays with the worker until it is recycled into the fixed pool.
        let snapshot = unsafe { slot.snapshot.assume_init_ref() };
        let admitted = admit_snapshot(
            snapshot,
            AdmissionContext {
                epoch: context.epoch,
                generation,
                clock: PipelineClockSample::new(slot.pipeline_running_now_ns, slot.monotonic_now),
                source_id: context.source_id,
                inference_component_id: context.inference_component_id,
            },
        );
        let admitted = match admitted {
            Ok(admitted) => admitted,
            Err(_) => {
                state
                    .metrics
                    .admission_rejections
                    .fetch_add(1, Ordering::Relaxed);
                exchange.recycle(slot);
                continue;
            }
        };
        if context.max_batch_age_ns.is_some_and(|maximum| {
            context
                .monotonic_clock
                .now()
                .0
                .saturating_sub(admitted.batch().stamp().captured_at.0)
                > maximum
        }) {
            state
                .metrics
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            exchange.recycle(slot);
            continue;
        }
        if state.closed.load(Ordering::Acquire) {
            exchange.recycle(slot);
            break;
        }
        match context.ingress.try_submit(admitted.into_batch()) {
            Ok(()) => {
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
                state.fault(format!("DeepStream ingress rejected a batch: {error}"));
                exchange.recycle(slot);
                break;
            }
        }
        exchange.recycle(slot);
    }
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
            return Err(SessionError::FirstBatchTimeout {
                timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
                current: pipeline.current_state(),
            });
        }
    }
}

fn monitor_pipeline(
    bus: &gst::Bus,
    commands: &Receiver<SessionCommand>,
    state: &ProbeState,
) -> Result<(), SessionError> {
    loop {
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

fn cleanup_pipeline(
    pipeline: &gst::Pipeline,
    probe_pad: &gst::Pad,
    probe_id: gst::PadProbeId,
    timeout: Duration,
    state: &ProbeState,
    exchange: &SnapshotExchange,
    perception_worker: &mut Option<JoinHandle<()>>,
) -> Result<(), SessionError> {
    state.closed.store(true, Ordering::Release);
    probe_pad.remove_probe(probe_id);
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
    #[error("DeepStream pipeline is missing inference element {name}")]
    MissingInferenceElement { name: String },
    #[error("DeepStream inference element {element} is missing probe pad {pad}")]
    MissingProbePad { element: String, pad: String },
    #[error("failed to install DeepStream metadata probe")]
    ProbeInstallFailed,
    #[error("DeepStream pipeline state change failed: {0}")]
    StateChange(String),
    #[error("DeepStream shutdown cleanup failed: {0}")]
    ShutdownCleanup(String),
    #[error("{primary}; DeepStream cleanup also failed: {cleanup}")]
    CleanupAfterFailure { primary: String, cleanup: String },
    #[error("DeepStream pipeline did not reach PLAYING; current state is {current:?}")]
    DidNotReachPlaying { current: gst::State },
    #[error(
        "DeepStream pipeline produced no admitted batch within {timeout_ms} ms; current state is {current:?}"
    )]
    FirstBatchTimeout {
        timeout_ms: u64,
        current: gst::State,
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
    #[error("DeepStream startup channel closed before readiness")]
    StartupChannelClosed,
    #[error("DeepStream startup failed: {0}")]
    StartupFailed(String),
    #[error("DeepStream owner thread panicked")]
    WorkerPanicked,
}

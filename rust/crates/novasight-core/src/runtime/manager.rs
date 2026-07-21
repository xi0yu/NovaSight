use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use tokio::sync::{mpsc, oneshot, watch};

use crate::{
    AppError, Detection, DetectionBatch, ErrorSnapshot, FrameStamp, OperationalSnapshot,
    ProportionalReplayControl, RecordingPointerDevice, ReplayPerceptionSource, RunIntent,
    RuntimeCommandReceipt, RuntimeEpoch, RuntimePhase,
};

use super::session::{EpochSnapshotPublisher, RuntimeSession, SessionStats};

const COMMAND_CAPACITY: usize = 16;

#[derive(Clone)]
enum ReplaySeed {
    Fixture,
    Rebind(Arc<[DetectionBatch]>),
    PreserveEpochs(Arc<[DetectionBatch]>),
}

/// Replay-only construction inputs retained by the manager so each epoch gets
/// a fresh source and an exclusively owned dry-run device lease.
#[derive(Clone)]
pub struct RuntimeDependencies {
    replay_seed: ReplaySeed,
}

impl RuntimeDependencies {
    pub fn replay_fixture() -> Self {
        Self {
            replay_seed: ReplaySeed::Fixture,
        }
    }

    /// Replays the supplied observations in every session, rebinding each
    /// frame stamp to the newly allocated runtime epoch.
    pub fn replay(batches: impl IntoIterator<Item = DetectionBatch>) -> Self {
        Self {
            replay_seed: ReplaySeed::Rebind(batches.into_iter().collect::<Vec<_>>().into()),
        }
    }

    /// Preserves supplied frame epochs for explicit stale-input fault tests.
    pub fn replay_preserving_epochs(batches: impl IntoIterator<Item = DetectionBatch>) -> Self {
        Self {
            replay_seed: ReplaySeed::PreserveEpochs(batches.into_iter().collect::<Vec<_>>().into()),
        }
    }

    fn source_for(&self, epoch: RuntimeEpoch) -> ReplayPerceptionSource {
        match &self.replay_seed {
            ReplaySeed::Fixture => ReplayPerceptionSource::new([fixture_batch(epoch)]),
            ReplaySeed::Rebind(batches) => {
                ReplayPerceptionSource::new(batches.iter().map(|batch| rebind_batch(batch, epoch)))
            }
            ReplaySeed::PreserveEpochs(batches) => {
                ReplayPerceptionSource::new(batches.iter().cloned())
            }
        }
    }
}

fn fixture_batch(epoch: RuntimeEpoch) -> DetectionBatch {
    let detection = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9)
        .expect("hard-coded replay detection is valid");
    DetectionBatch::fixture(FrameStamp::new(epoch, 1, 1), 640, 640, vec![detection])
        .expect("hard-coded replay batch is valid")
}

fn rebind_batch(batch: &DetectionBatch, epoch: RuntimeEpoch) -> DetectionBatch {
    let stamp = batch.stamp();
    DetectionBatch::fixture(
        FrameStamp {
            epoch,
            generation: stamp.generation,
            captured_at: stamp.captured_at,
        },
        batch.coordinate_width(),
        batch.coordinate_height(),
        batch.detections().to_vec(),
    )
    .expect("rebinding an admitted replay batch preserves its validation invariants")
}

/// Namespace for creating the sole runtime lifecycle owner.
pub struct RuntimeManager;

impl RuntimeManager {
    pub fn spawn(dependencies: RuntimeDependencies) -> RuntimeHandle {
        let initial = Arc::new(OperationalSnapshot::initial_replay());
        let (snapshot_tx, snapshot_rx) = watch::channel(initial);
        let publisher = EpochSnapshotPublisher::new(snapshot_tx);
        let (command_tx, command_rx) = mpsc::channel(COMMAND_CAPACITY);
        tokio::spawn(run_manager(command_rx, publisher, dependencies));

        RuntimeHandle {
            command_tx,
            snapshot_rx,
            start_in_flight: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Cloneable command/query surface; session internals and adapters never escape.
#[derive(Clone)]
pub struct RuntimeHandle {
    command_tx: mpsc::Sender<RuntimeCommand>,
    snapshot_rx: watch::Receiver<Arc<OperationalSnapshot>>,
    start_in_flight: Arc<AtomicBool>,
}

impl RuntimeHandle {
    pub async fn start(&self) -> Result<RuntimeCommandReceipt, AppError> {
        if self
            .start_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(AppError::RuntimeCommandConflict { command: "start" });
        }
        let _permit = StartPermit(&self.start_in_flight);
        self.command(RuntimeCommandKind::Start).await
    }

    pub async fn stop(&self) -> Result<RuntimeCommandReceipt, AppError> {
        self.command(RuntimeCommandKind::Stop).await
    }

    pub fn snapshot(&self) -> Arc<OperationalSnapshot> {
        self.snapshot_rx.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<Arc<OperationalSnapshot>> {
        self.snapshot_rx.clone()
    }

    async fn command(&self, kind: RuntimeCommandKind) -> Result<RuntimeCommandReceipt, AppError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(RuntimeCommand {
                kind,
                reply: reply_tx,
            })
            .await
            .map_err(|_| AppError::RuntimeManagerUnavailable)?;
        reply_rx
            .await
            .map_err(|_| AppError::RuntimeManagerUnavailable)?
    }
}

struct StartPermit<'a>(&'a AtomicBool);

impl Drop for StartPermit<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
enum RuntimeCommandKind {
    Start,
    Stop,
}

struct RuntimeCommand {
    kind: RuntimeCommandKind,
    reply: oneshot::Sender<Result<RuntimeCommandReceipt, AppError>>,
}

async fn run_manager(
    mut command_rx: mpsc::Receiver<RuntimeCommand>,
    publisher: EpochSnapshotPublisher,
    dependencies: RuntimeDependencies,
) {
    let mut next_epoch = 0_u64;
    let mut session: Option<RuntimeSession> = None;

    while let Some(command) = command_rx.recv().await {
        let result = match command.kind {
            RuntimeCommandKind::Start => {
                start_session(&publisher, &dependencies, &mut next_epoch, &mut session)
            }
            RuntimeCommandKind::Stop => stop_session(&publisher, &mut session).await,
        };
        let _ = command.reply.send(result);
    }

    if let Some(active_session) = session.take() {
        let current = publisher.current();
        let mut stopping = current.as_ref().clone();
        stopping.phase = RuntimePhase::Stopping;
        stopping.run_intent = RunIntent::Stopped;
        stopping.running = false;
        publisher.revoke_and_publish(stopping);
        let _ = active_session.stop().await;
    }
}

fn start_session(
    publisher: &EpochSnapshotPublisher,
    dependencies: &RuntimeDependencies,
    next_epoch: &mut u64,
    session: &mut Option<RuntimeSession>,
) -> Result<RuntimeCommandReceipt, AppError> {
    let current = publisher.current();
    match current.phase {
        RuntimePhase::Running | RuntimePhase::Standby => {
            return Ok(receipt(&current));
        }
        RuntimePhase::Stopped => {}
        RuntimePhase::Starting | RuntimePhase::Stopping | RuntimePhase::Faulted => {
            return Err(AppError::RuntimeCommandConflict { command: "start" });
        }
    }

    *next_epoch = next_epoch
        .checked_add(1)
        .ok_or(AppError::RuntimeEpochExhausted)?;
    let epoch = RuntimeEpoch(*next_epoch);

    let mut starting = OperationalSnapshot::initial_replay();
    starting.phase = RuntimePhase::Starting;
    starting.run_intent = RunIntent::Running;
    starting.epoch = Some(epoch);
    publisher.claim_and_publish(epoch, starting);

    let mut running = publisher.current().as_ref().clone();
    running.phase = RuntimePhase::Running;
    running.running = true;
    publisher.publish_manager(running);

    let source = dependencies.source_for(epoch);
    let device = Arc::new(RecordingPointerDevice::default());
    *session = Some(RuntimeSession::spawn(
        epoch,
        source,
        ProportionalReplayControl::new(1.0),
        device,
        publisher.clone(),
    ));

    Ok(receipt(&publisher.current()))
}

async fn stop_session(
    publisher: &EpochSnapshotPublisher,
    session: &mut Option<RuntimeSession>,
) -> Result<RuntimeCommandReceipt, AppError> {
    let current = publisher.current();
    if current.phase == RuntimePhase::Stopped {
        return Ok(receipt(&current));
    }

    let mut stopping = current.as_ref().clone();
    stopping.phase = RuntimePhase::Stopping;
    stopping.run_intent = RunIntent::Stopped;
    stopping.running = false;

    // Revocation and the Stopping publication are one critical section. Any
    // in-progress dry-run send is before this point; no later send can enter.
    publisher.revoke_and_publish(stopping);

    let stats = match session.take() {
        Some(active_session) => match active_session.stop().await {
            Ok(exit) => exit.stats,
            Err(error) => {
                let mut faulted = publisher.current().as_ref().clone();
                faulted.phase = RuntimePhase::Faulted;
                faulted.running = false;
                faulted.fatal_error = Some(ErrorSnapshot::from_error(&error));
                publisher.publish_manager(faulted);
                return Err(error);
            }
        },
        None => SessionStats {
            last_generation: current.last_generation,
            processed_batches: current.processed_batches,
            device_receipts: current.device_receipts,
        },
    };

    let mut stopped = publisher.current().as_ref().clone();
    stopped.phase = RuntimePhase::Stopped;
    stopped.run_intent = RunIntent::Stopped;
    stopped.running = false;
    stopped.last_generation = stats.last_generation;
    stopped.processed_batches = stats.processed_batches;
    stopped.device_receipts = stats.device_receipts;
    stopped.fatal_error = None;
    publisher.publish_manager(stopped);

    Ok(receipt(&publisher.current()))
}

fn receipt(snapshot: &OperationalSnapshot) -> RuntimeCommandReceipt {
    RuntimeCommandReceipt {
        epoch: snapshot.epoch,
        phase: snapshot.phase,
    }
}

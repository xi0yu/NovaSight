use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use tokio::sync::{mpsc, oneshot, watch};

use crate::{
    AppError, Detection, DetectionBatch, ErrorSnapshot, FrameStamp, OperationalSnapshot,
    ProportionalReplayControl, RecordingPointerDevice, ReplayPerceptionSource, RunIntent,
    RuntimeCommandReceipt, RuntimeEpoch, RuntimePhase,
};

use super::session::{EpochSnapshotPublisher, RuntimeSession, RuntimeSessionEvent, SessionStats};

const COMMAND_CAPACITY: usize = 16;
const EVENT_CAPACITY: usize = 16;

#[derive(Clone)]
enum ReplaySeed {
    Fixture,
    Rebind(Arc<[DetectionBatch]>),
    #[cfg(test)]
    FaultFirstThenRebind(Arc<[DetectionBatch]>),
}

/// Replay-only construction inputs retained by the manager so each epoch gets
/// a fresh source and an exclusively owned dry-run device lease.
#[derive(Clone)]
pub struct RuntimeDependencies {
    replay_seed: ReplaySeed,
    frame_interval: Duration,
}

impl RuntimeDependencies {
    pub fn replay_fixture(frame_interval: Duration) -> Self {
        Self {
            replay_seed: ReplaySeed::Fixture,
            frame_interval,
        }
    }

    /// Replays the supplied observations in every session, rebinding each
    /// frame stamp to the newly allocated runtime epoch.
    pub fn replay(
        batches: impl IntoIterator<Item = DetectionBatch>,
        frame_interval: Duration,
    ) -> Self {
        Self {
            replay_seed: ReplaySeed::Rebind(batches.into_iter().collect::<Vec<_>>().into()),
            frame_interval,
        }
    }

    #[cfg(test)]
    fn fault_first_then_rebind(batches: impl IntoIterator<Item = DetectionBatch>) -> Self {
        Self {
            replay_seed: ReplaySeed::FaultFirstThenRebind(
                batches.into_iter().collect::<Vec<_>>().into(),
            ),
            frame_interval: Duration::ZERO,
        }
    }

    fn source_for(&self, epoch: RuntimeEpoch) -> ReplayPerceptionSource {
        match &self.replay_seed {
            ReplaySeed::Fixture => ReplayPerceptionSource::new([fixture_batch(epoch)]),
            ReplaySeed::Rebind(batches) => {
                ReplayPerceptionSource::new(batches.iter().map(|batch| rebind_batch(batch, epoch)))
            }
            #[cfg(test)]
            ReplaySeed::FaultFirstThenRebind(batches) => {
                if epoch == RuntimeEpoch(1) {
                    ReplayPerceptionSource::new(batches.iter().cloned())
                } else {
                    ReplayPerceptionSource::new(
                        batches.iter().map(|batch| rebind_batch(batch, epoch)),
                    )
                }
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
        let (event_tx, event_rx) = mpsc::channel(EVENT_CAPACITY);
        tokio::spawn(run_manager(
            command_rx,
            event_rx,
            event_tx,
            publisher,
            dependencies,
        ));

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
    mut event_rx: mpsc::Receiver<RuntimeSessionEvent>,
    event_tx: mpsc::Sender<RuntimeSessionEvent>,
    publisher: EpochSnapshotPublisher,
    dependencies: RuntimeDependencies,
) {
    let mut next_epoch = 0_u64;
    let mut session: Option<RuntimeSession> = None;

    loop {
        tokio::select! {
            biased;
            Some(event) = event_rx.recv() => {
                handle_session_event(event, &publisher, &mut session).await;
            }
            command = command_rx.recv() => {
                let Some(command) = command else {
                    break;
                };
                let result = match command.kind {
                    RuntimeCommandKind::Start => {
                        start_session(
                            &publisher,
                            &dependencies,
                            &event_tx,
                            &mut next_epoch,
                            &mut session,
                        )
                    }
                    RuntimeCommandKind::Stop => stop_session(&publisher, &mut session).await,
                };
                let _ = command.reply.send(result);
            }
        }
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

async fn handle_session_event(
    event: RuntimeSessionEvent,
    publisher: &EpochSnapshotPublisher,
    session: &mut Option<RuntimeSession>,
) {
    let RuntimeSessionEvent::Faulted {
        epoch,
        error,
        stats,
    } = event;
    let current = publisher.current();
    if current.epoch != Some(epoch) || session.is_none() {
        return;
    }

    let mut faulted = current.as_ref().clone();
    faulted.phase = RuntimePhase::Faulted;
    faulted.running = false;
    faulted.last_generation = stats.last_generation;
    faulted.processed_batches = stats.processed_batches;
    faulted.device_receipts = stats.device_receipts;
    faulted.fatal_error = Some(ErrorSnapshot::from_error(&error));

    // Fault publication and epoch revocation share the ownership gate. Any
    // accepted send is before this point; no later send can acquire authority.
    publisher.revoke_and_publish(faulted);

    if let Some(faulted_session) = session.take() {
        let _ = faulted_session.stop().await;
    }
}

fn start_session(
    publisher: &EpochSnapshotPublisher,
    dependencies: &RuntimeDependencies,
    event_tx: &mpsc::Sender<RuntimeSessionEvent>,
    next_epoch: &mut u64,
    session: &mut Option<RuntimeSession>,
) -> Result<RuntimeCommandReceipt, AppError> {
    let current = publisher.current();
    match current.phase {
        RuntimePhase::Running | RuntimePhase::Standby => {
            return Ok(receipt(&current));
        }
        RuntimePhase::Stopped | RuntimePhase::Faulted => {}
        RuntimePhase::Starting | RuntimePhase::Stopping => {
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
        dependencies.frame_interval,
        ProportionalReplayControl::new(1.0),
        device,
        publisher.clone(),
        event_tx.clone(),
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

fn receipt(snapshot: &Arc<OperationalSnapshot>) -> RuntimeCommandReceipt {
    RuntimeCommandReceipt {
        epoch: snapshot.epoch,
        phase: snapshot.phase,
        snapshot: snapshot.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_batch(epoch: RuntimeEpoch, generation: u64) -> DetectionBatch {
        let detection = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9).expect("detection");
        DetectionBatch::fixture(
            FrameStamp::new(epoch, generation, generation),
            640,
            640,
            vec![detection],
        )
        .expect("batch")
    }

    async fn wait_for_snapshot(
        runtime: &RuntimeHandle,
        predicate: impl Fn(&OperationalSnapshot) -> bool,
    ) -> Arc<OperationalSnapshot> {
        let mut snapshots = runtime.subscribe();
        loop {
            let snapshot = snapshots.borrow_and_update().clone();
            if predicate(&snapshot) {
                return snapshot;
            }
            snapshots
                .changed()
                .await
                .expect("runtime keeps snapshot watch open");
        }
    }

    #[tokio::test]
    async fn manager_cleans_terminal_fault_before_allowing_restart() {
        let runtime = RuntimeManager::spawn(RuntimeDependencies::fault_first_then_rebind([
            target_batch(RuntimeEpoch(1), 1),
            target_batch(RuntimeEpoch(99), 2),
        ]));

        let first = runtime.start().await.expect("first epoch");
        let faulted =
            wait_for_snapshot(&runtime, |snapshot| snapshot.phase == RuntimePhase::Faulted).await;
        assert_eq!(faulted.epoch, first.epoch);
        assert_eq!(faulted.processed_batches, 1);
        assert_eq!(faulted.device_receipts, 1);
        assert_eq!(
            faulted
                .fatal_error
                .as_ref()
                .map(|error| error.code.as_str()),
            Some("runtime_epoch_mismatch")
        );

        let receipts_at_fault = faulted.device_receipts;
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert_eq!(runtime.snapshot().device_receipts, receipts_at_fault);

        let second = runtime.start().await.expect("clean restart after fault");
        assert!(second.epoch > first.epoch);
        let restarted = wait_for_snapshot(&runtime, |snapshot| {
            snapshot.epoch == second.epoch && snapshot.device_receipts == 2
        })
        .await;
        assert_eq!(restarted.phase, RuntimePhase::Running);
        assert_eq!(restarted.processed_batches, 2);
        assert_eq!(restarted.fatal_error, None);
        runtime.stop().await.expect("stop restarted epoch");
    }
}

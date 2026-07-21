use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use tokio::{sync::watch, task::JoinHandle};

use crate::{
    AppError, DeviceCommand, ErrorSnapshot, Generation, NearestCenterTargeting,
    OperationalSnapshot, PerceptionSource, ProportionalReplayControl, RecordingPointerDevice,
    ReplayPerceptionSource, RuntimeEpoch, RuntimePhase,
};

/// Serializes ownership changes with session publications and dry-run sends.
///
/// A session may mutate the latest snapshot or send only while holding this
/// gate and observing its epoch as active. Revoking an epoch at stop is the
/// linearization point after which that session can do neither.
#[derive(Clone)]
pub(crate) struct EpochSnapshotPublisher {
    snapshot_tx: watch::Sender<Arc<OperationalSnapshot>>,
    active_epoch: Arc<AtomicU64>,
    gate: Arc<Mutex<()>>,
}

impl EpochSnapshotPublisher {
    pub(crate) fn new(snapshot_tx: watch::Sender<Arc<OperationalSnapshot>>) -> Self {
        Self {
            snapshot_tx,
            active_epoch: Arc::new(AtomicU64::new(0)),
            gate: Arc::new(Mutex::new(())),
        }
    }

    pub(crate) fn current(&self) -> Arc<OperationalSnapshot> {
        self.snapshot_tx.borrow().clone()
    }

    pub(crate) fn claim_and_publish(&self, epoch: RuntimeEpoch, snapshot: OperationalSnapshot) {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.active_epoch.store(epoch.0, Ordering::Release);
        self.snapshot_tx.send_replace(Arc::new(snapshot));
    }

    pub(crate) fn publish_manager(&self, snapshot: OperationalSnapshot) {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.snapshot_tx.send_replace(Arc::new(snapshot));
    }

    pub(crate) fn revoke_and_publish(&self, snapshot: OperationalSnapshot) {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.active_epoch.store(0, Ordering::Release);
        self.snapshot_tx.send_replace(Arc::new(snapshot));
    }

    fn with_owner<T>(&self, epoch: RuntimeEpoch, operation: impl FnOnce() -> T) -> Option<T> {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.active_epoch.load(Ordering::Acquire) != epoch.0 {
            return None;
        }
        Some(operation())
    }

    fn publish_session(
        &self,
        epoch: RuntimeEpoch,
        update: impl FnOnce(&mut OperationalSnapshot),
    ) -> bool {
        self.with_owner(epoch, || {
            let mut snapshot = self.snapshot_tx.borrow().as_ref().clone();
            if snapshot.epoch != Some(epoch) {
                return false;
            }
            update(&mut snapshot);
            self.snapshot_tx.send_replace(Arc::new(snapshot));
            true
        })
        .unwrap_or(false)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SessionStats {
    pub(crate) last_generation: Option<Generation>,
    pub(crate) processed_batches: u64,
    pub(crate) device_receipts: u64,
}

#[derive(Debug)]
pub(crate) struct SessionExit {
    pub(crate) stats: SessionStats,
}

/// Owns the one replay source, dry-run device lease, and worker for an epoch.
pub(crate) struct RuntimeSession {
    cancellation_tx: watch::Sender<bool>,
    latest_command: Arc<Mutex<Option<DeviceCommand>>>,
    task: JoinHandle<SessionExit>,
}

impl RuntimeSession {
    pub(crate) fn spawn(
        epoch: RuntimeEpoch,
        source: ReplayPerceptionSource,
        control: ProportionalReplayControl,
        device: Arc<RecordingPointerDevice>,
        publisher: EpochSnapshotPublisher,
    ) -> Self {
        let (cancellation_tx, cancellation_rx) = watch::channel(false);
        let latest_command = Arc::new(Mutex::new(None));
        let task_latest_command = latest_command.clone();
        let task = tokio::spawn(run_session(
            epoch,
            source,
            control,
            device,
            publisher,
            cancellation_rx,
            task_latest_command,
        ));
        Self {
            cancellation_tx,
            latest_command,
            task,
        }
    }

    /// Cancellation is requested first, then the sole worker is joined, and
    /// only then is the latest command cleared. Returning proves no task owned
    /// by this session remains alive.
    pub(crate) async fn stop(self) -> Result<SessionExit, AppError> {
        self.cancellation_tx.send_replace(true);
        let result = self.task.await.map_err(|_| AppError::RuntimeTaskTerminated);
        self.latest_command
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        result
    }
}

async fn run_session(
    epoch: RuntimeEpoch,
    mut source: ReplayPerceptionSource,
    control: ProportionalReplayControl,
    device: Arc<RecordingPointerDevice>,
    publisher: EpochSnapshotPublisher,
    mut cancellation: watch::Receiver<bool>,
    latest_command: Arc<Mutex<Option<DeviceCommand>>>,
) -> SessionExit {
    let targeting = NearestCenterTargeting;
    let mut stats = SessionStats::default();

    loop {
        if *cancellation.borrow() {
            return SessionExit { stats };
        }

        let next_batch = tokio::select! {
            biased;
            changed = cancellation.changed() => {
                if changed.is_err() || *cancellation.borrow() {
                    return SessionExit { stats };
                }
                continue;
            }
            batch = source.next_batch() => batch,
        };

        let batch = match next_batch {
            Ok(Some(batch)) => batch,
            Ok(None) => {
                let _ = cancellation.changed().await;
                return SessionExit { stats };
            }
            Err(error) => return fault(epoch, stats, error, &publisher),
        };

        let actual_epoch = batch.stamp().epoch;
        if actual_epoch != epoch {
            return fault(
                epoch,
                stats,
                AppError::RuntimeEpochMismatch {
                    expected: epoch.0,
                    actual: actual_epoch.0,
                },
                &publisher,
            );
        }

        stats.last_generation = Some(batch.stamp().generation);
        stats.processed_batches += 1;

        let Some(target) = targeting.select(&batch) else {
            if !publish_stats(epoch, stats, &publisher) {
                return SessionExit { stats };
            }
            tokio::task::yield_now().await;
            continue;
        };

        let decision = match control.decide(&batch, &target, batch.stamp().captured_at.0) {
            Ok(decision) => decision,
            Err(error) => return fault(epoch, stats, error, &publisher),
        };
        let command = decision.into_command();

        let send_result = publisher.with_owner(epoch, || {
            if *cancellation.borrow() {
                return None;
            }
            *latest_command
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(command);
            Some(device.send(command))
        });

        let Some(Some(send_result)) = send_result else {
            return SessionExit { stats };
        };
        if let Err(error) = send_result {
            return fault(epoch, stats, error, &publisher);
        }

        stats.device_receipts += 1;
        if !publish_stats(epoch, stats, &publisher) {
            return SessionExit { stats };
        }
        tokio::task::yield_now().await;
    }
}

fn publish_stats(
    epoch: RuntimeEpoch,
    stats: SessionStats,
    publisher: &EpochSnapshotPublisher,
) -> bool {
    publisher.publish_session(epoch, |snapshot| {
        snapshot.last_generation = stats.last_generation;
        snapshot.processed_batches = stats.processed_batches;
        snapshot.device_receipts = stats.device_receipts;
    })
}

fn fault(
    epoch: RuntimeEpoch,
    stats: SessionStats,
    error: AppError,
    publisher: &EpochSnapshotPublisher,
) -> SessionExit {
    publisher.publish_session(epoch, |snapshot| {
        snapshot.phase = RuntimePhase::Faulted;
        snapshot.running = false;
        snapshot.last_generation = stats.last_generation;
        snapshot.processed_batches = stats.processed_batches;
        snapshot.device_receipts = stats.device_receipts;
        snapshot.fatal_error = Some(ErrorSnapshot::from_error(&error));
    });
    SessionExit { stats }
}

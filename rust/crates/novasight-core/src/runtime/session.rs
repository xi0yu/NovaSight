use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use tokio::{
    sync::{mpsc, watch},
    task::JoinHandle,
    time::{self, MissedTickBehavior},
};

use crate::{
    AppError, DeviceCommand, Generation, NearestCenterTargeting, OperationalSnapshot,
    PerceptionSource, RecordingPointerDevice, ReplayPerceptionSource, RuntimeAlgorithm,
    RuntimeEpoch,
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
pub(crate) enum RuntimeSessionEvent {
    Faulted {
        epoch: RuntimeEpoch,
        error: AppError,
        stats: SessionStats,
    },
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
        frame_interval: Duration,
        algorithm: RuntimeAlgorithm,
        device: Arc<RecordingPointerDevice>,
        publisher: EpochSnapshotPublisher,
        event_tx: mpsc::Sender<RuntimeSessionEvent>,
    ) -> Self {
        let (cancellation_tx, cancellation_rx) = watch::channel(false);
        let latest_command = Arc::new(Mutex::new(None));
        let task_latest_command = latest_command.clone();
        let task = tokio::spawn(run_session(
            epoch,
            PacedReplay {
                source,
                frame_interval,
            },
            algorithm,
            SessionIo {
                device,
                publisher,
                event_tx,
                latest_command: task_latest_command,
            },
            cancellation_rx,
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

struct PacedReplay {
    source: ReplayPerceptionSource,
    frame_interval: Duration,
}

struct SessionIo {
    device: Arc<RecordingPointerDevice>,
    publisher: EpochSnapshotPublisher,
    event_tx: mpsc::Sender<RuntimeSessionEvent>,
    latest_command: Arc<Mutex<Option<DeviceCommand>>>,
}

async fn run_session(
    epoch: RuntimeEpoch,
    mut replay: PacedReplay,
    mut algorithm: RuntimeAlgorithm,
    io: SessionIo,
    mut cancellation: watch::Receiver<bool>,
) -> SessionExit {
    let targeting = NearestCenterTargeting;
    let mut stats = SessionStats::default();
    let mut pace = (!replay.frame_interval.is_zero()).then(|| {
        let mut interval = time::interval(replay.frame_interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        interval
    });

    loop {
        if *cancellation.borrow() {
            return SessionExit { stats };
        }

        if let Some(interval) = pace.as_mut() {
            tokio::select! {
                biased;
                changed = cancellation.changed() => {
                    if changed.is_err() || *cancellation.borrow() {
                        return SessionExit { stats };
                    }
                    continue;
                }
                _ = interval.tick() => {}
            }
        }

        let next_batch = tokio::select! {
            biased;
            changed = cancellation.changed() => {
                if changed.is_err() || *cancellation.borrow() {
                    return SessionExit { stats };
                }
                continue;
            }
            batch = replay.source.next_batch() => batch,
        };

        let batch = match next_batch {
            Ok(Some(batch)) => batch,
            Ok(None) => {
                let _ = cancellation.changed().await;
                return SessionExit { stats };
            }
            Err(error) => return fault(epoch, stats, error, &io.event_tx).await,
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
                &io.event_tx,
            )
            .await;
        }

        stats.last_generation = Some(batch.stamp().generation);
        stats.processed_batches += 1;

        let Some(target) = targeting.select(&batch) else {
            if !publish_stats(epoch, stats, &io.publisher) {
                return SessionExit { stats };
            }
            tokio::task::yield_now().await;
            continue;
        };

        let command = match decide_with_algorithm(&mut algorithm, &batch, &target, epoch) {
            Ok(command) => command,
            Err(error) => return fault(epoch, stats, error, &io.event_tx).await,
        };
        let send_result = io.publisher.with_owner(epoch, || {
            if *cancellation.borrow() {
                return None;
            }
            *io.latest_command
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(command);
            Some(io.device.send(command))
        });

        let Some(Some(send_result)) = send_result else {
            return SessionExit { stats };
        };
        if let Err(error) = send_result {
            return fault(epoch, stats, error, &io.event_tx).await;
        }

        stats.device_receipts += 1;
        if !publish_stats(epoch, stats, &io.publisher) {
            return SessionExit { stats };
        }
        tokio::task::yield_now().await;
    }
}

/// Dispatch the next decision through whichever algorithm the runtime
/// session owns. The proportional path returns its typed
/// `ControlDecision` and converts to a `DeviceCommand`; the dual-phase
/// path produces a `DeviceCommand` directly. Both share the same
/// `DeviceCommand` shape so the device adapter can stay
/// algorithm-agnostic.
fn decide_with_algorithm(
    algorithm: &mut RuntimeAlgorithm,
    batch: &crate::perception::types::DetectionBatch,
    target: &crate::targeting::SelectedTarget,
    epoch: RuntimeEpoch,
) -> Result<DeviceCommand, AppError> {
    match algorithm {
        RuntimeAlgorithm::Proportional(controller) => controller
            .decide(batch, target, batch.stamp().captured_at.0)
            .map(|decision| decision.into_command()),
        RuntimeAlgorithm::DualPhase(controller) => {
            let observation = crate::control::dual_phase_v2::ControlObservation {
                generation: batch.stamp().generation.0,
                frame_id: batch.stamp().generation.0,
                target_id: target.object_id,
                capture_ts_ns: batch.stamp().captured_at.0,
                inference_end_ts_ns: batch.stamp().captured_at.0 + 5_000_000,
                control_now_ns: batch.stamp().captured_at.0 + 8_000_000,
                aim_x: target.center_x,
                aim_y: target.center_y,
                crosshair_x: 320.0,
                crosshair_y: 320.0,
                detection_confidence: f64::from(target.confidence),
                // This replay selector has no identity tracker; its selected-target
                // confidence is the only real quality signal available.
                track_confidence: f64::from(target.confidence),
                target_valid: true,
                trigger_active: true,
            };
            let _ = epoch;
            let decision = controller.calculate(observation);
            let issued_at = crate::perception::types::MonotonicNanos(batch.stamp().captured_at.0);
            Ok(DeviceCommand {
                epoch,
                generation: batch.stamp().generation,
                issued_at,
                target_object_id: if decision.emit_allowed {
                    target.object_id
                } else {
                    0
                },
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
            })
        }
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

async fn fault(
    epoch: RuntimeEpoch,
    stats: SessionStats,
    error: AppError,
    event_tx: &mpsc::Sender<RuntimeSessionEvent>,
) -> SessionExit {
    let _ = event_tx
        .send(RuntimeSessionEvent::Faulted {
            epoch,
            error,
            stats,
        })
        .await;
    SessionExit { stats }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Detection, DetectionBatch, FrameStamp, RunIntent, RuntimePhase};

    #[tokio::test]
    async fn stale_epoch_batch_faults_without_entering_current_state() {
        let expected_epoch = RuntimeEpoch(1);
        let stale_epoch = RuntimeEpoch(99);
        let detection = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9).expect("detection");
        let stale_batch = DetectionBatch::fixture(
            FrameStamp::new(stale_epoch, 1, 1),
            640,
            640,
            vec![detection],
        )
        .expect("batch");

        let initial = Arc::new(OperationalSnapshot::initial_replay());
        let (snapshot_tx, _snapshot_rx) = watch::channel(initial);
        let publisher = EpochSnapshotPublisher::new(snapshot_tx);
        let mut running = OperationalSnapshot::initial_replay();
        running.phase = RuntimePhase::Running;
        running.run_intent = RunIntent::Running;
        running.epoch = Some(expected_epoch);
        running.running = true;
        publisher.claim_and_publish(expected_epoch, running);

        let (cancellation_tx, cancellation_rx) = watch::channel(false);
        let latest_command = Arc::new(Mutex::new(None));
        let (event_tx, mut event_rx) = mpsc::channel(1);
        let exit = run_session(
            expected_epoch,
            PacedReplay {
                source: ReplayPerceptionSource::new([stale_batch]),
                frame_interval: Duration::ZERO,
            },
            crate::RuntimeAlgorithm::Proportional(crate::ProportionalReplayControl::new(1.0)),
            SessionIo {
                device: Arc::new(RecordingPointerDevice::default()),
                publisher: publisher.clone(),
                event_tx,
                latest_command,
            },
            cancellation_rx,
        )
        .await;
        drop(cancellation_tx);

        assert_eq!(exit.stats.processed_batches, 0);
        let current = publisher.current();
        assert_eq!(current.phase, RuntimePhase::Running);
        assert_eq!(current.epoch, Some(expected_epoch));
        assert_eq!(current.fatal_error, None);

        let event = event_rx.recv().await.expect("terminal event");
        let RuntimeSessionEvent::Faulted {
            epoch,
            error,
            stats,
        } = event;
        assert_eq!(epoch, expected_epoch);
        assert_eq!(stats.processed_batches, 0);
        assert!(matches!(
            error,
            AppError::RuntimeEpochMismatch {
                expected: 1,
                actual: 99
            }
        ));
    }
}

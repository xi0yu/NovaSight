use std::{sync::Arc, time::Duration};

use novasight_core::{
    AppError, Detection, DetectionBatch, FrameStamp, RunIntent, RuntimeDependencies, RuntimeEpoch,
    RuntimeManager, RuntimePhase,
};
use tokio::sync::Barrier;

fn one_target_batch(epoch: RuntimeEpoch, generation: u64) -> DetectionBatch {
    DetectionBatch::fixture(
        FrameStamp::new(epoch, generation, generation + 1),
        640,
        640,
        vec![Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9).unwrap()],
    )
    .unwrap()
}

fn empty_batch(epoch: RuntimeEpoch, generation: u64) -> DetectionBatch {
    DetectionBatch::fixture(
        FrameStamp::new(epoch, generation, generation + 1),
        640,
        640,
        Vec::new(),
    )
    .unwrap()
}

async fn wait_for_snapshot(
    runtime: &novasight_core::RuntimeHandle,
    predicate: impl Fn(&novasight_core::OperationalSnapshot) -> bool,
) -> Arc<novasight_core::OperationalSnapshot> {
    let mut snapshots = runtime.subscribe();
    loop {
        let snapshot = snapshots.borrow_and_update().clone();
        if predicate(&snapshot) {
            return snapshot;
        }
        snapshots
            .changed()
            .await
            .expect("runtime manager keeps the snapshot watch open");
    }
}

#[tokio::test]
async fn initial_snapshot_has_exact_stopped_status() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    let snapshot = runtime.snapshot();

    assert_eq!(snapshot.phase, RuntimePhase::Stopped);
    assert_eq!(snapshot.run_intent, RunIntent::Stopped);
    assert_eq!(snapshot.epoch, None);
    assert!(!snapshot.running);
    assert_eq!(snapshot.source, "replay");
    assert_eq!(snapshot.last_generation, None);
    assert_eq!(snapshot.processed_batches, 0);
    assert_eq!(snapshot.device_receipts, 0);
    assert_eq!(snapshot.fatal_error, None);
}

#[tokio::test]
async fn repeated_start_keeps_one_session_and_stop_blocks_old_receipts() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    let first = runtime.start().await.unwrap();
    let repeated = runtime.start().await.unwrap();
    assert_eq!(first.epoch, repeated.epoch);
    assert_eq!(runtime.snapshot().phase, RuntimePhase::Running);

    let running = wait_for_snapshot(&runtime, |snapshot| snapshot.device_receipts == 1).await;
    assert_eq!(running.processed_batches, 1);

    runtime.stop().await.unwrap();
    let stopped = runtime.snapshot();
    assert_eq!(stopped.phase, RuntimePhase::Stopped);
    assert_eq!(stopped.run_intent, RunIntent::Stopped);
    assert!(!stopped.running);
    let receipts_at_stop = stopped.device_receipts;

    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(runtime.snapshot().device_receipts, receipts_at_stop);
}

#[tokio::test]
async fn restart_allocates_new_epoch() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    let first = runtime.start().await.unwrap();
    runtime.stop().await.unwrap();
    let second = runtime.start().await.unwrap();
    assert!(second.epoch > first.epoch);
}

#[tokio::test]
async fn non_empty_replay_rebinds_batches_to_each_restart_epoch() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay(
        [one_target_batch(RuntimeEpoch(1), 7)],
        Duration::ZERO,
    ));

    let first = runtime.start().await.unwrap();
    wait_for_snapshot(&runtime, |snapshot| snapshot.device_receipts == 1).await;
    runtime.stop().await.unwrap();

    let second = runtime.start().await.unwrap();
    assert!(second.epoch > first.epoch);
    let restarted = wait_for_snapshot(&runtime, |snapshot| {
        snapshot.epoch == second.epoch
            && (snapshot.device_receipts == 1 || snapshot.phase == RuntimePhase::Faulted)
    })
    .await;

    assert_eq!(restarted.phase, RuntimePhase::Running);
    assert_eq!(
        restarted.last_generation,
        Some(novasight_core::Generation(7))
    );
    assert_eq!(restarted.processed_batches, 1);
    assert_eq!(restarted.device_receipts, 1);
    assert_eq!(restarted.fatal_error, None);
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_starts_accept_one_epoch_and_reject_one_conflict() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    let barrier = Arc::new(Barrier::new(3));

    let first_runtime = runtime.clone();
    let first_barrier = barrier.clone();
    let first = tokio::spawn(async move {
        first_barrier.wait().await;
        first_runtime.start().await
    });

    let second_runtime = runtime.clone();
    let second_barrier = barrier.clone();
    let second = tokio::spawn(async move {
        second_barrier.wait().await;
        second_runtime.start().await
    });

    barrier.wait().await;
    let results = [first.await.unwrap(), second.await.unwrap()];
    let accepted = results
        .iter()
        .filter_map(|result| result.as_ref().ok())
        .count();
    let conflicts = results
        .iter()
        .filter(|result| {
            matches!(
                result,
                Err(AppError::RuntimeCommandConflict { command: "start" })
            )
        })
        .count();

    assert_eq!(accepted, 1);
    assert_eq!(conflicts, 1);
    assert_eq!(runtime.snapshot().phase, RuntimePhase::Running);
}

#[tokio::test]
async fn stop_is_idempotent_and_snapshot_watch_is_latest_only() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    let snapshots = runtime.subscribe();

    let started = runtime.start().await.unwrap();
    let first_stop = runtime.stop().await.unwrap();
    let repeated_stop = runtime.stop().await.unwrap();

    assert_eq!(first_stop, repeated_stop);
    assert_eq!(first_stop.epoch, started.epoch);
    assert_eq!(first_stop.phase, RuntimePhase::Stopped);
    assert_eq!(snapshots.borrow().phase, RuntimePhase::Stopped);
}

#[tokio::test(flavor = "current_thread")]
async fn command_receipts_bind_the_snapshot_before_queued_opposing_commands() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));

    let (started, stopped) = tokio::join!(biased; runtime.start(), runtime.stop());
    let started = started.unwrap();
    let stopped = stopped.unwrap();
    assert_eq!(started.snapshot.phase, RuntimePhase::Running);
    assert!(started.snapshot.running);
    assert_eq!(stopped.snapshot.phase, RuntimePhase::Stopped);
    assert!(!stopped.snapshot.running);
    assert_eq!(runtime.snapshot().phase, RuntimePhase::Stopped);

    runtime.start().await.unwrap();
    let (stopped, restarted) = tokio::join!(biased; runtime.stop(), runtime.start());
    let stopped = stopped.unwrap();
    let restarted = restarted.unwrap();
    assert_eq!(stopped.snapshot.phase, RuntimePhase::Stopped);
    assert!(!stopped.snapshot.running);
    assert_eq!(restarted.snapshot.phase, RuntimePhase::Running);
    assert!(restarted.snapshot.running);
    assert_eq!(runtime.snapshot().phase, RuntimePhase::Running);
}

#[tokio::test(flavor = "current_thread")]
async fn stop_cancels_before_exhausting_a_ready_replay_source() {
    const BATCH_COUNT: u64 = 256;
    let batches = (1..=BATCH_COUNT).map(|generation| empty_batch(RuntimeEpoch(1), generation));
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay(batches, Duration::ZERO));

    runtime.start().await.unwrap();
    wait_for_snapshot(&runtime, |snapshot| snapshot.processed_batches > 0).await;
    runtime.stop().await.unwrap();

    let stopped = runtime.snapshot();
    assert_eq!(stopped.phase, RuntimePhase::Stopped);
    assert!(stopped.processed_batches < BATCH_COUNT);
}

#[tokio::test(start_paused = true)]
async fn paced_replay_waits_between_batches_and_stop_cancels_the_wait() {
    let batches = [
        empty_batch(RuntimeEpoch(1), 1),
        empty_batch(RuntimeEpoch(1), 2),
        empty_batch(RuntimeEpoch(1), 3),
    ];
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay(
        batches,
        Duration::from_secs(60),
    ));

    runtime.start().await.unwrap();
    wait_for_snapshot(&runtime, |snapshot| snapshot.processed_batches == 1).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(runtime.snapshot().processed_batches, 1);

    tokio::time::advance(Duration::from_secs(60)).await;
    wait_for_snapshot(&runtime, |snapshot| snapshot.processed_batches == 2).await;

    tokio::time::timeout(Duration::from_millis(1), runtime.stop())
        .await
        .expect("stop must cancel the pacing wait")
        .unwrap();
    assert_eq!(runtime.snapshot().processed_batches, 2);
}

#[tokio::test]
async fn stopped_epoch_cannot_overwrite_a_restarted_snapshot() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    let first = runtime.start().await.unwrap();
    runtime.stop().await.unwrap();
    let second = runtime.start().await.unwrap();
    assert!(second.epoch > first.epoch);

    for _ in 0..16 {
        tokio::task::yield_now().await;
        let current = runtime.snapshot();
        assert_eq!(current.epoch, second.epoch);
        assert_ne!(current.phase, RuntimePhase::Stopped);
        assert_eq!(current.fatal_error, None);
    }
}

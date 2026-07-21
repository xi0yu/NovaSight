//! `RuntimeHandle` is `Clone` and the supervisor types are
//! `Send + Sync` so the API server, the CLI client bridge, and the
//! background WebSocket broadcaster can each own their own copy.

use std::sync::Arc;

use novasight_runtime::{RuntimeHandle, RuntimeSnapshot, RuntimeSupervisor};
use tokio::sync::watch;

fn assert_send<T: Send + Sync>() {}

#[test]
fn runtime_handle_is_send_and_sync() {
    assert_send::<RuntimeHandle>();
}

#[tokio::test(flavor = "current_thread")]
async fn runtime_handle_subscribes_to_public_snapshots() {
    let (supervisor, handle) = RuntimeSupervisor::spawn_recording();
    let mut snapshots: watch::Receiver<Arc<RuntimeSnapshot>> = handle.subscribe();

    let reply = handle.start().await.expect("start");
    snapshots.changed().await.expect("snapshot update");
    assert_eq!(snapshots.borrow().as_ref(), &reply);

    handle.shutdown_daemon().await.expect("shutdown");
    supervisor.join().await.expect("supervisor join");
}

#[tokio::test(flavor = "current_thread")]
async fn runtime_handle_clone_round_trip() {
    let (supervisor, handle) = RuntimeSupervisor::spawn_recording();
    let cloned = handle.clone();
    // The cloned handle and the original handle must be able to
    // issue independent commands without contention. The supervisor
    // is single-threaded `current_thread` so back-to-back commands
    // are deterministic.
    let snapshot_a = handle.start().await.expect("start from a");
    let snapshot_b = cloned.start().await.expect("start from b (idempotent)");
    assert_eq!(snapshot_a.pipeline.epoch, snapshot_b.pipeline.epoch);
    assert_eq!(snapshot_a.pipeline.state, snapshot_b.pipeline.state);

    handle.shutdown_daemon().await.expect("shutdown");
    supervisor.join().await.expect("supervisor join");
}

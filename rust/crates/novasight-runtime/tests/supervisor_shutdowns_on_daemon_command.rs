//! Daemon shutdown tests. The supervisor replies to `ShutdownDaemon`,
//! then its task exits; the handle that follows up after the task
//! has exited surfaces `SupervisorClosed`.

use novasight_runtime::{RuntimeError, RuntimeErrorKind, RuntimeSupervisor};

#[tokio::test(flavor = "current_thread")]
async fn shutdown_daemon_replies_and_exits_the_supervisor() {
    let (supervisor, handle) = RuntimeSupervisor::spawn_recording();

    handle
        .shutdown_daemon()
        .await
        .expect("shutdown_daemon should reply Ok(())");

    // The supervisor task exits after the reply; `join` should
    // complete without surfacing an error.
    let join_result =
        tokio::time::timeout(std::time::Duration::from_secs(2), supervisor.join()).await;
    let join_result = join_result.expect("supervisor join must complete within 2s");
    join_result.expect("supervisor join must succeed after shutdown");
}

#[tokio::test(flavor = "current_thread")]
async fn handle_drops_when_supervisor_exits() {
    let (supervisor, handle) = RuntimeSupervisor::spawn_recording();

    handle.shutdown_daemon().await.expect("shutdown");
    supervisor.join().await.expect("supervisor join");

    // After the supervisor task is gone, the command channel is
    // closed. The next command must surface SupervisorClosed.
    let err = handle
        .start()
        .await
        .expect_err("start must error after shutdown");
    assert_eq!(err.kind, RuntimeErrorKind::SupervisorClosed);
}

#[tokio::test(flavor = "current_thread")]
async fn handle_clone_still_works_across_tasks() {
    let (supervisor, handle) = RuntimeSupervisor::spawn_recording();
    let handle_b = handle.clone();
    let task = tokio::spawn(async move { handle_b.start().await });
    let result = task.await.expect("join task").expect("start");
    assert_eq!(
        result.pipeline.epoch,
        Some(novasight_runtime::RuntimeEpoch(1))
    );

    handle.shutdown_daemon().await.expect("shutdown");
    supervisor.join().await.expect("supervisor join");
}

#[test]
fn error_kind_codes_match_documented_strings() {
    assert_eq!(
        RuntimeErrorKind::SupervisorUnavailable.code(),
        "supervisor_unavailable"
    );
    assert_eq!(
        RuntimeErrorKind::SupervisorClosed.code(),
        "supervisor_closed"
    );
    assert_eq!(
        RuntimeErrorKind::SupervisorReplyLost.code(),
        "supervisor_reply_lost"
    );
    assert_eq!(
        RuntimeErrorKind::InvalidPipelineState.code(),
        "invalid_pipeline_state"
    );
    assert_eq!(
        RuntimeErrorKind::RuntimeEpochExhausted.code(),
        "runtime_epoch_exhausted"
    );
    assert_eq!(
        RuntimeErrorKind::PipelineUnavailable.code(),
        "pipeline_unavailable"
    );
    assert_eq!(
        RuntimeErrorKind::PipelineRejected.code(),
        "pipeline_rejected"
    );
    assert_eq!(RuntimeErrorKind::Other.code(), "runtime_error");
}

#[test]
fn error_summary_round_trips_through_json() {
    let error = RuntimeError::invalid_pipeline_state("start called while stopping");
    let summary = error.summary();
    assert_eq!(summary.code, "invalid_pipeline_state");
    assert_eq!(summary.message, "start called while stopping");
    let json = serde_json::to_value(&summary).expect("serialize");
    assert_eq!(json["code"], "invalid_pipeline_state");
    assert_eq!(json["message"], "start called while stopping");
}

//! RuntimeSupervisor command round trip. Exercises the typed
//! Start/Stop/Restart/EmergencyStop commands and the oneshot reply
//! channel.

use novasight_runtime::{
    DaemonSnapshot, DaemonState, PipelineState, RuntimeEpoch, RuntimeHandle, RuntimeSupervisor,
    SubsystemSnapshot, SubsystemState,
};

async fn shutdown(supervisor: RuntimeSupervisor, handle: &RuntimeHandle) {
    handle.shutdown_daemon().await.expect("shutdown");
    supervisor.join().await.expect("supervisor join");
}

#[tokio::test(flavor = "current_thread")]
async fn start_transitions_stopped_to_running() {
    let (supervisor, handle) = RuntimeSupervisor::spawn();

    let initial = handle.snapshot();
    assert_eq!(initial.pipeline.state, PipelineState::Stopped);
    assert_eq!(initial.daemon.state, DaemonState::Starting);

    let after = handle.start().await.expect("start");
    assert_eq!(after.pipeline.state, PipelineState::Running);
    assert_eq!(after.pipeline.epoch, Some(RuntimeEpoch(1)));
    assert!(after.pipeline.started_at_ms.is_some());
    assert_eq!(after.subsystems.capture.state, SubsystemState::Running);
    assert_eq!(after.subsystems.inference.state, SubsystemState::Running);
    assert_eq!(after.subsystems.control.state, SubsystemState::Running);
    assert_eq!(after.subsystems.device.state, SubsystemState::Ready);

    shutdown(supervisor, &handle).await;
}

#[tokio::test(flavor = "current_thread")]
async fn stop_transitions_running_to_stopped() {
    let (supervisor, handle) = RuntimeSupervisor::spawn();
    handle.start().await.expect("start");

    let after = handle.stop().await.expect("stop");
    assert_eq!(after.pipeline.state, PipelineState::Stopped);
    assert_eq!(after.pipeline.epoch, Some(RuntimeEpoch(1)));
    assert_eq!(after.subsystems.capture.state, SubsystemState::Stopped);
    assert_eq!(after.subsystems.inference.state, SubsystemState::Stopped);
    assert_eq!(after.subsystems.control.state, SubsystemState::Stopped);
    assert_eq!(after.subsystems.device.state, SubsystemState::Stopped);

    shutdown(supervisor, &handle).await;
}

#[tokio::test(flavor = "current_thread")]
async fn restart_allocates_a_new_runtime_epoch() {
    let (supervisor, handle) = RuntimeSupervisor::spawn();
    let first = handle.start().await.expect("start");
    assert_eq!(first.pipeline.epoch, Some(RuntimeEpoch(1)));

    let after = handle.restart().await.expect("restart");
    assert_eq!(after.pipeline.state, PipelineState::Running);
    assert_eq!(after.pipeline.epoch, Some(RuntimeEpoch(2)));

    shutdown(supervisor, &handle).await;
}

#[tokio::test(flavor = "current_thread")]
async fn emergency_stop_marks_device_unavailable_and_increments_control_restart() {
    let (supervisor, handle) = RuntimeSupervisor::spawn();
    handle.start().await.expect("start");

    let after = handle.emergency_stop().await.expect("emergency_stop");
    assert_eq!(after.pipeline.state, PipelineState::Stopped);
    assert_eq!(after.subsystems.device.state, SubsystemState::Unavailable);
    assert_eq!(after.subsystems.control.restart_count, 1);
    assert!(
        after
            .subsystems
            .control
            .last_error
            .as_ref()
            .map(|e| e.code == "emergency_stop")
            .unwrap_or(false),
        "control last_error must record the emergency stop code"
    );

    shutdown(supervisor, &handle).await;
}

#[tokio::test(flavor = "current_thread")]
async fn start_is_idempotent_when_already_running() {
    let (supervisor, handle) = RuntimeSupervisor::spawn();
    let first = handle.start().await.expect("start");
    let second = handle.start().await.expect("start (idempotent)");
    assert_eq!(first.pipeline.epoch, second.pipeline.epoch);
    assert_eq!(first.pipeline.state, second.pipeline.state);
    shutdown(supervisor, &handle).await;
}

#[tokio::test(flavor = "current_thread")]
async fn stop_is_idempotent_when_already_stopped() {
    let (supervisor, handle) = RuntimeSupervisor::spawn();
    let first = handle.stop().await.expect("stop from stopped");
    assert_eq!(first.pipeline.state, PipelineState::Stopped);
    let second = handle.stop().await.expect("stop again");
    assert_eq!(second.pipeline.state, PipelineState::Stopped);
    shutdown(supervisor, &handle).await;
}

#[test]
fn snapshot_default_daemon_and_pipeline_states() {
    let initial = novasight_runtime::RuntimeSnapshot::default();
    assert_eq!(initial.daemon.state, DaemonState::Starting);
    assert_eq!(initial.daemon.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(initial.pipeline.state, PipelineState::Stopped);
    assert_eq!(initial.pipeline.epoch, None);
    assert!(initial.pipeline.started_at_ms.is_none());
    assert!(initial.pipeline.last_error.is_none());
    assert_eq!(initial.subsystems.capture.state, SubsystemState::Stopped);
    assert_eq!(initial.subsystems.inference.state, SubsystemState::Stopped);
    assert_eq!(initial.subsystems.control.state, SubsystemState::Stopped);
    assert_eq!(initial.subsystems.device.state, SubsystemState::Stopped);
    assert!(initial.daemon.uptime_ms < 1_000);
}

#[test]
fn pipeline_state_exposes_the_designed_lifecycle_vocabulary() {
    let states = [
        PipelineState::Stopped,
        PipelineState::Starting,
        PipelineState::Running,
        PipelineState::Standby,
        PipelineState::Stopping,
        PipelineState::Faulted,
    ];

    assert_eq!(states.len(), 6);
}

#[test]
fn subsystem_snapshot_default_is_stopped_without_restart_history() {
    let subsystem = SubsystemSnapshot::default();
    assert_eq!(subsystem.state, SubsystemState::Stopped);
    assert!(subsystem.last_error.is_none());
    assert_eq!(subsystem.restart_count, 0);
}

#[test]
fn daemon_snapshot_serializes_to_stable_keys() {
    let snapshot = DaemonSnapshot {
        state: DaemonState::Ready,
        version: "0.1.0".to_string(),
        uptime_ms: 42,
    };
    let json = serde_json::to_value(&snapshot).expect("serialize");
    assert_eq!(json["state"], "Ready");
    assert_eq!(json["version"], "0.1.0");
    assert_eq!(json["uptime_ms"], 42);
}

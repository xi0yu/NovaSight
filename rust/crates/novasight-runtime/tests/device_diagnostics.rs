use std::sync::Arc;

use novasight_core::{Clock, MonotonicNanos, PointerDevice, RecordingPointerDevice, RuntimeEpoch};
use novasight_pipeline::PipelineConfig;
use novasight_runtime::{RuntimeDependencies, RuntimeErrorKind, RuntimeSupervisor};

#[derive(Debug)]
struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(42)
    }
}

fn recording_runtime() -> (
    RuntimeSupervisor,
    novasight_runtime::RuntimeHandle,
    Arc<RecordingPointerDevice>,
) {
    let recording = Arc::new(RecordingPointerDevice::default());
    let device: Arc<dyn PointerDevice> = recording.clone();
    let clock: Arc<dyn Clock> = Arc::new(FixedClock);
    let dependencies = RuntimeDependencies::new(clock, device, PipelineConfig::default());
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
    (supervisor, runtime, recording)
}

async fn shutdown(supervisor: RuntimeSupervisor, runtime: &novasight_runtime::RuntimeHandle) {
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test]
async fn stopped_runtime_serializes_diagnostic_moves_through_the_supervisor() {
    let (supervisor, runtime, recording) = recording_runtime();

    let first = runtime.diagnose_device_move(12, -4).await.unwrap();
    let second = runtime.diagnose_device_move(-1, 3).await.unwrap();

    assert_eq!(first.epoch, RuntimeEpoch(0));
    assert_eq!(first.generation.0, 1);
    assert_eq!(first.issued_at, MonotonicNanos(42));
    assert_eq!((first.delta_x_counts, first.delta_y_counts), (12, -4));
    assert_eq!(second.generation.0, 2);
    assert_eq!(recording.receipts(), vec![first, second]);
    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn diagnostic_move_cannot_compete_with_a_running_pipeline() {
    let (supervisor, runtime, recording) = recording_runtime();
    runtime.start().await.unwrap();

    let error = runtime.diagnose_device_move(1, 0).await.unwrap_err();

    assert_eq!(error.kind, RuntimeErrorKind::InvalidPipelineState);
    assert!(recording.receipts().is_empty());
    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn emergency_stop_latches_diagnostics_closed() {
    let (supervisor, runtime, recording) = recording_runtime();
    runtime.emergency_stop().await.unwrap();

    let error = runtime.diagnose_device_move(1, 0).await.unwrap_err();

    assert_eq!(error.kind, RuntimeErrorKind::DeviceUnavailable);
    assert!(recording.receipts().is_empty());
    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn invalid_diagnostic_counts_do_not_fault_or_touch_the_device() {
    let (supervisor, runtime, recording) = recording_runtime();

    for (dx, dy) in [(0, 0), (i32::from(i16::MAX) + 1, 0)] {
        let error = runtime.diagnose_device_move(dx, dy).await.unwrap_err();
        assert_eq!(error.kind, RuntimeErrorKind::InvalidDeviceCommand);
    }

    assert!(recording.receipts().is_empty());
    assert_eq!(
        runtime.snapshot().subsystems.device.state,
        novasight_runtime::SubsystemState::Stopped
    );
    shutdown(supervisor, &runtime).await;
}

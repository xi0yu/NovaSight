use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::SyncSender,
};
use std::time::{Duration, Instant};

use novasight_core::{
    Clock, Detection, DetectionBatch, FrameStamp, MonotonicNanos, PointerButtons, PointerDevice,
    RecordingPointerDevice, RuntimeEpoch,
};
use novasight_pipeline::{
    PerceptionAdapter, PerceptionError, PerceptionEvent, PerceptionMetrics, PerceptionSession,
    PipelineConfig, PipelineIngress, PipelineStatus, TriggerMode,
};
use novasight_runtime::{PipelineState, RuntimeDependencies, RuntimeErrorKind, RuntimeSupervisor};

#[derive(Debug)]
struct ManualClock(AtomicU64);

#[derive(Debug, Default)]
struct BlockingFailDevice {
    entered: AtomicBool,
    release: AtomicBool,
}

impl BlockingFailDevice {
    async fn wait_until_entered(&self) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !self.entered.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("device send entered");
    }

    fn release_with_failure(&self) {
        self.release.store(true, Ordering::Release);
    }
}

impl PointerDevice for BlockingFailDevice {
    fn mode(&self) -> novasight_core::PointerDeviceMode {
        novasight_core::PointerDeviceMode::Commissioned
    }

    fn send(
        &self,
        _command: novasight_core::DeviceCommand,
    ) -> Result<novasight_core::DeviceReceipt, novasight_core::AppError> {
        self.entered.store(true, Ordering::Release);
        while !self.release.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        Err(novasight_core::AppError::RuntimeTaskTerminated)
    }
}

#[derive(Debug, Default)]
struct RecoverableHealthDevice {
    online: AtomicBool,
    recording: RecordingPointerDevice,
}

impl PointerDevice for RecoverableHealthDevice {
    fn mode(&self) -> novasight_core::PointerDeviceMode {
        novasight_core::PointerDeviceMode::Commissioned
    }

    fn send(
        &self,
        command: novasight_core::DeviceCommand,
    ) -> Result<novasight_core::DeviceReceipt, novasight_core::AppError> {
        self.recording.send(command)
    }

    fn buttons(&self) -> Result<Option<PointerButtons>, novasight_core::AppError> {
        if self.online.load(Ordering::Acquire) {
            Ok(Some(PointerButtons {
                left: false,
                right: false,
            }))
        } else {
            Err(novasight_core::AppError::PointerDevice {
                code: "driver_timeout",
                message: "simulated kmNet timeout".to_owned(),
            })
        }
    }
}

impl ManualClock {
    fn new(now_ns: u64) -> Self {
        Self(AtomicU64::new(now_ns))
    }
}

impl Clock for ManualClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(self.0.load(Ordering::Acquire))
    }
}

#[derive(Debug, Default)]
struct RecordingPerceptionAdapter {
    starts: AtomicU64,
    shutdowns: Arc<AtomicU64>,
    epochs: Mutex<Vec<RuntimeEpoch>>,
    events: Mutex<Option<SyncSender<PerceptionEvent>>>,
    pipeline_open_during_shutdown: Arc<AtomicBool>,
}

impl PerceptionAdapter for RecordingPerceptionAdapter {
    fn start(
        &self,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        events: SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        ingress
            .submit(batch(epoch, 1))
            .map_err(|error| PerceptionError::new(error.to_string()))?;
        self.starts.fetch_add(1, Ordering::AcqRel);
        self.epochs.lock().unwrap().push(epoch);
        *self.events.lock().unwrap() = Some(events.clone());
        Ok(Box::new(RecordingPerceptionSession {
            epoch,
            ingress,
            events,
            shutdowns: Arc::clone(&self.shutdowns),
            pipeline_open_during_shutdown: Arc::clone(&self.pipeline_open_during_shutdown),
        }))
    }
}

#[derive(Debug)]
struct RecordingPerceptionSession {
    epoch: RuntimeEpoch,
    ingress: PipelineIngress,
    events: SyncSender<PerceptionEvent>,
    shutdowns: Arc<AtomicU64>,
    pipeline_open_during_shutdown: Arc<AtomicBool>,
}

impl PerceptionSession for RecordingPerceptionSession {
    fn shutdown(&mut self) -> Result<(), PerceptionError> {
        if self.ingress.submit(batch(self.epoch, 2)).is_ok() {
            self.pipeline_open_during_shutdown
                .store(true, Ordering::Release);
        }
        self.shutdowns.fetch_add(1, Ordering::AcqRel);
        let _ = self.events.try_send(PerceptionEvent::Stopped);
        Ok(())
    }
}

#[derive(Debug, Default)]
struct SilentPerceptionAdapter {
    retained_senders: Mutex<Vec<SyncSender<PerceptionEvent>>>,
}

impl PerceptionAdapter for SilentPerceptionAdapter {
    fn start(
        &self,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        events: SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        ingress
            .submit(batch(epoch, 1))
            .map_err(|error| PerceptionError::new(error.to_string()))?;
        self.retained_senders.lock().unwrap().push(events);
        Ok(Box::new(SilentPerceptionSession))
    }
}

#[derive(Debug)]
struct SilentPerceptionSession;

impl PerceptionSession for SilentPerceptionSession {
    fn metrics(&self) -> PerceptionMetrics {
        PerceptionMetrics {
            probed_buffers: 7,
            published_batches: 5,
            overwritten_snapshots: 2,
            ..PerceptionMetrics::default()
        }
    }

    fn shutdown(&mut self) -> Result<(), PerceptionError> {
        Ok(())
    }
}

#[derive(Debug, Default)]
struct PolledFailureAdapter {
    fail_health: Arc<AtomicBool>,
    shutdowns: Arc<AtomicU64>,
}

impl PerceptionAdapter for PolledFailureAdapter {
    fn start(
        &self,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        _events: SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        ingress
            .submit(batch(epoch, 1))
            .map_err(|error| PerceptionError::new(error.to_string()))?;
        Ok(Box::new(PolledFailureSession {
            fail_health: Arc::clone(&self.fail_health),
            shutdowns: Arc::clone(&self.shutdowns),
        }))
    }
}

#[derive(Debug)]
struct PolledFailureSession {
    fail_health: Arc<AtomicBool>,
    shutdowns: Arc<AtomicU64>,
}

impl PerceptionSession for PolledFailureSession {
    fn poll_health(&mut self) -> Result<(), PerceptionError> {
        if self.fail_health.load(Ordering::Acquire) {
            Err(PerceptionError::new("simulated owner thread exit"))
        } else {
            Ok(())
        }
    }

    fn shutdown(&mut self) -> Result<(), PerceptionError> {
        self.shutdowns.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[derive(Debug)]
struct FailingPreflightAdapter;

impl PerceptionAdapter for FailingPreflightAdapter {
    fn preflight(&self) -> Result<(), PerceptionError> {
        Err(PerceptionError::new("candidate model contract rejected"))
    }

    fn start(
        &self,
        _epoch: RuntimeEpoch,
        _ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        _events: SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        unreachable!("failed preflight must not start perception")
    }
}

fn batch(epoch: RuntimeEpoch, generation: u64) -> DetectionBatch {
    DetectionBatch::new(
        FrameStamp::new(epoch, generation, 1_000_000_000),
        640,
        640,
        vec![Detection::new(41, 0, 380.0, 330.0, 40.0, 40.0, 0.95).expect("valid detection")],
    )
    .expect("valid batch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supervisor_start_stop_owns_the_real_pipeline_lifecycle() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_output_enabled(true);
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);

    let started = handle.start().await.expect("start real pipeline");
    let epoch = started.pipeline.epoch.expect("runtime epoch");
    assert_eq!(started.pipeline.state, PipelineState::Running);
    handle
        .set_trigger_active(true)
        .await
        .expect("activate trigger");
    handle
        .submit_detection_batch(batch(epoch, 1))
        .expect("submit to owned pipeline");

    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(device.receipts().len(), 1);

    let stopped = handle.stop().await.expect("stop joins pipeline");
    assert_eq!(stopped.pipeline.state, PipelineState::Stopped);
    assert_eq!(stopped.pipeline_metrics.status, PipelineStatus::Stopped);
    assert_eq!(stopped.pipeline_metrics.received_batches, 1);
    assert_eq!(stopped.pipeline_metrics.device_receipts, 1);
    assert_eq!(
        stopped
            .pipeline_metrics
            .last_generation
            .map(|generation| generation.0),
        Some(1)
    );
    let error = handle
        .submit_detection_batch(batch(epoch, 2))
        .expect_err("stopped pipeline rejects ingress");
    assert_eq!(error.kind, RuntimeErrorKind::PipelineUnavailable);

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recoverable_device_outage_is_degraded_until_hardware_recovers() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(RecoverableHealthDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let dependencies = RuntimeDependencies::new(
        clock,
        pointer,
        PipelineConfig {
            trigger_poll_interval_ms: Some(1),
            trigger_mode: TriggerMode::Hardware,
            ..PipelineConfig::default()
        },
    );
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);
    handle.start().await.expect("start pipeline");

    let mut snapshots = handle.subscribe();
    let degraded = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if snapshots.borrow().subsystems.device.state
                == novasight_runtime::SubsystemState::Degraded
            {
                return snapshots.borrow().as_ref().clone();
            }
            snapshots.changed().await.expect("device health snapshot");
        }
    })
    .await
    .expect("device outage becomes degraded");
    assert!(!degraded.pipeline_metrics.device_connected);
    assert!(degraded.pipeline_metrics.device_error_count > 0);
    assert!(
        degraded
            .subsystems
            .device
            .last_error
            .as_ref()
            .is_some_and(|error| {
                error.code == "device_reconnecting"
                    && error.message.contains("simulated kmNet timeout")
            })
    );

    device.online.store(true, Ordering::Release);
    let recovered = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if snapshots.borrow().subsystems.device.state
                == novasight_runtime::SubsystemState::Ready
                && snapshots.borrow().pipeline_metrics.device_recovery_count > 0
            {
                return snapshots.borrow().as_ref().clone();
            }
            snapshots.changed().await.expect("device recovery snapshot");
        }
    })
    .await
    .expect("device recovery becomes ready");
    assert!(recovered.pipeline_metrics.device_connected);
    assert!(recovered.subsystems.device.last_error.is_none());

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn perception_preflight_rejects_a_candidate_without_starting_the_pipeline() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_perception(Arc::new(FailingPreflightAdapter));
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);

    let error = handle
        .preflight_perception()
        .await
        .expect_err("invalid model must fail preflight");
    assert_eq!(error.kind, RuntimeErrorKind::PipelineRejected);
    assert!(error.message.contains("candidate model contract rejected"));
    assert_eq!(handle.snapshot().pipeline.state, PipelineState::Stopped);

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supervisor_owns_the_epoch_scoped_perception_session() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let perception = Arc::new(RecordingPerceptionAdapter::default());
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_perception(perception.clone());
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);

    let started = handle.start().await.expect("start perception and pipeline");
    let epoch = started.pipeline.epoch.expect("runtime epoch");
    assert_eq!(perception.starts.load(Ordering::Acquire), 1);
    assert_eq!(*perception.epochs.lock().unwrap(), vec![epoch]);
    tokio::time::sleep(Duration::from_millis(20)).await;
    handle
        .set_trigger_active(true)
        .await
        .expect("activate output before shutdown safety check");

    let stopped = handle.stop().await.expect("stop perception and pipeline");
    assert_eq!(stopped.pipeline.state, PipelineState::Stopped);
    assert_eq!(perception.shutdowns.load(Ordering::Acquire), 1);
    assert!(
        perception
            .pipeline_open_during_shutdown
            .load(Ordering::Acquire),
        "perception must stop before its downstream ingress is closed"
    );
    assert!(
        device.receipts().is_empty(),
        "output gate must close before perception can publish during shutdown"
    );

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_does_not_depend_on_a_perception_adapter_sending_a_terminal_event() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let perception = Arc::new(SilentPerceptionAdapter::default());
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_perception(perception.clone());
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);
    handle
        .start()
        .await
        .expect("start silent perception adapter");
    let mut snapshots = handle.subscribe();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if snapshots.borrow().perception_metrics.probed_buffers == 7 {
                break;
            }
            snapshots.changed().await.expect("metrics snapshot update");
        }
    })
    .await
    .expect("perception metrics reach the public snapshot");
    assert_eq!(handle.snapshot().perception_metrics.published_batches, 5);
    assert_eq!(
        handle.snapshot().perception_metrics.overwritten_snapshots,
        2
    );

    let stopped = tokio::time::timeout(Duration::from_secs(1), handle.stop())
        .await
        .expect("stop must cancel the perception event bridge")
        .expect("stop succeeds");
    assert_eq!(stopped.pipeline.state, PipelineState::Stopped);
    assert_eq!(perception.retained_senders.lock().unwrap().len(), 1);

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn perception_fault_closes_the_epoch_and_faults_the_supervisor() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let perception = Arc::new(RecordingPerceptionAdapter::default());
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_perception(perception.clone());
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);
    handle.start().await.expect("start perception and pipeline");
    let events = perception
        .events
        .lock()
        .unwrap()
        .clone()
        .expect("perception event sender");

    events
        .send(PerceptionEvent::Faulted {
            message: "simulated DeepStream bus error".to_owned(),
        })
        .unwrap();
    let mut snapshots = handle.subscribe();
    let faulted = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if snapshots.borrow().pipeline.state == PipelineState::Faulted {
                return snapshots.borrow().as_ref().clone();
            }
            snapshots.changed().await.expect("supervisor snapshot");
        }
    })
    .await
    .expect("perception fault reaches supervisor");

    assert_eq!(perception.shutdowns.load(Ordering::Acquire), 1);
    assert!(
        faulted
            .pipeline
            .last_error
            .as_ref()
            .is_some_and(|error| error.message.contains("DeepStream bus error"))
    );

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unexpected_perception_stop_is_a_runtime_fault() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let perception = Arc::new(RecordingPerceptionAdapter::default());
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_perception(perception.clone());
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);
    handle.start().await.expect("start perception and pipeline");
    let events = perception
        .events
        .lock()
        .unwrap()
        .clone()
        .expect("perception event sender");

    events.send(PerceptionEvent::Stopped).unwrap();
    let faulted = wait_for_pipeline_fault(&handle).await;
    assert!(
        faulted
            .pipeline
            .last_error
            .as_ref()
            .is_some_and(|error| error.message.contains("stopped unexpectedly"))
    );

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supervisor_polls_perception_liveness_when_no_event_can_be_sent() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let perception = Arc::new(PolledFailureAdapter::default());
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_perception(perception.clone());
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);
    handle.start().await.expect("start perception and pipeline");

    perception.fail_health.store(true, Ordering::Release);
    let faulted = wait_for_pipeline_fault(&handle).await;
    assert!(
        faulted
            .pipeline
            .last_error
            .as_ref()
            .is_some_and(|error| error.message.contains("simulated owner thread exit"))
    );
    assert_eq!(perception.shutdowns.load(Ordering::Acquire), 1);

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

async fn wait_for_pipeline_fault(
    handle: &novasight_runtime::RuntimeHandle,
) -> novasight_runtime::RuntimeSnapshot {
    let mut snapshots = handle.subscribe();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if snapshots.borrow().pipeline.state == PipelineState::Faulted {
                return snapshots.borrow().as_ref().clone();
            }
            snapshots.changed().await.expect("supervisor snapshot");
        }
    })
    .await
    .expect("perception failure reaches supervisor")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_fault_is_projected_into_the_supervisor_snapshot() {
    #[derive(Debug)]
    struct FailingDevice;

    impl PointerDevice for FailingDevice {
        fn mode(&self) -> novasight_core::PointerDeviceMode {
            novasight_core::PointerDeviceMode::Commissioned
        }

        fn send(
            &self,
            _command: novasight_core::DeviceCommand,
        ) -> Result<novasight_core::DeviceReceipt, novasight_core::AppError> {
            Err(novasight_core::AppError::RuntimeTaskTerminated)
        }
    }

    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn PointerDevice> = Arc::new(FailingDevice);
    let dependencies = RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
        .with_output_enabled(true);
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);
    let started = handle.start().await.expect("start real pipeline");
    let epoch = started.pipeline.epoch.expect("runtime epoch");
    handle
        .set_trigger_active(true)
        .await
        .expect("activate trigger");
    handle
        .submit_detection_batch(batch(epoch, 1))
        .expect("submit batch");

    let mut snapshots = handle.subscribe();
    let faulted = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if snapshots.borrow().pipeline.state == PipelineState::Faulted {
                return snapshots.borrow().as_ref().clone();
            }
            snapshots.changed().await.expect("supervisor snapshot");
        }
    })
    .await
    .expect("fault reaches supervisor");
    assert!(faulted.pipeline.last_error.is_some());

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_start_failure_is_published_as_faulted_not_left_starting() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let dependencies = RuntimeDependencies::new(
        clock,
        pointer,
        PipelineConfig {
            output_interval_ms: 0,
            ..PipelineConfig::default()
        },
    );
    let (supervisor, handle) = RuntimeSupervisor::spawn(dependencies);

    let error = handle.start().await.expect_err("invalid pipeline config");
    assert_eq!(error.kind, RuntimeErrorKind::PipelineRejected);
    let snapshot = handle.snapshot();
    assert_eq!(snapshot.pipeline.state, PipelineState::Faulted);
    assert!(snapshot.pipeline.last_error.is_some());

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delayed_fault_cannot_overwrite_a_completed_stop() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(BlockingFailDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (supervisor, handle) = RuntimeSupervisor::spawn(
        RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
            .with_output_enabled(true),
    );
    let started = handle.start().await.expect("start pipeline");
    let epoch = started.pipeline.epoch.expect("runtime epoch");
    handle
        .set_trigger_active(true)
        .await
        .expect("activate trigger");
    handle
        .submit_detection_batch(batch(epoch, 1))
        .expect("submit batch");
    device.wait_until_entered().await;

    let stop_handle = handle.clone();
    let stop_task = tokio::spawn(async move { stop_handle.stop().await });
    let mut snapshots = handle.subscribe();
    tokio::time::timeout(Duration::from_secs(1), async {
        while snapshots.borrow().pipeline.state != PipelineState::Stopping {
            snapshots.changed().await.expect("stopping snapshot");
        }
    })
    .await
    .expect("stop reaches stopping state");
    let stale_trigger_handle = handle.clone();
    let stale_trigger =
        tokio::spawn(async move { stale_trigger_handle.set_trigger_active(false).await });
    device.release_with_failure();

    let stopped = stop_task.await.expect("stop task").expect("stop pipeline");
    assert_eq!(stopped.pipeline.state, PipelineState::Stopped);
    assert_eq!(
        stale_trigger
            .await
            .expect("stale trigger task")
            .expect_err("retired epoch rejects trigger mutation")
            .kind,
        RuntimeErrorKind::PipelineUnavailable
    );
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(handle.snapshot().pipeline.state, PipelineState::Stopped);

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delayed_fault_cannot_overwrite_a_completed_emergency_stop() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(BlockingFailDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (supervisor, handle) = RuntimeSupervisor::spawn(
        RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
            .with_output_enabled(true),
    );
    let started = handle.start().await.expect("start pipeline");
    let epoch = started.pipeline.epoch.expect("runtime epoch");
    handle
        .set_trigger_active(true)
        .await
        .expect("activate trigger");
    handle
        .submit_detection_batch(batch(epoch, 1))
        .expect("submit batch");
    device.wait_until_entered().await;

    let emergency_handle = handle.clone();
    let emergency_task = tokio::spawn(async move { emergency_handle.emergency_stop().await });
    let mut snapshots = handle.subscribe();
    tokio::time::timeout(Duration::from_secs(1), async {
        while snapshots.borrow().pipeline.state != PipelineState::Stopping {
            snapshots.changed().await.expect("stopping snapshot");
        }
    })
    .await
    .expect("emergency stop reaches stopping state");
    device.release_with_failure();

    let stopped = emergency_task
        .await
        .expect("emergency task")
        .expect("emergency stop");
    assert_eq!(stopped.pipeline.state, PipelineState::Stopped);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(handle.snapshot().pipeline.state, PipelineState::Stopped);

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "current_thread")]
async fn trigger_release_does_not_block_the_tokio_executor() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(BlockingFailDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (supervisor, handle) = RuntimeSupervisor::spawn(
        RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
            .with_output_enabled(true),
    );
    let started = handle.start().await.expect("start pipeline");
    let epoch = started.pipeline.epoch.expect("runtime epoch");
    handle
        .set_trigger_active(true)
        .await
        .expect("activate trigger");
    handle
        .submit_detection_batch(batch(epoch, 1))
        .expect("submit batch");
    device.wait_until_entered().await;

    let release_handle = handle.clone();
    let release_task = tokio::spawn(async move { release_handle.set_trigger_active(false).await });
    tokio::time::timeout(
        Duration::from_millis(100),
        tokio::time::sleep(Duration::from_millis(10)),
    )
    .await
    .expect("Tokio timer progresses while release waits for the device lane");
    assert!(!release_task.is_finished());

    device.release_with_failure();
    release_task
        .await
        .expect("release task")
        .expect("release trigger");
    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_trigger_requests_are_serialized_and_retired_with_the_epoch() {
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(BlockingFailDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (supervisor, handle) = RuntimeSupervisor::spawn(
        RuntimeDependencies::new(clock, pointer, PipelineConfig::default())
            .with_output_enabled(true),
    );
    let started = handle.start().await.expect("start pipeline");
    let epoch = started.pipeline.epoch.expect("runtime epoch");
    handle
        .set_trigger_active(true)
        .await
        .expect("activate trigger");
    handle
        .submit_detection_batch(batch(epoch, 1))
        .expect("submit batch");
    device.wait_until_entered().await;

    let requests = (0..40)
        .map(|_| {
            let request_handle = handle.clone();
            tokio::spawn(async move { request_handle.set_trigger_active(false).await })
        })
        .collect::<Vec<_>>();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(requests.iter().all(|request| !request.is_finished()));

    device.release_with_failure();
    let mut succeeded = 0;
    let mut retired = 0;
    for request in requests {
        match request.await.expect("trigger task") {
            Ok(()) => succeeded += 1,
            Err(error) if error.kind == RuntimeErrorKind::PipelineUnavailable => retired += 1,
            Err(error) => panic!("unexpected trigger error: {error}"),
        }
    }
    assert_eq!(succeeded, 1);
    assert_eq!(retired, 39);

    handle.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("supervisor joins");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_supervisor_requests_orderly_pipeline_shutdown() {
    let (supervisor, handle) = RuntimeSupervisor::spawn_recording();
    handle.start().await.expect("start pipeline");
    let mut snapshots = handle.subscribe();

    drop(supervisor);

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let snapshot = snapshots.borrow().clone();
            if snapshot.daemon.state == novasight_runtime::DaemonState::ShuttingDown
                && snapshot.pipeline.state == PipelineState::Stopped
            {
                break;
            }
            snapshots.changed().await.expect("shutdown snapshot");
        }
    })
    .await
    .expect("drop triggers orderly shutdown");
    assert_eq!(
        handle
            .submit_detection_batch(batch(RuntimeEpoch(1), 2))
            .expect_err("closed ingress")
            .kind,
        RuntimeErrorKind::PipelineUnavailable
    );
}

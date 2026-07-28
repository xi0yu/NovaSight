use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use novasight_core::{
    AppError, Clock, Detection, DetectionBatch, DeviceCommand, DeviceReceipt, FrameStamp,
    MonotonicNanos, PointerButtons, PointerDevice, RecordingPointerDevice, RuntimeEpoch,
};
use novasight_pipeline::{
    PipelineConfig, PipelineError, PipelineEvent, PipelineRuntime, PipelineStatus, TriggerMode,
};

#[derive(Debug)]
struct FixedClock(AtomicU64);

impl FixedClock {
    fn new(now_ns: u64) -> Self {
        Self(AtomicU64::new(now_ns))
    }
}

impl Clock for FixedClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(self.0.load(std::sync::atomic::Ordering::Acquire))
    }
}

#[derive(Debug, Default)]
struct BlockingClock {
    state: Mutex<(u64, bool, bool)>,
    changed: Condvar,
}

impl BlockingClock {
    fn wait_until_device_check(&self) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (state, timeout) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(1), |state| !state.1)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            !timeout.timed_out(),
            "device lane did not reach clock check"
        );
        assert!(state.1);
    }

    fn release_device_check(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.2 = true;
        self.changed.notify_all();
    }
}

impl Clock for BlockingClock {
    fn now(&self) -> MonotonicNanos {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.0 += 1;
        if state.0 == 3 {
            state.1 = true;
            self.changed.notify_all();
            while !state.2 {
                state = self
                    .changed
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
        MonotonicNanos(1_008_000_000)
    }
}

#[derive(Debug)]
struct FailingDevice;

impl PointerDevice for FailingDevice {
    fn send(&self, _command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        Err(AppError::RuntimeTaskTerminated)
    }
}

#[derive(Debug)]
struct PanickingDevice;

impl PointerDevice for PanickingDevice {
    fn send(&self, _command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        panic!("simulated vendor panic")
    }
}

#[derive(Debug, Default)]
struct LifecycleDevice {
    connects: AtomicU64,
    disconnects: AtomicU64,
    recording: RecordingPointerDevice,
}

impl PointerDevice for LifecycleDevice {
    fn connect(&self) -> Result<(), AppError> {
        self.connects.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        self.recording.send(command)
    }

    fn disconnect(&self) -> Result<(), AppError> {
        self.disconnects.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[derive(Debug)]
struct ConnectFailingDevice;

impl PointerDevice for ConnectFailingDevice {
    fn connect(&self) -> Result<(), AppError> {
        Err(AppError::PointerDevice {
            code: "offline",
            message: "test device is offline".to_owned(),
        })
    }

    fn send(&self, _command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        unreachable!("failed connection must prevent worker startup")
    }
}

#[derive(Debug, Default)]
struct HardwareTriggerDevice {
    active: AtomicBool,
    recording: RecordingPointerDevice,
}

#[derive(Debug, Default)]
struct RecoveringTriggerDevice {
    polls: AtomicU64,
    recording: RecordingPointerDevice,
}

#[derive(Debug, Default)]
struct RightButtonDevice {
    recording: RecordingPointerDevice,
}

impl PointerDevice for RightButtonDevice {
    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        self.recording.send(command)
    }

    fn buttons(&self) -> Result<Option<PointerButtons>, AppError> {
        Ok(Some(PointerButtons {
            left: false,
            right: true,
        }))
    }
}

impl PointerDevice for RecoveringTriggerDevice {
    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        self.recording.send(command)
    }

    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        if self.polls.fetch_add(1, Ordering::AcqRel) == 0 {
            return Err(AppError::PointerDevice {
                code: "driver_timeout",
                message: "transient test timeout".to_owned(),
            });
        }
        Ok(Some(true))
    }
}

impl PointerDevice for HardwareTriggerDevice {
    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        self.recording.send(command)
    }

    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        Ok(Some(self.active.load(Ordering::Acquire)))
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

#[test]
fn output_scheduler_rejects_cadence_outside_one_to_ten_ms() {
    for output_interval_ms in [0, 11] {
        let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
        let device: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
        let result = PipelineRuntime::start(
            PipelineConfig {
                output_interval_ms,
                ..PipelineConfig::default()
            },
            clock,
            device,
        );
        assert!(matches!(
            result,
            Err(PipelineError::InvalidOutputInterval { actual_ms })
                if actual_ms == output_interval_ms
        ));
    }
}

#[test]
fn pipeline_epoch_owns_device_connect_and_disconnect() {
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device = Arc::new(LifecycleDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();

    let (mut runtime, _ingress) =
        PipelineRuntime::start(PipelineConfig::default(), clock, pointer).expect("pipeline starts");
    assert_eq!(device.connects.load(Ordering::Acquire), 1);
    assert_eq!(device.disconnects.load(Ordering::Acquire), 0);

    runtime.shutdown().expect("pipeline stops");
    assert_eq!(device.disconnects.load(Ordering::Acquire), 1);
}

#[test]
fn device_connection_failure_prevents_worker_startup() {
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device: Arc<dyn PointerDevice> = Arc::new(ConnectFailingDevice);

    let error = PipelineRuntime::start(PipelineConfig::default(), clock, device)
        .expect_err("offline device rejects epoch start");

    assert!(matches!(error, PipelineError::DeviceConnect(message) if message.contains("offline")));
}

#[test]
fn hardware_trigger_poller_owns_production_output_gate() {
    let epoch = RuntimeEpoch(31);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device = Arc::new(HardwareTriggerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            trigger_poll_interval_ms: Some(1),
            trigger_mode: TriggerMode::Hardware,
            ..PipelineConfig::default()
        },
        clock,
        pointer,
    )
    .expect("pipeline starts");
    ingress.submit(batch(epoch, 1)).expect("batch accepted");
    thread::sleep(Duration::from_millis(10));
    assert!(device.recording.receipts().is_empty());

    device.active.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(1);
    while !ingress.trigger_active() && Instant::now() < deadline {
        thread::yield_now();
    }
    assert!(ingress.trigger_active());
    ingress.submit(batch(epoch, 2)).expect("new batch accepted");
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.recording.receipts().is_empty() && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(device.recording.receipts().len(), 1);
    assert_eq!(device.recording.receipts()[0].generation, 2);
    runtime.shutdown().expect("workers join");
}

#[test]
fn hardware_button_state_is_monitored_even_while_trigger_mode_is_always() {
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device: Arc<dyn PointerDevice> = Arc::new(RightButtonDevice::default());
    let (mut runtime, _ingress) = PipelineRuntime::start(
        PipelineConfig {
            trigger_poll_interval_ms: Some(1),
            trigger_mode: TriggerMode::Always,
            ..PipelineConfig::default()
        },
        clock,
        device,
    )
    .expect("pipeline starts");

    let deadline = Instant::now() + Duration::from_secs(1);
    while !runtime.metrics().button_right && Instant::now() < deadline {
        thread::yield_now();
    }
    let metrics = runtime.metrics();
    assert!(metrics.buttons_available);
    assert!(!metrics.button_left);
    assert!(metrics.button_right);

    runtime.shutdown().expect("workers join");
    let metrics = runtime.metrics();
    assert!(!metrics.buttons_available);
    assert!(!metrics.button_left);
    assert!(!metrics.button_right);
}

#[test]
fn external_stop_signal_blocks_device_output_even_after_gate_open() {
    let epoch = RuntimeEpoch(33);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let cancelled = Arc::new(AtomicUsize::new(0));
    let (mut runtime, ingress) = PipelineRuntime::start_suspended_with_cancel(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        clock,
        pointer,
        cancelled.clone(),
    )
    .expect("pipeline starts suspended");
    ingress.set_trigger_active(true);
    ingress.submit(batch(epoch, 1)).unwrap();
    thread::sleep(Duration::from_millis(20));
    assert!(
        device.receipts().is_empty(),
        "suspended output must stay closed"
    );

    runtime.open_output_gate();
    ingress.submit(batch(epoch, 2)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(device.receipts().len(), 1);

    cancelled.store(1, Ordering::Release);
    ingress.submit(batch(epoch, 3)).unwrap();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(device.receipts().len(), 1, "cancel must retire output");
    runtime.shutdown().unwrap();
}

#[test]
fn paused_output_keeps_control_hot_and_reopen_requires_a_new_generation() {
    let epoch = RuntimeEpoch(34);
    let clock = Arc::new(FixedClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .unwrap();
    let timed_batch = |generation, captured_at_ns| {
        DetectionBatch::new(
            FrameStamp::new(epoch, generation, captured_at_ns),
            640,
            640,
            vec![Detection::new(41, 0, 380.0, 330.0, 40.0, 40.0, 0.95).expect("valid detection")],
        )
        .expect("valid batch")
    };
    ingress.set_trigger_active(true);
    ingress.submit(timed_batch(1, 1_000_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(device.receipts().len(), 1);

    runtime.pause_output_gate();
    assert!(!runtime.metrics().output_gate_open);
    assert!(
        ingress.trigger_active(),
        "pausing output must not stop control calculation"
    );
    clock.0.store(1_012_000_000, Ordering::Release);
    ingress.submit(timed_batch(2, 1_004_000_000)).unwrap();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(device.receipts().len(), 1);

    runtime.open_output_gate();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(
        device.receipts().len(),
        1,
        "a command calculated while paused must not leak after resume"
    );
    // The first post-resume sample establishes the current measurement
    // cadence. Its capture still predates the point where the previous device
    // move could be visible, so feedback gating must reject it as well.
    clock.0.store(1_038_000_000, Ordering::Release);
    ingress.submit(timed_batch(3, 1_030_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().control_decisions < 3 && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(device.receipts().len(), 1);

    clock.0.store(1_047_000_000, Ordering::Release);
    ingress.submit(timed_batch(4, 1_039_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().len() == 1 && Instant::now() < deadline {
        thread::yield_now();
    }
    let receipts = device.receipts();
    assert_eq!(receipts.len(), 2);
    assert_eq!(receipts[1].generation, 4);
    runtime.shutdown().unwrap();
}

#[test]
fn transient_trigger_failure_recovers_without_restarting_the_pipeline() {
    let epoch = RuntimeEpoch(32);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device = Arc::new(RecoveringTriggerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            trigger_poll_interval_ms: Some(1),
            trigger_mode: TriggerMode::Hardware,
            ..PipelineConfig::default()
        },
        clock,
        pointer,
    )
    .expect("pipeline starts");
    let deadline = Instant::now() + Duration::from_secs(1);
    while !ingress.trigger_active() && Instant::now() < deadline {
        thread::yield_now();
    }
    assert!(ingress.trigger_active());
    assert_eq!(runtime.status(), PipelineStatus::Running);
    let metrics = runtime.metrics();
    assert!(metrics.device_connected);
    assert_eq!(metrics.device_error_count, 1);
    assert_eq!(metrics.device_recovery_count, 1);
    assert!(
        metrics
            .last_device_error
            .as_deref()
            .is_some_and(|message| message.contains("transient test timeout"))
    );
    ingress.submit(batch(epoch, 1)).expect("batch accepted");
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.recording.receipts().is_empty() && Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(device.recording.receipts().len(), 1);
    runtime.shutdown().expect("workers join");
}

#[test]
fn ingress_rejects_cross_epoch_and_non_monotonic_observations() {
    let epoch = RuntimeEpoch(3);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        clock,
        device,
    )
    .expect("pipeline starts");
    assert!(matches!(
        ingress.submit(batch(RuntimeEpoch(99), 1)),
        Err(PipelineError::EpochMismatch {
            expected: 3,
            actual: 99
        })
    ));
    ingress.submit(batch(epoch, 2)).expect("first generation");
    assert!(matches!(
        ingress.submit(batch(epoch, 2)),
        Err(PipelineError::NonMonotonicGeneration {
            previous: 2,
            actual: 2
        })
    ));

    runtime.shutdown().expect("workers join");
}

#[test]
fn realtime_ingress_uses_the_same_epoch_and_generation_contract() {
    let epoch = RuntimeEpoch(30);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        clock,
        device,
    )
    .expect("pipeline starts");

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        match ingress.try_submit(batch(epoch, 1)) {
            Ok(()) => break,
            Err(PipelineError::IngressBusy) if Instant::now() < deadline => {
                thread::yield_now();
            }
            result => panic!("realtime batch was not accepted: {result:?}"),
        }
    }
    let duplicate = loop {
        match ingress.try_submit(batch(epoch, 1)) {
            Err(PipelineError::IngressBusy) => thread::yield_now(),
            result => break result,
        }
    };
    assert!(matches!(
        duplicate,
        Err(PipelineError::NonMonotonicGeneration {
            previous: 1,
            actual: 1
        })
    ));
    assert!(matches!(
        ingress.try_submit(batch(RuntimeEpoch(31), 2)),
        Err(PipelineError::EpochMismatch {
            expected: 30,
            actual: 31
        })
    ));

    runtime.shutdown().expect("workers join");
}

#[test]
fn device_failure_faults_the_pipeline_and_closes_ingress() {
    let epoch = RuntimeEpoch(4);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device: Arc<dyn PointerDevice> = Arc::new(FailingDevice);
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        clock,
        device,
    )
    .expect("pipeline starts");
    let events = runtime.take_event_receiver().expect("event receiver");
    ingress.set_trigger_active(true);
    ingress.submit(batch(epoch, 1)).expect("batch accepted");

    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.status() != PipelineStatus::Faulted && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    assert_eq!(runtime.status(), PipelineStatus::Faulted);
    assert!(runtime.metrics().last_fault.is_some());
    assert!(matches!(
        events.recv_timeout(Duration::from_secs(1)),
        Ok(PipelineEvent::Faulted { .. })
    ));
    assert!(matches!(
        ingress.submit(batch(epoch, 2)),
        Err(PipelineError::NotRunning)
    ));
    runtime.shutdown().expect("faulted workers still join");
}

#[test]
fn device_worker_panic_immediately_faults_and_closes_output() {
    let epoch = RuntimeEpoch(5);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_008_000_000));
    let device: Arc<dyn PointerDevice> = Arc::new(PanickingDevice);
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        clock,
        device,
    )
    .expect("pipeline starts");
    ingress.set_trigger_active(true);
    ingress.submit(batch(epoch, 1)).expect("batch accepted");

    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.status() != PipelineStatus::Faulted && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    assert_eq!(runtime.status(), PipelineStatus::Faulted);
    assert_eq!(
        runtime.metrics().last_fault.as_deref(),
        Some("device worker panicked")
    );
    assert!(!ingress.trigger_active());
    assert!(matches!(
        ingress.submit(batch(epoch, 2)),
        Err(PipelineError::NotRunning)
    ));
    runtime
        .shutdown()
        .expect("panic is contained inside worker");
}

#[test]
fn device_lane_drops_a_command_superseded_by_a_newer_generation() {
    let epoch = RuntimeEpoch(7);
    let clock = Arc::new(BlockingClock::default());
    let pipeline_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        pipeline_clock,
        pointer,
    )
    .expect("pipeline starts");
    ingress.set_trigger_active(true);
    ingress.submit(batch(epoch, 1)).expect("generation one");
    clock.wait_until_device_check();

    ingress.submit(batch(epoch, 2)).expect("generation two");
    clock.release_device_check();

    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().superseded_commands == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    assert_eq!(runtime.metrics().superseded_commands, 1);
    let receipts = device.receipts();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].generation, 2);
    runtime.shutdown().expect("workers join");
}

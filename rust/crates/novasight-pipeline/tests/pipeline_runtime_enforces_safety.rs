use std::sync::{Arc, Condvar, Mutex, atomic::AtomicU64};
use std::thread;
use std::time::{Duration, Instant};

use novasight_core::{
    AppError, Clock, Detection, DetectionBatch, DeviceCommand, DeviceReceipt, FrameStamp,
    MonotonicNanos, PointerDevice, RecordingPointerDevice, RuntimeEpoch,
};
use novasight_pipeline::{
    PipelineConfig, PipelineError, PipelineEvent, PipelineRuntime, PipelineStatus,
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
struct AgingClock(AtomicU64);

impl Clock for AgingClock {
    fn now(&self) -> MonotonicNanos {
        match self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel) {
            0 => MonotonicNanos(1_008_000_000),
            _ => MonotonicNanos(1_100_000_000),
        }
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
        if state.0 == 2 {
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
fn device_lane_drops_a_command_that_expired_after_control() {
    let epoch = RuntimeEpoch(6);
    let clock: Arc<dyn Clock> = Arc::new(AgingClock::default());
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            max_command_age_ns: 55_000_000,
            ..PipelineConfig::default()
        },
        clock,
        pointer,
    )
    .expect("pipeline starts");
    ingress.set_trigger_active(true);
    ingress.submit(batch(epoch, 1)).expect("batch accepted");

    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().stale_commands == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    assert!(device.receipts().is_empty());
    assert_eq!(runtime.metrics().stale_commands, 1);
    runtime.shutdown().expect("workers join");
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

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use novasight_core::{
    Clock, Detection, DetectionBatch, FrameStamp, MonotonicNanos, RecordingPointerDevice,
    RuntimeEpoch,
};
use novasight_pipeline::{PipelineConfig, PipelineRuntime, PipelineStatus};

#[derive(Debug)]
struct ManualClock(AtomicU64);

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

#[test]
fn pipeline_runtime_drives_phase2_algorithms_and_device_on_owned_threads() {
    let epoch = RuntimeEpoch(7);
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let config = PipelineConfig {
        epoch,
        ..PipelineConfig::default()
    };
    let (mut runtime, ingress) =
        PipelineRuntime::start(config, clock, pointer).expect("pipeline starts");
    ingress.set_trigger_active(true);

    let detection = Detection::new(41, 0, 380.0, 330.0, 40.0, 40.0, 0.95).expect("valid detection");
    let batch = DetectionBatch::new(
        FrameStamp::new(epoch, 1, 1_000_000_000),
        640,
        640,
        vec![detection],
    )
    .expect("valid batch");
    ingress.submit(batch).expect("pipeline accepts batch");

    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    let receipts = device.receipts();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].epoch, epoch);
    assert_eq!(receipts[0].generation, 1);
    // Output carries the Rust-owned stable TrackId, not the frame-local
    // detector candidate identity (41).
    assert_eq!(receipts[0].target_object_id, 1);
    assert_ne!(receipts[0].delta_x_counts, 0);

    let metrics = runtime.shutdown().expect("workers join");
    assert_eq!(metrics.status, PipelineStatus::Stopped);
    assert_eq!(metrics.received_batches, 1);
    assert_eq!(metrics.device_receipts, 1);
    assert_eq!(metrics.live_workers, 0);
}

#[test]
fn pipeline_output_is_closed_until_trigger_is_explicitly_active() {
    let epoch = RuntimeEpoch(8);
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        clock,
        pointer,
    )
    .expect("pipeline starts");
    let detection = Detection::new(42, 0, 380.0, 330.0, 40.0, 40.0, 0.95).expect("valid detection");
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 1, 1_000_000_000),
                640,
                640,
                vec![detection],
            )
            .expect("valid batch"),
        )
        .expect("batch accepted");

    thread::sleep(Duration::from_millis(20));
    assert!(device.receipts().is_empty());
    assert_eq!(runtime.metrics().blocked_decisions, 1);
    runtime.shutdown().expect("workers join");
}

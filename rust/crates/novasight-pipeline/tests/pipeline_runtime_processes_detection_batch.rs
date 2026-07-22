use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use image::{Rgb, RgbImage, codecs::jpeg::JpegEncoder};
use novasight_core::control::humanized_motion::{HumanizedMotionPhase, HumanizedSpeedCurveSource};
use novasight_core::control::recoil::{RecoilConfig, RecoilState};
use novasight_core::{
    Clock, Detection, DetectionBatch, FrameStamp, MonotonicNanos, RecordingPointerDevice,
    RuntimeEpoch,
};
use novasight_pipeline::{
    CrosshairConfig, CrosshairError, CrosshairHub, MotionProfileHub, PipelineConfig,
    PipelineRuntime, PipelineStatus,
};

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
    let control = runtime.metrics().dual_phase;
    assert!(control.sample_available);
    assert_eq!(control.generation, 1);
    assert_eq!(control.observation_width, 640);
    assert_eq!(control.observed_error_x, 80.0);
    assert_eq!(control.dx, receipts[0].delta_x_counts);
    assert!(control.full_error_counts_x.is_finite());
    assert!(control.float_demand_x.is_finite());

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

#[test]
fn vision_verified_crosshair_changes_the_real_control_origin() {
    let epoch = RuntimeEpoch(12);
    let template_path = std::env::temp_dir().join(format!(
        "novasight-pipeline-crosshair-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let crosshair = CrosshairHub::new(
        CrosshairConfig {
            enabled: true,
            use_for_control: true,
            search_size: 96,
            sample_frames: 3,
            confirm_duration: Duration::ZERO,
            max_age: Duration::from_secs(1),
            max_offset_px: 20.0,
            min_similarity: 0.60,
            max_step_px: 100.0,
        },
        &template_path,
    );
    let observer = crosshair.begin_epoch(epoch, 640, 640).unwrap();
    publish_crosshair_samples(&crosshair, &observer, 0, 3);
    crosshair.learn().unwrap();
    publish_crosshair_samples(&crosshair, &observer, 20, 1);
    let deadline = Instant::now() + Duration::from_secs(1);
    while crosshair.snapshot().state != "confirmed" && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(crosshair.resolve(320.0, 320.0, 640, 640).x, 340.0);

    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            crosshair: Some(crosshair),
            ..PipelineConfig::default()
        },
        clock,
        pointer,
    )
    .unwrap();
    ingress.set_trigger_active(true);
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 1, 1_000_000_000),
                640,
                640,
                vec![Detection::new(1, 0, 310.0, 300.0, 40.0, 40.0, 0.95).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(device.receipts()[0].delta_x_counts < 0);
    runtime.shutdown().unwrap();
    drop(observer);
    let _ = std::fs::remove_file(template_path);
}

#[test]
fn hot_active_motion_profile_shapes_the_real_device_command_lane() {
    let epoch = RuntimeEpoch(13);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let motion = MotionProfileHub::default();
    motion.activate_builtin();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            motion_profiles: Some(motion),
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .unwrap();
    ingress.set_trigger_active(true);

    let batch = |generation, captured_at| {
        DetectionBatch::new(
            FrameStamp::new(epoch, generation, captured_at),
            640,
            640,
            vec![Detection::new(generation, 0, 380.0, 300.0, 40.0, 40.0, 0.95).unwrap()],
        )
        .unwrap()
    };
    ingress.submit(batch(1, 1_000_000_000)).unwrap();
    thread::sleep(Duration::from_millis(20));
    assert!(
        device.receipts().is_empty(),
        "the first observation establishes the trajectory instead of emitting the static jump"
    );

    clock.0.store(1_038_000_000, Ordering::Release);
    ingress.submit(batch(2, 1_030_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let receipts = device.receipts();
    assert_eq!(receipts.len(), 1);
    assert!(receipts[0].delta_x_counts > 0);
    assert!(receipts[0].delta_x_counts < 600);
    let telemetry = runtime.metrics().humanized_motion;
    assert!(telemetry.enabled);
    assert_eq!(telemetry.phase, Some(HumanizedMotionPhase::Acceleration));
    assert_eq!(
        telemetry.speed_curve_source,
        Some(HumanizedSpeedCurveSource::MinimumJerk)
    );
    assert!(telemetry.progress > 0.0);
    assert!(telemetry.planned_duration_ms >= 35.0);
    ingress.set_trigger_active(false);
    assert!(!runtime.metrics().humanized_motion.enabled);
    runtime.shutdown().unwrap();
}

#[test]
fn recoil_runs_on_the_output_tick_without_a_tracking_command() {
    let epoch = RuntimeEpoch(14);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            recoil: RecoilConfig {
                enabled: true,
                base_rate_counts_s: 600.0,
                max_rate_counts_s: 600.0,
                startup_ms: 0.0,
                ..RecoilConfig::default()
            },
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .unwrap();
    ingress.set_trigger_active(true);
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 1, 1_000_000_000),
                640,
                640,
                vec![Detection::new(1, 0, 300.0, 311.2, 40.0, 40.0, 0.95).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let receipts = device.receipts();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].delta_x_counts, 0);
    assert_eq!(receipts[0].delta_y_counts, 2);
    let telemetry = runtime.metrics().recoil;
    assert_eq!(telemetry.state, RecoilState::Hold);
    assert_eq!(telemetry.source_generation, Some(1));
    assert_eq!(telemetry.emitted_counts_y, 2);

    clock.0.store(1_100_000_000, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().recoil.state != RecoilState::Stale && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(runtime.metrics().recoil.state, RecoilState::Stale);
    assert_eq!(device.receipts().len(), 1);
    runtime.shutdown().unwrap();
}

fn publish_crosshair_samples(
    hub: &CrosshairHub,
    observer: &novasight_pipeline::CrosshairEpoch,
    offset: i32,
    count: u64,
) {
    let expected = hub.snapshot().processed_frames + count;
    for sequence in 1..=count {
        loop {
            match observer
                .publisher()
                .publish_jpeg(sequence, crosshair_jpeg(offset))
            {
                Ok(()) => break,
                Err(CrosshairError::Busy) => thread::sleep(Duration::from_millis(2)),
                Err(error) => panic!("crosshair sample rejected: {error}"),
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(1);
    while hub.snapshot().processed_frames < expected && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
}

fn crosshair_jpeg(offset: i32) -> Vec<u8> {
    let mut image = RgbImage::from_pixel(96, 96, Rgb([20, 20, 20]));
    let center = 48 + offset;
    for delta in -12..=12 {
        image.put_pixel((center + delta) as u32, 48, Rgb([255, 30, 60]));
        image.put_pixel(center as u32, (48 + delta) as u32, Rgb([255, 30, 60]));
    }
    let mut output = Vec::new();
    JpegEncoder::new_with_quality(&mut output, 95)
        .encode_image(&image)
        .unwrap();
    output
}

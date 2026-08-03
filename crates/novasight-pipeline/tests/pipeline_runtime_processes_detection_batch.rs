use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use image::{Rgb, RgbImage, codecs::jpeg::JpegEncoder};
use novasight_core::controller::recoil::{RecoilConfig, RecoilState};
use novasight_core::{
    Clock, Detection, DetectionBatch, FrameStamp, MonotonicNanos, RecordingPointerDevice,
    RuntimeEpoch,
};
use novasight_pipeline::{
    CrosshairConfig, CrosshairError, CrosshairHub, PipelineConfig, PipelineRuntime, PipelineStatus,
    TriggerMode,
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

#[derive(Debug)]
struct LaneClock;

impl Clock for LaneClock {
    fn now(&self) -> MonotonicNanos {
        match thread::current().name() {
            Some("novasight-targeting") => MonotonicNanos(1_008_000_000),
            Some("novasight-control" | "novasight-device") => MonotonicNanos(1_100_000_000),
            _ => MonotonicNanos(1_100_000_000),
        }
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

    let deadline = Instant::now() + Duration::from_secs(3);
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
    let live_metrics = runtime.metrics();
    let control = live_metrics.dual_phase;
    assert!(control.sample_available);
    assert_eq!(control.generation, 1);
    assert_eq!(control.observation_width, 640);
    assert_eq!(control.observed_error_x, 80.0);
    assert_eq!(control.dx, receipts[0].delta_x_counts);
    assert!(control.full_error_counts_x.is_finite());
    assert!(control.float_demand_x.is_finite());
    assert_eq!(live_metrics.target_selection.target_class_id, Some(0));
    assert_eq!(live_metrics.target_selection.target_track_id.unwrap().0, 1);
    assert_eq!(live_metrics.target_selection.target_box_x, Some(380.0));
    assert_eq!(live_metrics.target_selection.target_aim_y, Some(338.8));
    assert_eq!(
        live_metrics.detections.generation.map(|value| value.0),
        Some(1)
    );
    assert_eq!(live_metrics.detections.coordinate_width, 640);
    assert_eq!(live_metrics.detections.items.len(), 1);
    assert_eq!(live_metrics.detections.items[0].object_id, 41);
    assert_eq!(live_metrics.detections.truncated, 0);

    let metrics = runtime.shutdown().expect("workers join");
    assert_eq!(metrics.status, PipelineStatus::Stopped);
    assert_eq!(metrics.received_batches, 1);
    assert_eq!(metrics.device_receipts, 1);
    assert_eq!(metrics.live_workers, 0);
}

#[test]
fn control_waits_until_a_successful_device_move_can_be_visible_in_capture() {
    let epoch = RuntimeEpoch(72);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .expect("pipeline starts");
    ingress.set_trigger_active(true);

    let batch = |generation, captured_at_ns| {
        DetectionBatch::new(
            FrameStamp::new(epoch, generation, captured_at_ns),
            640,
            640,
            vec![Detection::new(generation, 0, 380.0, 300.0, 40.0, 40.0, 0.95).unwrap()],
        )
        .unwrap()
    };
    ingress.submit(batch(1, 1_000_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(device.receipts().len(), 1);

    // This frame was captured before the first accepted movement and cannot
    // possibly contain its visual result. It must not authorize a duplicate.
    clock.0.store(1_012_000_000, Ordering::Release);
    ingress.submit(batch(2, 1_004_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().control_decisions < 2 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    thread::sleep(Duration::from_millis(10));
    assert_eq!(
        device.receipts().len(),
        1,
        "a frame captured before the previous move became visible must not emit again"
    );

    // The first later sample also establishes the current measurement cadence.
    // Once a subsequent frame is newer than send + delay + cadence, control
    // must resume instead of turning the feedback gate into a permanent latch.
    clock.0.store(1_038_000_000, Ordering::Release);
    ingress.submit(batch(3, 1_030_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().control_decisions < 3 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    clock.0.store(1_047_000_000, Ordering::Release);
    ingress.submit(batch(4, 1_039_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().len() < 2 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        device.receipts().len(),
        2,
        "control must resume once a captured frame can contain the successful move"
    );

    runtime.shutdown().unwrap();
}

#[test]
fn smooth_detection_motion_keeps_one_identity_through_control_and_device_output() {
    let epoch = RuntimeEpoch(71);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .expect("pipeline starts");
    ingress.set_trigger_active(true);

    for (index, x) in [340.0_f32, 356.0, 372.0, 388.0, 404.0]
        .into_iter()
        .enumerate()
    {
        let generation = index as u64 + 1;
        let captured_at_ns = 1_000_000_000 + index as u64 * 8_333_333;
        clock.0.store(captured_at_ns + 8_000_000, Ordering::Release);
        ingress
            .submit(
                DetectionBatch::new(
                    FrameStamp::new(epoch, generation, captured_at_ns),
                    640,
                    640,
                    vec![Detection::new(generation, 0, x, 300.0, 40.0, 40.0, 0.95).unwrap()],
                )
                .unwrap(),
            )
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        while runtime.metrics().targeting_batches < generation && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(runtime.metrics().targeting_batches, generation);
    }

    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let receipts = device.receipts();
    assert!(!receipts.is_empty());
    assert!(
        receipts.iter().all(|receipt| receipt.target_object_id == 1),
        "feedback gating may intentionally skip visually unconfirmed frames, but normal motion must keep one downstream identity: {receipts:?}"
    );
    assert_eq!(
        runtime
            .metrics()
            .target_selection
            .target_track_id
            .unwrap()
            .0,
        1
    );

    runtime.shutdown().expect("workers join");
}

#[test]
fn detection_telemetry_is_bounded_without_dropping_the_runtime_batch() {
    let epoch = RuntimeEpoch(8);
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let pointer: Arc<dyn novasight_core::PointerDevice> =
        Arc::new(RecordingPointerDevice::default());
    let config = PipelineConfig {
        epoch,
        ..PipelineConfig::default()
    };
    let (mut runtime, ingress) =
        PipelineRuntime::start(config, clock, pointer).expect("pipeline starts");
    let detections = (0..65)
        .map(|object_id| {
            Detection::new(object_id, 0, 300.0, 300.0, 20.0, 20.0, 0.9).expect("valid detection")
        })
        .collect();
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 1, 1_000_000_000),
                640,
                640,
                detections,
            )
            .expect("valid bounded batch"),
        )
        .expect("pipeline accepts batch");

    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().detections.generation.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let metrics = runtime.metrics();
    assert_eq!(metrics.target_selection.candidates, 65);
    assert_eq!(metrics.detections.items.len(), 64);
    assert_eq!(metrics.detections.truncated, 1);
    assert_eq!(metrics.detections.items[63].object_id, 63);

    runtime.shutdown().expect("workers join");
}

#[test]
fn pipeline_hot_trigger_mode_blocks_until_trigger_is_explicitly_active() {
    let epoch = RuntimeEpoch(8);
    let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(1_008_000_000));
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            trigger_mode: TriggerMode::Always,
            ..PipelineConfig::default()
        },
        clock,
        pointer,
    )
    .expect("pipeline starts");
    assert_eq!(ingress.trigger_mode(), TriggerMode::Always);
    ingress.set_trigger_mode(TriggerMode::Hardware);
    assert_eq!(ingress.trigger_mode(), TriggerMode::Hardware);
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

    ingress.set_trigger_active(true);
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 2, 1_000_000_001),
                640,
                640,
                vec![
                    Detection::new(42, 0, 380.0, 330.0, 40.0, 40.0, 0.95).expect("valid detection"),
                ],
            )
            .expect("valid batch"),
        )
        .expect("pipeline accepts triggered batch");
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && std::time::Instant::now() < deadline {
        thread::yield_now();
    }
    assert_eq!(device.receipts().len(), 1);
    runtime.shutdown().expect("workers join");
}

#[test]
fn control_lane_rejects_an_observation_that_aged_while_waiting_for_control() {
    let epoch = RuntimeEpoch(9);
    let clock: Arc<dyn Clock> = Arc::new(LaneClock);
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
    ingress.set_trigger_active(true);
    let detection = Detection::new(43, 0, 380.0, 330.0, 40.0, 40.0, 0.95).expect("valid detection");
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

    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let metrics = runtime.metrics();
        if metrics.control_decisions == 1 && metrics.blocked_decisions == 1 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "control/device lanes did not process the observation: {metrics:?}"
        );
        thread::sleep(Duration::from_millis(1));
    }

    let metrics = runtime.metrics();
    assert!(device.receipts().is_empty());
    assert_eq!(metrics.blocked_decisions, 1);
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
fn recoil_is_added_to_an_existing_tracking_command_after_its_interval() {
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
                require_target: true,
                interval_ms: 8,
                y_counts: 2,
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
                vec![Detection::new(1, 0, 310.0, 311.2, 40.0, 40.0, 0.95).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(device.receipts()[0].delta_y_counts, 0);

    clock.0.store(1_078_000_000, Ordering::Release);
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 2, 1_070_000_000),
                640,
                640,
                vec![Detection::new(1, 0, 310.0, 311.2, 40.0, 40.0, 0.95).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().len() < 2 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let receipts = device.receipts();
    assert_eq!(receipts.len(), 2);
    assert_eq!(receipts[1].delta_y_counts, 2);
    let telemetry = runtime.metrics().recoil;
    assert_eq!(telemetry.state, RecoilState::Applied);
    assert_eq!(telemetry.source_generation, Some(2));
    assert_eq!(telemetry.emitted_counts_y, 2);

    clock.0.store(1_100_000_000, Ordering::Release);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(device.receipts().len(), 2);
    runtime.shutdown().unwrap();
}

#[test]
fn due_recoil_emits_at_the_predicted_aim_point_when_tracking_is_settled() {
    let epoch = RuntimeEpoch(15);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let mut control = novasight_core::controller::DualPhaseConfig {
        prediction_enabled: true,
        ..Default::default()
    };
    control.arrival_radius_counts = 3.0;
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            control,
            recoil: RecoilConfig {
                enabled: true,
                require_target: true,
                interval_ms: 8,
                y_counts: 2,
            },
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .unwrap();
    ingress.set_trigger_active(true);

    let centered_batch = |generation, captured_at_ns| {
        DetectionBatch::new(
            FrameStamp::new(epoch, generation, captured_at_ns),
            640,
            640,
            // Default aim ratio is 0.22, so this box aims exactly at 320,320.
            vec![Detection::new(generation, 0, 300.0, 311.2, 40.0, 40.0, 0.95).unwrap()],
        )
        .unwrap()
    };
    ingress.submit(centered_batch(1, 1_000_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().control_decisions < 1 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(device.receipts().is_empty());

    clock.0.store(1_078_000_000, Ordering::Release);
    ingress.submit(centered_batch(2, 1_070_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    let receipts = device.receipts();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].delta_x_counts, 0);
    assert_eq!(receipts[0].delta_y_counts, 2);
    assert_eq!(runtime.metrics().recoil.state, RecoilState::Applied);

    // A recoil-only send is feed-forward compensation. It must not mark the
    // visual controller's Y correction as pending, otherwise a short recoil
    // interval can starve Y tracking for the entire firing period.
    clock.0.store(1_079_000_000, Ordering::Release);
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 3, 1_071_000_000),
                640,
                640,
                vec![Detection::new(3, 0, 300.0, 321.2, 40.0, 40.0, 0.95).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().len() < 2 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let receipts = device.receipts();
    assert_eq!(receipts.len(), 2);
    assert!(receipts[1].delta_y_counts > 0);
    runtime.shutdown().unwrap();
}

#[test]
fn recoil_without_target_uses_fresh_observations_when_target_guard_is_disabled() {
    let epoch = RuntimeEpoch(16);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            recoil: RecoilConfig {
                enabled: true,
                require_target: false,
                interval_ms: 8,
                y_counts: 2,
            },
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .unwrap();
    ingress.set_trigger_active(true);

    let empty_batch = |generation, captured_at_ns| {
        DetectionBatch::new(
            FrameStamp::new(epoch, generation, captured_at_ns),
            640,
            640,
            Vec::new(),
        )
        .unwrap()
    };
    ingress.submit(empty_batch(1, 1_000_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().control_decisions < 1 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(device.receipts().is_empty());

    clock.0.store(1_078_000_000, Ordering::Release);
    ingress.submit(empty_batch(2, 1_070_000_000)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    let receipts = device.receipts();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].target_object_id, 0);
    assert_eq!(receipts[0].delta_x_counts, 0);
    assert_eq!(receipts[0].delta_y_counts, 2);
    runtime.shutdown().unwrap();
}

#[test]
fn target_guard_uses_the_existing_tracker_loss_grace_without_predicted_control() {
    let epoch = RuntimeEpoch(18);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start(
        PipelineConfig {
            epoch,
            recoil: RecoilConfig {
                enabled: true,
                require_target: true,
                interval_ms: 8,
                y_counts: 2,
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
    while runtime.metrics().control_decisions < 1 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }

    // One missed observation retains the locked identity for the configured
    // tracker grace. Tracking emits no predicted-only command, while recoil
    // remains authorized long enough to avoid amplifying the visual loss.
    clock.0.store(1_078_000_000, Ordering::Release);
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 2, 1_070_000_000),
                640,
                640,
                Vec::new(),
            )
            .unwrap(),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while device.receipts().is_empty() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let receipts = device.receipts();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].target_object_id, 0);
    assert_eq!(receipts[0].delta_y_counts, 2);
    assert_eq!(
        runtime.metrics().dual_phase.block_reason,
        novasight_core::controller::BlockReason::TargetInvalid
    );

    // Once the existing 120 ms tracker grace expires, the target requirement
    // closes again and no further recoil-only move is authorized.
    clock.0.store(1_258_000_000, Ordering::Release);
    ingress
        .submit(
            DetectionBatch::new(
                FrameStamp::new(epoch, 3, 1_250_000_000),
                640,
                640,
                Vec::new(),
            )
            .unwrap(),
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while runtime.metrics().control_decisions < 3 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    thread::sleep(Duration::from_millis(10));
    assert_eq!(device.receipts().len(), 1);
    runtime.shutdown().unwrap();
}

#[test]
fn prediction_and_recoil_compose_once_without_mutating_the_predicted_aim() {
    let epoch = RuntimeEpoch(17);
    let clock = Arc::new(ManualClock::new(1_008_000_000));
    let daemon_clock: Arc<dyn Clock> = clock.clone();
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn novasight_core::PointerDevice> = device.clone();
    let (mut runtime, ingress) = PipelineRuntime::start_suspended(
        PipelineConfig {
            epoch,
            control: novasight_core::controller::DualPhaseConfig {
                prediction_enabled: true,
                ..Default::default()
            },
            recoil: RecoilConfig {
                enabled: true,
                require_target: true,
                interval_ms: 8,
                y_counts: 2,
            },
            ..PipelineConfig::default()
        },
        daemon_clock,
        pointer,
    )
    .unwrap();
    ingress.set_trigger_active(true);

    let moving_batch = |generation: u64| {
        let elapsed_ms = generation.saturating_sub(1) * 20;
        let captured_at_ns = 1_000_000_000 + elapsed_ms * 1_000_000;
        let aim_y = 330.0 + generation.saturating_sub(1) as f32 * 2.0;
        DetectionBatch::new(
            FrameStamp::new(epoch, generation, captured_at_ns),
            640,
            640,
            vec![
                Detection::new(generation, 0, 300.0, aim_y - 40.0 * 0.22, 40.0, 40.0, 0.95)
                    .unwrap(),
            ],
        )
        .unwrap()
    };

    // Warm the real X/Y predictor while physical output is safely suspended.
    for generation in 1..=4_u64 {
        clock.0.store(
            1_008_000_000 + generation.saturating_sub(1) * 20_000_000,
            Ordering::Release,
        );
        ingress.submit(moving_batch(generation)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while runtime.metrics().control_decisions < generation && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
    }
    assert!(device.receipts().is_empty());
    runtime.open_output_gate();

    for generation in 5..=7_u64 {
        clock.0.store(
            1_008_000_000 + generation.saturating_sub(1) * 20_000_000,
            Ordering::Release,
        );
        ingress.submit(moving_batch(generation)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while runtime.metrics().control_decisions < generation && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while (device.receipts().last().map(|item| item.generation.0) != Some(7)
        || runtime.metrics().recoil.state != RecoilState::Applied)
        && Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(1));
    }

    let metrics = runtime.metrics();
    let control = metrics.dual_phase;
    let recoil = metrics.recoil;
    assert_eq!(control.generation, 7);
    assert!(control.predicted_offset_y > 0.0);
    assert_eq!(recoil.state, RecoilState::Applied);
    assert_eq!(recoil.emitted_counts_y, 2);
    let final_command = device
        .receipts()
        .last()
        .copied()
        .expect("generation 7 move");
    assert_eq!(final_command.generation.0, 7);
    assert_eq!(
        final_command.delta_y_counts,
        control.dy + recoil.emitted_counts_y
    );
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

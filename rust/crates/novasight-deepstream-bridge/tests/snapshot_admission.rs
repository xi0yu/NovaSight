use std::mem::size_of;

use novasight_core::{Generation, MonotonicNanos, RuntimeEpoch};
use novasight_deepstream_bridge::{
    ABI_VERSION, AdmissionContext, AdmissionError, Detection, FRAME_DETECTIONS_TRUNCATED,
    FRAME_HAS_UNTRACKED_OBJECT, FRAME_INFERENCE_DONE, FRAME_META_PTS_VALID, FrameSnapshot,
    PipelineClockSample, admit_capture_snapshot, admit_snapshot,
};

fn context() -> AdmissionContext {
    AdmissionContext {
        epoch: RuntimeEpoch(7),
        generation: Generation(100),
        clock: PipelineClockSample::new(1_000_000, MonotonicNanos(5_000_000)),
        source_id: 3,
        inference_component_id: 1,
    }
}

fn valid_snapshot() -> FrameSnapshot {
    let mut snapshot = FrameSnapshot {
        abi_version: ABI_VERSION,
        struct_size: size_of::<FrameSnapshot>() as u32,
        frame_num: 42,
        frame_pts_ns: 900_000,
        source_id: 3,
        source_width: 1920,
        source_height: 1080,
        pipeline_width: 640,
        pipeline_height: 640,
        detection_count: 1,
        flags: FRAME_INFERENCE_DONE | FRAME_META_PTS_VALID,
        ..FrameSnapshot::default()
    };
    snapshot.detections[0] = Detection {
        object_id: 9,
        left: 100.0,
        top: 120.0,
        width: 80.0,
        height: 160.0,
        confidence: 0.91,
        class_id: 2,
        component_id: 1,
        flags: 0,
    };
    snapshot
}

#[test]
fn converts_owned_metadata_into_a_validated_pipeline_batch() {
    let admitted = admit_snapshot(&valid_snapshot(), context()).expect("admit snapshot");
    let batch = admitted.batch();

    assert_eq!(batch.stamp().epoch, RuntimeEpoch(7));
    assert_eq!(batch.stamp().generation, Generation(100));
    assert_eq!(batch.stamp().captured_at, MonotonicNanos(4_900_000));
    assert_eq!(batch.coordinate_width(), 640);
    assert_eq!(batch.coordinate_height(), 640);
    assert_eq!(batch.detections().len(), 1);
    assert_eq!(batch.detections()[0].object_id(), 9);
    assert_eq!(batch.detections()[0].class_id(), 2);
}

#[test]
fn capture_admission_does_not_invent_deepstream_inference_completion() {
    let mut snapshot = valid_snapshot();
    snapshot.flags &= !FRAME_INFERENCE_DONE;
    snapshot.detection_count = 0;

    let capture = admit_capture_snapshot(&snapshot, context()).expect("admit capture identity");
    assert_eq!(capture.stamp().generation, Generation(100));
    assert_eq!(capture.stamp().captured_at, MonotonicNanos(4_900_000));
    assert_eq!(capture.dimensions(), (640, 640));
}

#[test]
fn capture_admission_rejects_unexpected_object_metadata() {
    let mut snapshot = valid_snapshot();
    snapshot.flags &= !FRAME_INFERENCE_DONE;
    assert_eq!(
        admit_capture_snapshot(&snapshot, context()),
        Err(AdmissionError::UnexpectedCaptureMetadata {
            detections: 1,
            truncated: 0,
            invalid: 0,
        })
    );
}

#[test]
fn runtime_generation_does_not_follow_a_reset_vendor_frame_counter() {
    let mut snapshot = valid_snapshot();
    snapshot.frame_num = 0;
    let mut admission = context();
    admission.generation = Generation(101);

    let admitted = admit_snapshot(&snapshot, admission).expect("admit reconnected source frame");

    assert_eq!(admitted.batch().stamp().generation, Generation(101));
}

#[test]
fn filters_documented_missing_confidence_without_inventing_a_score() {
    let mut snapshot = valid_snapshot();
    snapshot.detection_count = 2;
    snapshot.detections[1] = Detection {
        object_id: 10,
        confidence: -0.1,
        ..snapshot.detections[0]
    };

    let admitted = admit_snapshot(&snapshot, context()).expect("filter tracker-only object");

    assert_eq!(admitted.batch().detections().len(), 1);
    assert_eq!(admitted.filtered_without_detector_confidence(), 1);
}

#[test]
fn requires_clock_evidence_and_rejects_future_pts() {
    let mut missing = valid_snapshot();
    missing.flags &= !FRAME_META_PTS_VALID;
    assert_eq!(
        admit_snapshot(&missing, context()),
        Err(AdmissionError::FramePtsMissing)
    );

    let mut future = valid_snapshot();
    future.frame_pts_ns = 1_000_001;
    assert!(matches!(
        admit_snapshot(&future, context()),
        Err(AdmissionError::FramePtsInFuture { .. })
    ));
}

#[test]
fn assigns_frame_local_candidate_identity_to_primary_detector_objects() {
    let mut snapshot = valid_snapshot();
    snapshot.flags |= FRAME_HAS_UNTRACKED_OBJECT;
    snapshot.detection_count = 2;
    snapshot.detections[0].object_id = u64::MAX;
    snapshot.detections[1] = Detection {
        object_id: u64::MAX,
        left: 300.0,
        ..snapshot.detections[0]
    };

    let admitted = admit_snapshot(&snapshot, context()).expect("admit detector candidates");
    let identities: Vec<_> = admitted
        .batch()
        .detections()
        .iter()
        .map(|detection| detection.object_id())
        .collect();
    assert_eq!(identities, [0, 1]);
}

#[test]
fn rejects_incomplete_ambiguous_or_lossy_vendor_metadata() {
    for (snapshot, expected) in [
        (
            {
                let mut value = valid_snapshot();
                value.flags &= !FRAME_INFERENCE_DONE;
                value
            },
            AdmissionError::InferenceIncomplete,
        ),
        (
            {
                let mut value = valid_snapshot();
                value.flags |= FRAME_DETECTIONS_TRUNCATED;
                value.truncated_count = 2;
                value
            },
            AdmissionError::Truncated { omitted: 2 },
        ),
        (
            {
                let mut value = valid_snapshot();
                value.invalid_object_count = 1;
                value
            },
            AdmissionError::InvalidObjectMetadata { count: 1 },
        ),
    ] {
        assert_eq!(admit_snapshot(&snapshot, context()), Err(expected));
    }
}

#[test]
fn rejects_wrong_source_component_and_coordinate_space() {
    let mut wrong_source = valid_snapshot();
    wrong_source.source_id = 4;
    assert!(matches!(
        admit_snapshot(&wrong_source, context()),
        Err(AdmissionError::SourceMismatch { .. })
    ));

    let mut wrong_component = valid_snapshot();
    wrong_component.detections[0].component_id = 5;
    assert!(matches!(
        admit_snapshot(&wrong_component, context()),
        Err(AdmissionError::ComponentMismatch { .. })
    ));

    let mut outside = valid_snapshot();
    outside.detections[0].left = 620.0;
    assert!(matches!(
        admit_snapshot(&outside, context()),
        Err(AdmissionError::Domain(_))
    ));
}

#[test]
fn rejects_abi_count_class_and_frame_number_corruption() {
    let mut wrong_abi = valid_snapshot();
    wrong_abi.abi_version += 1;
    assert!(matches!(
        admit_snapshot(&wrong_abi, context()),
        Err(AdmissionError::AbiVersion { .. })
    ));

    let mut excessive = valid_snapshot();
    excessive.detection_count = u32::MAX;
    assert!(matches!(
        admit_snapshot(&excessive, context()),
        Err(AdmissionError::DetectionCount { .. })
    ));

    let mut negative_class = valid_snapshot();
    negative_class.detections[0].class_id = -1;
    assert!(matches!(
        admit_snapshot(&negative_class, context()),
        Err(AdmissionError::NegativeClassId { .. })
    ));

    let mut negative_frame = valid_snapshot();
    negative_frame.frame_num = -1;
    assert_eq!(
        admit_snapshot(&negative_frame, context()),
        Err(AdmissionError::NegativeFrameNumber(-1))
    );
}

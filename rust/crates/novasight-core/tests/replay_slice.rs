use std::marker::PhantomData;

use novasight_core::{
    AppError, ControlDecision, Detection, DetectionBatch, FrameStamp, NearestCenterTargeting,
    ProportionalReplayControl, RecordingPointerDevice, RuntimeEpoch,
};

#[tokio::test]
async fn replay_batch_becomes_recorded_device_receipt() {
    let batch = DetectionBatch::fixture(
        FrameStamp::new(RuntimeEpoch(1), 7, 1_000_000_000),
        640,
        640,
        vec![Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9).unwrap()],
    )
    .unwrap();
    let target = NearestCenterTargeting.select(&batch).expect("target");
    let decision = ProportionalReplayControl::new(1.0)
        .decide(&batch, &target, 1_010_000_000)
        .expect("decision");
    let device = RecordingPointerDevice::default();
    let receipt = device.send(decision.into_command()).expect("receipt");

    assert_eq!(receipt.epoch, RuntimeEpoch(1));
    assert_eq!(receipt.generation, 7);
    assert_eq!(device.receipts().len(), 1);
}

#[test]
fn validated_domain_types_cannot_bypass_admission_through_deserialization() {
    struct TraitProbe<T: ?Sized>(PhantomData<T>);
    trait AmbiguousIfDeserialize<A> {
        fn marker() {}
    }

    impl<T: ?Sized> AmbiguousIfDeserialize<()> for TraitProbe<T> {}
    impl<T: serde::de::DeserializeOwned> AmbiguousIfDeserialize<u8> for TraitProbe<T> {}

    let _ = <TraitProbe<Detection> as AmbiguousIfDeserialize<_>>::marker;
    let _ = <TraitProbe<DetectionBatch> as AmbiguousIfDeserialize<_>>::marker;
    let _ = <TraitProbe<ControlDecision> as AmbiguousIfDeserialize<_>>::marker;
}

#[test]
fn batch_validates_full_box_edges_without_lossy_coordinate_bounds() {
    let partially_outside = Detection::new(1, 0, 630.0, 100.0, 18.0, 20.0, 0.9).expect("detection");
    let error = DetectionBatch::new(
        FrameStamp::new(RuntimeEpoch(1), 1, 1),
        640,
        640,
        vec![partially_outside],
    )
    .expect_err("right edge exceeds declared width");
    assert!(matches!(
        error,
        AppError::CoordinateSpaceMismatch { object_id: 1, .. }
    ));

    let exact_large_edge =
        Detection::new(2, 0, 16_777_216.0, 0.0, 1.0, 1.0, 0.9).expect("detection");
    DetectionBatch::new(
        FrameStamp::new(RuntimeEpoch(1), 2, 2),
        16_777_217,
        1,
        vec![exact_large_edge],
    )
    .expect("f64 admission preserves the exact u32 boundary");
}

#[test]
fn batch_rejects_duplicate_object_ids_before_target_selection() {
    let left = Detection::new(7, 0, 290.0, 310.0, 20.0, 20.0, 0.9).expect("detection");
    let right = Detection::new(7, 0, 330.0, 310.0, 20.0, 20.0, 0.9).expect("detection");

    let error = DetectionBatch::new(
        FrameStamp::new(RuntimeEpoch(1), 3, 3),
        640,
        640,
        vec![right, left],
    )
    .expect_err("duplicate IDs make tie-breaking input-order dependent");

    assert_eq!(error, AppError::DuplicateObjectId { object_id: 7 });
}

#[test]
fn large_geometry_preserves_exact_tie_break_and_rounded_command() {
    let smaller_id = Detection::new(1, 0, 8_388_608.0, 0.0, 1.0, 2.0, 0.9).expect("detection");
    let larger_id = Detection::new(2, 0, 8_388_610.0, 0.0, 1.0, 2.0, 0.9).expect("detection");
    let batch = DetectionBatch::new(
        FrameStamp::new(RuntimeEpoch(1), 4, 4),
        16_777_219,
        2,
        vec![larger_id, smaller_id],
    )
    .expect("batch");

    let target = NearestCenterTargeting.select(&batch).expect("target");
    assert_eq!(target.object_id, 1);

    let command = ProportionalReplayControl::new(1.0)
        .decide(&batch, &target, 5)
        .expect("decision")
        .into_command();
    assert_eq!(command.delta_x_counts, -1);
    assert_eq!(command.delta_y_counts, 0);
}

use novasight_core::{
    Detection, DetectionBatch, FrameStamp, NearestCenterTargeting, ProportionalReplayControl,
    RecordingPointerDevice, RuntimeEpoch,
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
    let target = NearestCenterTargeting::default()
        .select(&batch)
        .expect("target");
    let decision = ProportionalReplayControl::new(1.0)
        .decide(&batch, &target, 1_010_000_000)
        .expect("decision");
    let device = RecordingPointerDevice::default();
    let receipt = device.send(decision.into_command()).expect("receipt");

    assert_eq!(receipt.epoch, RuntimeEpoch(1));
    assert_eq!(receipt.generation, 7);
    assert_eq!(device.receipts().len(), 1);
}

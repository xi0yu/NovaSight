use novasight_pipeline::LatestSlot;

#[test]
fn latest_slot_keeps_only_the_newest_unconsumed_value() {
    let slot = LatestSlot::new();

    slot.publish(11).expect("slot is open");
    slot.publish(22).expect("slot is open");

    assert_eq!(*slot.try_take().expect("latest value"), 22);
    assert!(slot.try_take().is_none());

    let metrics = slot.metrics();
    assert_eq!(metrics.published, 2);
    assert_eq!(metrics.overwritten, 1);
    assert_eq!(metrics.consumed, 1);
}

#[test]
fn closing_a_slot_rejects_new_values_and_wakes_consumers() {
    let slot = LatestSlot::<u64>::new();
    slot.close();

    assert!(slot.publish(1).is_err());
    assert!(slot.wait_take().is_none());
}

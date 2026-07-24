use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use novasight_pipeline::{LatestSlot, TryPublishError};

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

#[test]
fn periodic_consumer_wakes_immediately_for_a_published_value() {
    let slot = LatestSlot::new();
    let consumer = slot.clone();
    let (started_tx, started_rx) = sync_channel(0);
    let (result_tx, result_rx) = sync_channel(0);
    let worker = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let result = consumer.wait_take_or_timeout(Duration::from_secs(1));
        result_tx
            .send(result.map(|value| value.map(|value| *value)))
            .unwrap();
    });

    started_rx.recv().unwrap();
    slot.publish(42).unwrap();
    assert_eq!(
        result_rx
            .recv_timeout(Duration::from_millis(250))
            .expect("publish must wake a periodic consumer before its idle interval"),
        Ok(Some(42))
    );
    worker.join().unwrap();
}

#[test]
fn replacing_a_value_does_not_hold_the_slot_lock_while_dropping_it() {
    #[derive(Debug)]
    struct BlockingDrop(Arc<(Mutex<(bool, bool)>, Condvar)>);

    impl Drop for BlockingDrop {
        fn drop(&mut self) {
            let (state, changed) = &*self.0;
            let mut state = state.lock().unwrap();
            state.0 = true;
            changed.notify_all();
            while !state.1 {
                state = changed.wait(state).unwrap();
            }
        }
    }

    let drop_state = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let slot = LatestSlot::new();
    slot.publish(BlockingDrop(Arc::clone(&drop_state))).unwrap();

    let publishing_slot = slot.clone();
    let publisher = thread::spawn(move || {
        publishing_slot.publish(BlockingDrop(Arc::new((
            Mutex::new((true, true)),
            Condvar::new(),
        ))))
    });

    let (state, changed) = &*drop_state;
    let mut state = state.lock().unwrap();
    while !state.0 {
        state = changed.wait(state).unwrap();
    }
    slot.try_publish(BlockingDrop(Arc::new((
        Mutex::new((true, true)),
        Condvar::new(),
    ))))
    .expect("slot mutex is released before the replaced value is dropped");
    state.1 = true;
    changed.notify_all();
    drop(state);

    publisher.join().unwrap().unwrap();
    slot.close();
}

#[test]
fn realtime_publish_rejects_a_closed_slot_without_waiting() {
    let slot = LatestSlot::new();
    slot.close();

    assert_eq!(slot.try_publish(1), Err(TryPublishError::Closed));
}

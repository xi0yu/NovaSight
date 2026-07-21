use novasight_core::Clock;
use novasight_platform_jetson::SystemMonotonicClock;

#[test]
fn system_clock_never_moves_backwards_in_sample_window() {
    let clock = SystemMonotonicClock::default();
    let first = clock.now();
    let second = clock.now();

    assert!(second >= first);
}

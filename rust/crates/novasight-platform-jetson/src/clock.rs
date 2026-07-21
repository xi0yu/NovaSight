use std::time::Instant;

use novasight_core::{Clock, MonotonicNanos};

/// Process-relative monotonic clock whose zero is captured when the value is created.
///
/// Samples are elapsed nanoseconds, never wall-clock timestamps. Elapsed values larger
/// than `u64::MAX` nanoseconds saturate at `u64::MAX`.
#[derive(Debug)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for SystemMonotonicClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(
            self.origin
                .elapsed()
                .as_nanos()
                .min(u64::MAX as u128) as u64,
        )
    }
}

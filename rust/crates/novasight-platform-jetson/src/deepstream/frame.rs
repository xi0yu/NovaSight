use std::time::Duration;

use gstreamer as gst;
use novasight_core::{Generation, MonotonicNanos, RuntimeEpoch};
use novasight_pipeline::{LatestSlot, SlotMetrics};
use thiserror::Error;

/// Owned reference to one real DeepStream/GStreamer buffer.
///
/// Holding this value keeps the underlying NVMM-backed `GstBuffer` alive
/// after the pad probe and even while the producing pipeline is stopped.
#[derive(Debug)]
pub struct FrameLease {
    buffer: gst::Buffer,
    epoch: RuntimeEpoch,
    generation: Generation,
    captured_at: MonotonicNanos,
    width: u32,
    height: u32,
}

impl FrameLease {
    pub(crate) fn new(
        buffer: gst::Buffer,
        epoch: RuntimeEpoch,
        generation: Generation,
        captured_at: MonotonicNanos,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            buffer,
            epoch,
            generation,
            captured_at,
            width,
            height,
        }
    }

    pub fn buffer(&self) -> &gst::BufferRef {
        self.buffer.as_ref()
    }

    pub(crate) fn buffer_ptr(&self) -> u64 {
        u64::try_from(self.buffer.as_ptr().addr())
            .expect("supported Jetson pointer width does not exceed 64 bits")
    }

    pub const fn epoch(&self) -> RuntimeEpoch {
        self.epoch
    }

    pub const fn generation(&self) -> Generation {
        self.generation
    }

    pub const fn captured_at(&self) -> MonotonicNanos {
        self.captured_at
    }

    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Runtime-shared, capacity-one exchange for the newest leased NVMM frame.
#[derive(Clone, Debug, Default)]
pub struct LatestFrameExchange {
    slot: LatestSlot<FrameLease>,
}

impl LatestFrameExchange {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn publish(&self, lease: FrameLease) -> u64 {
        self.slot
            .publish(lease)
            .expect("LatestFrameExchange does not expose a close operation")
    }

    pub fn take_latest(
        &self,
        expected_epoch: RuntimeEpoch,
        now: MonotonicNanos,
        maximum_age: Duration,
    ) -> Result<std::sync::Arc<FrameLease>, FrameLeaseError> {
        let lease = self.slot.try_take().ok_or(FrameLeaseError::Empty)?;
        if lease.epoch != expected_epoch {
            return Err(FrameLeaseError::EpochMismatch {
                expected: expected_epoch,
                actual: lease.epoch,
            });
        }
        let age_ns =
            now.0
                .checked_sub(lease.captured_at.0)
                .ok_or(FrameLeaseError::ClockRegression {
                    captured_at_ns: lease.captured_at.0,
                    observed_now_ns: now.0,
                })?;
        if age_ns > maximum_age.as_nanos().min(u64::MAX as u128) as u64 {
            return Err(FrameLeaseError::Stale {
                age_ns,
                maximum_age_ns: maximum_age.as_nanos().min(u64::MAX as u128) as u64,
            });
        }
        Ok(lease)
    }

    pub fn clear(&self) {
        drop(self.slot.try_take());
    }

    pub fn metrics(&self) -> SlotMetrics {
        self.slot.metrics()
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum FrameLeaseError {
    #[error("latest frame exchange has no frame")]
    Empty,
    #[error("latest frame epoch mismatch: expected {expected:?}, got {actual:?}")]
    EpochMismatch {
        expected: RuntimeEpoch,
        actual: RuntimeEpoch,
    },
    #[error(
        "latest frame clock regressed: captured at {captured_at_ns}ns but observed now is {observed_now_ns}ns"
    )]
    ClockRegression {
        captured_at_ns: u64,
        observed_now_ns: u64,
    },
    #[error("latest frame is stale: age {age_ns}ns exceeds {maximum_age_ns}ns")]
    Stale { age_ns: u64, maximum_age_ns: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    fn lease(epoch: u64, generation: u64, captured_at: u64) -> FrameLease {
        FrameLease::new(
            gst::Buffer::new(),
            RuntimeEpoch(epoch),
            Generation(generation),
            MonotonicNanos(captured_at),
            1920,
            1080,
        )
    }

    #[test]
    fn lease_and_exchange_can_cross_runtime_threads() {
        assert_send_sync::<FrameLease>();
        assert_send_sync::<LatestFrameExchange>();
    }

    #[test]
    fn exchange_returns_the_newest_owned_buffer() {
        let exchange = LatestFrameExchange::new();
        exchange.publish(lease(7, 1, 100));
        exchange.publish(lease(7, 2, 200));

        let frame = exchange
            .take_latest(
                RuntimeEpoch(7),
                MonotonicNanos(250),
                Duration::from_nanos(50),
            )
            .unwrap();

        assert_eq!(frame.epoch(), RuntimeEpoch(7));
        assert_eq!(frame.generation(), Generation(2));
        assert_eq!(frame.captured_at(), MonotonicNanos(200));
        assert_eq!(frame.dimensions(), (1920, 1080));
        assert_eq!(
            exchange.metrics(),
            SlotMetrics {
                published: 2,
                overwritten: 1,
                consumed: 1,
            }
        );
    }

    #[test]
    fn epoch_mismatch_is_rejected_and_consumed() {
        let exchange = LatestFrameExchange::new();
        exchange.publish(lease(4, 1, 100));

        assert_eq!(
            exchange
                .take_latest(RuntimeEpoch(5), MonotonicNanos(100), Duration::from_secs(1),)
                .unwrap_err(),
            FrameLeaseError::EpochMismatch {
                expected: RuntimeEpoch(5),
                actual: RuntimeEpoch(4),
            }
        );
        assert_eq!(
            exchange
                .take_latest(RuntimeEpoch(5), MonotonicNanos(100), Duration::from_secs(1),)
                .unwrap_err(),
            FrameLeaseError::Empty
        );
    }

    #[test]
    fn stale_frame_is_rejected_at_the_consumer_boundary() {
        let exchange = LatestFrameExchange::new();
        exchange.publish(lease(3, 9, 100));

        assert_eq!(
            exchange
                .take_latest(
                    RuntimeEpoch(3),
                    MonotonicNanos(201),
                    Duration::from_nanos(100),
                )
                .unwrap_err(),
            FrameLeaseError::Stale {
                age_ns: 101,
                maximum_age_ns: 100,
            }
        );
    }

    #[test]
    fn future_frame_timestamp_is_rejected_as_clock_regression() {
        let exchange = LatestFrameExchange::new();
        exchange.publish(lease(3, 9, 201));

        assert_eq!(
            exchange
                .take_latest(RuntimeEpoch(3), MonotonicNanos(200), Duration::from_secs(1),)
                .unwrap_err(),
            FrameLeaseError::ClockRegression {
                captured_at_ns: 201,
                observed_now_ns: 200,
            }
        );
    }

    #[test]
    fn clear_releases_the_previous_epoch_before_restart() {
        let exchange = LatestFrameExchange::new();
        exchange.publish(lease(1, 1, 100));

        exchange.clear();

        assert_eq!(
            exchange
                .take_latest(RuntimeEpoch(2), MonotonicNanos(100), Duration::from_secs(1),)
                .unwrap_err(),
            FrameLeaseError::Empty
        );
    }
}

use std::sync::{Arc, Condvar, Mutex, TryLockError};
use std::time::Duration;

use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SlotMetrics {
    pub published: u64,
    pub overwritten: u64,
    pub consumed: u64,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("latest slot is closed")]
pub struct SlotClosed;

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum TryPublishError {
    #[error("latest slot is busy")]
    Busy,
    #[error("latest slot is closed")]
    Closed,
}

#[derive(Debug)]
struct SlotState<T> {
    value: Option<Arc<T>>,
    last_monotonic_key: Option<u64>,
    closed: bool,
    metrics: SlotMetrics,
}

impl<T> Default for SlotState<T> {
    fn default() -> Self {
        Self {
            value: None,
            last_monotonic_key: None,
            closed: false,
            metrics: SlotMetrics::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MonotonicPublishError {
    Closed,
    NonMonotonic { previous: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TryMonotonicPublishError {
    Busy,
    Closed,
    NonMonotonic { previous: u64 },
}

#[derive(Debug)]
struct SlotInner<T> {
    state: Mutex<SlotState<T>>,
    changed: Condvar,
}

/// Single-consumer, capacity-one handoff.
///
/// Producers replace an unread value instead of waiting. The consumer
/// blocks without polling and receives ownership through an `Arc`, so
/// publishing does not require `T: Clone`.
#[derive(Debug)]
pub struct LatestSlot<T> {
    inner: Arc<SlotInner<T>>,
}

impl<T> Clone for LatestSlot<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LatestSlot<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SlotInner {
                state: Mutex::new(SlotState::default()),
                changed: Condvar::new(),
            }),
        }
    }

    pub fn publish(&self, value: T) -> Result<u64, SlotClosed> {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.closed {
            return Err(SlotClosed);
        }
        if state.value.is_some() {
            state.metrics.overwritten = state.metrics.overwritten.saturating_add(1);
        }
        state.metrics.published = state.metrics.published.saturating_add(1);
        let published = state.metrics.published;
        let replaced = state.value.replace(Arc::new(value));
        self.inner.changed.notify_one();
        drop(state);
        // `T::drop` is arbitrary user code. Never execute it while holding the
        // slot mutex or a realtime try-publisher could be delayed by cleanup
        // from an older value.
        drop(replaced);
        Ok(published)
    }

    /// Publish without ever waiting for the consumer or another producer.
    /// Realtime callbacks should drop the incoming value on [`TryPublishError::Busy`].
    pub fn try_publish(&self, value: T) -> Result<u64, TryPublishError> {
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(TryPublishError::Busy),
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        if state.closed {
            return Err(TryPublishError::Closed);
        }
        if state.value.is_some() {
            state.metrics.overwritten = state.metrics.overwritten.saturating_add(1);
        }
        state.metrics.published = state.metrics.published.saturating_add(1);
        let published = state.metrics.published;
        let replaced = state.value.replace(Arc::new(value));
        self.inner.changed.notify_one();
        drop(state);
        drop(replaced);
        Ok(published)
    }

    /// Publish a value only when its key is newer than every value previously
    /// accepted by this slot. The check and replacement share the slot lock,
    /// so cloned producers cannot publish an older value after a newer one.
    pub(crate) fn publish_monotonic(
        &self,
        key: u64,
        value: T,
    ) -> Result<u64, MonotonicPublishError> {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.closed {
            return Err(MonotonicPublishError::Closed);
        }
        if let Some(previous) = state.last_monotonic_key
            && key <= previous
        {
            return Err(MonotonicPublishError::NonMonotonic { previous });
        }
        if state.value.is_some() {
            state.metrics.overwritten = state.metrics.overwritten.saturating_add(1);
        }
        state.metrics.published = state.metrics.published.saturating_add(1);
        state.last_monotonic_key = Some(key);
        let published = state.metrics.published;
        let replaced = state.value.replace(Arc::new(value));
        self.inner.changed.notify_one();
        drop(state);
        drop(replaced);
        Ok(published)
    }

    /// Realtime variant of [`Self::publish_monotonic`]. Busy means the caller
    /// may retry the same key because it was not accepted by the slot.
    pub(crate) fn try_publish_monotonic(
        &self,
        key: u64,
        value: T,
    ) -> Result<u64, TryMonotonicPublishError> {
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(TryMonotonicPublishError::Busy),
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        if state.closed {
            return Err(TryMonotonicPublishError::Closed);
        }
        if let Some(previous) = state.last_monotonic_key
            && key <= previous
        {
            return Err(TryMonotonicPublishError::NonMonotonic { previous });
        }
        if state.value.is_some() {
            state.metrics.overwritten = state.metrics.overwritten.saturating_add(1);
        }
        state.metrics.published = state.metrics.published.saturating_add(1);
        state.last_monotonic_key = Some(key);
        let published = state.metrics.published;
        let replaced = state.value.replace(Arc::new(value));
        self.inner.changed.notify_one();
        drop(state);
        drop(replaced);
        Ok(published)
    }

    pub fn try_take(&self) -> Option<Arc<T>> {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let value = state.value.take();
        if value.is_some() {
            state.metrics.consumed = state.metrics.consumed.saturating_add(1);
        }
        value
    }

    /// Wait for the next value. Returns `None` only after the slot is
    /// closed and no unread value remains.
    pub fn wait_take(&self) -> Option<Arc<T>> {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if let Some(value) = state.value.take() {
                state.metrics.consumed = state.metrics.consumed.saturating_add(1);
                return Some(value);
            }
            if state.closed {
                return None;
            }
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    pub fn close(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.closed = true;
        self.inner.changed.notify_all();
    }

    /// Take a newly published value immediately, or return `None` when the
    /// interval elapses. Closing the slot interrupts the wait without polling.
    ///
    /// This preserves capacity-one/latest-only delivery while allowing a
    /// periodic consumer to distinguish real work from its idle tick.
    pub fn wait_take_or_timeout(&self, interval: Duration) -> Result<Option<Arc<T>>, SlotClosed> {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(value) = state.value.take() {
            state.metrics.consumed = state.metrics.consumed.saturating_add(1);
            return Ok(Some(value));
        }
        if state.closed {
            return Err(SlotClosed);
        }
        let (mut state, timeout) = self
            .inner
            .changed
            .wait_timeout_while(state, interval, |state| {
                state.value.is_none() && !state.closed
            })
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(value) = state.value.take() {
            state.metrics.consumed = state.metrics.consumed.saturating_add(1);
            return Ok(Some(value));
        }
        if state.closed {
            return Err(SlotClosed);
        }
        debug_assert!(timeout.timed_out());
        Ok(None)
    }

    pub fn metrics(&self) -> SlotMetrics {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .metrics
    }
}

#[cfg(test)]
mod tests {
    use super::{LatestSlot, MonotonicPublishError, TryMonotonicPublishError};

    #[test]
    fn monotonic_publish_never_replaces_newer_data_with_older_data() {
        let slot = LatestSlot::new();
        slot.publish_monotonic(7, "newest").unwrap();

        assert_eq!(
            slot.publish_monotonic(6, "older"),
            Err(MonotonicPublishError::NonMonotonic { previous: 7 })
        );
        assert_eq!(slot.try_take().as_deref(), Some(&"newest"));
    }

    #[test]
    fn busy_try_publish_does_not_consume_the_generation() {
        let slot = LatestSlot::new();
        let competing_producer = slot.clone();
        let state = slot
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        assert_eq!(
            competing_producer.try_publish_monotonic(9, "retryable"),
            Err(TryMonotonicPublishError::Busy)
        );
        drop(state);

        competing_producer
            .try_publish_monotonic(9, "accepted")
            .unwrap();
        assert_eq!(slot.try_take().as_deref(), Some(&"accepted"));
    }
}

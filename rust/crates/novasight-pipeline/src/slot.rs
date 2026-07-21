use std::sync::{Arc, Condvar, Mutex};
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

#[derive(Debug)]
struct SlotState<T> {
    value: Option<Arc<T>>,
    closed: bool,
    metrics: SlotMetrics,
}

impl<T> Default for SlotState<T> {
    fn default() -> Self {
        Self {
            value: None,
            closed: false,
            metrics: SlotMetrics::default(),
        }
    }
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
        state.value = Some(Arc::new(value));
        self.inner.changed.notify_one();
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

    /// Wait for one scheduler interval while still waking promptly when
    /// the slot closes. Published values remain capacity-one and are
    /// deliberately not consumed until the interval expires.
    pub(crate) fn wait_interval(&self, interval: Duration) -> bool {
        let state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.closed {
            return false;
        }
        let (state, _) = self
            .inner
            .changed
            .wait_timeout_while(state, interval, |state| !state.closed)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !state.closed
    }

    pub fn metrics(&self) -> SlotMetrics {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .metrics
    }
}

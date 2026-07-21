//! Phase 2 capacity-one latest command slot.
//!
//! Buffer depth of exactly one with overwrite semantics. The slot
//! tracks epoch/generation provenance and an `expiry_at` monotonic
//! timestamp; a stale or wrong-epoch command can never reach a
//! `PointerDevice`. The runtime uses this slot to absorb back-to-back
//! control decisions and ensure only the most recent emit-eligible
//! decision is forwarded to the device adapter.

use std::cell::Cell;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::output::DeviceCommand;
use crate::perception::types::{Generation, MonotonicNanos, RuntimeEpoch};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SlotTakeError {
    Empty,
    EpochMismatch {
        stored: RuntimeEpoch,
        actual: RuntimeEpoch,
    },
    Expired {
        now: MonotonicNanos,
        expiry: MonotonicNanos,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SlotPushError {
    InvalidExpiry,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct SlotEntry {
    command: DeviceCommand,
    expiry_at: MonotonicNanos,
}

#[derive(Debug)]
pub struct LatestCommandSlot {
    inner: Mutex<Option<SlotEntry>>,
    overwrite_count: Cell<u64>,
    take_count: Cell<u64>,
    expired_drops: Cell<u64>,
}

impl LatestCommandSlot {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            overwrite_count: Cell::new(0),
            take_count: Cell::new(0),
            expired_drops: Cell::new(0),
        }
    }

    pub fn push(
        &self,
        command: DeviceCommand,
        expiry_at: MonotonicNanos,
    ) -> Result<(), SlotPushError> {
        if expiry_at.0 <= command.issued_at.0 {
            return Err(SlotPushError::InvalidExpiry);
        }
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_some() {
            self.overwrite_count
                .set(self.overwrite_count.get().saturating_add(1));
        }
        *guard = Some(SlotEntry { command, expiry_at });
        Ok(())
    }

    pub fn take(
        &self,
        now: MonotonicNanos,
        expected_epoch: RuntimeEpoch,
    ) -> Result<DeviceCommand, SlotTakeError> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = match guard.take() {
            Some(entry) => entry,
            None => return Err(SlotTakeError::Empty),
        };
        self.take_count.set(self.take_count.get().saturating_add(1));
        if entry.command.epoch != expected_epoch {
            return Err(SlotTakeError::EpochMismatch {
                stored: entry.command.epoch,
                actual: expected_epoch,
            });
        }
        if now.0 > entry.expiry_at.0 {
            self.expired_drops
                .set(self.expired_drops.get().saturating_add(1));
            return Err(SlotTakeError::Expired {
                now,
                expiry: entry.expiry_at,
            });
        }
        Ok(entry.command)
    }

    pub fn peek(
        &self,
        expected_epoch: RuntimeEpoch,
    ) -> Result<(Generation, MonotonicNanos), AppError> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = guard.as_ref().ok_or(AppError::RuntimeManagerUnavailable)?;
        if entry.command.epoch != expected_epoch {
            return Err(AppError::RuntimeEpochMismatch {
                expected: expected_epoch.0,
                actual: entry.command.epoch.0,
            });
        }
        Ok((entry.command.generation, entry.expiry_at))
    }

    pub fn clear(&self) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = None;
    }

    pub fn overwrite_count(&self) -> u64 {
        self.overwrite_count.get()
    }

    pub fn take_count(&self) -> u64 {
        self.take_count.get()
    }

    pub fn expired_drops(&self) -> u64 {
        self.expired_drops.get()
    }
}

impl Default for LatestCommandSlot {
    fn default() -> Self {
        Self::new()
    }
}

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

pub mod latest_command;
pub mod quantizer;

use crate::error::AppError;
use crate::perception::types::{Generation, MonotonicNanos, RuntimeEpoch};
use crate::ports::PointerDevice;

/// Fully typed pointer movement derived from one runtime observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceCommand {
    pub epoch: RuntimeEpoch,
    pub generation: Generation,
    pub issued_at: MonotonicNanos,
    pub target_object_id: u64,
    pub delta_x_counts: i32,
    pub delta_y_counts: i32,
}

/// Durable replay evidence that a particular typed command was accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceReceipt {
    pub attempt: u64,
    pub epoch: RuntimeEpoch,
    pub generation: Generation,
    pub issued_at: MonotonicNanos,
    pub target_object_id: u64,
    pub delta_x_counts: i32,
    pub delta_y_counts: i32,
}

impl DeviceReceipt {
    /// Record a successfully accepted command at a concrete device adapter.
    pub const fn accepted(attempt: u64, command: DeviceCommand) -> Self {
        Self {
            attempt,
            epoch: command.epoch,
            generation: command.generation,
            issued_at: command.issued_at,
            target_object_id: command.target_object_id,
            delta_x_counts: command.delta_x_counts,
            delta_y_counts: command.delta_y_counts,
        }
    }

    fn from_command(attempt: u64, command: DeviceCommand) -> Self {
        Self::accepted(attempt, command)
    }
}

/// Deterministic dry-run adapter retaining one real receipt per attempted command.
#[derive(Debug, Default)]
pub struct RecordingPointerDevice {
    receipts: Mutex<Vec<DeviceReceipt>>,
}

impl RecordingPointerDevice {
    pub fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        <Self as PointerDevice>::send(self, command)
    }

    pub fn receipts(&self) -> Vec<DeviceReceipt> {
        self.receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl PointerDevice for RecordingPointerDevice {
    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        let mut receipts = self
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let receipt = DeviceReceipt::from_command(receipts.len() as u64, command);
        receipts.push(receipt);
        Ok(receipt)
    }
}

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

pub mod humanized_motion;
pub mod latest_command;

use crate::error::AppError;
use crate::perception::types::{Generation, MonotonicNanos, RuntimeEpoch};
use crate::ports::{PointerDevice, PointerDeviceMode};

/// Fully typed pointer movement derived from one runtime observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceCommand {
    pub epoch: RuntimeEpoch,
    pub generation: Generation,
    /// Capture timestamp of the perception sample that authorized this
    /// movement. Output freshness must be measured from here, not from the
    /// later control-decision timestamp.
    pub source_captured_at: MonotonicNanos,
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
    fn mode(&self) -> PointerDeviceMode {
        PointerDeviceMode::Commissioned
    }

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

/// Inert production adapter used before a physical pointer device has been
/// provisioned. It lets the runtime own a complete dependency graph without
/// silently substituting a recording or network device.
#[derive(Clone, Copy, Debug, Default)]
pub struct UncommissionedPointerDevice;

fn uncommissioned_device_error() -> AppError {
    AppError::PointerDevice {
        code: "device_uncommissioned",
        message:
            "configure hardware.auto_connect with a provisioned host and UUID before opening output"
                .to_owned(),
    }
}

impl PointerDevice for UncommissionedPointerDevice {
    fn mode(&self) -> PointerDeviceMode {
        PointerDeviceMode::Uncommissioned
    }

    fn send(&self, _command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        Err(uncommissioned_device_error())
    }

    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        Err(uncommissioned_device_error())
    }

    fn buttons(&self) -> Result<Option<crate::ports::PointerButtons>, AppError> {
        Err(uncommissioned_device_error())
    }
}

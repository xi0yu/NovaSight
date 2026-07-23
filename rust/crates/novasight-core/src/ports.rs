use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{AppError, DetectionBatch, DeviceCommand, DeviceReceipt, MonotonicNanos};

/// Monotonic time seam for runtime code; implementations must not return wall time.
pub trait Clock: Send + Sync {
    fn now(&self) -> MonotonicNanos;
}

/// Ordered perception input seam. `None` means the finite source is exhausted.
#[async_trait]
pub trait PerceptionSource: Send {
    async fn next_batch(&mut self) -> Result<Option<DetectionBatch>, AppError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PointerButtons {
    pub left: bool,
    pub right: bool,
}

/// Provisioning state exposed by a pointer adapter. Runtime composition uses
/// this capability instead of maintaining a second configuration boolean that
/// could drift from the selected adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PointerDeviceMode {
    Commissioned,
    #[default]
    Uncommissioned,
}

impl PointerButtons {
    pub const fn trigger_active(self) -> bool {
        self.left || self.right
    }
}

/// Device transition seam; successful sends return a typed receipt for the same command.
pub trait PointerDevice: Send + Sync {
    fn mode(&self) -> PointerDeviceMode {
        PointerDeviceMode::Uncommissioned
    }

    /// Acquire the concrete device session for one runtime epoch. Stateless
    /// adapters may keep the default no-op implementation.
    fn connect(&self) -> Result<(), AppError> {
        Ok(())
    }

    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError>;

    /// Read the hardware-owned output trigger when the adapter supports it.
    /// `None` means this device has no hardware trigger source.
    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        Ok(None)
    }

    /// Read both hardware buttons when the adapter exposes them. Existing
    /// adapters that only expose one combined trigger retain safe compatibility.
    fn buttons(&self) -> Result<Option<PointerButtons>, AppError> {
        self.trigger_active()
            .map(|active| active.map(|left| PointerButtons { left, right: false }))
    }

    /// Release the epoch-owned device session after every worker has stopped.
    fn disconnect(&self) -> Result<(), AppError> {
        Ok(())
    }
}

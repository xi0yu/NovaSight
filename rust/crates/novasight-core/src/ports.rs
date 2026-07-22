use async_trait::async_trait;

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

/// Device transition seam; successful sends return a typed receipt for the same command.
pub trait PointerDevice: Send + Sync {
    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError>;

    /// Read the hardware-owned output trigger when the adapter supports it.
    /// `None` means this device has no hardware trigger source.
    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        Ok(None)
    }
}

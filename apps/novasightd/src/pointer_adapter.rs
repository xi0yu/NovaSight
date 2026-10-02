//! Platform-neutral pointer commissioning selection.
//!
//! Concrete kmNet construction remains in the Jetson composition root. This
//! module owns the safe branch so it is compiled and tested on every platform.

use std::sync::Arc;

use novasight_core::{PointerDevice, PointerDeviceMode, UncommissionedPointerDevice};
use novasight_store::config::DeviceConfig;

pub(crate) struct PointerAdapterSelection {
    pub(crate) device: Arc<dyn PointerDevice>,
    pub(crate) trigger_poll_interval_ms: Option<u64>,
}

pub(crate) fn configured_pointer_device_mode(device: Option<&DeviceConfig>) -> PointerDeviceMode {
    if device.is_some_and(|device| device.auto_connect) {
        PointerDeviceMode::Commissioned
    } else {
        PointerDeviceMode::Uncommissioned
    }
}

pub(crate) fn select_uncommissioned_pointer_adapter(
    device: &DeviceConfig,
) -> Option<PointerAdapterSelection> {
    (configured_pointer_device_mode(Some(device)) == PointerDeviceMode::Uncommissioned).then(|| {
        PointerAdapterSelection {
            device: Arc::new(UncommissionedPointerDevice),
            trigger_poll_interval_ms: None,
        }
    })
}

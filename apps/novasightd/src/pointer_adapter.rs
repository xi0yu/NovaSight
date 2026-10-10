//! Platform-neutral pointer commissioning selection.
//!
//! Concrete kmNet construction remains in the Jetson composition root. This
//! module owns the safe branch so it is compiled and tested on every platform.

use std::sync::Arc;

use novasight_core::{
    NullPointerDevice, PointerDevice, PointerDeviceMode, UncommissionedPointerDevice,
};
use novasight_store::config::{DeviceBackend, DeviceConfig};

pub(crate) struct PointerAdapterSelection {
    pub(crate) device: Arc<dyn PointerDevice>,
    pub(crate) trigger_poll_interval_ms: Option<u64>,
}

pub(crate) fn configured_pointer_device_mode(device: Option<&DeviceConfig>) -> PointerDeviceMode {
    if device.is_some_and(|device| device.backend == DeviceBackend::Null || device.auto_connect) {
        PointerDeviceMode::Commissioned
    } else {
        PointerDeviceMode::Uncommissioned
    }
}

pub(crate) fn select_builtin_pointer_adapter(
    device: &DeviceConfig,
) -> Option<PointerAdapterSelection> {
    if device.backend == DeviceBackend::Null {
        return Some(PointerAdapterSelection {
            device: Arc::new(NullPointerDevice::default()),
            trigger_poll_interval_ms: None,
        });
    }
    (configured_pointer_device_mode(Some(device)) == PointerDeviceMode::Uncommissioned).then(|| {
        PointerAdapterSelection {
            device: Arc::new(UncommissionedPointerDevice),
            trigger_poll_interval_ms: None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_protocol_is_commissioned_without_native_network_or_button_polling() {
        let config: DeviceConfig = serde_json::from_value(serde_json::json!({
            "backend": "null", "host": "", "uuid": ""
        }))
        .unwrap();
        assert_eq!(
            configured_pointer_device_mode(Some(&config)),
            PointerDeviceMode::Commissioned
        );
        let selected = select_builtin_pointer_adapter(&config).unwrap();
        assert_eq!(selected.device.mode(), PointerDeviceMode::Commissioned);
        assert_eq!(selected.trigger_poll_interval_ms, None);
        assert_eq!(selected.device.buttons().unwrap(), None);
        let native: DeviceConfig = serde_json::from_value(serde_json::json!({
            "auto_connect": true
        }))
        .unwrap();
        assert!(select_builtin_pointer_adapter(&native).is_none());
    }
}

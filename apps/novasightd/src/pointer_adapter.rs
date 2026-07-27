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

#[cfg(test)]
mod tests {
    use novasight_core::PointerDeviceMode;
    use novasight_store::config::DeviceConfig;

    use super::{configured_pointer_device_mode, select_uncommissioned_pointer_adapter};

    #[test]
    fn disabled_auto_connect_selects_an_inert_adapter_without_a_trigger_worker() {
        let device = DeviceConfig::default();

        let selected = select_uncommissioned_pointer_adapter(&device).expect("safe adapter");

        assert_eq!(selected.device.mode(), PointerDeviceMode::Uncommissioned);
        assert_eq!(selected.trigger_poll_interval_ms, None);
    }

    #[test]
    fn commissioned_configuration_is_left_for_the_platform_composition_root() {
        let mut device = DeviceConfig::default();
        device.auto_connect = true;

        assert_eq!(
            configured_pointer_device_mode(Some(&device)),
            PointerDeviceMode::Commissioned
        );
        assert!(select_uncommissioned_pointer_adapter(&device).is_none());
    }
}

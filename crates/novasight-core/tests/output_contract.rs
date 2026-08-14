//! Output contract tests. Pins the typed behavior of the
//! `AxisCountQuantizer` / `DeviceCountQuantizer` contract used to convert a
//! fractional control demand into a delivery-ready integer command.

use novasight_core::output::DeviceCommand;
use novasight_core::quantizer::{AxisCountQuantizer, DeviceCountQuantizer};
use novasight_core::{
    AppError, Generation, MonotonicNanos, PointerDevice, PointerDeviceMode, RuntimeEpoch,
    UncommissionedPointerDevice,
};

fn command(epoch: u64, generation: u64, issued_at: u64, _expiry_at: u64) -> DeviceCommand {
    DeviceCommand {
        epoch: RuntimeEpoch(epoch),
        generation: Generation(generation),
        source_captured_at: MonotonicNanos(issued_at),
        issued_at: MonotonicNanos(issued_at),
        target_object_id: 1,
        delta_x_counts: 5,
        delta_y_counts: 0,
    }
}

#[test]
fn uncommissioned_device_is_inert_and_rejects_every_send() {
    let device = UncommissionedPointerDevice;

    assert_eq!(device.mode(), PointerDeviceMode::Uncommissioned);
    device
        .connect()
        .expect("disabled adapter has no session to open");
    for error in [
        device.send(command(1, 1, 100, 200)).unwrap_err(),
        device.buttons().unwrap_err(),
        device.trigger_active().unwrap_err(),
    ] {
        assert_eq!(
            error,
            AppError::PointerDevice {
                code: "device_uncommissioned",
                message: "configure hardware.auto_connect with a provisioned host and UUID before opening output"
                    .to_owned(),
            }
        );
    }
    device
        .disconnect()
        .expect("disabled adapter has no session to close");
}

#[test]
fn pointer_device_capability_defaults_to_uncommissioned() {
    assert_eq!(
        PointerDeviceMode::default(),
        PointerDeviceMode::Uncommissioned
    );
}

#[test]
fn quantizer_keeps_fractional_residual_below_one_count() {
    let mut quantizer = AxisCountQuantizer::new();
    let first = quantizer.quantize(5.0).expect("quantize 5");
    assert_eq!(first, 5);
    assert!(quantizer.residual().abs() <= 1.0);
    for _ in 0..5 {
        let _ = quantizer.quantize(200.0).expect("quantize 200");
    }
    assert!(quantizer.residual().abs() <= 1.0);
}

#[test]
fn quantizer_rejects_nan_and_infinity() {
    let mut quantizer = AxisCountQuantizer::new();
    assert!(quantizer.quantize(f64::NAN).is_err());
    assert!(quantizer.quantize(f64::INFINITY).is_err());
    assert!(quantizer.quantize(f64::NEG_INFINITY).is_err());
    assert_eq!(quantizer.residual(), 0.0);
}

#[test]
fn quantizer_carries_residual_across_calls() {
    let mut quantizer = AxisCountQuantizer::new();
    let first = quantizer.quantize(0.4).expect("quantize 0.4");
    assert_eq!(first, 0);
    let residual_after_first = quantizer.residual();
    assert!((residual_after_first - 0.4).abs() < 1e-9);
    let second = quantizer.quantize(0.4).expect("quantize 0.4");
    assert_eq!(second, 0);
    assert!((quantizer.residual() - 0.8).abs() < 1e-9);
    let third = quantizer.quantize(0.4).expect("quantize 0.4");
    assert_eq!(third, 1);
    assert!((quantizer.residual() - 0.2).abs() < 1e-9);
}

#[test]
fn quantizer_sign_inversion_round_trip() {
    let mut quantizer = AxisCountQuantizer::new();
    let positive = quantizer.quantize(3.5).expect("+3.5");
    assert_eq!(positive, 3);
    assert!((quantizer.residual() - 0.5).abs() < 1e-9);
    let negative = quantizer.quantize(-3.5).expect("-3.5");
    assert_eq!(negative, -3);
    assert!((quantizer.residual() + 0.5).abs() < 1e-9);
}

#[test]
fn device_count_quantizer_clears_both_axes_on_partial_failure() {
    let mut quantizer = DeviceCountQuantizer::new();
    let _ = quantizer.quantize(0.6, 0.4).expect("first quantization");
    let (before_x, before_y) = quantizer.residuals();
    assert!(before_x != 0.0 || before_y != 0.0);
    let result = quantizer.quantize(0.1, f64::NAN);
    assert!(result.is_err());
    let (after_x, after_y) = quantizer.residuals();
    assert_eq!(after_x, 0.0);
    assert_eq!(after_y, 0.0);
}

#[test]
fn reset_axis_quantizer_drops_residual() {
    let mut quantizer = AxisCountQuantizer::new();
    let _ = quantizer.quantize(0.7).expect("quantize");
    assert!(quantizer.residual().abs() > 0.0);
    quantizer.reset();
    assert_eq!(quantizer.residual(), 0.0);
}

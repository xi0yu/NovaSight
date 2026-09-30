//! Deterministic host simulations; not a substitute for capture/device calibration.
use novasight_core::controller::{AimAlgorithm, AimAlgorithmConfig, AimSample, BlockReason};

fn sample(generation: u64, elapsed_ms: f64, error: f64) -> AimSample {
    let capture_ts_ns = 1_000_000_000 + (elapsed_ms * 1_000_000.0).round() as u64;
    AimSample {
        generation,
        target_id: 1,
        capture_ts_ns,
        control_now_ns: capture_ts_ns + 4_000_000,
        aim_x: 320.0 + error,
        aim_y: 320.0,
        crosshair_x: 320.0,
        crosshair_y: 320.0,
        target_valid: true,
        trigger_active: true,
    }
}

#[test]
fn prediction_history_window_does_not_control_entry_ramp() {
    for history_gap in [10.0, 500.0] {
        let mut control = AimAlgorithm::new(AimAlgorithmConfig {
            prediction_enabled: false,
            entry_ramp_ms: 100.0,
            velocity_history_reset_gap_ms: history_gap,
            ..Default::default()
        });
        control.step(sample(1, 0.0, 80.0));
        assert!(control.step(sample(2, 20.0, 80.0)).float_demand_x > 0.0);
        // 60ms between captures is not a stale 60ms-old frame. Fresh frames
        // must continue entry even when their period exceeds the 50ms age gate.
        let at_80 = control.step(sample(3, 80.0, 80.0));
        let at_140 = control.step(sample(4, 140.0, 80.0));
        assert!(at_140.float_demand_x > at_80.float_demand_x);
        assert!(at_80.float_demand_x > 0.0);
        // Capture loss still resets entry, even with a generous prediction history.
        assert_eq!(control.step(sample(5, 240.0, 80.0)).dx, 0);
    }
}

#[test]
fn time_normalization_preserves_constant_error_budget_and_discards_gate_debt() {
    let config = AimAlgorithmConfig {
        response_reference_hz: 60.0,
        entry_ramp_ms: 0.0,
        prediction_enabled: false,
        ..Default::default()
    };
    let full = AimAlgorithm::new(AimAlgorithmConfig {
        response_reference_hz: 0.0,
        ..config
    })
    .step(sample(1, 0.0, 80.0))
    .float_demand_x;
    for fps in [30_u64, 60, 120, 240] {
        let mut control = AimAlgorithm::new(config);
        let mut total = 0.0;
        for i in 0..=fps {
            let result = control.step(sample(i + 1, i as f64 * 1000.0 / fps as f64, 80.0));
            if i == 0 {
                assert_eq!(result.dx, 0);
            }
            total += result.float_demand_x;
        }
        assert!((total - full * 60.0).abs() < 1e-7, "fps={fps}");
        control.release_trigger();
        assert_eq!(control.step(sample(fps + 2, 1010.0, 80.0)).dx, 0);
        // Valid captures with a controller stall cannot repay more than 50 ms.
        let mut delayed = sample(fps + 3, 1020.0, 80.0);
        delayed.control_now_ns = delayed.capture_ts_ns + 50_000_000;
        let result = control.step(delayed);
        assert!((result.float_demand_x - full * 3.0).abs() < 1e-9);
        assert_eq!(control.step(sample(fps + 4, 1200.0, 80.0)).dx, 0);
    }
    for invalid in [-1.0, 241.0, f64::NAN] {
        let result = AimAlgorithm::new(AimAlgorithmConfig {
            response_reference_hz: invalid,
            ..config
        })
        .step(sample(1, 0.0, 80.0));
        assert_eq!(result.block_reason, BlockReason::GeometryInvalid);
    }
}

#[test]
fn entry_is_smoothstep_and_restarts_after_each_gate() {
    let config = AimAlgorithmConfig {
        prediction_enabled: false,
        ..Default::default()
    };
    let full = AimAlgorithm::new(AimAlgorithmConfig {
        entry_ramp_ms: 0.0,
        ..config
    })
    .step(sample(1, 0.0, 80.0))
    .float_demand_x;
    let mut control = AimAlgorithm::new(config);
    for (i, gain) in [0.0, 0.15625, 0.5, 0.84375, 1.0, 1.0]
        .into_iter()
        .enumerate()
    {
        let result = control.step(sample(i as u64 + 1, i as f64 * 50.0, 80.0));
        assert!((result.float_demand_x / full - gain).abs() < 1e-12);
        if i == 0 {
            assert_eq!(result.block_reason, BlockReason::EntryRampPending);
        }
    }
    // Release, invalid/out-of-range target, identity switch, stale data, and capture gap.
    for gate in 0..5 {
        let mut control = AimAlgorithm::new(config);
        for i in 0..=4 {
            control.step(sample(i + 1, i as f64 * 50.0, 80.0));
        }
        let mut event = sample(6, 210.0, 80.0);
        match gate {
            0 => event.trigger_active = false,
            1 => event.target_valid = false,
            2 => event.target_id = 2,
            3 => event.control_now_ns += 100_000_000,
            _ => event = sample(6, 400.0, 80.0),
        }
        let result = control.step(event);
        assert_eq!((result.dx, result.dy), (0, 0), "gate {gate}");
        if gate < 2 || gate == 3 {
            let result = control.step(sample(7, 220.0, 80.0));
            assert_eq!(result.block_reason, BlockReason::EntryRampPending);
        }
    }
    for invalid in [f64::NAN, -1.0, 2001.0] {
        let result = AimAlgorithm::new(AimAlgorithmConfig {
            entry_ramp_ms: invalid,
            ..config
        })
        .step(sample(1, 0.0, 80.0));
        assert_eq!(result.block_reason, BlockReason::GeometryInvalid);
    }
}

#[test]
fn configured_hundred_ms_ramp_reaches_full_kp_and_stays_there() {
    let config = AimAlgorithmConfig {
        entry_ramp_ms: 100.0,
        prediction_enabled: false,
        ..Default::default()
    };
    let full = AimAlgorithm::new(AimAlgorithmConfig {
        entry_ramp_ms: 0.0,
        ..config
    })
    .step(sample(1, 0.0, 80.0))
    .float_demand_x;
    assert!(full > 0.0);
    let mut control = AimAlgorithm::new(config);
    for (i, gain) in [0.0, 0.15625, 0.5, 0.84375, 1.0, 1.0]
        .into_iter()
        .enumerate()
    {
        let result = control.step(sample(i as u64 + 1, i as f64 * 25.0, 80.0));
        assert!((result.float_demand_x - full * gain).abs() < 1e-12);
    }
}

#[test]
fn next_frame_prediction_matches_constant_velocity_at_multiple_frame_rates() {
    for fps in [30.0, 60.0, 120.0, 240.0] {
        let dt = 1000.0 / fps;
        let mut control = AimAlgorithm::new(AimAlgorithmConfig {
            prediction_lead_ms: dt,
            prediction_actuation_delay_ms: 0.0,
            ..Default::default()
        });
        for i in 0..8 {
            let mut input = sample(i + 1, i as f64 * dt, 20.0 + i as f64 * dt * 0.1);
            input.control_now_ns = input.capture_ts_ns;
            let result = control.step(input);
            if i >= 3 {
                assert!((result.prediction_horizon_ms - dt).abs() < 0.000_001);
                assert!((result.predicted_offset_x - dt * 0.1).abs() < 0.000_001);
            }
        }
    }
}

#[test]
fn closed_loop_settles_after_entry_motion_reversal_and_stop() {
    // Fixed local linear device calibration, one delayed capture, four-second trials.
    // Both prediction modes are measured, so a passing endpoint cannot hide worse tracking.
    for fps in [30.0, 60.0, 120.0, 240.0] {
        for (moving, noise) in [(false, 0.0), (true, 0.0), (false, 0.4), (true, 0.4)] {
            for prediction_enabled in [false, true] {
                let config = AimAlgorithmConfig {
                    prediction_enabled,
                    ..Default::default()
                };
                let focal = 320.0 / (config.projection_fov_x_deg.to_radians() / 2.0).tan();
                let px_per_count =
                    focal * (std::f64::consts::TAU / config.projection_counts_per_360).tan();
                let mut control = AimAlgorithm::new(config);
                let dt = 1000.0 / fps;
                let mut error = 100.0;
                let mut previous_error = error;
                let mut tail_sum = 0.0;
                let mut tail_count = 0;
                let mut moving_sum = 0.0;
                let mut moving_count = 0;
                let mut peak_overshoot: f64 = 0.0;
                for i in 0..(fps as u64 * 4) {
                    let t = i as f64 * dt;
                    let velocity = if !moving || t >= 2500.0 {
                        0.0
                    } else if t < 1500.0 {
                        0.04
                    } else {
                        -0.04
                    };
                    error += velocity * dt;
                    let mut input =
                        sample(i + 1, t, previous_error + noise * (i as f64 * 1.6).sin());
                    input.control_now_ns += (dt * 1_000_000.0).round() as u64;
                    previous_error = error;
                    let result = control.step(input);
                    if i == 0 {
                        assert_eq!((result.dx, result.dy), (0, 0));
                    }
                    assert!(result.float_demand_x.is_finite());
                    error -= f64::from(result.dx) * px_per_count;
                    peak_overshoot = peak_overshoot.max((-error).max(0.0));
                    if t > 750.0 && t < 1400.0 {
                        moving_sum += error.abs();
                        moving_count += 1;
                    }
                    if t > 3500.0 {
                        tail_sum += error.abs();
                        tail_count += 1;
                    }
                }
                let tail = tail_sum / tail_count as f64;
                let tracking = moving_sum / moving_count as f64;
                println!(
                    "fps={fps} moving={moving} noise={noise} prediction={prediction_enabled} tail={tail:.4}px tracking={tracking:.4}px negative_peak={peak_overshoot:.4}px"
                );
                assert!(tail < 0.5, "settling fps={fps} moving={moving}: {tail}");
                assert!(tracking < 8.0, "tracking fps={fps}: {tracking}");
                if !moving {
                    assert!(
                        peak_overshoot < 2.0,
                        "overshoot fps={fps}: {peak_overshoot}"
                    );
                }
            }
        }
    }
}

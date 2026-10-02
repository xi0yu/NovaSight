use novasight_core::controller::{AimControlInput, AimControlLaw, AimControlParameters, AxisPair};

#[test]
fn prediction_is_applied_before_projection_and_nonlinear_response() {
    let parameters = AimControlParameters {
        source_width: 640,
        roi_width: 640,
        roi_height: 640,
        observation_width: 640,
        observation_height: 640,
        projection_fov_x_deg: 90.0,
        projection_counts_per_360: std::f64::consts::TAU * 100.0,
        response_scale: 0.25,
    };
    let law = AimControlLaw::new(parameters).expect("valid control law");

    let result = law
        .evaluate(AimControlInput {
            measured_error_px: AxisPair::new(24.0, -12.0),
            predicted_offset_px: AxisPair::new(8.0, 4.0),
        })
        .expect("finite control result");

    assert_eq!(result.predicted_error_px, AxisPair::new(32.0, -8.0));

    let expected_counts = AxisPair::new(
        (32.0_f64 / 320.0).atan() * 100.0,
        (-8.0_f64 / 320.0).atan() * 100.0,
    );
    assert!((result.projected_error_counts.x - expected_counts.x).abs() < 1e-12);
    assert!((result.projected_error_counts.y - expected_counts.y).abs() < 1e-12);

    let expected_x = 0.25 * 256.0 * (expected_counts.x / 256.0).atan();
    let expected_y = 0.25 * 256.0 * (expected_counts.y / 256.0).atan();
    assert!((result.demand_counts.x - expected_x).abs() < 1e-12);
    assert!((result.demand_counts.y - expected_y).abs() < 1e-12);
    assert_eq!(result.effective_gain, 0.25);

    // Kp is constant at every distance; changing Y must not boost X.
    for kp in [0.0, 0.2, 0.8] {
        let law = AimControlLaw::new(AimControlParameters {
            response_scale: kp,
            ..parameters
        })
        .unwrap();
        let mut previous_x = None;
        for y in [0.0, 10.0, -1000.0, 10000.0] {
            let result = law
                .evaluate(AimControlInput {
                    measured_error_px: AxisPair::new(32.0, y),
                    predicted_offset_px: AxisPair::default(),
                })
                .unwrap();
            assert_eq!(result.effective_gain, kp);
            for (counts, demand) in [
                (result.projected_error_counts.x, result.demand_counts.x),
                (result.projected_error_counts.y, result.demand_counts.y),
            ] {
                assert!((demand - kp * 256.0 * (counts / 256.0).atan()).abs() < 1e-12);
            }
            if let Some(x) = previous_x {
                assert_eq!(result.demand_counts.x, x);
            }
            previous_x = Some(result.demand_counts.x);
        }
    }
}

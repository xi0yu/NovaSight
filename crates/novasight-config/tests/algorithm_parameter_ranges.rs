use novasight_config::AppConfig;

#[test]
fn algorithm_parameters_accept_endpoints_and_reject_invalid_values() {
    for (key, max) in [
        ("target_fov_radius_px", 1000),
        ("entry_ramp_ms", 1000),
        ("max_output_x_counts", 1000),
        ("max_output_y_counts", 1000),
        ("fire_delay_ms", 2000),
    ] {
        for value in ["0".to_string(), max.to_string()] {
            let config: AppConfig =
                serde_yaml::from_str(&format!("pipeline:\n  {key}: {value}\n")).unwrap();
            config.validate_configured_adapters().unwrap();
        }
        for value in [
            "-1".to_string(),
            (max + 1).to_string(),
            ".nan".to_string(),
            ".inf".to_string(),
        ] {
            let config =
                serde_yaml::from_str::<AppConfig>(&format!("pipeline:\n  {key}: {value}\n"));
            if let Ok(config) = config {
                let error = config.validate_configured_adapters().unwrap_err();
                assert_eq!(error.field, format!("pipeline.{key}"), "{key}={value}");
            }
        }
    }
}

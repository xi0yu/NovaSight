use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_runtime::compose_pipeline_config;
use novasight_store::config::YamlConfigRepository;

static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let unique = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "novasight-runtime-config-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn join(&self, path: impl AsRef<Path>) -> PathBuf {
        self.0.join(path)
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn compose_pipeline_config_uses_configured_kalman_prediction_window() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let config = YamlConfigRepository::new(&path)
        .save_field(
            "pipeline",
            "tracker_kalman_max_predict_dt_ms",
            serde_yaml::Value::Number(42_u64.into()),
            0,
        )
        .unwrap();
    let config = YamlConfigRepository::new(&path)
        .save_field(
            "pipeline",
            "tracker_kalman_max_predict_missing_ms",
            serde_yaml::Value::Number(125_u64.into()),
            config.revision,
        )
        .unwrap();
    let config = YamlConfigRepository::new(&path)
        .save_field(
            "pipeline",
            "tracker_kalman_max_predict_steps",
            serde_yaml::Value::Number(7_u64.into()),
            config.revision,
        )
        .unwrap();

    let pipeline = compose_pipeline_config(&config, None).unwrap();

    assert_eq!(pipeline.targeting.kalman.max_predict_dt_ms, 42.0);
    assert_eq!(pipeline.targeting.kalman.max_predict_missing_ms, 125.0);
    assert_eq!(pipeline.targeting.kalman.max_predict_steps, 7);
}

#[test]
fn compose_pipeline_config_uses_continuous_response_fields() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let mut revision = 0;
    for (key, value) in [
        ("p_response_scale", 0.42),
        ("p_response_gain_floor", 0.55),
        ("p_response_gain_ceiling", 1.15),
        ("p_response_curve_width_ratio", 0.40),
        ("p_response_curve_shape", 1.50),
    ] {
        let config = YamlConfigRepository::new(&path)
            .save_field(
                "pipeline",
                key,
                serde_yaml::to_value(value).unwrap(),
                revision,
            )
            .unwrap();
        revision = config.revision;
    }
    let config = YamlConfigRepository::load(&path).unwrap();

    let pipeline = compose_pipeline_config(&config, None).unwrap();

    assert_eq!(pipeline.control.response_scale, 0.42);
    assert_eq!(pipeline.control.response_gain_floor, 0.55);
    assert_eq!(pipeline.control.response_gain_ceiling, 1.15);
    assert_eq!(pipeline.control.response_curve_width_ratio, 0.40);
    assert_eq!(pipeline.control.response_curve_shape, 1.50);
}

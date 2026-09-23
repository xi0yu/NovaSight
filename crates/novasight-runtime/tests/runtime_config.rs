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
        ("p_response_boost", 0.60),
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
    assert_eq!(pipeline.control.response_boost, 0.60);
    assert_eq!(pipeline.control.response_curve_shape, 1.50);
}

#[test]
fn compose_pipeline_config_wires_target_decision_policy() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let mut config = YamlConfigRepository::load(&path).unwrap();
    config.pipeline.tracker_class_cost_weight = 0.7;
    config.pipeline.target_selection_distance_weight = 0.1;
    config.pipeline.target_selection_class_weight = 0.2;
    config.pipeline.target_selection_confidence_weight = 0.3;
    config.pipeline.target_selection_size_weight = 0.4;
    config.pipeline.target_selection_continuity_weight = 0.5;
    config.pipeline.target_selection_motion_weight = 0.6;
    config.pipeline.target_selection_motion_horizon_ms = 45.0;

    let pipeline = compose_pipeline_config(&config, None).unwrap();

    assert_eq!(pipeline.targeting.tracker_class_cost_weight, 0.7);
    assert_eq!(pipeline.targeting.selection_weights.distance, 0.1);
    assert_eq!(pipeline.targeting.selection_weights.class, 0.2);
    assert_eq!(pipeline.targeting.selection_weights.confidence, 0.3);
    assert_eq!(pipeline.targeting.selection_weights.size, 0.4);
    assert_eq!(pipeline.targeting.selection_weights.continuity, 0.5);
    assert_eq!(pipeline.targeting.selection_weights.motion, 0.6);
    assert_eq!(pipeline.targeting.selection_motion_horizon_ms, 45.0);
}

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_runtime::compose_pipeline_config;
use novasight_store::config::{AppConfig, CaptureConfig, YamlConfigRepository};

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

fn with_test_capture(mut config: AppConfig) -> AppConfig {
    config.capture = Some(CaptureConfig::manual(
        "/dev/video-test".into(),
        "MJPG".to_owned(),
        (1920, 1080, 120),
        (640, 220, 640, 640),
    ));
    config
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

    let pipeline = compose_pipeline_config(&with_test_capture(config), None).unwrap();

    assert_eq!(pipeline.targeting.kalman.max_predict_dt_ms, 42.0);
    assert_eq!(pipeline.targeting.kalman.max_predict_missing_ms, 125.0);
}

#[test]
fn compose_pipeline_config_uses_continuous_response_fields() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let mut revision = 0;
    for (key, value) in [
        ("p_response_scale", 0.42),
        ("entry_ramp_ms", 250.0),
        ("response_reference_hz", 60.0),
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

    let pipeline = compose_pipeline_config(&with_test_capture(config), None).unwrap();

    assert_eq!(pipeline.control.response_scale, 0.42);
    assert_eq!(pipeline.control.entry_ramp_ms, 250.0);
    assert_eq!(pipeline.control.response_reference_hz, 60.0);
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
    config.pipeline.target_class_weights = "0:0.8,1:0.4".into();
    config.pipeline.target_class_aim_x_ratios = "0:0.3,1:0.7".into();
    config.inference.as_mut().unwrap().confidence_threshold = 0.65;
    config.pipeline.frame_max_age_ms = 35.0;

    let pipeline = compose_pipeline_config(&with_test_capture(config), None).unwrap();

    assert_eq!(pipeline.targeting.tracker_class_cost_weight, 0.7);
    assert_eq!(pipeline.targeting.selection_weights.distance, 0.1);
    assert_eq!(pipeline.targeting.selection_weights.class, 0.2);
    assert_eq!(pipeline.targeting.selection_weights.confidence, 0.3);
    assert_eq!(pipeline.targeting.class_weights[&0], 0.8);
    assert_eq!(pipeline.targeting.class_weights[&1], 0.4);
    assert_eq!(pipeline.targeting.class_aim_x_ratios[&0], 0.3);
    assert_eq!(pipeline.targeting.min_confidence, 0.65_f32);
    assert_eq!(pipeline.control.freshness_threshold_ms, 35.0);
}

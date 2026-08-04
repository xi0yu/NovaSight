use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_store::config::YamlConfigRepository;
use serde_yaml::Value;

static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let unique = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "novasight-config-repository-{}-{unique}",
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
fn bundled_runtime_config_loads_current_algorithm_defaults() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();

    let config = YamlConfigRepository::load(path).unwrap();

    assert_eq!(config.schema_version, 11);
    assert_eq!(config.pipeline.p_response_scale, 0.30);
    assert_eq!(config.pipeline.p_response_curve_shape, 1.0);
    assert_eq!(config.pipeline.atan_scale_counts, 256.0);
    assert_eq!(config.pipeline.prediction_lead_ms, 16.0);
    assert!(config.pipeline.prediction_enabled);
}

#[test]
fn schema_ten_prediction_frame_lead_migrates_to_time_lead() {
    let directory = TempDirectory::new();
    let path = directory.join("schema-ten-prediction-lead.yaml");
    fs::write(
        &path,
        r#"schema_version: 10
limits:
  stream_fps: 40
pipeline:
  prediction_lead_frames: 2.0
  velocity_smoothing_frames: 5.0
"#,
    )
    .unwrap();

    let config = YamlConfigRepository::load(&path).unwrap();

    assert_eq!(config.schema_version, 11);
    assert!((config.pipeline.prediction_lead_ms - 50.0).abs() < 1e-12);
    assert!(
        !config
            .pipeline
            .legacy
            .contains_key("prediction_lead_frames")
    );
    assert!(
        !config
            .pipeline
            .legacy
            .contains_key("velocity_smoothing_frames")
    );

    YamlConfigRepository::new(&path)
        .save_field("pipeline", "residual_cap", Value::from(0.75), 0)
        .unwrap();
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(persisted["schema_version"], 11);
    assert_eq!(persisted["pipeline"]["prediction_lead_ms"], 50.0);
    assert!(
        persisted["pipeline"]
            .get("prediction_lead_frames")
            .is_none()
    );
    assert!(
        persisted["pipeline"]
            .get("velocity_smoothing_frames")
            .is_none()
    );
}

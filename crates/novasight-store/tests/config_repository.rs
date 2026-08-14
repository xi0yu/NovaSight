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

    let config = YamlConfigRepository::load(&path).unwrap();

    assert_eq!(config.schema_version, 13);
    assert_eq!(config.pipeline.p_response_scale, 0.20);
    assert_eq!(config.pipeline.p_response_boost, 0.50);
    assert_eq!(config.pipeline.p_response_curve_shape, 1.0);
    assert_eq!(config.pipeline.max_output_x_counts, 127.0);
    assert_eq!(config.pipeline.max_output_y_counts, 127.0);
    assert_eq!(config.pipeline.prediction_lead_ms, 16.0);
    assert_eq!(config.pipeline.prediction_cap_px, 10.0);
    assert!(config.pipeline.prediction_enabled);

    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(persisted["pipeline"]["atan_scale_counts"].is_null());
}

#[test]
fn older_schema_uses_current_prediction_defaults() {
    let directory = TempDirectory::new();
    let path = directory.join("schema-ten-current-prediction.yaml");
    fs::write(
        &path,
        r#"schema_version: 10
limits:
  stream_fps: 40
pipeline:
  prediction_enabled: true
  atan_scale_counts: 999.0
"#,
    )
    .unwrap();

    let config = YamlConfigRepository::load(&path).unwrap();

    assert_eq!(config.schema_version, 13);
    assert_eq!(config.pipeline.prediction_lead_ms, 16.0);
    assert!(config.pipeline.extra.is_empty());

    YamlConfigRepository::new(&path)
        .save_field("pipeline", "max_output_x_counts", Value::from(128.0), 0)
        .unwrap();
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(persisted["schema_version"], 13);
    assert_eq!(persisted["pipeline"]["prediction_lead_ms"], 16.0);
    assert_eq!(persisted["pipeline"]["p_response_boost"], 0.5);
    assert_eq!(persisted["pipeline"]["prediction_cap_px"], 10.0);
    assert!(persisted["pipeline"]["atan_scale_counts"].is_null());
}

#[test]
fn retired_velocity_change_field_migrates_on_load_and_save() {
    let directory = TempDirectory::new();
    let path = directory.join("retired-velocity-change.yaml");
    fs::write(
        &path,
        r#"schema_version: 12
revision: 0
limits:
  stream_fps: 40
pipeline:
  prediction_enabled: true
  velocity_change_base_px_ms: 0.42
  velocity_change_relative: 0.61
"#,
    )
    .unwrap();

    let config = YamlConfigRepository::load(&path).unwrap();

    assert_eq!(config.pipeline.velocity_spread_base_px_ms, 0.42);
    assert_eq!(config.pipeline.velocity_spread_relative, 0.61);
    assert!(config.pipeline.extra.is_empty());

    YamlConfigRepository::new(&path)
        .save_field("pipeline", "max_output_x_counts", Value::from(128.0), 0)
        .unwrap();
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(persisted["pipeline"]["velocity_spread_base_px_ms"], 0.42);
    assert_eq!(persisted["pipeline"]["velocity_spread_relative"], 0.61);
    assert!(persisted["pipeline"]["velocity_change_base_px_ms"].is_null());
    assert!(persisted["pipeline"]["velocity_change_relative"].is_null());
}

#[test]
fn retired_velocity_change_field_migrates_on_replacement() {
    let directory = TempDirectory::new();
    let path = directory.join("replace-retired-velocity-change.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let replacement: Value = serde_yaml::from_str(
        r#"revision: 0
pipeline:
  velocity_change_base_px_ms: 0.37
  velocity_change_relative: 0.58
"#,
    )
    .unwrap();

    let config = YamlConfigRepository::new(&path)
        .replace_document(replacement, 0)
        .unwrap();

    assert_eq!(config.pipeline.velocity_spread_base_px_ms, 0.37);
    assert_eq!(config.pipeline.velocity_spread_relative, 0.58);
    assert!(config.pipeline.extra.is_empty());
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(persisted["pipeline"]["velocity_spread_base_px_ms"], 0.37);
    assert_eq!(persisted["pipeline"]["velocity_spread_relative"], 0.58);
    assert!(persisted["pipeline"]["velocity_change_base_px_ms"].is_null());
    assert!(persisted["pipeline"]["velocity_change_relative"].is_null());
}

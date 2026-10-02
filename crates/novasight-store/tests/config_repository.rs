use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_store::config::YamlConfigRepository;
use serde_yaml::Value;

static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[test]
fn response_reference_defaults_off_and_validates_saved_updates() {
    let directory = TempDirectory::new();
    let path = directory.join("response-time.yaml");
    fs::write(
        &path,
        "schema_version: 17\npipeline:\n  p_response_scale: 0.42\n",
    )
    .unwrap();
    let loaded = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(loaded.pipeline.response_reference_hz, 0.0);
    let repository = YamlConfigRepository::new(&path);
    for value in [-1.0, 241.0] {
        assert!(
            repository
                .save_field(
                    "pipeline",
                    "response_reference_hz",
                    Value::from(value),
                    loaded.revision
                )
                .is_err()
        );
    }
    repository
        .save_field(
            "pipeline",
            "response_reference_hz",
            Value::from(60.0),
            loaded.revision,
        )
        .unwrap();
    let saved = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(saved.pipeline.response_reference_hz, 60.0);
    assert_eq!(saved.pipeline.p_response_scale, 0.42);
}

#[test]
fn retired_distance_gain_is_removed_without_changing_kp_or_ramp() {
    let directory = TempDirectory::new();
    let path = directory.join("distance-gain.yaml");
    fs::write(&path, "schema_version: 17\npipeline:\n  p_response_scale: 0.42\n  entry_ramp_ms: 100\n  p_response_boost: 2.0\n  p_response_curve_shape: 1.5\n").unwrap();
    let loaded = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(loaded.pipeline.p_response_scale, 0.42);
    assert_eq!(loaded.pipeline.entry_ramp_ms, 100.0);
    let repository = YamlConfigRepository::new(&path);
    let before = fs::read(&path).unwrap();
    for key in ["p_response_boost", "p_response_curve_shape"] {
        assert!(!loaded.pipeline.extra.contains_key(key));
        assert!(
            repository
                .save_field("pipeline", key, Value::from(1.0), loaded.revision)
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), before);
    }
    repository
        .save_field(
            "pipeline",
            "entry_ramp_ms",
            Value::from(100.0),
            loaded.revision,
        )
        .unwrap();
    let document = fs::read_to_string(&path).unwrap();
    assert!(!document.contains("p_response_boost"));
    assert!(!document.contains("p_response_curve_shape"));
    let saved = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(saved.pipeline.p_response_scale, 0.42);
    assert_eq!(saved.pipeline.entry_ramp_ms, 100.0);
}

#[test]
fn retired_recoil_is_removed_but_legacy_trigger_delay_survives() {
    for canonical in [false, true] {
        let directory = TempDirectory::new();
        let path = directory.join("retired-recoil.yaml");
        let pipeline = if canonical {
            "pipeline:\n  fire_delay_enabled: false\n  fire_delay_ms: 12\n"
        } else {
            ""
        };
        fs::write(&path, format!(
            "schema_version: 17\n{pipeline}control:\n  recoil:\n    enabled: true\n    interval_ms: 0\n    y_counts: -1\n    fire_delay_enabled: true\n    fire_delay_ms: 40\n"
        )).unwrap();
        let repository = YamlConfigRepository::new(&path);
        let loaded = YamlConfigRepository::load(&path).unwrap();
        assert!(!loaded.control.extra.contains_key("recoil"));
        assert_eq!(loaded.pipeline.fire_delay_enabled, !canonical);
        assert_eq!(
            loaded.pipeline.fire_delay_ms,
            if canonical { 12 } else { 40 }
        );
        let before = fs::read(&path).unwrap();
        let error = repository
            .save_field("control", "recoil", Value::Null, loaded.revision)
            .unwrap_err();
        assert_eq!(error.code(), "CONFIG_UNSUPPORTED_CONFIG_KEY");
        assert_eq!(fs::read(&path).unwrap(), before);
        let saved = repository
            .save_field(
                "pipeline",
                "target_fov_radius_px",
                Value::from(240.0),
                loaded.revision,
            )
            .unwrap();
        assert_eq!(saved.pipeline.fire_delay_ms, loaded.pipeline.fire_delay_ms);
        assert!(!fs::read_to_string(&path).unwrap().contains("recoil:"));
        let reloaded = YamlConfigRepository::load(&path).unwrap();
        assert_eq!(
            reloaded.pipeline.fire_delay_ms,
            loaded.pipeline.fire_delay_ms
        );
    }
}

#[test]
fn search_radius_migration_preserves_value_and_retires_capsule_settings() {
    let directory = TempDirectory::new();
    let path = directory.join("search-range.yaml");
    fs::write(
        &path,
        "schema_version: 18\npipeline:\n  target_fov_radius_px: 180.0\n  target_range_scale: 5.0\n  target_range_shape: circle\n",
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    let original = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(
        original.schema_version,
        novasight_store::config::CURRENT_SCHEMA_VERSION
    );
    assert_eq!(original.pipeline.target_fov_radius_px, 180.0);
    assert!(!original.pipeline.extra.contains_key("target_range_scale"));
    assert!(!original.pipeline.extra.contains_key("target_range_shape"));
    assert_eq!(
        fs::read(&path).unwrap(),
        before,
        "loading must not rewrite user configuration"
    );
    let repository = YamlConfigRepository::new(&path);
    let revision = YamlConfigRepository::load(&path).unwrap().revision;
    repository
        .save_field(
            "pipeline",
            "target_fov_radius_px",
            Value::from(240.0),
            revision,
        )
        .unwrap();
    let saved = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(saved.pipeline.target_fov_radius_px, 240.0);
    let document = fs::read_to_string(&path).unwrap();
    assert!(!document.contains("target_range_scale"));
    assert!(!document.contains("target_range_shape"));
    for invalid in [0.0, -1.0, 100_000.1, f64::NAN] {
        assert!(
            repository
                .save_field(
                    "pipeline",
                    "target_fov_radius_px",
                    Value::from(invalid),
                    saved.revision
                )
                .is_err()
        );
    }
    for retired in ["target_range_shape", "target_range_scale"] {
        let error = repository
            .save_field("pipeline", retired, Value::from(1.0), saved.revision)
            .unwrap_err();
        assert_eq!(error.code(), "CONFIG_UNSUPPORTED_CONFIG_KEY");
    }
    let unchanged = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(unchanged.revision, saved.revision);
    assert_eq!(unchanged.pipeline.target_fov_radius_px, 240.0);
    assert_eq!(fs::read_to_string(&path).unwrap(), document);
}

struct TempDirectory(PathBuf);

#[test]
fn class_values_persist_and_reject_malformed_pairs() {
    let directory = TempDirectory::new();
    let path = directory.join("classes.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let repository = YamlConfigRepository::new(&path);
    for key in [
        "target_class_weights",
        "target_class_aim_x_ratios",
        "target_class_aim_y_ratios",
    ] {
        let revision = YamlConfigRepository::load(&path).unwrap().revision;
        let saved = repository
            .save_field("pipeline", key, Value::from("0:0.8,1:0.4"), revision)
            .unwrap();
        for invalid in ["0:1.1", "0:NaN", "-1:0.4", "0:0.4,0:0.8"] {
            assert!(
                repository
                    .save_field("pipeline", key, Value::from(invalid), saved.revision)
                    .is_err()
            );
        }
    }
    let loaded = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(loaded.pipeline.target_class_weights, "0:0.8,1:0.4");
    assert_eq!(loaded.pipeline.target_class_aim_x_ratios, "0:0.8,1:0.4");
    assert_eq!(loaded.pipeline.target_class_aim_y_ratios, "0:0.8,1:0.4");
}

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

    assert_eq!(
        config.schema_version,
        novasight_store::config::CURRENT_SCHEMA_VERSION
    );
    assert_eq!(config.pipeline.p_response_scale, 0.20);
    assert_eq!(config.pipeline.max_output_x_counts, 127.0);
    assert_eq!(config.pipeline.max_output_y_counts, 127.0);
    assert_eq!(config.pipeline.prediction_lead_ms, 16.0);
    assert!(config.pipeline.prediction_enabled);
    assert!(config.capture.is_none());

    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(persisted["capture"].is_null());
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

    assert_eq!(
        config.schema_version,
        novasight_store::config::CURRENT_SCHEMA_VERSION
    );
    assert_eq!(config.pipeline.prediction_lead_ms, 16.0);
    assert!(config.pipeline.extra.is_empty());

    YamlConfigRepository::new(&path)
        .save_field("pipeline", "max_output_x_counts", Value::from(128.0), 0)
        .unwrap();
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(
        persisted["schema_version"],
        novasight_store::config::CURRENT_SCHEMA_VERSION
    );
    assert_eq!(persisted["pipeline"]["prediction_lead_ms"], 16.0);
    assert!(persisted["pipeline"].get("prediction_cap_px").is_none());
    assert!(persisted["pipeline"]["atan_scale_counts"].is_null());
}

#[test]
fn retired_velocity_change_fields_are_removed_on_load_and_save() {
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

    assert!(config.pipeline.extra.is_empty());

    YamlConfigRepository::new(&path)
        .save_field("pipeline", "max_output_x_counts", Value::from(128.0), 0)
        .unwrap();
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(persisted["pipeline"]["velocity_spread_base_px_ms"].is_null());
    assert!(persisted["pipeline"]["velocity_spread_relative"].is_null());
    assert!(persisted["pipeline"]["velocity_change_base_px_ms"].is_null());
    assert!(persisted["pipeline"]["velocity_change_relative"].is_null());
}

#[test]
fn retired_velocity_change_fields_are_removed_on_replacement() {
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

    assert!(config.pipeline.extra.is_empty());
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(persisted["pipeline"]["velocity_spread_base_px_ms"].is_null());
    assert!(persisted["pipeline"]["velocity_spread_relative"].is_null());
    assert!(persisted["pipeline"]["velocity_change_base_px_ms"].is_null());
    assert!(persisted["pipeline"]["velocity_change_relative"].is_null());
}

#[test]
fn class_profile_maps_are_replaced_when_a_profile_is_deleted() {
    let directory = TempDirectory::new();
    let path = directory.join("replace-class-profiles.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let repository = YamlConfigRepository::new(&path);
    let old_profiles: Value =
        serde_yaml::from_str("{default: [enemy], obsolete: [enemy]}").unwrap();
    repository
        .save_field("inference", "detection_class_profiles", old_profiles, 0)
        .unwrap();
    let old_roles: Value =
        serde_yaml::from_str("{class_roles: {default: {'0': head}, obsolete: {'0': body}}}")
            .unwrap();
    repository
        .save_field("control", "aim", old_roles, 1)
        .unwrap();
    let replacement: Value = serde_yaml::from_str(
        "revision: 2\ninference:\n  detection_class_profiles:\n    default: [enemy]\ncontrol:\n  aim:\n    class_roles:\n      default: {'0': head}\n",
    )
    .unwrap();

    repository.replace_document(replacement, 2).unwrap();

    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(persisted["inference"]["detection_class_profiles"]["obsolete"].is_null());
    assert!(persisted["control"]["aim"]["class_roles"]["obsolete"].is_null());
}

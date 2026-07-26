use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

use novasight_store::config::{
    AppConfig, ConfigRepository, DeepStreamBackend, InferenceBackend, TriggerMode,
    YamlConfigRepository,
};
use serde_yaml::Value;

static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[test]
fn trigger_mode_is_part_of_the_public_config_contract() {
    assert_eq!(TriggerMode::default(), TriggerMode::Always);
}

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
fn tracked_project_example_preserves_commissioned_hardware_and_legacy_sections() {
    let project_config =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../config/novasight.example.yaml");
    let config = YamlConfigRepository::load(project_config).unwrap();

    assert_eq!(config.server.port, 5174);
    let capture = config.capture.as_ref().unwrap();
    assert_eq!(capture.device, Path::new("/dev/video0"));
    assert_eq!(capture.backend, DeepStreamBackend::DeepstreamNvinfer);
    assert!(capture.latest_only);
    let inference = config.inference.as_ref().unwrap();
    assert_eq!(inference.backend, InferenceBackend::DeepstreamNvinfer);
    assert!(inference.require_gpu);
    assert!(config.device.as_ref().unwrap().auto_connect);
    assert_eq!(config.device.as_ref().unwrap().uuid, "12345678");
    assert!(config.require_production_adapters().is_err());
    assert!(config.legacy.contains_key("source"));
}

#[test]
fn loads_the_complete_rust_owned_example() {
    let example =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");

    let config = YamlConfigRepository::load(example).unwrap();

    assert_eq!(config.schema_version, 6);
    assert_eq!(config.revision, 0);
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 5174);
    assert_eq!(config.server.control_socket, Path::new("@novasightd-dev"));
    assert!(config.replay.enabled);
    assert_eq!(config.replay.frame_interval_ms, 16);
    assert!(!config.replay.output_gate_open);
    assert!(config.control.output_enabled);
    let adapters = config.require_production_adapters().unwrap();
    assert_eq!(adapters.capture.appsink_max_buffers, 1);
    assert_eq!(adapters.inference.deepstream_component_id, 1);
    assert_eq!(adapters.inference.deepstream_probe_element, "primary-infer");
    assert_eq!(
        adapters.inference.deepstream_parser_library,
        Path::new("auto")
    );
    assert!(adapters.device.auto_connect);
    assert_eq!(adapters.device.host, "192.168.2.188");
    assert_eq!(adapters.device.uuid, "12345678");
    assert_eq!(adapters.device.send_timeout_ms, 25);
    assert_eq!(adapters.pipeline.freshness_threshold_ms, 55.0);
    assert_eq!(adapters.pipeline.max_command_age_ms, 55);
    assert_eq!(adapters.pipeline.output_interval_ms, 4);
    assert_eq!(config.paths.database, Path::new("data/novasight.db"));
    assert_eq!(config.paths.license, Path::new("data/license.json"));
}

#[test]
fn production_output_cannot_open_before_hardware_is_commissioned() {
    let directory = TempDirectory::new();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let path = directory.join("unsafe-output.yaml");
    let document = fs::read_to_string(source)
        .unwrap()
        .replace("output_enabled: false", "output_enabled: true")
        .replace("auto_connect: true", "auto_connect: false");
    fs::write(&path, document).unwrap();

    let error = YamlConfigRepository::load(path).unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("control.output_enabled"));
    assert!(error.to_string().contains("hardware.auto_connect"));
}

#[test]
fn field_update_cannot_decommission_hardware_while_output_is_enabled() {
    let directory = TempDirectory::new();
    let path = directory.join("unsafe-decommission.yaml");
    fs::write(
        &path,
        "revision: 0\ncontrol:\n  output_enabled: true\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 192.168.2.188\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n",
    )
    .unwrap();
    let before = fs::read(&path).unwrap();

    let error = YamlConfigRepository::new(&path)
        .save_field("hardware", "auto_connect", Value::Bool(false), 0)
        .unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("control.output_enabled"));
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn document_replacement_cannot_persist_an_uncommissioned_open_gate() {
    let directory = TempDirectory::new();
    let path = directory.join("unsafe-replacement.yaml");
    fs::write(
        &path,
        "revision: 0\ncontrol:\n  output_enabled: false\nhardware:\n  auto_connect: false\n",
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    let replacement: Value = serde_yaml::from_str(
        "revision: 0\ncontrol:\n  output_enabled: true\nhardware:\n  auto_connect: false\n",
    )
    .unwrap();

    let error = YamlConfigRepository::new(&path)
        .replace_document(replacement, 0)
        .unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("control.output_enabled"));
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn document_replacement_cannot_remove_hardware_while_output_is_enabled() {
    let directory = TempDirectory::new();
    let path = directory.join("removed-hardware.yaml");
    fs::write(
        &path,
        "revision: 0\ncontrol:\n  output_enabled: true\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 192.168.2.188\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n",
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    let replacement: Value = serde_yaml::from_str("revision: 0\nhardware: null\n").unwrap();

    let error = YamlConfigRepository::new(&path)
        .replace_document(replacement, 0)
        .unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("control.output_enabled"));
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn production_preflight_rejects_an_unauthenticated_public_http_binding() {
    let directory = TempDirectory::new();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let path = directory.join("public-http.yaml");
    let document = fs::read_to_string(source)
        .unwrap()
        .replace("host: 127.0.0.1", "host: 0.0.0.0");
    fs::write(&path, document).unwrap();

    let config = YamlConfigRepository::load(path).unwrap();
    let error = config.require_production_adapters().unwrap_err();

    assert_eq!(error.field, "server.host");
    assert!(error.message.contains("loopback"));
}

#[test]
fn production_does_not_silently_invent_rust_pipeline_parameters() {
    let directory = TempDirectory::new();
    let example =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let mut document: Value = serde_yaml::from_str(&fs::read_to_string(example).unwrap()).unwrap();
    document
        .as_mapping_mut()
        .unwrap()
        .remove(Value::String("pipeline".to_owned()));
    let path = directory.join("missing-pipeline.yaml");
    fs::write(&path, serde_yaml::to_string(&document).unwrap()).unwrap();

    let config = YamlConfigRepository::load(path).unwrap();
    let error = config.require_production_adapters().unwrap_err();

    assert_eq!(error.field, "pipeline");
    assert!(error.message.contains("must be explicit"));
}

#[test]
fn invalid_rust_pipeline_safety_bounds_fail_during_load() {
    let directory = TempDirectory::new();
    let path = directory.join("invalid-pipeline.yaml");
    fs::write(&path, "pipeline:\n  output_interval_ms: 11\n").unwrap();

    let error = YamlConfigRepository::load(path).unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("pipeline.output_interval_ms"));
}

#[test]
fn missing_file_is_an_error_not_silent_defaults() {
    let directory = TempDirectory::new();
    let error = YamlConfigRepository::load(directory.join("missing.yaml")).unwrap_err();

    assert_eq!(error.code(), "CONFIG_NOT_FOUND");
}

#[test]
fn infrastructure_defaults_do_not_invent_missing_production_adapters() {
    let directory = TempDirectory::new();
    let path = directory.join("minimal.yaml");
    fs::write(&path, "{}\n").unwrap();

    let config = YamlConfigRepository::load(path).unwrap();

    assert_eq!(config.schema_version, 6);
    assert_eq!(config.revision, 0);
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 5174);
    assert_eq!(
        config.server.control_socket,
        Path::new("/run/novasight/novasightd.sock")
    );
    assert!(config.replay.enabled);
    assert_eq!(config.replay.frame_interval_ms, 16);
    assert!(!config.replay.output_gate_open);
    assert_eq!(config.paths.data_dir, Path::new("data"));
    assert_eq!(config.paths.model_dir, Path::new("data/models"));
    assert_eq!(config.paths.database, Path::new("data/novasight.db"));
    assert_eq!(config.paths.license, Path::new("data/license.json"));
    assert_eq!(config.paths.python_executable, Path::new("python3"));
    assert!(config.capture.is_none());
    assert!(config.inference.is_none());
    assert!(config.device.is_none());
    let error = config.require_production_adapters().unwrap_err();
    assert_eq!(error.field, "capture");
}

#[test]
fn schema_three_aggressive_profile_restores_delay_stable_defaults() {
    let directory = TempDirectory::new();
    let path = directory.join("legacy-control.yaml");
    fs::write(
        &path,
        r#"schema_version: 3
pipeline:
  atan_scale_counts: 1024.0
  far_kp: 0.90
  far_max_counts_per_update: 600.0
  near_kp: 0.30
  near_max_counts_per_update: 120.0
"#,
    )
    .unwrap();

    let config = YamlConfigRepository::load(&path).unwrap();

    assert_eq!(config.schema_version, 6);
    assert!(!config.pipeline.prediction_enabled);
    assert_eq!(config.pipeline.atan_scale_counts, 256.0);
    assert_eq!(config.pipeline.far_kp, 0.22);
    assert_eq!(config.pipeline.far_max_counts_per_update, 127.0);
    assert_eq!(config.pipeline.near_kp, 0.20);
    assert_eq!(config.pipeline.near_max_counts_per_update, 72.0);
    assert!(!config.control.humanized_motion.enabled);

    YamlConfigRepository::new(&path)
        .save_field("pipeline", "residual_cap", Value::from(0.75), 0)
        .unwrap();
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(persisted["schema_version"], 6);
    assert_eq!(persisted["pipeline"]["prediction_enabled"], false);
    assert_eq!(persisted["pipeline"]["atan_scale_counts"], 256.0);
    assert_eq!(persisted["pipeline"]["far_kp"], 0.22);
    assert_eq!(persisted["pipeline"]["far_max_counts_per_update"], 127.0);
    assert_eq!(persisted["pipeline"]["near_kp"], 0.20);
    assert_eq!(persisted["pipeline"]["near_max_counts_per_update"], 72.0);
    assert_eq!(persisted["control"]["humanized_motion"]["enabled"], false);
}

#[test]
fn schema_five_generated_far_gain_migrates_without_overwriting_custom_profiles() {
    let directory = TempDirectory::new();
    let generated_path = directory.join("generated-control.yaml");
    fs::write(
        &generated_path,
        r#"schema_version: 5
pipeline:
  atan_scale_counts: 256.0
  far_kp: 0.45
  far_max_counts_per_update: 127.0
  near_kp: 0.22
  near_max_counts_per_update: 72.0
"#,
    )
    .unwrap();
    let migrated = YamlConfigRepository::load(generated_path).unwrap();
    assert_eq!(migrated.schema_version, 6);
    assert_eq!(migrated.pipeline.far_kp, 0.22);
    assert_eq!(migrated.pipeline.near_kp, 0.20);

    let custom_path = directory.join("custom-control.yaml");
    fs::write(
        &custom_path,
        r#"schema_version: 5
pipeline:
  atan_scale_counts: 256.0
  far_kp: 0.30
  far_max_counts_per_update: 127.0
  near_kp: 0.22
  near_max_counts_per_update: 72.0
"#,
    )
    .unwrap();
    let custom = YamlConfigRepository::load(custom_path).unwrap();
    assert_eq!(custom.schema_version, 6);
    assert_eq!(custom.pipeline.far_kp, 0.30);
}

#[test]
fn parse_and_io_errors_have_distinct_stable_codes() {
    let directory = TempDirectory::new();
    let malformed = directory.join("malformed.yaml");
    fs::write(&malformed, "server: [unterminated\n").unwrap();

    let parse_error = YamlConfigRepository::load(&malformed).unwrap_err();
    let io_error = YamlConfigRepository::load(&directory.0).unwrap_err();

    assert_eq!(parse_error.code(), "CONFIG_PARSE_ERROR");
    assert_eq!(io_error.code(), "CONFIG_IO_ERROR");
}

#[test]
fn invalid_production_adapter_configuration_fails_closed() {
    let directory = TempDirectory::new();
    let path = directory.join("invalid-adapter.yaml");
    fs::write(
        &path,
        "capture:\n  latest_only: false\ninference:\n  allow_cpu_fallback: true\n",
    )
    .unwrap();

    let error = YamlConfigRepository::load(path).unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("capture.latest_only"));
}

#[test]
fn present_production_sections_cannot_invent_adapter_parameters() {
    let directory = TempDirectory::new();
    let path = directory.join("empty-adapters.yaml");
    fs::write(&path, "capture: {}\ninference: {}\nhardware: {}\n").unwrap();

    let config = YamlConfigRepository::load(path).unwrap();
    let error = config.require_production_adapters().unwrap_err();

    assert_eq!(error.field, "capture");
    assert!(error.message.contains("must be explicit"));
}

#[test]
fn saving_partial_adapter_sections_cannot_materialize_defaults() {
    let directory = TempDirectory::new();
    let path = directory.join("partial-adapters.yaml");
    let original = "revision: 0\ncapture: {}\ninference: {}\nhardware: {}\n";
    fs::write(&path, original).unwrap();
    let config = YamlConfigRepository::load(&path).unwrap();

    let error = YamlConfigRepository::save(&path, &config, 0).unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("capture"));
    assert_eq!(fs::read_to_string(path).unwrap(), original);
}

#[test]
fn whitespace_only_parser_library_fails_closed() {
    let directory = TempDirectory::new();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let path = directory.join("blank-parser.yaml");
    let document = fs::read_to_string(source).unwrap().replace(
        "deepstream_parser_library: auto",
        "deepstream_parser_library: '   '",
    );
    fs::write(&path, document).unwrap();

    let error = YamlConfigRepository::load(path).unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(
        error
            .to_string()
            .contains("inference.deepstream_parser_library")
    );
}

#[test]
fn each_production_adapter_reports_its_invalid_field() {
    let directory = TempDirectory::new();
    let path = directory.join("invalid-adapter-field.yaml");
    for (document, field) in [
        (
            "inference:\n  allow_cpu_fallback: true\n",
            "inference.device",
        ),
        (
            "hardware:\n  auto_connect: true\n  host: 10.0.0.8\n  uuid: A1B2C3D4\n  port: 0\n",
            "hardware.port",
        ),
        (
            "hardware:\n  auto_connect: true\n  host: 10.0.0.8\n  uuid: A1B2C3D4\n  monitor_port: 1023\n",
            "hardware.monitor_port",
        ),
        ("capture:\n  preference: manual\n", "capture.preference"),
    ] {
        fs::write(&path, document).unwrap();

        let error = YamlConfigRepository::load(&path).unwrap_err();

        assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
        assert!(
            error.to_string().contains(field),
            "expected {field} in {error}"
        );
    }
}

#[test]
fn device_alias_migrates_to_hardware_without_duplicate_fields() {
    let directory = TempDirectory::new();
    let path = directory.join("device-alias.yaml");
    fs::write(
        &path,
        "revision: 0\ndevice:\n  auto_connect: true\n  backend: native_udp\n  host: 10.0.0.8\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n",
    )
    .unwrap();
    let config = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(config.device.as_ref().unwrap().host, "10.0.0.8");

    let saved = YamlConfigRepository::save(&path, &config, 0).unwrap();
    let document: Value = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert_eq!(saved.device.as_ref().unwrap().host, "10.0.0.8");
    assert!(document.get("device").is_none());
    assert_eq!(document["hardware"]["host"].as_str(), Some("10.0.0.8"));
    assert_eq!(
        YamlConfigRepository::load(path)
            .unwrap()
            .device
            .unwrap()
            .host,
        "10.0.0.8"
    );
}

#[test]
fn conflicting_device_and_hardware_sections_fail_closed() {
    let directory = TempDirectory::new();
    let path = directory.join("device-conflict.yaml");
    fs::write(&path, "device: {}\nhardware: {}\n").unwrap();

    let error = YamlConfigRepository::load(path).unwrap_err();

    assert_eq!(error.code(), "CONFIG_VALIDATION_ERROR");
    assert!(error.to_string().contains("cannot both be present"));
}

#[test]
fn save_increments_revision_and_preserves_unknown_root_and_nested_sections() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(
        &path,
        r#"schema_version: 1
revision: 7
server:
  host: 127.0.0.1
  port: 9000
  tls:
    certificate: deploy/server.pem
replay:
  enabled: true
  frame_interval_ms: 20
  output_gate_open: false
  decoder:
    queue_depth: 3
paths:
  data_dir: deploy/data
  model_dir: deploy/models
  database: deploy/novasight.db
  license: deploy/license.json
  python_executable: deploy/python
  plugins:
    directory: deploy/plugins
capture:
  device: /dev/video9
  backend: deepstream_nvinfer
  memory: nvmm
  preference: auto_high_fps
  latest_only: true
  appsink_max_buffers: 1
  queue_leaky: downstream
  width: 0
  height: 0
  fps: 0
  pixel_format: ""
  roi_left: 0
  roi_top: 0
  roi_width: 0
  roi_height: 0
"#,
    )
    .unwrap();

    let mut config = YamlConfigRepository::load(&path).unwrap();
    config.server.port = 5174;
    config.replay.output_gate_open = true;

    let saved = YamlConfigRepository::save(&path, &config, 7).unwrap();

    assert_eq!(saved.revision, 8);
    let document: Value = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(document["revision"].as_u64(), Some(8));
    assert_eq!(document["server"]["port"].as_u64(), Some(5174));
    assert_eq!(
        document["server"]["tls"]["certificate"].as_str(),
        Some("deploy/server.pem")
    );
    assert_eq!(
        document["replay"]["decoder"]["queue_depth"].as_u64(),
        Some(3)
    );
    assert_eq!(
        document["paths"]["plugins"]["directory"].as_str(),
        Some("deploy/plugins")
    );
    assert_eq!(document["capture"]["device"].as_str(), Some("/dev/video9"));

    let reloaded = YamlConfigRepository::load(path).unwrap();
    assert_eq!(reloaded.revision, 8);
    assert_eq!(
        reloaded.capture.as_ref().unwrap().device,
        Path::new("/dev/video9")
    );
    assert!(reloaded.server.legacy.contains_key("tls"));
    assert!(reloaded.replay.legacy.contains_key("decoder"));
    assert!(reloaded.paths.legacy.contains_key("plugins"));
}

#[test]
fn stale_expected_revision_is_typed_and_does_not_modify_the_file() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 4\nserver:\n  port: 6000\nlegacy_key: keep-me\n",
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    let mut config = YamlConfigRepository::load(&path).unwrap();
    config.server.port = 7000;

    let error = YamlConfigRepository::save(&path, &config, 3).unwrap_err();

    assert_eq!(error.code(), "CONFIG_REVISION_CONFLICT");
    assert_eq!(error.expected_revision(), Some(3));
    assert_eq!(error.actual_revision(), Some(4));
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn save_rejects_reserved_keys_in_every_legacy_map_without_modifying_the_file() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\nfuture:\n  enabled: true\n").unwrap();
    let before = fs::read(&path).unwrap();

    for (section, key) in [
        ("root", "revision"),
        ("server", "port"),
        ("replay", "enabled"),
        ("paths", "database"),
    ] {
        let mut config = YamlConfigRepository::load(&path).unwrap();
        let legacy = match section {
            "root" => &mut config.legacy,
            "server" => &mut config.server.legacy,
            "replay" => &mut config.replay.legacy,
            "paths" => &mut config.paths.legacy,
            _ => unreachable!(),
        };
        legacy.insert(key.to_owned(), Value::Null);

        let error = YamlConfigRepository::save(&path, &config, 0).unwrap_err();

        assert_eq!(error.code(), "CONFIG_RESERVED_LEGACY_KEY");
        assert_eq!(fs::read(&path).unwrap(), before);
    }
}

#[test]
fn save_returns_the_same_configuration_that_immediate_reload_observes() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 2\nfuture:\n  enabled: true\nserver:\n  extension: retained\n",
    )
    .unwrap();

    let saved = YamlConfigRepository::save(&path, &AppConfig::default(), 2).unwrap();
    let reloaded = YamlConfigRepository::load(path).unwrap();

    assert_eq!(
        serde_yaml::to_value(saved).unwrap(),
        serde_yaml::to_value(reloaded).unwrap()
    );
}

#[cfg(unix)]
#[test]
fn symlink_configuration_paths_are_rejected_without_replacing_the_link() {
    let directory = TempDirectory::new();
    let referent = directory.join("referent.yaml");
    let link = directory.join("novasight.yaml");
    fs::write(&referent, "revision: 0\nserver:\n  port: 5174\n").unwrap();
    std::os::unix::fs::symlink(&referent, &link).unwrap();

    let load_error = YamlConfigRepository::load(&link).unwrap_err();
    let save_error = YamlConfigRepository::save(&link, &AppConfig::default(), 0).unwrap_err();

    assert_eq!(load_error.code(), "CONFIG_SYMLINK_UNSUPPORTED");
    assert_eq!(save_error.code(), "CONFIG_SYMLINK_UNSUPPORTED");
    assert!(
        fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        fs::read_to_string(referent).unwrap(),
        "revision: 0\nserver:\n  port: 5174\n"
    );
}

#[cfg(unix)]
#[test]
fn hard_linked_configuration_paths_are_rejected_without_splitting_aliases() {
    use std::os::unix::fs::MetadataExt;

    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    let alias = directory.join("novasight-alias.yaml");
    let contents = "revision: 0\nserver:\n  port: 5174\n";
    fs::write(&path, contents).unwrap();
    fs::hard_link(&path, &alias).unwrap();
    let config = YamlConfigRepository::load(&path).unwrap();

    let error = YamlConfigRepository::save(&path, &config, 0).unwrap_err();

    assert_eq!(error.code(), "CONFIG_HARDLINK_UNSUPPORTED");
    assert_eq!(fs::read_to_string(&path).unwrap(), contents);
    assert_eq!(fs::read_to_string(&alias).unwrap(), contents);
    assert_eq!(
        fs::metadata(&path).unwrap().ino(),
        fs::metadata(&alias).unwrap().ino()
    );
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn save_fails_closed_when_destination_has_unpreservable_extended_metadata() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\nserver:\n  port: 5174\n").unwrap();
    #[cfg(target_os = "linux")]
    let attribute = "user.novasight.config-test";
    #[cfg(target_os = "macos")]
    let attribute = "com.novasight.config-test";
    xattr::set(&path, attribute, b"retain-access-policy").unwrap();
    let before = fs::read(&path).unwrap();
    let config = YamlConfigRepository::load(&path).unwrap();

    let error = YamlConfigRepository::save(&path, &config, 0).unwrap_err();

    assert_eq!(error.code(), "CONFIG_SECURITY_METADATA_UNSUPPORTED");
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(
        xattr::get(&path, attribute).unwrap().as_deref(),
        Some(b"retain-access-policy".as_slice())
    );
}

#[test]
fn concurrent_writers_have_one_winner_and_fail_fast_when_the_lock_is_busy() {
    const WRITER_COUNT: usize = 8;

    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\nserver:\n  port: 5174\n").unwrap();
    let config = Arc::new(YamlConfigRepository::load(&path).unwrap());
    let barrier = Arc::new(Barrier::new(WRITER_COUNT));

    let writers: Vec<_> = (0..WRITER_COUNT)
        .map(|_| {
            let path = path.clone();
            let config = Arc::clone(&config);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                YamlConfigRepository::save(path, &config, 0)
            })
        })
        .collect();
    let results: Vec<_> = writers
        .into_iter()
        .map(|writer| writer.join().unwrap())
        .collect();

    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert!(
        results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .all(|error| match error.code() {
                "CONFIG_BUSY" => true,
                "CONFIG_REVISION_CONFLICT" => {
                    error.expected_revision() == Some(0) && error.actual_revision() == Some(1)
                }
                _ => false,
            })
    );
    assert_eq!(YamlConfigRepository::load(path).unwrap().revision, 1);
}

#[test]
fn configuration_save_reports_a_busy_writer_after_a_bounded_lock_wait() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\n").unwrap();
    let config = YamlConfigRepository::load(&path).unwrap();
    let directory_lock = fs::File::open(&directory.0).unwrap();
    directory_lock.lock().unwrap();

    let error = YamlConfigRepository::save(&path, &config, 0).unwrap_err();

    assert_eq!(error.code(), "CONFIG_BUSY");
    assert_eq!(error.path(), path);
}

#[test]
fn successful_save_replaces_the_document_without_leaving_temporary_files() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\n").unwrap();
    let config = YamlConfigRepository::load(&path).unwrap();

    YamlConfigRepository::save(&path, &config, 0).unwrap();

    let entries: Vec<_> = fs::read_dir(&directory.0).unwrap().collect();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].as_ref().unwrap().path(), path);
}

#[test]
fn path_bound_repository_implements_the_store_neutral_contract() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 11\nserver:\n  port: 5175\n").unwrap();
    let repository = YamlConfigRepository::new(&path);

    let mut config = ConfigRepository::load_config(&repository).unwrap();
    config.server.port = 5176;
    let saved = ConfigRepository::save_config(&repository, &config, 11).unwrap();

    assert_eq!(saved.revision, 12);
    assert_eq!(YamlConfigRepository::load(path).unwrap().server.port, 5176);
}

#[test]
fn field_update_is_atomic_and_preserves_unknown_document_fields() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 4\nserver:\n  port: 5174\n  tls:\n    certificate: keep.pem\nfuture:\n  enabled: true\n",
    )
    .unwrap();
    let repository = YamlConfigRepository::new(&path);

    let saved = repository
        .save_field("server", "port", Value::Number(6000_u64.into()), 4)
        .unwrap();

    assert_eq!(saved.revision, 5);
    assert_eq!(saved.server.port, 6000);
    let document: Value = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(document["revision"].as_u64(), Some(5));
    assert_eq!(document["server"]["port"].as_u64(), Some(6000));
    assert_eq!(
        document["server"]["tls"]["certificate"].as_str(),
        Some("keep.pem")
    );
    assert_eq!(document["future"]["enabled"].as_bool(), Some(true));
}

#[test]
fn field_update_rejects_stale_revision_without_modifying_file() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 7\nserver:\n  port: 5174\n").unwrap();
    let before = fs::read(&path).unwrap();

    let error = YamlConfigRepository::new(&path)
        .save_field("server", "port", Value::Number(6000_u64.into()), 6)
        .unwrap_err();

    assert_eq!(error.code(), "CONFIG_REVISION_CONFLICT");
    assert_eq!(error.actual_revision(), Some(7));
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn invalid_field_update_rolls_back_the_complete_document() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\nserver:\n  port: 5174\n").unwrap();
    let before = fs::read(&path).unwrap();

    let error = YamlConfigRepository::new(&path)
        .save_field("server", "port", Value::String("not-a-port".to_owned()), 0)
        .unwrap_err();

    assert_eq!(error.code(), "CONFIG_PARSE_ERROR");
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn field_update_rejects_ambiguous_or_non_mapping_targets() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\nfuture: disabled\n").unwrap();
    let before = fs::read(&path).unwrap();
    let repository = YamlConfigRepository::new(&path);

    let non_mapping = repository
        .save_field("future", "enabled", Value::Bool(true), 0)
        .unwrap_err();
    let alias = repository
        .save_field("device", "host", Value::String("127.0.0.1".to_owned()), 0)
        .unwrap_err();
    let nested = repository
        .save_field("server.tls", "certificate", Value::Null, 0)
        .unwrap_err();

    assert_eq!(non_mapping.code(), "CONFIG_INVALID_FIELD_TARGET");
    assert_eq!(alias.code(), "CONFIG_INVALID_FIELD_TARGET");
    assert_eq!(nested.code(), "CONFIG_INVALID_FIELD_TARGET");
    assert_eq!(fs::read(path).unwrap(), before);
}

#[test]
fn document_replacement_uses_revision_guard_and_preserves_unsubmitted_extensions() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(
        &path,
        "schema_version: 1\nrevision: 5\nserver:\n  port: 5174\n  tls:\n    certificate: keep.pem\nfuture:\n  retained: true\n",
    )
    .unwrap();
    let replacement: Value =
        serde_yaml::from_str("schema_version: 999\nrevision: 5\nserver:\n  port: 7000\n").unwrap();

    let saved = YamlConfigRepository::new(&path)
        .replace_document(replacement, 5)
        .unwrap();

    assert_eq!(saved.schema_version, 6);
    assert_eq!(saved.revision, 6);
    assert_eq!(saved.server.port, 7000);
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(persisted["server"]["tls"]["certificate"], "keep.pem");
    assert_eq!(persisted["future"]["retained"], true);
}

#[test]
fn invalid_document_replacement_does_not_modify_the_file() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "revision: 0\nserver:\n  port: 5174\n").unwrap();
    let before = fs::read(&path).unwrap();

    let error = YamlConfigRepository::new(&path)
        .replace_document(Value::String("invalid".to_owned()), 0)
        .unwrap_err();

    assert_eq!(error.code(), "CONFIG_REPLACEMENT_INVALID");
    assert_eq!(fs::read(path).unwrap(), before);
}

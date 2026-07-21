use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

use novasight_store::config::{ConfigRepository, YamlConfigRepository};
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
fn loads_current_project_yaml_without_dropping_legacy_sections() {
    let project_config =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../config/novasight.yaml");

    let config = YamlConfigRepository::load(project_config).unwrap();

    assert_eq!(config.server.port, 5174);
    assert!(config.legacy.contains_key("capture"));
    assert!(config.legacy.contains_key("inference"));
}

#[test]
fn loads_the_complete_rust_owned_example() {
    let example =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");

    let config = YamlConfigRepository::load(example).unwrap();

    assert_eq!(config.schema_version, 1);
    assert_eq!(config.revision, 0);
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 5174);
    assert!(config.replay.enabled);
    assert_eq!(config.replay.frame_interval_ms, 16);
    assert!(!config.replay.output_gate_open);
    assert_eq!(config.paths.database, Path::new("data/novasight.db"));
    assert_eq!(config.paths.license, Path::new("data/license.json"));
}

#[test]
fn missing_file_is_an_error_not_silent_defaults() {
    let directory = TempDirectory::new();
    let error = YamlConfigRepository::load(directory.join("missing.yaml")).unwrap_err();

    assert_eq!(error.code(), "CONFIG_NOT_FOUND");
}

#[test]
fn defaults_match_the_current_deployment_after_a_document_is_opened() {
    let directory = TempDirectory::new();
    let path = directory.join("minimal.yaml");
    fs::write(&path, "{}\n").unwrap();

    let config = YamlConfigRepository::load(path).unwrap();

    assert_eq!(config.schema_version, 1);
    assert_eq!(config.revision, 0);
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 5174);
    assert!(config.replay.enabled);
    assert_eq!(config.replay.frame_interval_ms, 16);
    assert!(!config.replay.output_gate_open);
    assert_eq!(config.paths.data_dir, Path::new("data"));
    assert_eq!(config.paths.model_dir, Path::new("data/models"));
    assert_eq!(config.paths.database, Path::new("data/novasight.db"));
    assert_eq!(config.paths.license, Path::new("data/license.json"));
    assert_eq!(config.paths.python_executable, Path::new("python3"));
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
    assert!(reloaded.legacy.contains_key("capture"));
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
fn concurrent_writers_with_the_same_expected_revision_have_one_winner() {
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
            .all(|error| error.code() == "CONFIG_REVISION_CONFLICT"
                && error.expected_revision() == Some(0)
                && error.actual_revision() == Some(1))
    );
    assert_eq!(YamlConfigRepository::load(path).unwrap().revision, 1);
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

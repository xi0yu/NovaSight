use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use uuid::Uuid;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_novasightd")
}

fn daemon_command() -> Command {
    Command::new(binary())
}

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let path = PathBuf::from("/tmp").join(format!("novasightd-cli-{}", Uuid::new_v4()));
        fs::create_dir(&path).expect("create temp directory");
        Self(path)
    }

    fn join(&self, path: impl AsRef<Path>) -> PathBuf {
        self.0.join(path)
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn temp_config() -> (TempDirectory, PathBuf) {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    let socket = directory.join("novasightd.sock");
    fs::write(
        &path,
        format!(
            "server:\n  host: 127.0.0.1\n  port: 0\n  control_socket: {}\npaths:\n  data_dir: {}\n  model_dir: {}\n  database: {}\n  license: {}\n",
            socket.display(),
            directory.join("data").display(),
            directory.join("data/models").display(),
            directory.join("data/novasight.db").display(),
            directory.join("license.json").display()
        ),
    )
    .expect("write config");
    (directory, path)
}

fn production_config() -> (TempDirectory, PathBuf) {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    novasight_store::config::YamlConfigRepository::initialize_default(&path)
        .expect("initialize bundled production configuration");
    (directory, path)
}

#[test]
fn help_documents_yaml_check_without_test_runtime_flags() {
    let output = daemon_command().arg("--help").output().expect("run help");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 help");

    assert!(output.status.success());
    assert!(stdout.contains(".config/novasight.yaml"));
    assert!(stdout.contains("--check"));
    assert!(!stdout.contains("--dry-run"));
}

#[test]
fn build_info_is_machine_readable_and_reports_compiled_capabilities() {
    let output = Command::new(binary())
        .arg("--build-info-json")
        .output()
        .expect("read daemon build identity");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let info: serde_json::Value = serde_json::from_slice(&output.stdout).expect("build info JSON");
    assert_eq!(info["schema_version"], 1);
    assert_eq!(info["binary"], "novasightd");
    assert_eq!(info["version"], env!("CARGO_PKG_VERSION"));
    assert!(
        info["source_revision"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
    assert!(info["source_dirty"].is_boolean());
    assert!(
        info["target"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
    assert!(
        info["profile"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
    assert!(info["features"].is_array());
}

#[test]
fn missing_config_exits_nonzero_with_stable_code() {
    let directory = TempDirectory::new();
    let missing = directory.join("missing.yaml");
    let output = daemon_command()
        .args(["--config", missing.to_str().expect("UTF-8 path"), "--check"])
        .output()
        .expect("run missing config");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("CONFIG_NOT_FOUND"));
}

#[test]
fn missing_default_config_is_created_before_preflight() {
    let directory = TempDirectory::new();
    let output = daemon_command()
        .current_dir(&directory.0)
        .arg("--check")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run first-start preflight");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    let config_path = directory.join(".config/novasight.yaml");

    assert!(
        config_path.is_file(),
        "first start should create the default configuration before preflight completion: {stderr}"
    );
    let config = novasight_store::config::YamlConfigRepository::load(config_path)
        .expect("load generated default configuration");
    config
        .require_production_adapters()
        .expect("generated configuration must contain the explicit Jetson contract");
}

#[test]
fn production_check_rejects_incomplete_adapter_configuration() {
    let (_directory, path) = temp_config();
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .output()
        .expect("run production check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("PRODUCTION_CONFIG_INVALID"));
    assert!(stderr.contains("capture"));
}

#[test]
fn normal_mode_fails_closed_when_production_adapter_sections_are_missing() {
    let (_directory, path) = temp_config();
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run production mode");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("PRODUCTION_CONFIG_INVALID"));
    assert!(stderr.contains("capture"));
}

#[test]
fn development_hardware_check_is_not_blocked_by_formal_license_configuration() {
    let (_directory, path) = production_config();
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .env("NOVASIGHT_LICENSE_PUBLIC_KEY", "not a PEM public key")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run production check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(!stderr.contains("LICENSE_PUBLIC_KEY_INVALID"));
    assert!(stderr.contains("PRODUCTION_RUNTIME_UNAVAILABLE"));
}

#[test]
fn development_hardware_check_ignores_an_unreadable_formal_key_file() {
    let (_directory, path) = production_config();
    let missing_key = path.with_file_name("missing-license-public.pem");
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE", &missing_key)
        .output()
        .expect("run production check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(!stderr.contains("LICENSE_PUBLIC_KEY_READ_FAILED"));
    assert!(stderr.contains("PRODUCTION_RUNTIME_UNAVAILABLE"));
}

#[test]
fn valid_production_authority_reaches_the_platform_build_boundary() {
    let (_directory, path) = production_config();
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run production check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("PRODUCTION_RUNTIME_UNAVAILABLE"));
}

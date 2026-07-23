use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::sync::mpsc;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::net::UnixStream;

use uuid::Uuid;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_novasightd")
}

fn daemon_command() -> Command {
    static PYTHON: OnceLock<PathBuf> = OnceLock::new();
    let python = PYTHON.get_or_init(|| {
        let output = Command::new("python3")
            .args(["-c", "import sys; print(sys.executable)"])
            .output()
            .expect("resolve test Python");
        assert!(output.status.success(), "test Python must be runnable");
        PathBuf::from(
            String::from_utf8(output.stdout)
                .expect("UTF-8 Python path")
                .trim(),
        )
    });
    let mut command = Command::new(binary());
    command.arg("--model-job-python").arg(python);
    command
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

#[test]
fn help_documents_yaml_check_and_explicit_dry_run() {
    let output = daemon_command().arg("--help").output().expect("run help");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 help");

    assert!(output.status.success());
    assert!(stdout.contains("/etc/novasight/novasight.yaml"));
    assert!(stdout.contains("--check"));
    assert!(stdout.contains("--dry-run"));
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
fn check_loads_config_and_exits_without_starting_the_daemon() {
    let (_directory, path) = temp_config();
    let output = daemon_command()
        .args([
            "--config",
            path.to_str().expect("UTF-8 path"),
            "--check",
            "--dry-run",
        ])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run check");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");

    assert!(output.status.success());
    assert!(stdout.contains("PASS mode=dry_run"));
    assert!(stdout.contains("hardware_not_started=true"));
}

#[cfg(unix)]
#[test]
fn check_rejects_a_model_worker_that_cannot_execute_its_protocol() {
    use std::os::unix::fs::PermissionsExt;

    let (directory, path) = temp_config();
    let broken_python = directory.join("broken-python");
    fs::write(
        &broken_python,
        "#!/bin/sh\necho worker-import-failed >&2\nexit 17\n",
    )
    .expect("write broken interpreter");
    fs::set_permissions(&broken_python, fs::Permissions::from_mode(0o755))
        .expect("make broken interpreter executable");
    let output = Command::new(binary())
        .args([
            "--model-job-python",
            broken_python.to_str().expect("UTF-8 interpreter"),
            "--config",
            path.to_str().expect("UTF-8 path"),
            "--check",
            "--dry-run",
        ])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("MODEL_INGRESS_PREFLIGHT_FAILED"));
    assert!(stderr.contains("worker-import-failed"));
}

#[cfg(unix)]
#[test]
fn check_rejects_an_incompatible_model_worker_protocol() {
    use std::os::unix::fs::PermissionsExt;

    let (directory, path) = temp_config();
    let incompatible_python = directory.join("incompatible-python");
    fs::write(
        &incompatible_python,
        "#!/bin/sh\necho '{\"protocol\":2,\"worker\":\"other\",\"operations\":[]}'\n",
    )
    .expect("write incompatible interpreter");
    fs::set_permissions(&incompatible_python, fs::Permissions::from_mode(0o755))
        .expect("make incompatible interpreter executable");
    let output = Command::new(binary())
        .args([
            "--model-job-python",
            incompatible_python.to_str().expect("UTF-8 interpreter"),
            "--config",
            path.to_str().expect("UTF-8 path"),
            "--check",
            "--dry-run",
        ])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("MODEL_INGRESS_PREFLIGHT_FAILED"));
    assert!(stderr.contains("incompatible protocol"));
}

#[test]
fn check_rejects_a_missing_named_motion_profile_instead_of_silently_using_builtin() {
    let (_directory, path) = temp_config();
    let mut config = fs::read_to_string(&path).expect("read config");
    config.push_str(
        "control:\n  humanized_motion:\n    enabled: true\n    active_profile: missing-profile\n",
    );
    fs::write(&path, config).expect("write motion config");

    let output = daemon_command()
        .args([
            "--config",
            path.to_str().expect("UTF-8 path"),
            "--check",
            "--dry-run",
        ])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("MOTION_PROFILE_LOAD_FAILED"));
    assert!(stderr.contains("missing-profile"));
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
fn production_check_rejects_a_missing_license_public_key_before_platform_startup() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path")])
        .arg("--check")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run production mode");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("LICENSE_PUBLIC_KEY_MISSING"));
}

#[test]
fn production_check_rejects_an_invalid_license_public_key() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .env("NOVASIGHT_LICENSE_PUBLIC_KEY", "not a PEM public key")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .output()
        .expect("run production check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("LICENSE_PUBLIC_KEY_INVALID"));
}

#[test]
fn production_check_requires_the_configured_instance_lock() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let public_key =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/license-public.pem");
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE", public_key)
        .env_remove("NOVASIGHT_INSTANCE_LOCK")
        .output()
        .expect("run production check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("INSTANCE_LOCK_MISSING"));
}

#[test]
fn valid_production_authority_reaches_the_platform_build_boundary() {
    let directory = TempDirectory::new();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
    let public_key =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/license-public.pem");
    let output = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE", public_key)
        .env("NOVASIGHT_INSTANCE_LOCK", directory.join("instance.lock"))
        .output()
        .expect("run production check");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("PRODUCTION_RUNTIME_UNAVAILABLE"));
}

#[cfg(unix)]
#[test]
fn explicit_dry_run_exits_cleanly_on_sigterm() {
    let (_directory, path) = temp_config();
    let mut child = daemon_command()
        .args(["--config", path.to_str().expect("UTF-8 path"), "--dry-run"])
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .env_remove("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dry-run daemon");
    let stderr = child.stderr.take().expect("capture stderr");
    let mut child = ChildGuard(Some(child));
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut log = Vec::new();
        let mut ready = None;
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let mut address = None;
            let mut socket = None;
            for field in line.split_whitespace() {
                if let Some(value) = field.strip_prefix("address=") {
                    address = value.parse::<SocketAddr>().ok();
                } else if let Some(value) = field.strip_prefix("socket=") {
                    socket = Some(PathBuf::from(value));
                }
            }
            log.push(line);
            if let Some(value) = address.zip(socket) {
                ready = Some(value);
                break;
            }
        }
        let _ = ready_tx.send((ready, log));
    });
    let (ready, log) = ready_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("readiness timeout");
    let (address, control_socket) =
        ready.unwrap_or_else(|| panic!("daemon exited before readiness: {}", log.join(" | ")));
    assert!(address.ip().is_loopback());
    assert_ne!(address.port(), 0);
    assert!(control_socket.exists(), "control socket was not created");
    let (status, health) = unix_http_request(&control_socket, "GET", "/healthz");
    assert_eq!(status, 200);
    assert!(health.contains("\"ok\":true"));

    let (status, license) = unix_http_request(&control_socket, "GET", "/api/license");
    assert_eq!(status, 200);
    assert!(license.contains("\"configured\":false"));
    let (status, blocked) = unix_http_request(&control_socket, "GET", "/api/v1/status");
    assert_eq!(status, 401);
    assert!(blocked.contains("license required"));
    let (status, activated) = unix_http_json_request(
        &control_socket,
        "POST",
        "/api/license/activate",
        r#"{"key":"NOVASIGHT-TEST-MAX-ACCESS-2026"}"#,
    );
    assert_eq!(status, 200);
    assert!(activated.contains("\"valid\":true"));
    let (status, initial) = unix_http_request(&control_socket, "GET", "/api/v1/status");
    assert_eq!(status, 200);
    assert!(initial.contains("\"state\":\"stopped\""));
    let (status, started) = unix_http_request(&control_socket, "POST", "/api/v1/runtime/start");
    assert_eq!(status, 200);
    assert!(started.contains("\"state\":\"running\""));
    let (status, shared) = unix_http_request(&control_socket, "GET", "/api/v1/status");
    assert_eq!(status, 200);
    assert!(shared.contains("\"state\":\"running\""));

    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .expect("send SIGTERM");
    assert!(signal.success());
    let status = child
        .wait_timeout(Duration::from_secs(5))
        .expect("wait for graceful exit");
    assert!(status.success(), "novasightd exited with {status}");
    assert!(
        !control_socket.exists(),
        "control socket was not removed after shutdown"
    );
}

#[cfg(unix)]
fn unix_http_request(socket: &Path, method: &str, path: &str) -> (u16, String) {
    unix_http_request_with_body(socket, method, path, "", None)
}

#[cfg(unix)]
fn unix_http_json_request(socket: &Path, method: &str, path: &str, body: &str) -> (u16, String) {
    unix_http_request_with_body(socket, method, path, body, Some("application/json"))
}

#[cfg(unix)]
fn unix_http_request_with_body(
    socket: &Path,
    method: &str,
    path: &str,
    body: &str,
    content_type: Option<&str>,
) -> (u16, String) {
    let mut stream = UnixStream::connect(socket).expect("connect Unix control socket");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let content_type = content_type
        .map(|value| format!("Content-Type: {value}\r\n"))
        .unwrap_or_default();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\n{content_type}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("write Unix HTTP request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read Unix HTTP response");
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .expect("Unix HTTP response headers");
    let status = headers
        .split_whitespace()
        .nth(1)
        .expect("Unix HTTP status")
        .parse()
        .expect("numeric Unix HTTP status");
    (status, body.to_owned())
}

struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn id(&self) -> u32 {
        self.0.as_ref().expect("child present").id()
    }

    fn wait_timeout(&mut self, timeout: Duration) -> std::io::Result<std::process::ExitStatus> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let child = self.0.as_mut().expect("child present");
            if let Some(status) = child.try_wait()? {
                self.0.take();
                return Ok(status);
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let status = child.wait()?;
                self.0.take();
                panic!("novasightd did not exit within {timeout:?}; killed with {status}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

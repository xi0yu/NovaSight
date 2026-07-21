use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use uuid::Uuid;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_novasightd")
}

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("novasightd-cli-{}", Uuid::new_v4()));
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
    fs::write(&path, "{}\n").expect("write config");
    (directory, path)
}

#[test]
fn help_documents_yaml_check_and_explicit_dry_run() {
    let output = Command::new(binary())
        .arg("--help")
        .output()
        .expect("run help");
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
    let output = Command::new(binary())
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
    let output = Command::new(binary())
        .args(["--config", path.to_str().expect("UTF-8 path"), "--check"])
        .output()
        .expect("run check");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");

    assert!(output.status.success());
    assert!(stdout.contains("PASS config readable"));
}

#[test]
fn normal_mode_fails_closed_without_a_real_device_backend() {
    let (_directory, path) = temp_config();
    let output = Command::new(binary())
        .args(["--config", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run production mode");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("DEVICE_BACKEND_NOT_CONFIGURED"));
}

#[cfg(unix)]
#[test]
fn explicit_dry_run_exits_cleanly_on_sigterm() {
    let (_directory, path) = temp_config();
    let mut child = Command::new(binary())
        .args(["--config", path.to_str().expect("UTF-8 path"), "--dry-run"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dry-run daemon");
    let stderr = child.stderr.take().expect("capture stderr");
    let mut child = ChildGuard(Some(child));
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let ready = BufReader::new(stderr)
            .lines()
            .map_while(Result::ok)
            .any(|line| line.contains("novasightd ready mode=dry-run"));
        let _ = ready_tx.send(ready);
    });
    assert!(
        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("readiness timeout"),
        "daemon exited before readiness"
    );

    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .expect("send SIGTERM");
    assert!(signal.success());
    let status = child.wait().expect("wait for graceful exit");
    assert!(status.success(), "novasightd exited with {status}");
}

struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn id(&self) -> u32 {
        self.0.as_ref().expect("child present").id()
    }

    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.0.take().expect("child present").wait()
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

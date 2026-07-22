use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::net::UnixStream;

use uuid::Uuid;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_novasightd")
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
            "server:\n  host: 127.0.0.1\n  port: 0\n  control_socket: {}\n",
            socket.display()
        ),
    )
    .expect("write config");
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
fn normal_mode_fails_closed_when_production_adapter_sections_are_missing() {
    let (_directory, path) = temp_config();
    let output = Command::new(binary())
        .args(["--config", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run production mode");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("PRODUCTION_CONFIG_INVALID"));
    assert!(stderr.contains("capture"));
}

#[test]
fn complete_adapter_config_reaches_the_unimplemented_device_boundary() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/novasightd.example.yaml");
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
    assert!(control_socket.exists(), "control socket was not created");

    let (status, initial) = http_request(address, "GET", "/api/v1/status");
    assert_eq!(status, 200);
    assert!(initial.contains("\"state\":\"stopped\""));
    let (status, started) = unix_http_request(&control_socket, "POST", "/api/v1/runtime/start");
    assert_eq!(status, 200);
    assert!(started.contains("\"state\":\"running\""));
    let (status, shared) = http_request(address, "GET", "/api/v1/status");
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

fn http_request(address: SocketAddr, method: &str, path: &str) -> (u16, String) {
    let mut stream =
        TcpStream::connect_timeout(&address, Duration::from_secs(5)).expect("connect HTTP server");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
    .expect("write HTTP request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read HTTP response");
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .expect("HTTP response headers");
    let status = headers
        .split_whitespace()
        .nth(1)
        .expect("HTTP status")
        .parse()
        .expect("numeric HTTP status");
    (status, body.to_owned())
}

#[cfg(unix)]
fn unix_http_request(socket: &Path, method: &str, path: &str) -> (u16, String) {
    let mut stream = UnixStream::connect(socket).expect("connect Unix control socket");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
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

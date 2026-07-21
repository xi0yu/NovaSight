use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[test]
fn help_describes_the_external_config_cli() {
    let output = binary().arg("--help").output().expect("run --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage: relink_server [OPTIONS]"));
    assert!(stdout.contains("--config <CONFIG>"));
    assert!(stdout.contains("config/novasight.yaml"));
}

#[test]
fn missing_config_exits_nonzero_with_stable_reason() {
    let missing = unique_path("missing", "yaml");
    let output = run_with_config(&missing);

    assert!(!output.status.success());
    assert_stderr_code(&output, "CONFIG_NOT_FOUND");
}

#[test]
fn invalid_config_exits_nonzero_with_stable_reason() {
    let config = TempConfig::raw("invalid", "replay: [not valid yaml");
    let output = run_with_config(config.path());

    assert!(!output.status.success());
    assert_stderr_code(&output, "CONFIG_PARSE_ERROR");
}

#[test]
fn disabled_replay_is_rejected_before_binding() {
    let occupied = TcpListener::bind("127.0.0.1:0").expect("reserve address");
    let port = occupied.local_addr().expect("reserved address").port();
    let config = TempConfig::server("disabled-replay", port, false, false);
    let output = run_with_config(config.path());

    assert!(!output.status.success());
    assert_stderr_code(&output, "REPLAY_DISABLED");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("SERVER_BIND_FAILED"));
}

#[test]
fn open_output_gate_aborts_startup() {
    let config = TempConfig::server("open-output-gate", unused_port(), true, true);
    let output = run_with_config(config.path());

    assert!(!output.status.success());
    assert_stderr_code(&output, "OUTPUT_GATE_OPEN");
}

#[test]
fn bind_failure_exits_nonzero_with_stable_reason() {
    let occupied = TcpListener::bind("127.0.0.1:0").expect("reserve address");
    let port = occupied.local_addr().expect("reserved address").port();
    let config = TempConfig::server("bind-failure", port, true, false);
    let output = run_with_config(config.path());

    assert!(!output.status.success());
    assert_stderr_code(&output, "SERVER_BIND_FAILED");
}

#[cfg(unix)]
#[test]
fn actual_binary_composes_replay_api_websocket_and_sigterm_shutdown() {
    let port = unused_port();
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let config = TempConfig::server("composition-smoke", port, true, false);
    let child = binary()
        .args(["--config", config.path().to_str().expect("utf-8 path")])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn relink_server");
    let mut server = ChildGuard::new(child);

    wait_until_healthy(address, Duration::from_secs(5));
    let (status, health) = http_request(address, "GET", "/healthz");
    assert_eq!(status, 200);
    assert_eq!(health, "{\"ok\":true}");

    let (_, first_start) = http_request(address, "POST", "/api/runtime/start");
    let first_start: Value = serde_json::from_str(&first_start).expect("first start JSON");
    assert_eq!(first_start["accepted"], true);
    assert_eq!(first_start["epoch"], 1);

    let (_, duplicate_start) = http_request(address, "POST", "/api/runtime/start");
    let duplicate_start: Value =
        serde_json::from_str(&duplicate_start).expect("duplicate start JSON");
    assert_eq!(duplicate_start["accepted"], true);
    assert_eq!(duplicate_start["epoch"], 1);

    let state = wait_for_processed_batch(address, Duration::from_secs(2));
    assert_eq!(state["source"], "replay");
    assert_eq!(state["running"], true);
    assert_eq!(state["pipeline"]["epoch"], 1);
    assert!(state["pipeline"]["processed_batches"].as_u64().unwrap_or(0) >= 1);
    assert_eq!(state["executor"]["selected"], "dry_run");
    assert_eq!(state["executor"]["executors"]["kmnet"]["available"], false);

    let status_frame = websocket_status_frame(address);
    assert_eq!(status_frame["kind"], "runtime_snapshot");
    assert_eq!(status_frame["state"]["pipeline"]["epoch"], 1);

    let (_, stopped) = http_request(address, "POST", "/api/runtime/stop");
    let stopped: Value = serde_json::from_str(&stopped).expect("stop JSON");
    assert_eq!(stopped["running"], false);

    let signal_status = Command::new("kill")
        .args(["-TERM", &server.id().to_string()])
        .status()
        .expect("send SIGTERM");
    assert!(signal_status.success());
    let status = server.wait().expect("wait for graceful process exit");
    assert!(status.success(), "relink_server exited with {status}");
}

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_relink_server"))
}

fn run_with_config(path: &Path) -> Output {
    binary()
        .args(["--config", path.to_str().expect("utf-8 path")])
        .output()
        .expect("run relink_server")
}

fn assert_stderr_code(output: &Output, code: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(code),
        "stderr did not contain {code}: {stderr}"
    );
}

fn unused_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("reserve ephemeral port")
        .local_addr()
        .expect("ephemeral address")
        .port()
}

fn wait_until_healthy(address: SocketAddr, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok((200, body)) = try_http_request(address, "GET", "/healthz")
            && body == "{\"ok\":true}"
        {
            return;
        }
        assert!(Instant::now() < deadline, "server did not become healthy");
        thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_processed_batch(address: SocketAddr, timeout: Duration) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        let (_, body) = http_request(address, "GET", "/api/runtime/state");
        let state: Value = serde_json::from_str(&body).expect("runtime state JSON");
        if state["pipeline"]["processed_batches"].as_u64().unwrap_or(0) >= 1 {
            return state;
        }
        assert!(Instant::now() < deadline, "replay batch was not processed");
        thread::yield_now();
    }
}

fn http_request(address: SocketAddr, method: &str, path: &str) -> (u16, String) {
    try_http_request(address, method, path).expect("HTTP request")
}

fn try_http_request(
    address: SocketAddr,
    method: &str,
    path: &str,
) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(100))?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("HTTP response had no header terminator"))?;
    let status = headers
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| std::io::Error::other("HTTP response had no status"))?;
    Ok((status, body.to_owned()))
}

fn websocket_status_frame(address: SocketAddr) -> Value {
    let mut stream =
        TcpStream::connect_timeout(&address, Duration::from_secs(1)).expect("connect WebSocket");
    stream
        .set_read_timeout(Some(Duration::from_secs(1)))
        .expect("set WebSocket timeout");
    write!(
        stream,
        "GET /ws/status HTTP/1.1\r\nHost: {address}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    .expect("write WebSocket handshake");

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .expect("read WebSocket status");
    assert!(
        status_line.contains(" 101 "),
        "WebSocket upgrade failed: {status_line}"
    );
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect("read WebSocket header");
        if line == "\r\n" {
            break;
        }
    }

    let payload = read_server_websocket_text(&mut reader);
    serde_json::from_slice(&payload).expect("WebSocket status JSON")
}

fn read_server_websocket_text(reader: &mut impl Read) -> Vec<u8> {
    let mut header = [0_u8; 2];
    reader
        .read_exact(&mut header)
        .expect("read WebSocket frame header");
    assert_eq!(header[0] & 0x0f, 1, "expected text WebSocket frame");
    assert_eq!(header[1] & 0x80, 0, "server frame must not be masked");
    let length = match header[1] & 0x7f {
        length @ 0..=125 => u64::from(length),
        126 => {
            let mut bytes = [0_u8; 2];
            reader
                .read_exact(&mut bytes)
                .expect("read 16-bit frame length");
            u64::from(u16::from_be_bytes(bytes))
        }
        127 => {
            let mut bytes = [0_u8; 8];
            reader
                .read_exact(&mut bytes)
                .expect("read 64-bit frame length");
            u64::from_be_bytes(bytes)
        }
        _ => unreachable!(),
    };
    let mut payload = vec![0_u8; usize::try_from(length).expect("frame length fits usize")];
    reader
        .read_exact(&mut payload)
        .expect("read WebSocket payload");
    payload
}

struct TempConfig {
    path: PathBuf,
}

impl TempConfig {
    fn raw(label: &str, content: &str) -> Self {
        let path = unique_path(label, "yaml");
        fs::write(&path, content).expect("write temporary config");
        Self { path }
    }

    fn server(label: &str, port: u16, replay_enabled: bool, output_gate_open: bool) -> Self {
        Self::raw(
            label,
            &format!(
                "server:\n  host: 127.0.0.1\n  port: {port}\nreplay:\n  enabled: {replay_enabled}\n  frame_interval_ms: 1\n  output_gate_open: {output_gate_open}\n"
            ),
        )
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempConfig {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn unique_path(label: &str, extension: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "novasight-relink-server-{label}-{}-{timestamp}-{id}.{extension}",
        std::process::id()
    ))
}

struct ChildGuard {
    child: Option<Child>,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn id(&self) -> u32 {
        self.child.as_ref().expect("child present").id()
    }

    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.take().expect("child present").wait()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

use std::path::{Path, PathBuf};
use std::process::Command;

use novasight_api::build_control_router;
use novasight_runtime::{PipelineState, RuntimeSnapshot, RuntimeSupervisor};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_novasightctl")
}

struct SocketPath(PathBuf);

impl SocketPath {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "novasightctl-{}-{}.sock",
            std::process::id(),
            unique_id()
        )))
    }
}

impl Drop for SocketPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[test]
fn help_documents_the_real_command_surface() {
    let output = Command::new(binary())
        .arg("--help")
        .output()
        .expect("run help");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 help");

    assert!(output.status.success());
    for command in ["status", "start", "stop", "restart", "emergency-stop"] {
        assert!(stdout.contains(command), "help omitted {command}");
    }
    assert!(!stdout.contains("diagnose"));
}

#[test]
fn missing_daemon_exits_nonzero_with_a_stable_error_code() {
    let socket = SocketPath::new();
    let output = run_cli(&socket.0, "status");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("daemon_connect_failed"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commands_print_the_snapshot_returned_by_the_daemon() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let server_runtime = runtime.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_control_router(server_runtime))
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let started = tokio::task::spawn_blocking(move || run_cli(&socket_path, "start"))
        .await
        .expect("join CLI process");
    assert!(started.status.success());
    let snapshot: RuntimeSnapshot = serde_json::from_slice(&started.stdout).expect("snapshot JSON");
    assert_eq!(snapshot.pipeline.state, PipelineState::Running);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Running);

    let socket_path = socket.0.clone();
    let stopped = tokio::task::spawn_blocking(move || run_cli(&socket_path, "emergency-stop"))
        .await
        .expect("join CLI process");
    assert!(stopped.status.success());
    let snapshot: RuntimeSnapshot = serde_json::from_slice(&stopped.stdout).expect("snapshot JSON");
    assert_eq!(snapshot.pipeline.state, PipelineState::Stopped);

    server.abort();
    runtime
        .shutdown_daemon()
        .await
        .expect("shutdown supervisor");
    supervisor.join().await.expect("join supervisor");
}

fn run_cli(socket: &Path, command: &str) -> std::process::Output {
    Command::new(binary())
        .args(["--socket", socket.to_str().expect("UTF-8 socket"), command])
        .output()
        .expect("run novasightctl")
}

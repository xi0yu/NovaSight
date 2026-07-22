use std::path::{Path, PathBuf};
use std::process::Command;

use novasight_api::{build_control_router, build_control_router_with_services};
use novasight_runtime::{
    AppConfig, ConfigService, ConfigUpdate, PipelineState, RuntimeSnapshot, RuntimeSupervisor,
};
use novasight_store::config::YamlConfigRepository;

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
    for command in [
        "status",
        "start",
        "stop",
        "restart",
        "emergency-stop",
        "config",
    ] {
        assert!(stdout.contains(command), "help omitted {command}");
    }
    assert!(!stdout.contains("diagnose"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_commands_share_the_daemon_revisioned_config_service() {
    let socket = SocketPath::new();
    let config_path = socket.0.with_extension("yaml");
    std::fs::write(&config_path, "revision: 3\nserver:\n  port: 5174\n").unwrap();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let config = ConfigService::new(
        &config_path,
        YamlConfigRepository::load(&config_path).unwrap(),
    );
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let shown =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["config", "show"]))
            .await
            .unwrap();
    assert!(shown.status.success());
    let shown: AppConfig = serde_json::from_slice(&shown.stdout).unwrap();
    assert_eq!(shown.revision, 3);

    let socket_path = socket.0.clone();
    let updated = tokio::task::spawn_blocking(move || {
        run_cli_args(
            &socket_path,
            &[
                "config",
                "set",
                "server",
                "port",
                "6002",
                "--expected-revision",
                "3",
            ],
        )
    })
    .await
    .unwrap();
    assert!(updated.status.success());
    let updated: ConfigUpdate = serde_json::from_slice(&updated.stdout).unwrap();
    assert_eq!(updated.config.revision, 4);
    assert_eq!(updated.config.server.port, 6002);
    assert!(updated.restart_required);
    assert!(!updated.applied);

    server.abort();
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_file(config_path).unwrap();
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
    run_cli_args(socket, &[command])
}

fn run_cli_args(socket: &Path, arguments: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(["--socket", socket.to_str().expect("UTF-8 socket")])
        .args(arguments)
        .output()
        .expect("run novasightctl")
}
